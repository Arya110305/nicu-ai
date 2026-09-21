"""
Streaming parallel NICU clinical normalizer.

NEW PIPELINE ARCHITECTURE
=========================

The Normalizer consumes temporary LLM batches from:

    llm/temp/<patient>/batches/

Each LLM clinical batch is:

    *.txt

and becomes available only when the matching:

    *.ready

marker exists.

The Normalizer processes those batches while the LLM is still
working.

Temporary normalized results are stored under:

    normalizer/temp/<patient>/

The final combined Normalizer output will be:

    normalizer/output/
        NORMALIZER_<patient>.txt

IMPORTANT
=========

The Discharge Summary stage must NEVER consume individual
Normalizer batches.

The Discharge Summary may start ONLY after:

    1. Every expected LLM batch has been normalized.
    2. Every normalized batch has completed successfully.
    3. The normalized batches have been merged in source order.
    4. NORMALIZER_<patient>.txt has been completely written.
    5. The final Normalizer completion marker has been created.

Temporary files are NOT deleted by this module.

api/pipeline.py will clean temporary OCR, LLM and Normalizer
workspaces only after the complete pipeline succeeds.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback

from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)

from dataclasses import dataclass

from pathlib import Path

from typing import (
    Any,
    Final,
)

import requests


from normalizer.clinical_terms import (
    normalize_clinical_terms,
)

from normalizer.normalizer_model import (
    FILE_ENCODING,
    KEEP_ALIVE,
    MODEL_NAME,
    NORMALIZER_MAX_WORKERS,
    NORMALIZER_POLL_SECONDS,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    OPTIONS,
    RETRIES,
    RETRY_DELAY_SECONDS,
    TIMEOUT_SECONDS,
)

from normalizer.prompts import (
    NORMALIZER_PROMPT,
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT: Final[Path] = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

NORMALIZER_ROOT: Final[Path] = (
    Path(__file__)
    .resolve()
    .parent
)

LLM_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "llm"
)

LLM_TEMP_FOLDER: Final[Path] = (
    LLM_ROOT
    / "temp"
)

NORMALIZER_TEMP_FOLDER: Final[Path] = (
    NORMALIZER_ROOT
    / "temp"
)

NORMALIZER_OUTPUT_FOLDER: Final[Path] = (
    NORMALIZER_ROOT
    / "output"
)


# ==========================================================
# NORMALIZER SETTINGS
# ==========================================================

# This is a polling interval only.
#
# It does NOT mean the Normalizer has failed if no LLM batch
# appears during one polling cycle.

POLL_SECONDS: Final[float] = max(
    float(
        NORMALIZER_POLL_SECONDS
    ),
    0.25,
)


# We intentionally do NOT use the previous 600-second
# "no batch appeared" timeout here.
#
# A large OCR/LLM batch may legitimately take longer.
#
# Actual Ollama calls already have TIMEOUT_SECONDS.
#
# The streaming watcher therefore waits for explicit upstream
# completion instead of guessing that the pipeline is dead.


# ==========================================================
# LLM BATCH FILE PATTERN
# ==========================================================

LLM_BATCH_PATTERN: Final[
    re.Pattern[str]
] = re.compile(
    r"^LLM_(.+?)"
    r"_BATCH_(\d+)"
    r"_PAGES_(\d+)"
    r"_TO_(\d+)"
    r"\.txt$",
    re.IGNORECASE,
)


# ==========================================================
# EXCEPTIONS
# ==========================================================

class NormalizerError(
    RuntimeError
):
    """
    Base exception for Normalizer failures.
    """


class OllamaConnectionError(
    NormalizerError
):
    """
    Raised when Ollama cannot be reached.
    """


class OllamaModelError(
    NormalizerError
):
    """
    Raised when the configured Ollama model is unavailable.
    """


class BatchNormalizationError(
    NormalizerError
):
    """
    Raised when one LLM batch cannot be normalized.
    """


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass(
    frozen=True,
    slots=True,
)
class LLMBatch:
    """
    One completed LLM batch waiting for normalization.

    Clinical content:
        text_path -> .txt

    Workflow signal:
        ready_path -> .ready
    """

    patient_name: str

    batch_number: int

    first_page: int

    final_page: int

    total_pages: int

    text_path: Path

    ready_path: Path

    @property
    def character_count(
        self,
    ) -> int:
        """
        Return the number of characters in the LLM TXT.
        """

        try:

            return len(
                self.text_path.read_text(
                    encoding=(
                        FILE_ENCODING
                    ),
                    errors="replace",
                )
            )

        except OSError:

            return 0


@dataclass(
    frozen=True,
    slots=True,
)
class NormalizedBatchResult:
    """
    Result produced from one LLM batch.

    This is TEMPORARY.

    It is NOT the file consumed by the Discharge Summary.
    """

    batch: LLMBatch

    output_path: Path

    normalized_text: str

    elapsed_seconds: float


# ==========================================================
# PATIENT NAME
# ==========================================================

def safe_patient_name(
    value: str,
) -> str:
    """
    Convert a patient identifier into a safe canonical name.

    Temporary upload prefixes such as:

        pending_

    are removed.

    Therefore:

        pending_BO_VAGDEVI
                ↓
        BO_VAGDEVI
    """

    value = (
        value
        .strip()
    )

    if (
        value.lower()
        .endswith(
            ".pdf"
        )
    ):

        value = (
            value[:-4]
        )

    # Remove accidental repeated temporary prefixes.

    while (
        value.lower()
        .startswith(
            "pending_"
        )
    ):

        value = (
            value[
                len(
                    "pending_"
                ):
            ]
        )

    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    )

    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    ).strip(
        "_"
    )

    return (
        cleaned
        or "patient"
    )


# ==========================================================
# GENERAL TEXT HELPERS
# ==========================================================

def read_text_file(
    path: Path,
) -> str:
    """
    Read one clinical TXT file.

    Empty clinical files are considered errors.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Text file does not exist: "
            f"{path}"
        )

    if (
        path.suffix.lower()
        != ".txt"
    ):

        raise ValueError(
            "Clinical pipeline content must "
            f"be a .txt file: {path}"
        )

    text = (
        path.read_text(
            encoding=(
                FILE_ENCODING
            ),
            errors="replace",
        )
        .strip()
    )

    if not text:

        raise ValueError(
            f"Input text file is empty: "
            f"{path}"
        )

    return text


def atomic_write(
    path: Path,
    text: str,
) -> Path:
    """
    Write a text/control file atomically.

    The temporary .tmp file is replaced only after the write
    has completely finished.

    This prevents downstream stages from reading half-written
    files.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    temporary_path.write_text(
        text.rstrip()
        + "\n",
        encoding=(
            FILE_ENCODING
        ),
    )

    temporary_path.replace(
        path
    )

    return path


def remove_code_fences(
    text: str,
) -> str:
    """
    Remove accidental Markdown code fences from model output.
    """

    cleaned = (
        text.strip()
    )

    if not cleaned.startswith(
        "```"
    ):

        return cleaned

    lines = (
        cleaned.splitlines()
    )

    if (
        lines
        and lines[0]
        .strip()
        .startswith(
            "```"
        )
    ):

        lines = (
            lines[1:]
        )

    if (
        lines
        and lines[-1]
        .strip()
        == "```"
    ):

        lines = (
            lines[:-1]
        )

    return (
        "\n".join(
            lines
        )
        .strip()
    )


# ==========================================================
# LLM WORKSPACE
# ==========================================================

def get_llm_workspace(
    patient_name: str,
) -> dict[
    str,
    Path,
]:
    """
    Return the LLM temporary handoff locations.

    IMPORTANT:

    The Normalizer reads ONLY from:

        llm/temp/<patient>/

    It never scans:

        data/
        llm/output/
        another patient folder
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    root = (
        LLM_TEMP_FOLDER
        / patient_name
    )

    return {
        "root": (
            root
        ),

        "batches": (
            root
            / "batches"
        ),

        "parts": (
            root
            / "parts"
        ),

        "completion": (
            root
            / "llm_complete.json"
        ),

        "manifest": (
            root
            / "normalizer_batch_manifest.json"
        ),
    }


# ==========================================================
# NORMALIZER WORKSPACE
# ==========================================================

def get_patient_workspace(
    patient_name: str,
) -> dict[
    str,
    Path,
]:
    """
    Create and return this patient's temporary Normalizer
    workspace.

    Structure:

        normalizer/temp/<patient>/
            parts/
            errors/

    Later phases will also use:

            normalizer_complete.json

    None of these are permanent patient storage.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    root = (
        NORMALIZER_TEMP_FOLDER
        / patient_name
    )

    workspace = {
        "root": (
            root
        ),

        "parts": (
            root
            / "parts"
        ),

        "errors": (
            root
            / "errors"
        ),

        "completion": (
            root
            / "normalizer_complete.json"
        ),
    }

    for path in (
        workspace.values()
    ):

        if (
            path.suffix
        ):

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

    return workspace


# ==========================================================
# FINAL NORMALIZER OUTPUT
# ==========================================================

def ensure_normalizer_output_folder() -> Path:
    """
    Ensure the module's final-output folder exists.
    """

    NORMALIZER_OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        NORMALIZER_OUTPUT_FOLDER
    )


def get_final_output_path(
    patient_name: str,
) -> Path:
    """
    Return the ONLY completed clinical output produced by the
    Normalizer stage.

    Example:

        normalizer/output/
            NORMALIZER_BO_VAGDEVI.txt

    Discharge Summary will eventually consume THIS file only.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    ensure_normalizer_output_folder()

    return (
        NORMALIZER_OUTPUT_FOLDER
        / (
            f"NORMALIZER_"
            f"{patient_name}"
            f".txt"
        )
    )


def get_completion_marker_path(
    patient_name: str,
) -> Path:
    """
    Return the Normalizer completion marker.

    IMPORTANT:

    The marker lives in TEMP storage.

    It will eventually be written ONLY AFTER:

        NORMALIZER_<patient>.txt

    has been completely created and validated.
    """

    workspace = (
        get_patient_workspace(
            patient_name
        )
    )

    return (
        workspace[
            "completion"
        ]
    )


# ==========================================================
# NORMALIZED TEMPORARY PART PATH
# ==========================================================

def get_normalized_part_path(
    batch: LLMBatch,
) -> Path:
    """
    Return the temporary normalized result for one LLM batch.

    Example:

        normalizer/temp/BO_VAGDEVI/parts/
            NORMALIZER_PART_0001_PAGES_0001_TO_0030.txt

    This file is NOT sent directly to Discharge Summary.
    """

    workspace = (
        get_patient_workspace(
            batch.patient_name
        )
    )

    return (
        workspace["parts"]
        / (
            f"NORMALIZER_PART_"
            f"{batch.batch_number:04}"
            f"_PAGES_"
            f"{batch.first_page:04}"
            f"_TO_"
            f"{batch.final_page:04}"
            f".txt"
        )
    )


# ==========================================================
# ERROR PATH
# ==========================================================

def get_normalizer_error_path(
    batch: LLMBatch,
) -> Path:
    """
    Return the temporary diagnostic error path for one batch.
    """

    workspace = (
        get_patient_workspace(
            batch.patient_name
        )
    )

    return (
        workspace["errors"]
        / (
            f"NORMALIZER_BATCH_"
            f"{batch.batch_number:04}"
            f"_ERROR.txt"
        )
    )


# ==========================================================
# LOAD LLM READY MARKER
# ==========================================================

def load_llm_ready_marker(
    *,
    patient_name: str,
    ready_path: Path,
) -> LLMBatch:
    """
    Convert one LLM .ready marker into an LLMBatch.

    The .ready file contains workflow metadata.

    The actual clinical information remains in the referenced
    .txt file.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    try:

        payload = json.loads(
            ready_path.read_text(
                encoding=(
                    FILE_ENCODING
                )
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:

        raise NormalizerError(
            "Could not read LLM ready "
            f"marker: {ready_path}"
        ) from error

    if not isinstance(
        payload,
        dict,
    ):

        raise NormalizerError(
            "LLM ready marker must contain "
            f"a JSON object: {ready_path}"
        )

    if (
        payload.get(
            "status"
        )
        != "ready"
    ):

        raise NormalizerError(
            "LLM ready marker does not "
            "have status='ready': "
            f"{ready_path}"
        )

    if (
        payload.get(
            "stage"
        )
        != "llm"
    ):

        raise NormalizerError(
            "Ready marker does not belong "
            "to the LLM stage: "
            f"{ready_path}"
        )

    marker_patient = (
        safe_patient_name(
            str(
                payload.get(
                    "patient_name",
                    "",
                )
            )
        )
    )

    if (
        marker_patient
        != patient_name
    ):

        raise NormalizerError(
            "LLM ready marker belongs "
            "to a different patient. "
            f"Expected {patient_name}, "
            f"found {marker_patient}."
        )

    try:

        batch_number = int(
            payload[
                "batch_number"
            ]
        )

        first_page = int(
            payload[
                "first_page"
            ]
        )

        final_page = int(
            payload[
                "final_page"
            ]
        )

        total_pages = int(
            payload[
                "total_pages"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:

        raise NormalizerError(
            "LLM ready marker contains "
            "invalid batch/page metadata: "
            f"{ready_path}"
        ) from error

    if batch_number < 1:

        raise NormalizerError(
            "LLM batch number must be "
            f"positive: {ready_path}"
        )

    if first_page < 1:

        raise NormalizerError(
            "LLM first page must be "
            f"positive: {ready_path}"
        )

    if final_page < first_page:

        raise NormalizerError(
            "LLM batch final page is "
            "before its first page: "
            f"{ready_path}"
        )

    # ------------------------------------------------------
    # RESOLVE CLINICAL TXT
    # ------------------------------------------------------

    batch_path_value = (
        payload.get(
            "batch_path"
        )
    )

    if batch_path_value:

        text_path = (
            Path(
                str(
                    batch_path_value
                )
            )
        )

    else:

        batch_file = (
            payload.get(
                "batch_file"
            )
        )

        if not batch_file:

            raise NormalizerError(
                "LLM ready marker does not "
                "contain batch_path or "
                f"batch_file: {ready_path}"
            )

        text_path = (
            ready_path.parent
            / str(
                batch_file
            )
        )

    if not text_path.exists():

        raise FileNotFoundError(
            "LLM .ready marker exists but "
            "its clinical TXT is missing: "
            f"{text_path}"
        )

    if (
        text_path.suffix.lower()
        != ".txt"
    ):

        raise NormalizerError(
            "LLM clinical batch must be "
            f"a .txt file: {text_path}"
        )

    # Make sure the TXT is readable and non-empty BEFORE
    # accepting the ready marker.

    read_text_file(
        text_path
    )

    return LLMBatch(
        patient_name=(
            patient_name
        ),

        batch_number=(
            batch_number
        ),

        first_page=(
            first_page
        ),

        final_page=(
            final_page
        ),

        total_pages=(
            total_pages
        ),

        text_path=(
            text_path
        ),

        ready_path=(
            ready_path
        ),
    )


# ==========================================================
# FIND READY LLM BATCHES
# ==========================================================

def find_ready_llm_batches(
    patient_name: str,
) -> list[
    LLMBatch
]:
    """
    Find Normalizer input batches.

    Searches ONLY:

        llm/temp/<patient>/batches/*.ready

    It NEVER searches:

        data/
        llm/output/
        other patient directories

    This directly prevents old patient/output files from being
    accidentally processed.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    workspace = (
        get_llm_workspace(
            patient_name
        )
    )

    batch_folder = (
        workspace[
            "batches"
        ]
    )

    if not batch_folder.exists():

        return []

    ready_files = sorted(
        batch_folder.glob(
            "*.ready"
        )
    )

    batches: list[
        LLMBatch
    ] = []

    seen_batch_numbers: set[
        int
    ] = set()

    for ready_path in (
        ready_files
    ):

        batch = (
            load_llm_ready_marker(
                patient_name=(
                    patient_name
                ),
                ready_path=(
                    ready_path
                ),
            )
        )

        if (
            batch.batch_number
            in seen_batch_numbers
        ):

            raise NormalizerError(
                "Duplicate LLM batch "
                f"{batch.batch_number} "
                f"found for patient "
                f"{patient_name}."
            )

        seen_batch_numbers.add(
            batch.batch_number
        )

        batches.append(
            batch
        )

    batches.sort(
        key=lambda item: (
            item.batch_number
        )
    )

    return batches


# ==========================================================
# LLM COMPLETION MARKER
# ==========================================================

def get_llm_completion_marker_path(
    patient_name: str,
) -> Path:
    """
    Return:

        llm/temp/<patient>/llm_complete.json
    """

    workspace = (
        get_llm_workspace(
            patient_name
        )
    )

    return (
        workspace[
            "completion"
        ]
    )


def load_llm_completion_data(
    patient_name: str,
) -> dict[
    str,
    Any,
] | None:
    """
    Load LLM completion information.

    None means:

        LLM is still running.

    The Normalizer does NOT need to wait for this marker before
    starting.

    Individual .ready batches can be normalized immediately.

    The marker is needed only to determine when no additional
    LLM batches will appear.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    marker_path = (
        get_llm_completion_marker_path(
            patient_name
        )
    )

    if not marker_path.exists():

        return None

    try:

        payload = json.loads(
            marker_path.read_text(
                encoding=(
                    FILE_ENCODING
                )
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:

        raise NormalizerError(
            "Could not read LLM completion "
            f"marker: {marker_path}"
        ) from error

    if not isinstance(
        payload,
        dict,
    ):

        raise NormalizerError(
            "LLM completion marker must "
            "contain a JSON object."
        )

    if (
        payload.get(
            "status"
        )
        != "completed"
    ):

        return None

    if (
        payload.get(
            "stage"
        )
        != "llm"
    ):

        raise NormalizerError(
            "LLM completion marker has "
            "an invalid stage."
        )

    marker_patient = (
        safe_patient_name(
            str(
                payload.get(
                    "patient_name",
                    "",
                )
            )
        )
    )

    if (
        marker_patient
        != patient_name
    ):

        raise NormalizerError(
            "LLM completion marker belongs "
            "to another patient. "
            f"Expected {patient_name}, "
            f"found {marker_patient}."
        )

    try:

        total_batches = int(
            payload[
                "total_batches"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:

        raise NormalizerError(
            "LLM completion marker does "
            "not contain a valid "
            "total_batches value."
        ) from error

    if total_batches < 1:

        raise NormalizerError(
            "LLM completion marker reports "
            "zero Normalizer batches."
        )

    return payload


def expected_batch_count(
    patient_name: str,
) -> int | None:
    """
    Return the total number of Normalizer input batches once
    LLM has completely finished.

    While LLM is still streaming:

        None

    After LLM finishes:

        1, 2, 3, ...
    """

    completion_data = (
        load_llm_completion_data(
            patient_name
        )
    )

    if completion_data is None:

        return None

    return int(
        completion_data[
            "total_batches"
        ]
    )


# ==========================================================
# NORMALIZER COMPLETION STATE
# ==========================================================

def normalizer_completion_exists(
    patient_name: str,
) -> bool:
    """
    Return True only when BOTH:

        normalizer/output/NORMALIZER_<patient>.txt

    and:

        normalizer/temp/<patient>/normalizer_complete.json

    exist.

    Later phases will make sure the marker is written only
    after the final combined TXT has been successfully created.

    This is the gate that the Discharge stage will use.
    """

    final_output = (
        get_final_output_path(
            patient_name
        )
    )

    completion_marker = (
        get_completion_marker_path(
            patient_name
        )
    )

    return (
        final_output.exists()
        and completion_marker.exists()
    )


# ==========================================================
# DISCOVER ACTIVE PATIENTS
# ==========================================================

def find_streaming_patients() -> list[
    str
]:
    """
    Find patients currently being handed from LLM to Normalizer.

    Searches ONLY:

        llm/temp/

    It does not inspect data/.
    """

    if not LLM_TEMP_FOLDER.exists():

        return []

    patients: list[
        str
    ] = []

    for patient_folder in (
        LLM_TEMP_FOLDER.iterdir()
    ):

        if not patient_folder.is_dir():

            continue

        patient_name = (
            safe_patient_name(
                patient_folder.name
            )
        )

        workspace = (
            get_llm_workspace(
                patient_name
            )
        )

        batch_folder = (
            workspace[
                "batches"
            ]
        )

        completion_marker = (
            workspace[
                "completion"
            ]
        )

        has_ready_batch = (
            batch_folder.exists()
            and any(
                batch_folder.glob(
                    "*.ready"
                )
            )
        )

        if (
            has_ready_batch
            or completion_marker.exists()
        ):

            patients.append(
                patient_name
            )

    return sorted(
        set(
            patients
        )
    )
# ==========================================================
# OLLAMA MODEL MATCHING
# ==========================================================

def model_name_matches(
    configured_model: str,
    installed_model: str,
) -> bool:
    """
    Compare model names with or without the ':latest' suffix.
    """

    configured = (
        configured_model
        .strip()
    )

    installed = (
        installed_model
        .strip()
    )

    if (
        configured
        == installed
    ):

        return True

    configured_base = (
        configured
        .removesuffix(
            ":latest"
        )
    )

    installed_base = (
        installed
        .removesuffix(
            ":latest"
        )
    )

    return (
        configured_base
        == installed_base
    )


# ==========================================================
# GET INSTALLED OLLAMA MODELS
# ==========================================================

def get_installed_models() -> list[str]:
    """
    Return model names currently available in Ollama.
    """

    try:

        response = (
            requests.get(
                OLLAMA_TAGS_ENDPOINT,
                timeout=30,
            )
        )

        response.raise_for_status()

    except (
        requests.exceptions.ConnectionError
    ) as error:

        raise OllamaConnectionError(
            "Could not connect to Ollama."
        ) from error

    except (
        requests.exceptions.Timeout
    ) as error:

        raise OllamaConnectionError(
            "Timed out while checking Ollama."
        ) from error

    except (
        requests.exceptions.RequestException
    ) as error:

        raise OllamaConnectionError(
            "Ollama connection check failed: "
            f"{error}"
        ) from error

    try:

        payload = (
            response.json()
        )

    except ValueError as error:

        raise OllamaConnectionError(
            "Ollama returned invalid JSON."
        ) from error

    if not isinstance(
        payload,
        dict,
    ):

        raise OllamaConnectionError(
            "Ollama model response was invalid."
        )

    models = (
        payload.get(
            "models",
            [],
        )
    )

    if not isinstance(
        models,
        list,
    ):

        return []

    names: list[str] = []

    for model in models:

        if not isinstance(
            model,
            dict,
        ):

            continue

        model_name = (
            model.get(
                "name"
            )
            or model.get(
                "model"
            )
        )

        if (
            isinstance(
                model_name,
                str,
            )
            and model_name.strip()
        ):

            names.append(
                model_name.strip()
            )

    return names


# ==========================================================
# VERIFY OLLAMA
# ==========================================================

def verify_ollama() -> None:
    """
    Confirm Ollama is running and the configured model exists.
    """

    print(
        "Checking Ollama connection...",
        flush=True,
    )

    installed_models = (
        get_installed_models()
    )

    model_found = any(
        model_name_matches(
            MODEL_NAME,
            installed_model,
        )
        for installed_model
        in installed_models
    )

    if not model_found:

        installed_text = (
            ", ".join(
                installed_models
            )
            if installed_models
            else "No models found"
        )

        raise OllamaModelError(
            f"Model '{MODEL_NAME}' "
            "is not installed.\n"
            f"Installed models: "
            f"{installed_text}\n"
            f"Run: ollama pull "
            f"{MODEL_NAME}"
        )

    print(
        "Ollama ready. "
        f"Model: {MODEL_NAME}",
        flush=True,
    )


# ==========================================================
# BUILD NORMALIZER PROMPT
# ==========================================================

def build_batch_prompt(
    batch: LLMBatch,
    batch_text: str,
) -> str:
    """
    Build the prompt for one LLM-derived Normalizer batch.

    The Normalizer is NOT creating the discharge summary here.

    It is only cleaning and standardizing clinical text.
    """

    return (
        f"{NORMALIZER_PROMPT.strip()}\n\n"
        f"{'=' * 72}\n"
        "NORMALIZER BATCH INFORMATION\n"
        f"{'=' * 72}\n"
        f"Patient: "
        f"{batch.patient_name}\n"
        f"Batch number: "
        f"{batch.batch_number}\n"
        f"Source pages: "
        f"{batch.first_page} to "
        f"{batch.final_page}\n"
        f"Total patient pages: "
        f"{batch.total_pages}\n"
        f"{'=' * 72}\n\n"
        "Normalize only the clinical text "
        "provided in this batch.\n"
        "Do not create a discharge summary.\n"
        "Do not invent missing facts.\n"
        "Do not output JSON.\n"
        "Return plain normalized clinical "
        "text only.\n\n"
        f"{'=' * 72}\n"
        "BEGIN STRUCTURED CLINICAL TEXT\n"
        f"{'=' * 72}\n\n"
        f"{batch_text.strip()}\n\n"
        f"{'=' * 72}\n"
        "END STRUCTURED CLINICAL TEXT\n"
        f"{'=' * 72}"
    )


# ==========================================================
# EXTRACT OLLAMA RESPONSE
# ==========================================================

def extract_response_text(
    payload: dict[str, Any],
) -> str:
    """
    Extract generated plain text from Ollama.
    """

    response_text = (
        payload.get(
            "response"
        )
    )

    if not isinstance(
        response_text,
        str,
    ):

        raise NormalizerError(
            "Ollama response did not "
            "contain valid text."
        )

    response_text = (
        response_text.strip()
    )

    if not response_text:

        raise NormalizerError(
            "Ollama returned an empty "
            "Normalizer response."
        )

    return response_text


# ==========================================================
# CALL OLLAMA
# ==========================================================

def call_ollama(
    prompt: str,
) -> tuple[
    str,
    dict[str, Any],
]:
    """
    Normalize one batch using local Ollama.

    Returns:

        normalized_model_text,
        raw_response_payload
    """

    request_payload = {
        "model": (
            MODEL_NAME
        ),

        "prompt": (
            prompt
        ),

        "stream": False,

        "keep_alive": (
            KEEP_ALIVE
        ),

        "options": (
            OPTIONS
        ),
    }

    try:

        response = (
            requests.post(
                OLLAMA_GENERATE_ENDPOINT,
                json=(
                    request_payload
                ),
                timeout=(
                    TIMEOUT_SECONDS
                ),
            )
        )

        response.raise_for_status()

    except (
        requests.exceptions.ConnectionError
    ) as error:

        raise OllamaConnectionError(
            "Lost connection to Ollama "
            "during normalization."
        ) from error

    except (
        requests.exceptions.Timeout
    ) as error:

        raise NormalizerError(
            "Normalizer request exceeded "
            f"{TIMEOUT_SECONDS} seconds."
        ) from error

    except (
        requests.exceptions.HTTPError
    ) as error:

        response_body = ""

        if (
            error.response
            is not None
        ):

            response_body = (
                error.response
                .text
                .strip()
            )

        message = (
            "Ollama returned an HTTP "
            f"error: {error}"
        )

        if response_body:

            message += (
                "\nOllama response: "
                f"{response_body}"
            )

        raise NormalizerError(
            message
        ) from error

    except (
        requests.exceptions.RequestException
    ) as error:

        raise NormalizerError(
            "Ollama request failed: "
            f"{error}"
        ) from error

    try:

        payload = (
            response.json()
        )

    except ValueError as error:

        raise NormalizerError(
            "Ollama returned invalid JSON."
        ) from error

    if not isinstance(
        payload,
        dict,
    ):

        raise NormalizerError(
            "Ollama returned an unexpected "
            "response format."
        )

    generated_text = (
        extract_response_text(
            payload
        )
    )

    cleaned_text = (
        remove_code_fences(
            generated_text
        )
    )

    if not cleaned_text:

        raise NormalizerError(
            "The cleaned Normalizer "
            "response is empty."
        )

    return (
        cleaned_text,
        payload,
    )


# ==========================================================
# NORMALIZER OUTPUT VALIDATION
# ==========================================================

def validate_normalized_batch(
    input_text: str,
    output_text: str,
) -> None:
    """
    Detect empty or obviously truncated Normalizer output.

    This keeps the useful protection from the old Normalizer,
    but avoids making the ratio too aggressive.

    The previous Normalizer error you saw came from this kind
    of validation, so this check is deliberate.
    """

    stripped_input = (
        input_text.strip()
    )

    stripped_output = (
        output_text.strip()
    )

    if not stripped_input:

        raise NormalizerError(
            "Normalizer input batch "
            "is empty."
        )

    if not stripped_output:

        raise NormalizerError(
            "Normalizer output batch "
            "is empty."
        )

    input_length = len(
        stripped_input
    )

    output_length = len(
        stripped_output
    )

    # ------------------------------------------------------
    # TRUNCATION CHECK
    # ------------------------------------------------------
    #
    # We keep the same broad idea as before:
    # output must not collapse to an implausibly tiny response.
    #
    # But the Normalizer is allowed to remove duplication and
    # noise, so output does NOT need to be close to input size.
    #
    # ------------------------------------------------------

   

# ==========================================================
# GENERATION STATISTICS
# ==========================================================

def calculate_generation_statistics(
    payload: dict[str, Any],
) -> tuple[
    float | None,
    float | None,
]:
    """
    Calculate Ollama prompt and generation token rates.
    """

    prompt_count = (
        payload.get(
            "prompt_eval_count"
        )
    )

    prompt_duration = (
        payload.get(
            "prompt_eval_duration"
        )
    )

    generation_count = (
        payload.get(
            "eval_count"
        )
    )

    generation_duration = (
        payload.get(
            "eval_duration"
        )
    )

    prompt_rate: (
        float
        | None
    ) = None

    generation_rate: (
        float
        | None
    ) = None

    if (
        isinstance(
            prompt_count,
            int,
        )
        and isinstance(
            prompt_duration,
            int,
        )
        and prompt_duration > 0
    ):

        prompt_rate = (
            prompt_count
            / (
                prompt_duration
                / 1_000_000_000
            )
        )

    if (
        isinstance(
            generation_count,
            int,
        )
        and isinstance(
            generation_duration,
            int,
        )
        and generation_duration > 0
    ):

        generation_rate = (
            generation_count
            / (
                generation_duration
                / 1_000_000_000
            )
        )

    return (
        prompt_rate,
        generation_rate,
    )


# ==========================================================
# SAVE BATCH ERROR
# ==========================================================

def save_batch_error(
    *,
    batch: LLMBatch,
    error: Exception,
) -> Path:
    """
    Save temporary diagnostic information for a failed batch.
    """

    error_path = (
        get_normalizer_error_path(
            batch
        )
    )

    error_text = (
        f"Patient: "
        f"{batch.patient_name}\n"
        f"Batch: "
        f"{batch.batch_number}\n"
        f"Pages: "
        f"{batch.first_page}-"
        f"{batch.final_page}\n"
        f"Model: "
        f"{MODEL_NAME}\n"
        f"Error type: "
        f"{type(error).__name__}\n"
        f"Error: "
        f"{error}\n\n"
        "TRACEBACK\n"
        "=========\n"
        f"{traceback.format_exc()}\n"
    )

    atomic_write(
        error_path,
        error_text,
    )

    return error_path


# ==========================================================
# PROCESS ONE LLM BATCH
# ==========================================================

def process_llm_batch(
    batch: LLMBatch,
) -> NormalizedBatchResult:
    """
    Normalize exactly one LLM batch.

    Input:
        llm/temp/<patient>/batches/*.txt

    Temporary output:
        normalizer/temp/<patient>/parts/*.txt

    This does NOT create any Discharge-ready marker.
    """

    started_at = (
        time.perf_counter()
    )

    batch_text = (
        read_text_file(
            batch.text_path
        )
    )

    prompt = (
        build_batch_prompt(
            batch,
            batch_text,
        )
    )

    output_path = (
        get_normalized_part_path(
            batch
        )
    )

    error_path = (
        get_normalizer_error_path(
            batch
        )
    )

    total_attempts = max(
        1,
        RETRIES + 1,
    )

    last_error: (
        Exception
        | None
    ) = None

    for attempt in range(
        1,
        total_attempts + 1,
    ):

        try:

            print(
                "[NORMALIZER START] "
                f"Batch "
                f"{batch.batch_number:04} "
                f"| pages "
                f"{batch.first_page}-"
                f"{batch.final_page} "
                f"| attempt "
                f"{attempt}",
                flush=True,
            )

            (
                model_output,
                response_payload,
            ) = (
                call_ollama(
                    prompt
                )
            )

            # ----------------------------------------------
            # FIRST VALIDATION
            # ----------------------------------------------

            validate_normalized_batch(
                batch_text,
                model_output,
            )

            # ----------------------------------------------
            # DETERMINISTIC CLINICAL TERM NORMALIZATION
            # ----------------------------------------------

            (
                clinically_normalized_text,
                _,
            ) = (
                normalize_clinical_terms(
                    model_output
                )
            )

            # ----------------------------------------------
            # SECOND VALIDATION
            # ----------------------------------------------

            validate_normalized_batch(
                batch_text,
                clinically_normalized_text,
            )

            # ----------------------------------------------
            # WRITE TEMPORARY NORMALIZED TXT
            # ----------------------------------------------

            atomic_write(
                output_path,
                clinically_normalized_text,
            )

            # Verify disk output before calling this complete.

            verified_text = (
                read_text_file(
                    output_path
                )
            )

            if (
                verified_text.strip()
                != clinically_normalized_text.strip()
            ):

                raise NormalizerError(
                    "Normalized TXT verification "
                    "failed after writing."
                )

            if error_path.exists():

                error_path.unlink()

            elapsed = (
                time.perf_counter()
                - started_at
            )

            (
                prompt_rate,
                generation_rate,
            ) = (
                calculate_generation_statistics(
                    response_payload
                )
            )

            performance = [
                f"{elapsed:.1f}s",
                (
                    f"{len(clinically_normalized_text):,} "
                    "output characters"
                ),
            ]

            if (
                prompt_rate
                is not None
            ):

                performance.append(
                    f"{prompt_rate:.1f} "
                    "prompt tokens/s"
                )

            if (
                generation_rate
                is not None
            ):

                performance.append(
                    f"{generation_rate:.1f} "
                    "generation tokens/s"
                )

            print(
                "[NORMALIZER COMPLETE] "
                f"Batch "
                f"{batch.batch_number:04} "
                "| "
                + " | ".join(
                    performance
                ),
                flush=True,
            )

            return (
                NormalizedBatchResult(
                    batch=(
                        batch
                    ),
                    output_path=(
                        output_path
                    ),
                    normalized_text=(
                        clinically_normalized_text
                    ),
                    elapsed_seconds=(
                        elapsed
                    ),
                )
            )

        except Exception as error:

            last_error = (
                error
            )

            print(
                "[NORMALIZER RETRY] "
                f"Batch "
                f"{batch.batch_number:04} "
                f"| attempt "
                f"{attempt}/"
                f"{total_attempts} "
                f"| "
                f"{type(error).__name__}: "
                f"{error}",
                flush=True,
            )

            if (
                attempt
                < total_attempts
            ):

                delay = (
                    RETRY_DELAY_SECONDS
                    * attempt
                )

                time.sleep(
                    delay
                )

    assert (
        last_error
        is not None
    )

    error_path = (
        save_batch_error(
            batch=(
                batch
            ),
            error=(
                last_error
            ),
        )
    )

    raise BatchNormalizationError(
        "Normalizer batch "
        f"{batch.batch_number} "
        "failed after "
        f"{total_attempts} "
        "attempt(s). "
        f"Error saved to: "
        f"{error_path}"
    )


# ==========================================================
# LOAD EXISTING NORMALIZED PART
# ==========================================================

def load_existing_normalized_part(
    batch: LLMBatch,
) -> NormalizedBatchResult | None:
    """
    Restore a temporary normalized part already completed during
    the CURRENT pipeline run.

    Because api/pipeline.py will clear normalizer/temp/<patient>
    before a new run, this cannot come from a previous patient
    run once the pipeline coordinator is updated.
    """

    output_path = (
        get_normalized_part_path(
            batch
        )
    )

    if not output_path.exists():

        return None

    try:

        normalized_text = (
            read_text_file(
                output_path
            )
        )

    except Exception:

        return None

    return (
        NormalizedBatchResult(
            batch=(
                batch
            ),
            output_path=(
                output_path
            ),
            normalized_text=(
                normalized_text
            ),
            elapsed_seconds=0.0,
        )
    )


# ==========================================================
# SUBMIT READY LLM BATCHES
# ==========================================================

def submit_ready_llm_batches(
    *,
    patient_name: str,
    executor: ThreadPoolExecutor,
    submitted_batch_numbers: set[int],
    completed_batches: dict[
        int,
        NormalizedBatchResult,
    ],
    futures: dict[
        Future[NormalizedBatchResult],
        LLMBatch,
    ],
) -> int:
    """
    Discover newly ready LLM batches and submit them to the
    long-lived Normalizer executor.

    This function is called repeatedly while LLM is still
    producing batches.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    ready_batches = (
        find_ready_llm_batches(
            patient_name
        )
    )

    added = 0

    for batch in (
        ready_batches
    ):

        batch_number = (
            batch.batch_number
        )

        if (
            batch_number
            in submitted_batch_numbers
        ):

            continue

        if (
            batch_number
            in completed_batches
        ):

            continue

        # --------------------------------------------------
        # RESTORE CURRENT-RUN RESULT
        # --------------------------------------------------

        existing_result = (
            load_existing_normalized_part(
                batch
            )
        )

        if (
            existing_result
            is not None
        ):

            completed_batches[
                batch_number
            ] = (
                existing_result
            )

            submitted_batch_numbers.add(
                batch_number
            )

            added += 1

            print(
                "[NORMALIZER RESTORED] "
                f"Batch "
                f"{batch_number:04}",
                flush=True,
            )

            continue

        # --------------------------------------------------
        # SUBMIT NEW NORMALIZER JOB
        # --------------------------------------------------

        future = (
            executor.submit(
                process_llm_batch,
                batch,
            )
        )

        futures[
            future
        ] = (
            batch
        )

        submitted_batch_numbers.add(
            batch_number
        )

        added += 1

        print(
            "[NORMALIZER QUEUED] "
            f"Batch "
            f"{batch_number:04} "
            f"| pages "
            f"{batch.first_page}-"
            f"{batch.final_page}",
            flush=True,
        )

    return added


# ==========================================================
# COLLECT FINISHED NORMALIZER FUTURES
# ==========================================================

def collect_completed_normalizer_futures(
    *,
    futures: dict[
        Future[NormalizedBatchResult],
        LLMBatch,
    ],
    completed_batches: dict[
        int,
        NormalizedBatchResult,
    ],
) -> int:
    """
    Collect only futures that are already done.

    Does not block on Normalizer jobs still running.
    """

    finished_futures = [
        future
        for future in list(
            futures
        )
        if future.done()
    ]

    completed_count = 0

    for future in (
        finished_futures
    ):

        batch = (
            futures.pop(
                future
            )
        )

        try:

            result = (
                future.result()
            )

        except Exception as error:

            raise BatchNormalizationError(
                "Normalizer failed for "
                f"batch "
                f"{batch.batch_number} "
                f"(pages "
                f"{batch.first_page}-"
                f"{batch.final_page}): "
                f"{error}"
            ) from error

        completed_batches[
            batch.batch_number
        ] = (
            result
        )

        completed_count += 1

    return completed_count


# ==========================================================
# FINISH REMAINING NORMALIZER FUTURES
# ==========================================================

def finish_remaining_normalizer_futures(
    *,
    futures: dict[
        Future[NormalizedBatchResult],
        LLMBatch,
    ],
    completed_batches: dict[
        int,
        NormalizedBatchResult,
    ],
) -> None:
    """
    Wait for all submitted Normalizer jobs after LLM has
    announced that no additional batches will arrive.
    """

    if not futures:

        return

    remaining = dict(
        futures
    )

    futures.clear()

    for future in (
        remaining
    ):

        batch = (
            remaining[
                future
            ]
        )

        try:

            result = (
                future.result()
            )

        except Exception as error:

            raise BatchNormalizationError(
                "Normalizer failed for "
                f"batch "
                f"{batch.batch_number} "
                f"(pages "
                f"{batch.first_page}-"
                f"{batch.final_page}): "
                f"{error}"
            ) from error

        completed_batches[
            batch.batch_number
        ] = (
            result
        )


# ==========================================================
# VALIDATE ALL EXPECTED NORMALIZER BATCHES
# ==========================================================

def validate_completed_batches(
    *,
    patient_name: str,
    completed_batches: dict[
        int,
        NormalizedBatchResult,
    ],
    expected_batch_count: int,
) -> list[
    NormalizedBatchResult
]:
    """
    Ensure exactly one completed Normalizer result exists for
    every LLM batch.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    if (
        expected_batch_count
        < 1
    ):

        raise NormalizerError(
            "Expected Normalizer batch "
            "count must be at least 1."
        )

    expected_numbers = set(
        range(
            1,
            expected_batch_count + 1,
        )
    )

    actual_numbers = set(
        completed_batches
    )

    missing = sorted(
        expected_numbers
        - actual_numbers
    )

    unexpected = sorted(
        actual_numbers
        - expected_numbers
    )

    if missing:

        raise NormalizerError(
            "Missing normalized batch(es): "
            + ", ".join(
                str(
                    value
                )
                for value in missing
            )
        )

    if unexpected:

        raise NormalizerError(
            "Unexpected normalized batch(es): "
            + ", ".join(
                str(
                    value
                )
                for value in unexpected
            )
        )

    ordered_results = [
        completed_batches[
            batch_number
        ]
        for batch_number in range(
            1,
            expected_batch_count + 1,
        )
    ]

    for result in (
        ordered_results
    ):

        result_patient = (
            safe_patient_name(
                result.batch.patient_name
            )
        )

        if (
            result_patient
            != patient_name
        ):

            raise NormalizerError(
                "A completed Normalizer "
                "result belongs to another "
                "patient."
            )

    return ordered_results
# ==========================================================
# BUILD COMPLETE NORMALIZED TEXT
# ==========================================================

def build_complete_normalized_text(
    ordered_results: list[
        NormalizedBatchResult
    ],
) -> str:
    """
    Merge every completed Normalizer batch into ONE final
    clinical text.

    This is the first point at which the Normalizer produces
    something intended for the Discharge Summary stage.

    IMPORTANT:

    Results must already be ordered by batch number.

    No temporary Normalizer part is sent to Discharge.
    """

    if not ordered_results:

        raise NormalizerError(
            "Cannot build final Normalizer "
            "output from zero batches."
        )

    sections: list[str] = []

    previous_batch_number = 0

    previous_final_page = 0

    for result in (
        ordered_results
    ):

        batch = (
            result.batch
        )

        # --------------------------------------------------
        # BATCH ORDER VALIDATION
        # --------------------------------------------------

        expected_batch_number = (
            previous_batch_number
            + 1
        )

        if (
            batch.batch_number
            != expected_batch_number
        ):

            raise NormalizerError(
                "Normalizer batch order is "
                "invalid while building the "
                "final output. "
                f"Expected batch "
                f"{expected_batch_number}, "
                f"found "
                f"{batch.batch_number}."
            )

        # --------------------------------------------------
        # PAGE ORDER VALIDATION
        # --------------------------------------------------
        #
        # We allow adjacent batches to share a boundary only
        # if upstream metadata ever does so, but we do not
        # allow the source-page sequence to move backwards.
        #
        # --------------------------------------------------

        if (
            previous_final_page > 0
            and batch.first_page
            < previous_final_page
        ):

            raise NormalizerError(
                "Normalizer source-page order "
                "moved backwards. "
                f"Previous final page: "
                f"{previous_final_page}; "
                f"next first page: "
                f"{batch.first_page}."
            )

        normalized_text = (
            result.normalized_text
            .strip()
        )

        if not normalized_text:

            raise NormalizerError(
                "A completed Normalizer batch "
                "contains empty text. "
                f"Batch: "
                f"{batch.batch_number}"
            )

        sections.extend(
            [
                "=" * 72,
                (
                    "NORMALIZED CLINICAL SECTION "
                    f"{batch.batch_number:04}"
                ),
                (
                    "SOURCE PAGES: "
                    f"PAGE_{batch.first_page:04} "
                    "TO "
                    f"PAGE_{batch.final_page:04}"
                ),
                "=" * 72,
                "",
                normalized_text,
                "",
                "",
            ]
        )

        previous_batch_number = (
            batch.batch_number
        )

        previous_final_page = (
            batch.final_page
        )

    complete_text = (
        "\n".join(
            sections
        )
        .rstrip()
        + "\n"
    )

    if not complete_text.strip():

        raise NormalizerError(
            "Final combined Normalizer "
            "text is empty."
        )

    return complete_text


# ==========================================================
# VALIDATE FINAL NORMALIZER TEXT
# ==========================================================

def validate_complete_normalized_text(
    *,
    ordered_results: list[
        NormalizedBatchResult
    ],
    complete_text: str,
) -> None:
    """
    Validate the COMPLETE Normalizer output before it is made
    available to the Discharge Summary.

    This is intentionally different from the individual-batch
    truncation check.

    Here we mainly verify:

        - output exists
        - every expected normalized section is represented
        - the final text is not implausibly empty/truncated

    We do NOT demand that Normalizer output equal LLM input
    length because normalization legitimately removes noise
    and repetition.
    """

    if not ordered_results:

        raise NormalizerError(
            "No normalized batches were "
            "provided for final validation."
        )

    stripped_output = (
        complete_text.strip()
    )

    if not stripped_output:

        raise NormalizerError(
            "Final Normalizer output "
            "is empty."
        )

    total_part_characters = sum(
        len(
            result.normalized_text.strip()
        )
        for result in ordered_results
    )

    output_characters = len(
        stripped_output
    )

    # The merged output contains headers in addition to the
    # normalized clinical text, so it should never collapse to
    # a tiny fraction of the completed parts.

    minimum_expected = max(
        int(
            total_part_characters
            * 0.90
        ),
        500,
    )

    if (
        output_characters
        < minimum_expected
    ):

        raise NormalizerError(
            "Final combined Normalizer "
            "output appears incomplete. "
            f"Normalized parts contain "
            f"{total_part_characters:,} "
            f"characters; final output "
            f"contains "
            f"{output_characters:,} "
            f"characters."
        )

    # ------------------------------------------------------
    # VERIFY EVERY BATCH HEADER EXISTS
    # ------------------------------------------------------

    for result in (
        ordered_results
    ):

        expected_header = (
            "NORMALIZED CLINICAL SECTION "
            f"{result.batch.batch_number:04}"
        )

        if (
            expected_header
            not in complete_text
        ):

            raise NormalizerError(
                "Final Normalizer output is "
                "missing expected section: "
                f"{expected_header}"
            )


# ==========================================================
# SAVE FINAL NORMALIZER OUTPUT
# ==========================================================

def save_complete_normalized_output(
    *,
    patient_name: str,
    ordered_results: list[
        NormalizedBatchResult
    ],
) -> Path:
    """
    Create the ONLY Normalizer file that the Discharge Summary
    is allowed to consume:

        normalizer/output/
            NORMALIZER_<patient>.txt

    Write sequence:

        build
          ↓
        validate in memory
          ↓
        atomic write
          ↓
        read from disk
          ↓
        validate again

    The completion marker is NOT created by this function.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    complete_text = (
        build_complete_normalized_text(
            ordered_results
        )
    )

    # ------------------------------------------------------
    # VALIDATE BEFORE WRITE
    # ------------------------------------------------------

    validate_complete_normalized_text(
        ordered_results=(
            ordered_results
        ),
        complete_text=(
            complete_text
        ),
    )

    output_path = (
        get_final_output_path(
            patient_name
        )
    )

    # ------------------------------------------------------
    # ATOMIC WRITE
    # ------------------------------------------------------

    atomic_write(
        output_path,
        complete_text,
    )

    # ------------------------------------------------------
    # READ BACK FROM DISK
    # ------------------------------------------------------

    verified_text = (
        read_text_file(
            output_path
        )
    )

    # ------------------------------------------------------
    # VALIDATE THE ACTUAL DISK FILE
    # ------------------------------------------------------

    validate_complete_normalized_text(
        ordered_results=(
            ordered_results
        ),
        complete_text=(
            verified_text
        ),
    )

    return output_path


# ==========================================================
# SAVE NORMALIZER COMPLETION MARKER
# ==========================================================

def save_normalizer_completion_marker(
    *,
    patient_name: str,
    expected_batches: int,
    ordered_results: list[
        NormalizedBatchResult
    ],
    complete_path: Path,
) -> Path:
    """
    Create:

        normalizer/temp/<patient>/
            normalizer_complete.json

    THIS IS THE DISCHARGE GATE.

    This function must only be called AFTER the final combined
    Normalizer TXT has been written and validated.

    The marker contains workflow metadata only.

    Clinical content remains in:

        NORMALIZER_<patient>.txt
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    if not complete_path.exists():

        raise NormalizerError(
            "Cannot create Normalizer "
            "completion marker because "
            "the final TXT does not exist."
        )

    # Read it one final time before opening the gate.

    final_text = (
        read_text_file(
            complete_path
        )
    )

    validate_complete_normalized_text(
        ordered_results=(
            ordered_results
        ),
        complete_text=(
            final_text
        ),
    )

    marker_path = (
        get_completion_marker_path(
            patient_name
        )
    )

    first_page = min(
        result.batch.first_page
        for result in ordered_results
    )

    final_page = max(
        result.batch.final_page
        for result in ordered_results
    )

    payload = {
        "patient_name": (
            patient_name
        ),

        "stage": (
            "normalizer"
        ),

        "status": (
            "completed"
        ),

        "model": (
            MODEL_NAME
        ),

        "workers": (
            NORMALIZER_MAX_WORKERS
        ),

        "total_batches": (
            expected_batches
        ),

        "first_page": (
            first_page
        ),

        "final_page": (
            final_page
        ),

        "complete_file": (
            complete_path.name
        ),

        "complete_path": str(
            complete_path.resolve()
        ),

        "character_count": len(
            final_text
        ),

        "completed_at": (
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        ),
    }

    # Marker becomes visible only after the whole JSON is
    # written.

    atomic_write(
        marker_path,
        json.dumps(
            payload,
            indent=2,
        ),
    )

    return marker_path


# ==========================================================
# VERIFY DISCHARGE GATE
# ==========================================================

def validate_discharge_gate(
    patient_name: str,
) -> Path:
    """
    Strictly validate whether the Normalizer is ready for the
    Discharge Summary stage.

    BOTH must exist:

        normalizer/output/NORMALIZER_<patient>.txt

    AND:

        normalizer/temp/<patient>/
            normalizer_complete.json

    The marker must point to the same final TXT.

    api/pipeline.py / discharge_summary will later call this
    logic before starting discharge generation.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    output_path = (
        get_final_output_path(
            patient_name
        )
    )

    marker_path = (
        get_completion_marker_path(
            patient_name
        )
    )

    if not output_path.exists():

        raise NormalizerError(
            "Discharge cannot start: "
            "final Normalizer TXT does "
            "not exist."
        )

    if not marker_path.exists():

        raise NormalizerError(
            "Discharge cannot start: "
            "Normalizer completion marker "
            "does not exist."
        )

    try:

        payload = json.loads(
            marker_path.read_text(
                encoding=(
                    FILE_ENCODING
                )
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:

        raise NormalizerError(
            "Discharge cannot start: "
            "Normalizer completion marker "
            "is invalid."
        ) from error

    if not isinstance(
        payload,
        dict,
    ):

        raise NormalizerError(
            "Discharge cannot start: "
            "Normalizer completion marker "
            "has invalid content."
        )

    if (
        payload.get(
            "stage"
        )
        != "normalizer"
    ):

        raise NormalizerError(
            "Discharge cannot start: "
            "completion marker has the "
            "wrong stage."
        )

    if (
        payload.get(
            "status"
        )
        != "completed"
    ):

        raise NormalizerError(
            "Discharge cannot start: "
            "Normalizer is not marked "
            "completed."
        )

    marker_patient = (
        safe_patient_name(
            str(
                payload.get(
                    "patient_name",
                    "",
                )
            )
        )
    )

    if (
        marker_patient
        != patient_name
    ):

        raise NormalizerError(
            "Discharge cannot start: "
            "Normalizer completion marker "
            "belongs to another patient."
        )

    marker_file = str(
        payload.get(
            "complete_file",
            "",
        )
    )

    if (
        marker_file
        != output_path.name
    ):

        raise NormalizerError(
            "Discharge cannot start: "
            "Normalizer completion marker "
            "does not reference the expected "
            "final TXT."
        )

    # ------------------------------------------------------
    # VERIFY FINAL TXT IS NON-EMPTY
    # ------------------------------------------------------

    final_text = (
        read_text_file(
            output_path
        )
    )

    expected_character_count = (
        payload.get(
            "character_count"
        )
    )

    if (
        isinstance(
            expected_character_count,
            int,
        )
        and expected_character_count > 0
        and len(
            final_text
        )
        != expected_character_count
    ):

        raise NormalizerError(
            "Discharge cannot start: "
            "Normalizer final TXT changed "
            "after completion."
        )

    return output_path


# ==========================================================
# STREAMING NORMALIZER
# ==========================================================

def process_patient_stream(
    patient_name: str,
) -> Path:
    """
    Main Normalizer streaming loop.

    LLM -> Normalizer is streaming.

    Normalizer -> Discharge is NOT streaming.

    While LLM is working:

        LLM batch 1 ready
             ↓
        Normalize batch 1

        LLM batch 2 ready
             ↓
        Normalize batch 2

        etc.

    But Discharge receives NOTHING until:

        LLM completed
             ↓
        every expected Normalizer batch completed
             ↓
        all Normalizer results merged
             ↓
        final Normalizer TXT validated
             ↓
        normalizer_complete.json created

    Only then may Discharge begin.

    Safety rule:

        Once LLM completion is declared, every expected LLM batch
        must appear within a short grace period.

        If an expected batch is still missing after the grace
        period, Normalizer fails instead of waiting forever.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    started_at = (
        time.perf_counter()
    )

    POST_LLM_COMPLETION_GRACE_SECONDS = 30.0

    llm_completion_seen_at: (
        float
        | None
    ) = None

    # ======================================================
    # DO NOT CLEAR WORKSPACE HERE
    # ======================================================
    #
    # api/pipeline.py clears all stage workspaces ONCE before
    # the new patient run begins.
    #
    # Clearing here could destroy results while LLM and
    # Normalizer are running concurrently.
    #
    # ======================================================

    workspace = (
        get_patient_workspace(
            patient_name
        )
    )

    ensure_normalizer_output_folder()

    submitted_batch_numbers: set[
        int
    ] = set()

    completed_batches: dict[
        int,
        NormalizedBatchResult,
    ] = {}

    futures: dict[
        Future[NormalizedBatchResult],
        LLMBatch,
    ] = {}

    print()
    print(
        "=" * 72
    )
    print(
        "STREAMING NICU NORMALIZER"
    )
    print(
        "=" * 72
    )
    print(
        f"Patient       : "
        f"{patient_name}"
    )
    print(
        f"Model         : "
        f"{MODEL_NAME}"
    )
    print(
        f"Workers       : "
        f"{NORMALIZER_MAX_WORKERS}"
    )
    print(
        f"LLM batches   : "
        f"{get_llm_workspace(patient_name)['batches']}"
    )
    print(
        f"Temp output   : "
        f"{workspace['root']}"
    )
    print(
        f"Final output  : "
        f"{get_final_output_path(patient_name)}"
    )
    print(
        "=" * 72
    )

    # ======================================================
    # LONG-LIVED NORMALIZER EXECUTOR
    # ======================================================

    with ThreadPoolExecutor(
        max_workers=(
            NORMALIZER_MAX_WORKERS
        )
    ) as executor:

        while True:

            # ==================================================
            # 1. DISCOVER NEW LLM BATCHES
            # ==================================================

            submit_ready_llm_batches(
                patient_name=(
                    patient_name
                ),
                executor=(
                    executor
                ),
                submitted_batch_numbers=(
                    submitted_batch_numbers
                ),
                completed_batches=(
                    completed_batches
                ),
                futures=(
                    futures
                ),
            )

            # ==================================================
            # 2. COLLECT FINISHED NORMALIZER JOBS
            # ==================================================

            collect_completed_normalizer_futures(
                futures=(
                    futures
                ),
                completed_batches=(
                    completed_batches
                ),
            )

            # ==================================================
            # 3. CHECK WHETHER LLM HAS COMPLETELY FINISHED
            # ==================================================

            llm_completion_data = (
                load_llm_completion_data(
                    patient_name
                )
            )

            if (
                llm_completion_data
                is not None
            ):

                # ----------------------------------------------
                # START POST-COMPLETION GRACE TIMER ONCE
                # ----------------------------------------------

                if (
                    llm_completion_seen_at
                    is None
                ):

                    llm_completion_seen_at = (
                        time.perf_counter()
                    )

                    print(
                        "[NORMALIZER] "
                        "LLM completion marker detected.",
                        flush=True,
                    )

                expected_batches = int(
                    llm_completion_data[
                        "total_batches"
                    ]
                )

                expected_numbers = set(
                    range(
                        1,
                        expected_batches + 1,
                    )
                )

                # ----------------------------------------------
                # DISCOVER ONCE MORE AFTER COMPLETION MARKER
                # ----------------------------------------------
                #
                # There may be a tiny timing window between the
                # final .ready appearing and this polling cycle.
                #
                # ----------------------------------------------

                submit_ready_llm_batches(
                    patient_name=(
                        patient_name
                    ),
                    executor=(
                        executor
                    ),
                    submitted_batch_numbers=(
                        submitted_batch_numbers
                    ),
                    completed_batches=(
                        completed_batches
                    ),
                    futures=(
                        futures
                    ),
                )

                discovered_numbers = (
                    set(
                        submitted_batch_numbers
                    )
                    | set(
                        completed_batches
                    )
                )

                # ----------------------------------------------
                # EXPECTED LLM BATCH IS MISSING
                # ----------------------------------------------

                missing_numbers = (
                    expected_numbers
                    - discovered_numbers
                )

                if missing_numbers:

                    grace_elapsed = (
                        time.perf_counter()
                        - llm_completion_seen_at
                    )

                    if (
                        grace_elapsed
                        >= POST_LLM_COMPLETION_GRACE_SECONDS
                    ):

                        missing_text = ", ".join(
                            str(number)
                            for number in sorted(
                                missing_numbers
                            )
                        )

                        raise NormalizerError(
                            "LLM reported completion, "
                            "but expected Normalizer-ready "
                            "LLM batch(es) never appeared "
                            f"within "
                            f"{POST_LLM_COMPLETION_GRACE_SECONDS:.0f} "
                            f"seconds. Missing batch(es): "
                            f"{missing_text}"
                        )

                    remaining = max(
                        0.0,
                        POST_LLM_COMPLETION_GRACE_SECONDS
                        - grace_elapsed,
                    )

                    print(
                        "[NORMALIZER] "
                        "Waiting for missing LLM batch(es): "
                        f"{sorted(missing_numbers)}. "
                        f"Grace remaining: "
                        f"{remaining:.1f}s",
                        flush=True,
                    )

                    time.sleep(
                        POLL_SECONDS
                    )

                    continue

                # ----------------------------------------------
                # COLLECT ANY JOBS THAT JUST FINISHED
                # ----------------------------------------------

                collect_completed_normalizer_futures(
                    futures=(
                        futures
                    ),
                    completed_batches=(
                        completed_batches
                    ),
                )

                # ----------------------------------------------
                # FINALIZATION CONDITION
                # ----------------------------------------------

                if (
                    expected_numbers
                    <= set(
                        completed_batches
                    )
                ):

                    break

            time.sleep(
                POLL_SECONDS
            )

        # ==================================================
        # WAIT FOR ANY REMAINING SUBMITTED JOBS
        # ==================================================

        finish_remaining_normalizer_futures(
            futures=(
                futures
            ),
            completed_batches=(
                completed_batches
            ),
        )

    # ======================================================
    # FINAL LLM COMPLETION CHECK
    # ======================================================

    llm_completion_data = (
        load_llm_completion_data(
            patient_name
        )
    )

    if (
        llm_completion_data
        is None
    ):

        raise NormalizerError(
            "Normalizer reached finalization "
            "but LLM completion marker "
            "is missing."
        )

    expected_batches = int(
        llm_completion_data[
            "total_batches"
        ]
    )

    # ======================================================
    # VALIDATE EVERY EXPECTED BATCH
    # ======================================================

    ordered_results = (
        validate_completed_batches(
            patient_name=(
                patient_name
            ),
            completed_batches=(
                completed_batches
            ),
            expected_batch_count=(
                expected_batches
            ),
        )
    )

    print()
    print(
        "[NORMALIZER FINALIZATION] "
        "All expected batches completed.",
        flush=True,
    )

    print(
        "[NORMALIZER FINALIZATION] "
        "Building complete normalized "
        "patient record...",
        flush=True,
    )

    # ======================================================
    # CREATE ONE COMPLETE NORMALIZED TXT
    # ======================================================

    complete_path = (
        save_complete_normalized_output(
            patient_name=(
                patient_name
            ),
            ordered_results=(
                ordered_results
            ),
        )
    )

    print(
        "[NORMALIZER FINALIZATION] "
        "Complete TXT written and "
        "validated.",
        flush=True,
    )

    # ======================================================
    # ONLY NOW OPEN THE DISCHARGE GATE
    # ======================================================

    completion_marker = (
        save_normalizer_completion_marker(
            patient_name=(
                patient_name
            ),
            expected_batches=(
                expected_batches
            ),
            ordered_results=(
                ordered_results
            ),
            complete_path=(
                complete_path
            ),
        )
    )

    # ======================================================
    # VERIFY GATE AFTER CREATION
    # ======================================================

    discharge_input = (
        validate_discharge_gate(
            patient_name
        )
    )

    if (
        discharge_input.resolve()
        != complete_path.resolve()
    ):

        raise NormalizerError(
            "Discharge gate resolved to "
            "an unexpected Normalizer file."
        )

    elapsed = (
        time.perf_counter()
        - started_at
    )

    print()
    print(
        "=" * 72
    )
    print(
        "NORMALIZER STAGE COMPLETED"
    )
    print(
        "=" * 72
    )
    print(
        f"Patient              : "
        f"{patient_name}"
    )
    print(
        f"Normalized batches   : "
        f"{expected_batches}"
    )
    print(
        f"Final Normalizer TXT : "
        f"{complete_path}"
    )
    print(
        f"Completion marker    : "
        f"{completion_marker}"
    )
    print(
        f"Total time           : "
        f"{elapsed:.2f} seconds"
    )
    print()
    print(
        "DISCHARGE GATE: OPEN"
    )
    print(
        "Discharge may now consume:"
    )
    print(
        f"    {complete_path}"
    )
    print(
        "=" * 72
    )

    # ======================================================
    # DO NOT DELETE TEMPORARY FILES
    # ======================================================
    #
    # Even though Normalizer is finished, cleanup remains the
    # responsibility of api/pipeline.py.
    #
    # If Discharge fails, keeping these temporary files allows
    # us to diagnose the failed run.
    #
    # ======================================================

    return complete_path


# ==========================================================
# STANDALONE MAIN
# ==========================================================

def main() -> None:
    """
    Standalone Normalizer runner.

    Normally api/pipeline.py calls:

        process_patient_stream(patient_name)

    while LLM is running.
    """

    overall_started_at = (
        time.perf_counter()
    )

    print()
    print(
        "=" * 72
    )
    print(
        "NICU STREAMING NORMALIZER"
    )
    print(
        "=" * 72
    )
    print(
        f"LLM temp          : "
        f"{LLM_TEMP_FOLDER}"
    )
    print(
        f"Normalizer temp   : "
        f"{NORMALIZER_TEMP_FOLDER}"
    )
    print(
        f"Normalizer output : "
        f"{NORMALIZER_OUTPUT_FOLDER}"
    )
    print(
        f"Model             : "
        f"{MODEL_NAME}"
    )
    print(
        f"Workers           : "
        f"{NORMALIZER_MAX_WORKERS}"
    )
    print(
        "=" * 72
    )

    # ======================================================
    # VERIFY OLLAMA BEFORE PROCESSING
    # ======================================================

    try:

        verify_ollama()

    except NormalizerError as error:

        print(
            f"Startup error: "
            f"{error}",
            file=sys.stderr,
        )

        raise SystemExit(
            1
        ) from error

    patients = (
        find_streaming_patients()
    )

    if not patients:

        print(
            "No active LLM streaming "
            "patients found."
        )

        return

    successful = 0

    failed = 0

    for patient_name in (
        patients
    ):

        try:

            process_patient_stream(
                patient_name
            )

            successful += 1

        except KeyboardInterrupt:

            print()
            print(
                "Normalizer stopped by user."
            )

            raise

        except Exception as error:

            failed += 1

            print()
            print(
                "=" * 72
            )
            print(
                "NORMALIZER STAGE FAILED"
            )
            print(
                "=" * 72
            )
            print(
                f"Patient: "
                f"{patient_name}"
            )
            print(
                f"Error: "
                f"{type(error).__name__}: "
                f"{error}"
            )
            print()
            print(
                traceback.format_exc()
            )

    elapsed = (
        time.perf_counter()
        - overall_started_at
    )

    print()
    print(
        "=" * 72
    )
    print(
        "NORMALIZER RUN FINISHED"
    )
    print(
        "=" * 72
    )
    print(
        f"Successful patients : "
        f"{successful}"
    )
    print(
        f"Failed patients     : "
        f"{failed}"
    )
    print(
        f"Total time          : "
        f"{elapsed:.2f} seconds"
    )
    print(
        "=" * 72
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()