"""
Parallel streaming LLM structurer for the NICU AI pipeline.

NEW PIPELINE ARCHITECTURE
=========================

OCR temporary streaming input:

    ocr/temp/<patient>/batches/
        OCR_<patient>_BATCH_0001_....txt
        OCR_<patient>_BATCH_0001_....ready
        OCR_<patient>_BATCH_0002_....txt
        OCR_<patient>_BATCH_0002_....ready


LLM temporary workspace:

    llm/temp/<patient>/

        parts/
            LLM_PART_0001_....txt
            LLM_PART_0002_....txt

        batches/
            LLM_<patient>_BATCH_0001_....txt
            LLM_<patient>_BATCH_0001_....ready

        errors/

        normalizer_batch_manifest.json
        llm_complete.json


Final LLM output:

    llm/output/
        LLM_<patient>.txt


IMPORTANT
=========

1. Clinical content is always stored as plain TXT.

2. .ready files contain ONLY workflow metadata.

3. Normalizer reads the TXT referenced by the .ready file.

4. llm_complete.json is ONLY a temporary completion signal.

5. LLM does NOT write into data/.

6. LLM does NOT delete its Normalizer-ready batches after finishing.

7. Final cleanup is controlled later by api/pipeline.py.

8. OCR and LLM can overlap:
   as soon as an OCR .ready batch appears, LLM can process it.

9. LLM and Normalizer can overlap:
   as soon as an LLM .ready batch appears, Normalizer can process it.

10. Source-page ordering is preserved.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback

from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import ollama

from llm import models as model_config
from llm.prompts import STRUCTURER_PROMPT


# ==========================================================
# ENVIRONMENT
# ==========================================================

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "VECLIB_MAXIMUM_THREADS",
    "1",
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "1",
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

LLM_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "llm"
)

LLM_TEMP_FOLDER: Final[Path] = (
    LLM_ROOT
    / "temp"
)

LLM_OUTPUT_FOLDER: Final[Path] = (
    LLM_ROOT
    / "output"
)

OCR_TEMP_FOLDER: Final[Path] = (
    PROJECT_ROOT
    / "ocr"
    / "temp"
)


# ==========================================================
# MODEL SETTINGS
# ==========================================================

MODEL_NAME: Final[str] = (
    model_config.MODEL_NAME
)

STRUCTURER_OPTIONS: Final[
    dict[str, Any]
] = (
    model_config.STRUCTURER_OPTIONS
)

OLLAMA_HOST: Final[str] = getattr(
    model_config,
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

OLLAMA_KEEP_ALIVE: Final[str] = getattr(
    model_config,
    "OLLAMA_KEEP_ALIVE",
    "30m",
)

MAX_LLM_WORKERS: Final[int] = int(
    getattr(
        model_config,
        "EXTRACTOR_MAX_WORKERS",
        1,
    )
)

TIMEOUT_SECONDS: Final[int] = int(
    getattr(
        model_config,
        "EXTRACTOR_TIMEOUT_SECONDS",
        1800,
    )
)

MAX_RETRIES: Final[int] = int(
    getattr(
        model_config,
        "EXTRACTOR_RETRIES",
        2,
    )
)

RETRY_DELAY_SECONDS: Final[float] = float(
    getattr(
        model_config,
        "EXTRACTOR_RETRY_DELAY_SECONDS",
        5,
    )
)

POLL_SECONDS: Final[float] = float(
    getattr(
        model_config,
        "EXTRACTOR_POLL_SECONDS",
        1.0,
    )
)


# ==========================================================
# LLM -> NORMALIZER CHARACTER RANGE
# ==========================================================
#
# IMPORTANT:
#
# These values do NOT control OCR batching.
#
# OCR decides how much OCR text goes into each LLM request.
#
# These values control how completed LLM results are grouped
# before being handed to Normalizer.
#
# ==========================================================

NORMALIZER_MIN_BATCH_CHARACTERS: Final[int] = int(
    getattr(
        model_config,
        "NORMALIZER_MIN_BATCH_CHARACTERS",
        15_000,
    )
)

NORMALIZER_MAX_BATCH_CHARACTERS: Final[int] = int(
    getattr(
        model_config,
        "NORMALIZER_MAX_BATCH_CHARACTERS",
        20_000,
    )
)


if (
    NORMALIZER_MIN_BATCH_CHARACTERS
    < 1
):
    raise ValueError(
        "NORMALIZER_MIN_BATCH_CHARACTERS "
        "must be at least 1."
    )


if (
    NORMALIZER_MAX_BATCH_CHARACTERS
    < NORMALIZER_MIN_BATCH_CHARACTERS
):
    raise ValueError(
        "NORMALIZER_MAX_BATCH_CHARACTERS "
        "must be greater than or equal to "
        "NORMALIZER_MIN_BATCH_CHARACTERS."
    )


# ==========================================================
# PROMPT
# ==========================================================

DEFAULT_STRUCTURER_PROMPT = """
You are an expert NICU clinical-document structuring assistant.

You will receive one consecutive OCR batch from a single patient record.

This is a structuring stage only.

Do not create a discharge summary.
Do not output JSON.
Do not invent missing information.
Do not guess unclear values.
Do not silently correct uncertain dates, values, names or doses.

Preserve all reliable patient-specific clinical information.

Remove non-clinical document clutter such as blank forms, repeated hospital
headers, revision numbers, billing instructions, procedure menus, laboratory
methodology and meaningless OCR fragments when they contain no
patient-specific clinical fact.

For every source page use:

SOURCE PAGE: PAGE_XXXX
DOCUMENT TYPE: <document type or UNCERTAIN>

Keep dates and clinical chronology.

Return plain English clinical text only.
Do not use Markdown code fences.
Do not include an introduction or conclusion.
""".strip()


if (
    isinstance(
        STRUCTURER_PROMPT,
        str,
    )
    and STRUCTURER_PROMPT.strip()
):
    ACTIVE_STRUCTURER_PROMPT = (
        STRUCTURER_PROMPT.strip()
    )

else:
    ACTIVE_STRUCTURER_PROMPT = (
        DEFAULT_STRUCTURER_PROMPT
    )


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass(
    frozen=True,
    slots=True,
)
class OCRBatch:
    """
    One temporary OCR TXT batch ready for LLM processing.
    """

    patient_name: str

    batch_number: int

    first_page: int

    final_page: int

    total_pages: int

    text_path: Path

    ready_path: Path


@dataclass(
    frozen=True,
    slots=True,
)
class LLMPartResult:
    """
    One LLM result corresponding to exactly one OCR batch.

    This is temporary internal data.
    """

    batch: OCRBatch

    output_path: Path

    structured_text: str

    elapsed_seconds: float


    @property
    def character_count(
        self,
    ) -> int:
        return len(
            self.structured_text
        )


@dataclass(
    frozen=True,
    slots=True,
)
class NormalizerReadyBatch:
    """
    One TXT batch ready for Normalizer.

    The actual medical content is in:

        output_path -> .txt

    A separate .ready file only tells Normalizer that this
    TXT has been completely written and is safe to read.
    """

    batch_number: int

    first_page: int

    final_page: int

    total_pages: int

    output_path: Path

    ready_path: Path

    character_count: int

    source_part_numbers: tuple[
        int,
        ...,
    ]


# ==========================================================
# PATIENT NAME
# ==========================================================

def safe_patient_name(
    value: str,
) -> str:
    """
    Return one canonical patient name.

    Examples:

        pending_BO_VAGDEVI
            -> BO_VAGDEVI

        BO VAGDEVI
            -> BO_VAGDEVI

        BO_VAGDEVI
            -> BO_VAGDEVI
    """

    cleaned = (
        value.strip()
    )

    cleaned = re.sub(
        r"\.pdf$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    if cleaned.lower().startswith(
        "pending_"
    ):
        cleaned = cleaned[
            len("pending_"):
        ]

    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        cleaned,
    )

    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    )

    cleaned = (
        cleaned.strip("_")
    )

    return (
        cleaned
        or "patient"
    )


# ==========================================================
# TEXT FILE READER
# ==========================================================

def read_text_file(
    path: Path,
) -> str:
    """
    Read clinical TXT.

    Empty files are rejected so a zero-length OCR/LLM batch
    cannot silently continue through the pipeline.
    """

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"TXT file not found: {path}"
        )

    text = (
        path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        .strip()
    )

    if not text:
        raise ValueError(
            f"TXT file is empty: {path}"
        )

    return text


# ==========================================================
# ATOMIC TEXT WRITE
# ==========================================================

def atomic_write(
    path: Path,
    text: str,
) -> Path:
    """
    Write a TXT/JSON/ready file safely.

    The destination becomes visible only after the complete
    content has been written.

    This prevents downstream stages from reading a half-written
    file while stages are running concurrently.
    """

    path = Path(
        path
    )

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
        text.strip()
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )

    return path


# ==========================================================
# REMOVE MODEL CODE FENCES
# ==========================================================

def remove_code_fences(
    text: str,
) -> str:
    """
    Ensure model output remains normal plain clinical text.
    """

    cleaned = (
        text.strip()
    )

    if cleaned.startswith(
        "```"
    ):

        lines = (
            cleaned.splitlines()
        )

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        )

    return (
        cleaned.strip()
    )


# ==========================================================
# LLM PATIENT WORKSPACE
# ==========================================================

def get_patient_workspace(
    patient_name: str,
) -> dict[str, Path]:
    """
    Return temporary LLM directories.

    NOTHING here is permanent patient storage.

    llm/temp/<patient>/
        parts/
        batches/
        errors/
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

    folders = {
        "root": (
            root
        ),

        "parts": (
            root
            / "parts"
        ),

        "batches": (
            root
            / "batches"
        ),

        "errors": (
            root
            / "errors"
        ),
    }

    for folder in (
        folders.values()
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    return folders


# ==========================================================
# OCR WORKSPACE
# ==========================================================

def get_ocr_workspace(
    patient_name: str,
) -> dict[str, Path]:
    """
    Return the OCR temporary workspace created by ocr/ocr.py.

    We READ from this location.

    We do not use data/.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    root = (
        OCR_TEMP_FOLDER
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

        "completion": (
            root
            / "ocr_complete.json"
        ),
    }


# ==========================================================
# FINAL LLM OUTPUT
# ==========================================================

def ensure_llm_output_folder() -> Path:
    """
    Ensure llm/output exists.
    """

    LLM_OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        LLM_OUTPUT_FOLDER
    )


def get_final_llm_output_path(
    patient_name: str,
) -> Path:
    """
    Final stage output:

        llm/output/LLM_<patient>.txt

    This remains here until the entire NICU pipeline succeeds.

    api/pipeline.py will later MOVE this file into:

        data/<patient>/LLM_<patient>.txt
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    ensure_llm_output_folder()

    return (
        LLM_OUTPUT_FOLDER
        / f"LLM_{patient_name}.txt"
    )


# ==========================================================
# LLM COMPLETION MARKER
# ==========================================================

def get_llm_completion_marker_path(
    patient_name: str,
) -> Path:
    """
    Temporary workflow marker:

        llm/temp/<patient>/llm_complete.json

    Normalizer uses this to know when no more LLM batches
    will arrive.

    This JSON is NOT medical output.
    """

    workspace = (
        get_patient_workspace(
            patient_name
        )
    )

    return (
        workspace["root"]
        / "llm_complete.json"
    )


# ==========================================================
# NORMALIZER MANIFEST
# ==========================================================

def get_normalizer_manifest_path(
    patient_name: str,
) -> Path:
    """
    Temporary workflow metadata used to remember which LLM
    parts have already been emitted to Normalizer batches.

        llm/temp/<patient>/
            normalizer_batch_manifest.json

    This prevents duplicate Normalizer batches during the
    same processing run.
    """

    workspace = (
        get_patient_workspace(
            patient_name
        )
    )

    return (
        workspace["root"]
        / "normalizer_batch_manifest.json"
    )


# ==========================================================
# CLEAN LLM WORKSPACE
# ==========================================================

def clear_patient_llm_workspace(
    patient_name: str,
) -> None:
    """
    Delete stale LLM TEMPORARY files before a completely new
    patient processing run.

    This removes:

        old parts
        old Normalizer-ready batches
        old .ready markers
        old errors
        old manifest
        old llm_complete.json

    This is the protection against the stale-output problem
    we experienced previously.

    IMPORTANT:

    api/pipeline.py should call this ONCE before starting a
    completely new pipeline.

    process_patient_stream() itself should NOT randomly clear
    the workspace after OCR has started producing data.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    workspace_root = (
        LLM_TEMP_FOLDER
        / patient_name
    )

    if workspace_root.exists():

        shutil.rmtree(
            workspace_root
        )


def clear_patient_llm_output(
    patient_name: str,
) -> None:
    """
    Remove stale final LLM output before a completely new run.

    Normally a successful previous run will already have had
    this file moved into data/<patient>/ by api/pipeline.py.
    """

    output_path = (
        get_final_llm_output_path(
            patient_name
        )
    )

    if output_path.exists():

        output_path.unlink()

# ==========================================================
# LOAD OCR COMPLETION DATA
# ==========================================================

def load_ocr_completion_data(
    patient_name: str,
) -> dict[str, Any] | None:
    """
    Load OCR's temporary completion marker:

        ocr/temp/<patient>/ocr_complete.json

    Returns None while OCR is still running.

    IMPORTANT:

    LLM does NOT wait for this file before starting.

    Individual OCR .ready files are enough to start LLM work.

    This marker only tells LLM:

        "OCR has completely finished.
         No additional OCR batches will appear."
    """

    workspace = (
        get_ocr_workspace(
            patient_name
        )
    )

    completion_path = (
        workspace[
            "completion"
        ]
    )

    if not completion_path.exists():
        return None

    try:

        payload = json.loads(
            completion_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:

        raise RuntimeError(
            "Could not read OCR completion "
            f"marker: {completion_path}"
        ) from error

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "OCR completion marker does not "
            "contain a JSON object."
        )

    if (
        payload.get(
            "status"
        )
        != "completed"
    ):
        return None

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

    expected_patient = (
        safe_patient_name(
            patient_name
        )
    )

    if (
        marker_patient
        != expected_patient
    ):
        raise RuntimeError(
            "OCR completion marker patient "
            "does not match the active patient. "
            f"Expected {expected_patient}, "
            f"found {marker_patient}."
        )

    return payload


# ==========================================================
# PARSE OCR READY MARKER
# ==========================================================

def _load_ocr_ready_marker(
    *,
    patient_name: str,
    ready_path: Path,
) -> OCRBatch:
    """
    Convert one OCR .ready marker into an OCRBatch.

    The .ready file contains metadata only.

    The actual clinical content remains in the TXT referenced
    by "batch_path" / "batch_file".
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    try:

        payload = json.loads(
            ready_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:

        raise RuntimeError(
            "Could not read OCR ready marker: "
            f"{ready_path}"
        ) from error

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "OCR ready marker is not a JSON object: "
            f"{ready_path}"
        )

    if (
        payload.get(
            "status"
        )
        != "ready"
    ):
        raise RuntimeError(
            "OCR ready marker does not have "
            f"status='ready': {ready_path}"
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
        raise RuntimeError(
            "OCR ready marker belongs to the "
            "wrong patient. "
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

        raise RuntimeError(
            "OCR ready marker contains invalid "
            f"batch/page metadata: {ready_path}"
        ) from error

    # ------------------------------------------------------
    # RESOLVE THE CLINICAL TXT
    # ------------------------------------------------------

    batch_path_value = (
        payload.get(
            "batch_path"
        )
    )

    if batch_path_value:

        text_path = Path(
            str(
                batch_path_value
            )
        )

    else:

        batch_file = (
            payload.get(
                "batch_file"
            )
        )

        if not batch_file:
            raise RuntimeError(
                "OCR ready marker does not contain "
                "batch_path or batch_file: "
                f"{ready_path}"
            )

        text_path = (
            ready_path.parent
            / str(
                batch_file
            )
        )

    if not text_path.exists():
        raise FileNotFoundError(
            "OCR ready marker exists but its "
            "clinical TXT does not exist: "
            f"{text_path}"
        )

    if (
        text_path.suffix.lower()
        != ".txt"
    ):
        raise RuntimeError(
            "OCR batch clinical content must be "
            f"a .txt file: {text_path}"
        )

    return OCRBatch(
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
# FIND READY OCR BATCHES
# ==========================================================

def find_ready_ocr_batches(
    patient_name: str,
) -> list[OCRBatch]:
    """
    Find all OCR batches that are completely written and ready
    for LLM processing.

    Searches ONLY:

        ocr/temp/<patient>/batches/*.ready

    It does NOT search:

        data/
        ocr/output/
        another patient's folder

    This is one of the main protections against stale data.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    workspace = (
        get_ocr_workspace(
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

    ready_paths = sorted(
        batch_folder.glob(
            "*.ready"
        )
    )

    batches: list[
        OCRBatch
    ] = []

    seen_batch_numbers: set[
        int
    ] = set()

    for ready_path in ready_paths:

        batch = (
            _load_ocr_ready_marker(
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
            raise RuntimeError(
                "Duplicate OCR batch number "
                f"{batch.batch_number} found for "
                f"{patient_name}."
            )

        seen_batch_numbers.add(
            batch.batch_number
        )

        batches.append(
            batch
        )

    batches.sort(
        key=lambda batch: (
            batch.batch_number
        )
    )

    return batches


# ==========================================================
# LLM PART PATH
# ==========================================================

def get_llm_part_path(
    batch: OCRBatch,
) -> Path:
    """
    Return the temporary LLM result path corresponding to one
    OCR batch.

    Example:

        llm/temp/BO_VAGDEVI/parts/
            LLM_PART_0001_PAGES_0001_TO_0015.txt
    """

    workspace = (
        get_patient_workspace(
            batch.patient_name
        )
    )

    return (
        workspace["parts"]
        / (
            f"LLM_PART_"
            f"{batch.batch_number:04}"
            f"_PAGES_"
            f"{batch.first_page:04}"
            f"_TO_"
            f"{batch.final_page:04}"
            f".txt"
        )
    )


# ==========================================================
# LLM ERROR PATH
# ==========================================================

def get_llm_error_path(
    batch: OCRBatch,
) -> Path:
    """
    Temporary diagnostic error file for one failed LLM batch.
    """

    workspace = (
        get_patient_workspace(
            batch.patient_name
        )
    )

    return (
        workspace["errors"]
        / (
            f"LLM_BATCH_"
            f"{batch.batch_number:04}"
            "_ERROR.txt"
        )
    )


# ==========================================================
# BUILD LLM PROMPT
# ==========================================================

def build_structurer_prompt(
    batch: OCRBatch,
    ocr_text: str,
) -> str:
    """
    Build the prompt for one OCR batch.

    The batch boundaries and source page range are explicitly
    included so the model knows this is only one section of a
    larger patient record.
    """

    return (
        f"{ACTIVE_STRUCTURER_PROMPT}\n\n"
        f"{'=' * 70}\n"
        f"PATIENT RECORD BATCH INFORMATION\n"
        f"{'=' * 70}\n"
        f"Patient: {batch.patient_name}\n"
        f"OCR batch: {batch.batch_number}\n"
        f"Source pages: "
        f"{batch.first_page} to "
        f"{batch.final_page}\n"
        f"Total PDF pages: "
        f"{batch.total_pages}\n"
        f"{'=' * 70}\n\n"
        f"OCR TEXT START\n\n"
        f"{ocr_text}\n\n"
        f"OCR TEXT END"
    )


# ==========================================================
# EXTRACT OLLAMA RESPONSE TEXT
# ==========================================================

def _extract_ollama_response_text(
    response: Any,
) -> str:
    """
    Extract plain assistant text from the Ollama Python client
    response.

    Supports both object-style and dictionary-style responses.
    """

    # Newer Ollama client response object.

    message = getattr(
        response,
        "message",
        None,
    )

    if message is not None:

        content = getattr(
            message,
            "content",
            None,
        )

        if content:
            return str(
                content
            ).strip()

    # Dictionary fallback.

    if isinstance(
        response,
        dict,
    ):

        message_data = (
            response.get(
                "message"
            )
        )

        if isinstance(
            message_data,
            dict,
        ):

            content = (
                message_data.get(
                    "content"
                )
            )

            if content:
                return str(
                    content
                ).strip()

        content = (
            response.get(
                "response"
            )
        )

        if content:
            return str(
                content
            ).strip()

    raise RuntimeError(
        "Ollama returned no readable text."
    )


# ==========================================================
# CREATE OLLAMA CLIENT
# ==========================================================

def create_ollama_client() -> ollama.Client:
    """
    Create one Ollama client.

    Each LLM worker can safely create/use its own client while
    ThreadPoolExecutor coordinates parallel requests.
    """

    return ollama.Client(
        host=(
            OLLAMA_HOST
        ),
        timeout=(
            TIMEOUT_SECONDS
        ),
    )


# ==========================================================
# RUN ONE OLLAMA REQUEST
# ==========================================================

def _run_ollama_request(
    prompt: str,
) -> str:
    """
    Send one OCR batch to Ollama.

    The clinical structurer instructions remain the system
    message, while the actual OCR batch is the user message.
    """

    client = ollama.Client(
        host=OLLAMA_HOST
    )

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": ACTIVE_STRUCTURER_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options=STRUCTURER_OPTIONS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    message = response.get(
        "message",
        {}
    )

    content = str(
        message.get(
            "content",
            ""
        )
    ).strip()

    if not content:

        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return content

# ==========================================================
# PROCESS ONE OCR BATCH
# ==========================================================

def process_ocr_batch(
    batch: OCRBatch,
) -> LLMPartResult:
    """
    Process exactly one OCR batch through the LLM.

    Input:
        OCR temporary .txt

    Output:
        LLM temporary part .txt

    This function can run concurrently for multiple OCR batches.
    """

    start_time = (
        time.perf_counter()
    )

    # ------------------------------------------------------
    # READ ACTUAL CLINICAL TXT
    # ------------------------------------------------------

    ocr_text = (
        read_text_file(
            batch.text_path
        )
    )

    prompt = (
        build_structurer_prompt(
            batch,
            ocr_text,
        )
    )

    output_path = (
        get_llm_part_path(
            batch
        )
    )

    error_path = (
        get_llm_error_path(
            batch
        )
    )

    last_error: Exception | None = (
        None
    )

    # ======================================================
    # RETRIES
    # ======================================================

    for attempt in range(
        1,
        MAX_RETRIES + 2,
    ):

        try:

            print(
                f"[LLM START] "
                f"Batch "
                f"{batch.batch_number} "
                f"| pages "
                f"{batch.first_page}-"
                f"{batch.final_page} "
                f"| attempt "
                f"{attempt}",
                flush=True,
            )

            structured_text = (
                _run_ollama_request(
                    prompt=(
                        prompt
                    ),
                )
            )

            # ----------------------------------------------
            # SAVE CLINICAL OUTPUT AS TXT
            # ----------------------------------------------

            atomic_write(
                output_path,
                structured_text,
            )

            # A previous failed attempt may have left an
            # error diagnostic.

            if error_path.exists():

                error_path.unlink()

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"[LLM COMPLETE] "
                f"Batch "
                f"{batch.batch_number} "
                f"| pages "
                f"{batch.first_page}-"
                f"{batch.final_page} "
                f"| "
                f"{len(structured_text):,} "
                f"characters "
                f"| "
                f"{elapsed:.2f}s",
                flush=True,
            )

            return LLMPartResult(
                batch=(
                    batch
                ),
                output_path=(
                    output_path
                ),
                structured_text=(
                    structured_text
                ),
                elapsed_seconds=(
                    elapsed
                ),
            )

        except Exception as error:

            last_error = (
                error
            )

            error_text = (
                f"Patient: "
                f"{batch.patient_name}\n"
                f"OCR batch: "
                f"{batch.batch_number}\n"
                f"Pages: "
                f"{batch.first_page}-"
                f"{batch.final_page}\n"
                f"Attempt: "
                f"{attempt}\n"
                f"Error type: "
                f"{type(error).__name__}\n"
                f"Error: "
                f"{error}\n\n"
                f"{traceback.format_exc()}"
            )

            atomic_write(
                error_path,
                error_text,
            )

            print(
                f"[LLM ERROR] "
                f"Batch "
                f"{batch.batch_number} "
                f"| attempt "
                f"{attempt} "
                f"| "
                f"{type(error).__name__}: "
                f"{error}",
                flush=True,
            )

            if (
                attempt
                <= MAX_RETRIES
            ):

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        f"LLM batch "
        f"{batch.batch_number} "
        f"failed after "
        f"{MAX_RETRIES + 1} "
        f"attempt(s): "
        f"{last_error}"
    )


# ==========================================================
# RESTORE COMPLETED LLM PART
# ==========================================================

def load_existing_llm_part(
    batch: OCRBatch,
) -> LLMPartResult | None:
    """
    Reload a temporary LLM part that was already successfully
    written during THIS pipeline run.

    This is useful if the streaming loop sees the same OCR
    .ready marker more than once.

    It prevents duplicate model calls.

    Because api/pipeline.py will clear llm/temp/<patient> before
    a NEW patient run, these files cannot accidentally belong to
    an older run.
    """

    output_path = (
        get_llm_part_path(
            batch
        )
    )

    if not output_path.exists():
        return None

    try:

        structured_text = (
            read_text_file(
                output_path
            )
        )

    except Exception:
        return None

    return LLMPartResult(
        batch=(
            batch
        ),
        output_path=(
            output_path
        ),
        structured_text=(
            structured_text
        ),
        elapsed_seconds=0.0,
    )


# ==========================================================
# SUBMIT READY OCR BATCHES
# ==========================================================

def submit_ready_ocr_batches(
    *,
    patient_name: str,
    executor: ThreadPoolExecutor,
    submitted_batch_numbers: set[int],
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
    futures: dict[
        Future[LLMPartResult],
        OCRBatch,
    ],
) -> int:
    """
    Discover newly ready OCR batches and submit them to the
    LLM ThreadPoolExecutor.

    Returns the number of newly submitted/restored batches.

    This function is designed to be called repeatedly while
    OCR is still running.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    ready_batches = (
        find_ready_ocr_batches(
            patient_name
        )
    )

    added = 0

    for batch in ready_batches:

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
            in completed_parts
        ):
            continue

        # --------------------------------------------------
        # CHECK FOR PART ALREADY PRODUCED THIS RUN
        # --------------------------------------------------

        existing_part = (
            load_existing_llm_part(
                batch
            )
        )

        if existing_part is not None:

            completed_parts[
                batch_number
            ] = (
                existing_part
            )

            submitted_batch_numbers.add(
                batch_number
            )

            added += 1

            print(
                f"[LLM RESTORED] "
                f"Batch "
                f"{batch_number}",
                flush=True,
            )

            continue

        # --------------------------------------------------
        # SUBMIT NEW MODEL CALL
        # --------------------------------------------------

        future = (
            executor.submit(
                process_ocr_batch,
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
            f"[LLM QUEUED] "
            f"Batch "
            f"{batch_number} "
            f"| pages "
            f"{batch.first_page}-"
            f"{batch.final_page}",
            flush=True,
        )

    return added


# ==========================================================
# COLLECT FINISHED LLM FUTURES
# ==========================================================

def collect_completed_llm_futures(
    *,
    futures: dict[
        Future[LLMPartResult],
        OCRBatch,
    ],
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
) -> int:
    """
    Collect LLM futures that have finished without blocking on
    requests that are still running.

    Returns the number of newly completed parts.
    """

    finished_futures = [
        future
        for future in list(
            futures
        )
        if future.done()
    ]

    completed_count = 0

    for future in finished_futures:

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

            raise RuntimeError(
                "LLM processing failed for "
                f"OCR batch "
                f"{batch.batch_number} "
                f"(pages "
                f"{batch.first_page}-"
                f"{batch.final_page}): "
                f"{error}"
            ) from error

        completed_parts[
            batch.batch_number
        ] = (
            result
        )

        completed_count += 1

    return completed_count


# ==========================================================
# WAIT FOR REMAINING LLM FUTURES
# ==========================================================

def finish_remaining_llm_futures(
    *,
    futures: dict[
        Future[LLMPartResult],
        OCRBatch,
    ],
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
) -> None:
    """
    Wait for all remaining submitted LLM requests after OCR has
    announced that no more OCR batches will arrive.
    """

    if not futures:
        return

    remaining = dict(
        futures
    )

    futures.clear()

    for future in as_completed(
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

            raise RuntimeError(
                "LLM processing failed for "
                f"OCR batch "
                f"{batch.batch_number} "
                f"(pages "
                f"{batch.first_page}-"
                f"{batch.final_page}): "
                f"{error}"
            ) from error

        completed_parts[
            batch.batch_number
        ] = (
            result
        )


# ==========================================================
# VALIDATE COMPLETED LLM PARTS
# ==========================================================

def validate_completed_llm_parts(
    *,
    patient_name: str,
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
    expected_batch_count: int,
) -> list[LLMPartResult]:
    """
    Validate that every OCR batch has exactly one LLM result.

    This prevents the pipeline from silently continuing when
    one batch is missing.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    if expected_batch_count < 1:
        raise RuntimeError(
            "OCR reported zero batches."
        )

    expected_numbers = set(
        range(
            1,
            expected_batch_count + 1,
        )
    )

    actual_numbers = set(
        completed_parts
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

        raise RuntimeError(
            "LLM is missing completed OCR "
            "batch(es): "
            + ", ".join(
                str(value)
                for value in missing
            )
        )

    if unexpected:

        raise RuntimeError(
            "LLM produced unexpected batch(es): "
            + ", ".join(
                str(value)
                for value in unexpected
            )
        )

    ordered_parts = [
        completed_parts[
            batch_number
        ]
        for batch_number in range(
            1,
            expected_batch_count + 1,
        )
    ]

    # Extra patient consistency protection.

    for part in ordered_parts:

        if (
            safe_patient_name(
                part.batch.patient_name
            )
            != patient_name
        ):

            raise RuntimeError(
                "Completed LLM part belongs "
                "to the wrong patient."
            )

    return ordered_parts
# ==========================================================
# NORMALIZER MANIFEST
# ==========================================================

def load_normalizer_manifest(
    patient_name: str,
) -> dict[str, Any]:
    """
    Load temporary LLM -> Normalizer batching state.

    This file exists only during the active pipeline:

        llm/temp/<patient>/normalizer_batch_manifest.json

    It prevents the same LLM part from being emitted twice.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    path = (
        get_normalizer_manifest_path(
            patient_name
        )
    )

    if not path.exists():

        return {
            "patient_name": (
                patient_name
            ),
            "emitted_parts": [],
            "batches": [],
        }

    try:

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:

        raise RuntimeError(
            "Could not read Normalizer "
            f"batch manifest: {path}"
        ) from error

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Normalizer batch manifest "
            "must contain a JSON object."
        )

    marker_patient = (
        safe_patient_name(
            str(
                payload.get(
                    "patient_name",
                    patient_name,
                )
            )
        )
    )

    if marker_patient != patient_name:

        raise RuntimeError(
            "Normalizer manifest belongs "
            "to a different patient."
        )

    payload.setdefault(
        "emitted_parts",
        [],
    )

    payload.setdefault(
        "batches",
        [],
    )

    return payload


def save_normalizer_manifest(
    *,
    patient_name: str,
    emitted_parts: set[int],
    batches: list[dict[str, Any]],
) -> Path:
    """
    Save temporary batching state atomically.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    path = (
        get_normalizer_manifest_path(
            patient_name
        )
    )

    payload = {
        "patient_name": (
            patient_name
        ),
        "min_batch_characters": (
            NORMALIZER_MIN_BATCH_CHARACTERS
        ),
        "max_batch_characters": (
            NORMALIZER_MAX_BATCH_CHARACTERS
        ),
        "emitted_parts": sorted(
            emitted_parts
        ),
        "batches": (
            batches
        ),
        "updated_at": (
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        ),
    }

    atomic_write(
        path,
        json.dumps(
            payload,
            indent=2,
        ),
    )

    return path


# ==========================================================
# NORMALIZER BATCH PATH
# ==========================================================

def get_normalizer_batch_path(
    *,
    patient_name: str,
    batch_number: int,
    first_page: int,
    final_page: int,
) -> Path:
    """
    Actual clinical TXT consumed by Normalizer.

    Example:

        llm/temp/BO_VAGDEVI/batches/
            LLM_BO_VAGDEVI_BATCH_0001_PAGES_0001_TO_0030.txt
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    workspace = (
        get_patient_workspace(
            patient_name
        )
    )

    return (
        workspace["batches"]
        / (
            f"LLM_{patient_name}"
            f"_BATCH_{batch_number:04}"
            f"_PAGES_{first_page:04}"
            f"_TO_{final_page:04}"
            ".txt"
        )
    )


# ==========================================================
# NORMALIZER READY PATH
# ==========================================================

def get_normalizer_ready_path(
    batch_path: Path,
) -> Path:
    """
    Workflow signal corresponding to one completely written
    Normalizer input TXT.

    Example:

        LLM_BO_VAGDEVI_BATCH_0001_....txt
        LLM_BO_VAGDEVI_BATCH_0001_....ready

    The .ready file is NOT clinical content.
    """

    return (
        batch_path.with_suffix(
            ".ready"
        )
    )


# ==========================================================
# BUILD NORMALIZER BATCH TEXT
# ==========================================================

def build_normalizer_batch_text(
    parts: list[LLMPartResult],
) -> str:
    """
    Combine consecutive LLM parts while preserving source order.

    The resulting plain TXT is what Normalizer will actually read.
    """

    if not parts:
        raise ValueError(
            "Cannot build Normalizer "
            "batch from zero LLM parts."
        )

    ordered_parts = sorted(
        parts,
        key=lambda item: (
            item.batch.batch_number
        ),
    )

    sections: list[str] = []

    for part in ordered_parts:

        sections.extend(
            [
                "=" * 72,
                (
                    "STRUCTURED SOURCE PART "
                    f"{part.batch.batch_number:04}"
                ),
                (
                    "SOURCE PAGES: "
                    f"PAGE_{part.batch.first_page:04} "
                    "TO "
                    f"PAGE_{part.batch.final_page:04}"
                ),
                "=" * 72,
                "",
                (
                    part.structured_text.strip()
                ),
                "",
            ]
        )

    return (
        "\n".join(
            sections
        ).strip()
        + "\n"
    )


# ==========================================================
# EMIT ONE NORMALIZER BATCH
# ==========================================================

def emit_normalizer_batch(
    *,
    patient_name: str,
    output_batch_number: int,
    parts: list[LLMPartResult],
) -> NormalizerReadyBatch:
    """
    Create one Normalizer-ready clinical TXT.

    VERY IMPORTANT ORDER:

        1. Build text.
        2. Atomically write .txt.
        3. Only after TXT exists, write .ready.

    Therefore Normalizer can never legitimately see .ready
    before the corresponding TXT is completely available.
    """

    if not parts:
        raise ValueError(
            "Cannot emit an empty "
            "Normalizer batch."
        )

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    ordered_parts = sorted(
        parts,
        key=lambda item: (
            item.batch.batch_number
        ),
    )

    first_page = (
        ordered_parts[0]
        .batch
        .first_page
    )

    final_page = (
        ordered_parts[-1]
        .batch
        .final_page
    )

    total_pages = max(
        part.batch.total_pages
        for part in ordered_parts
    )

    source_part_numbers = tuple(
        part.batch.batch_number
        for part in ordered_parts
    )

    combined_text = (
        build_normalizer_batch_text(
            ordered_parts
        )
    )

    output_path = (
        get_normalizer_batch_path(
            patient_name=(
                patient_name
            ),
            batch_number=(
                output_batch_number
            ),
            first_page=(
                first_page
            ),
            final_page=(
                final_page
            ),
        )
    )

    ready_path = (
        get_normalizer_ready_path(
            output_path
        )
    )

    # ======================================================
    # STEP 1 — WRITE CLINICAL TXT COMPLETELY
    # ======================================================

    atomic_write(
        output_path,
        combined_text,
    )

    # ======================================================
    # STEP 2 — VERIFY TXT
    # ======================================================

    verified_text = (
        read_text_file(
            output_path
        )
    )

    if not verified_text:

        raise RuntimeError(
            "Normalizer batch TXT was "
            "written but is empty."
        )

    # ======================================================
    # STEP 3 — ONLY NOW CREATE .READY
    # ======================================================

    ready_payload = {
        "patient_name": (
            patient_name
        ),
        "stage": "llm",
        "status": "ready",
        "batch_number": (
            output_batch_number
        ),
        "first_page": (
            first_page
        ),
        "final_page": (
            final_page
        ),
        "total_pages": (
            total_pages
        ),
        "character_count": (
            len(
                combined_text
            )
        ),
        "source_llm_parts": list(
            source_part_numbers
        ),
        "batch_file": (
            output_path.name
        ),
        "batch_path": str(
            output_path.resolve()
        ),
        "created_at": (
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        ),
    }

    atomic_write(
        ready_path,
        json.dumps(
            ready_payload,
            indent=2,
        ),
    )

    print()
    print(
        f"[NORMALIZER INPUT READY] "
        f"Batch "
        f"{output_batch_number:04} "
        f"| pages "
        f"{first_page}-"
        f"{final_page} "
        f"| "
        f"{len(combined_text):,} chars",
        flush=True,
    )

    print(
        f"[NORMALIZER TXT] "
        f"{output_path}",
        flush=True,
    )

    return NormalizerReadyBatch(
        batch_number=(
            output_batch_number
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
        output_path=(
            output_path
        ),
        ready_path=(
            ready_path
        ),
        character_count=(
            len(
                combined_text
            )
        ),
        source_part_numbers=(
            source_part_numbers
        ),
    )


# ==========================================================
# EMIT AVAILABLE NORMALIZER BATCHES
# ==========================================================

def emit_ready_normalizer_batches(
    *,
    patient_name: str,
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
    llm_stage_complete: bool,
) -> list[NormalizerReadyBatch]:
    """
    Turn completed LLM parts into Normalizer-ready TXT batches.

    Rules
    =====

    1. Parts are ALWAYS consumed in OCR/source order.

    2. Only a contiguous sequence can be emitted.

       Example:

           part 1 complete
           part 2 complete
           part 4 complete

       We do NOT jump from 2 to 4.

       We wait for part 3.

    3. Target batch size:

           15,000 - 20,000 characters

       using configured values.

    4. We do not split an individual LLM part in the middle.

    5. Once >= minimum size, if adding another whole part would
       exceed maximum, emit the current batch.

    6. A single LLM part is allowed to exceed maximum because
       splitting clinical text arbitrarily is less desirable.

    7. When the entire LLM stage finishes, the final smaller
       leftover is ALWAYS emitted.

    This final rule is especially important because otherwise
    Normalizer could wait forever for a final batch that never
    becomes large enough.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    manifest = (
        load_normalizer_manifest(
            patient_name
        )
    )

    emitted_parts: set[int] = {
        int(
            value
        )
        for value in (
            manifest.get(
                "emitted_parts",
                [],
            )
        )
    }

    existing_batches: list[
        dict[str, Any]
    ] = list(
        manifest.get(
            "batches",
            [],
        )
    )

    next_output_batch_number = (
        len(
            existing_batches
        )
        + 1
    )

    # ======================================================
    # FIND NEXT CONTIGUOUS PARTS
    # ======================================================

    expected_next_part = (
        max(
            emitted_parts
        )
        + 1
        if emitted_parts
        else 1
    )

    contiguous_parts: list[
        LLMPartResult
    ] = []

    while (
        expected_next_part
        in completed_parts
    ):

        contiguous_parts.append(
            completed_parts[
                expected_next_part
            ]
        )

        expected_next_part += 1

    if not contiguous_parts:
        return []

    created_batches: list[
        NormalizerReadyBatch
    ] = []

    current_parts: list[
        LLMPartResult
    ] = []

    current_characters = 0

    # ======================================================
    # INTERNAL FLUSH
    # ======================================================

    def flush_current() -> None:

        nonlocal current_parts
        nonlocal current_characters
        nonlocal next_output_batch_number

        if not current_parts:
            return

        emitted = (
            emit_normalizer_batch(
                patient_name=(
                    patient_name
                ),
                output_batch_number=(
                    next_output_batch_number
                ),
                parts=(
                    current_parts
                ),
            )
        )

        created_batches.append(
            emitted
        )

        for part in current_parts:

            emitted_parts.add(
                part.batch.batch_number
            )

        existing_batches.append(
            {
                "batch_number": (
                    emitted.batch_number
                ),
                "first_page": (
                    emitted.first_page
                ),
                "final_page": (
                    emitted.final_page
                ),
                "character_count": (
                    emitted.character_count
                ),
                "source_llm_parts": list(
                    emitted.source_part_numbers
                ),
                "batch_file": (
                    emitted.output_path.name
                ),
            }
        )

        save_normalizer_manifest(
            patient_name=(
                patient_name
            ),
            emitted_parts=(
                emitted_parts
            ),
            batches=(
                existing_batches
            ),
        )

        next_output_batch_number += 1

        current_parts = []

        current_characters = 0

    # ======================================================
    # CHARACTER-RANGE AGGREGATION
    # ======================================================

    for part in contiguous_parts:

        part_characters = (
            part.character_count
        )

        # --------------------------------------------------
        # EMPTY CURRENT BATCH
        # --------------------------------------------------

        if not current_parts:

            current_parts.append(
                part
            )

            current_characters = (
                part_characters
            )

            # One LLM result alone may already exceed MAX.

            if (
                current_characters
                >= NORMALIZER_MAX_BATCH_CHARACTERS
            ):

                flush_current()

            continue

        # --------------------------------------------------
        # PROPOSE ADDING NEXT PART
        # --------------------------------------------------

        proposed_characters = (
            current_characters
            + part_characters
        )

        # Current batch is already >= MIN.
        #
        # Adding this part would exceed MAX.
        #
        # Therefore emit current batch first.

        if (
            current_characters
            >= NORMALIZER_MIN_BATCH_CHARACTERS
            and proposed_characters
            > NORMALIZER_MAX_BATCH_CHARACTERS
        ):

            flush_current()

            current_parts.append(
                part
            )

            current_characters = (
                part_characters
            )

            if (
                current_characters
                >= NORMALIZER_MAX_BATCH_CHARACTERS
            ):

                flush_current()

            continue

        # Otherwise add it.

        current_parts.append(
            part
        )

        current_characters = (
            proposed_characters
        )

        if (
            current_characters
            >= NORMALIZER_MAX_BATCH_CHARACTERS
        ):

            flush_current()

    # ======================================================
    # HANDLE REMAINING PARTS
    # ======================================================

    if current_parts:

        if (
            current_characters
            >= NORMALIZER_MIN_BATCH_CHARACTERS
        ):

            flush_current()

        elif llm_stage_complete:

            # CRITICAL:
            #
            # The final leftover may be only 3k, 7k, 10k etc.
            #
            # We MUST still release it once LLM is completely
            # finished.
            #
            # Otherwise Normalizer could wait indefinitely.

            flush_current()

    return created_batches


# ==========================================================
# BUILD COMPLETE LLM TEXT
# ==========================================================

def build_complete_llm_text(
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
) -> str:
    """
    Build the final human-readable LLM stage TXT in exact
    source order.
    """

    if not completed_parts:

        raise ValueError(
            "Cannot build complete LLM output "
            "because there are no completed parts."
        )

    sections: list[str] = []

    for part_number in sorted(
        completed_parts
    ):

        part = (
            completed_parts[
                part_number
            ]
        )

        sections.extend(
            [
                "=" * 72,
                (
                    f"STRUCTURED PART "
                    f"{part_number:04}"
                ),
                (
                    "SOURCE PAGES: "
                    f"PAGE_{part.batch.first_page:04} "
                    "TO "
                    f"PAGE_{part.batch.final_page:04}"
                ),
                "=" * 72,
                "",
                (
                    part.structured_text.strip()
                ),
                "",
                "",
            ]
        )

    return (
        "\n".join(
            sections
        ).rstrip()
        + "\n"
    )


# ==========================================================
# SAVE COMPLETE LLM OUTPUT
# ==========================================================

def save_complete_llm_output(
    *,
    patient_name: str,
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
) -> Path:
    """
    Save final LLM stage output:

        llm/output/LLM_<patient>.txt

    This is NOT moved to data/ here.

    api/pipeline.py will move it only after the entire pipeline
    reaches successful discharge completion.
    """

    output_path = (
        get_final_llm_output_path(
            patient_name
        )
    )

    complete_text = (
        build_complete_llm_text(
            completed_parts
        )
    )

    atomic_write(
        output_path,
        complete_text,
    )

    return output_path


# ==========================================================
# SAVE LLM COMPLETION MARKER
# ==========================================================

def save_llm_completion_marker(
    *,
    patient_name: str,
    total_ocr_batches: int,
    total_normalizer_batches: int,
    complete_path: Path,
) -> Path:
    """
    Create:

        llm/temp/<patient>/llm_complete.json

    Normalizer uses this ONLY to determine:

        - LLM has finished.
        - No additional LLM batches will appear.
        - How many Normalizer-ready batches must exist.

    Clinical information remains in TXT.
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

    payload = {
        "patient_name": (
            patient_name
        ),
        "stage": "llm",
        "status": "completed",
        "model": (
            MODEL_NAME
        ),
        "workers": (
            MAX_LLM_WORKERS
        ),
        "total_ocr_batches": (
            total_ocr_batches
        ),
        "normalizer_min_batch_characters": (
            NORMALIZER_MIN_BATCH_CHARACTERS
        ),
        "normalizer_max_batch_characters": (
            NORMALIZER_MAX_BATCH_CHARACTERS
        ),

        # IMPORTANT:
        #
        # Normalizer uses this as the number of expected
        # Normalizer-ready LLM batches.

        "total_batches": (
            total_normalizer_batches
        ),

        "complete_file": (
            complete_path.name
        ),
        "complete_path": str(
            complete_path.resolve()
        ),
        "completed_at": (
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        ),
    }

    atomic_write(
        marker_path,
        json.dumps(
            payload,
            indent=2,
        ),
    )

    return marker_path


# ==========================================================
# CHECK ALL OCR BATCHES HAVE LLM PARTS
# ==========================================================

def all_expected_llm_parts_complete(
    *,
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
    ocr_completion_data: dict[
        str,
        Any,
    ],
) -> bool:
    """
    Once OCR finishes, determine whether every expected OCR
    batch has a completed LLM result.
    """

    total_batches = int(
        ocr_completion_data[
            "total_batches"
        ]
    )

    if total_batches < 1:
        raise RuntimeError(
            "OCR completion marker reports "
            "zero OCR batches."
        )

    expected = set(
        range(
            1,
            total_batches + 1,
        )
    )

    actual = set(
        completed_parts
    )

    return (
        expected
        <= actual
    )


# ==========================================================
# STREAMING PATIENT PROCESSING
# ==========================================================

def process_patient_stream(
    patient_name: str,
) -> Path:
    """
    Main streaming LLM stage.

    This function may start BEFORE OCR finishes.

    FLOW:

        OCR creates .txt + .ready
                 ↓
        LLM discovers it
                 ↓
        ThreadPoolExecutor processes it
                 ↓
        completed LLM part
                 ↓
        consecutive LLM parts are grouped
                 ↓
        Normalizer TXT + .ready created
                 ↓
        Normalizer can begin immediately

    Once OCR announces completion and all OCR batches have been
    processed:

        final smaller Normalizer batch is released
                 ↓
        llm/output/LLM_<patient>.txt
                 ↓
        llm_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    started_at = (
        time.perf_counter()
    )

    # ======================================================
    # IMPORTANT: DO NOT CLEAR HERE
    # ======================================================
    #
    # api/pipeline.py will eventually clear the workspace
    # BEFORE starting OCR/LLM/Normalizer.
    #
    # Clearing here could delete files while another stage
    # is already running.
    #
    # ======================================================

    get_patient_workspace(
        patient_name
    )

    ensure_llm_output_folder()

    submitted_batch_numbers: set[
        int
    ] = set()

    completed_parts: dict[
        int,
        LLMPartResult,
    ] = {}

    futures: dict[
        Future[LLMPartResult],
        OCRBatch,
    ] = {}

    print()
    print(
        "=" * 72
    )
    print(
        "STREAMING LLM STRUCTURER"
    )
    print(
        "=" * 72
    )
    print(
        f"Patient                : "
        f"{patient_name}"
    )
    print(
        f"Model                  : "
        f"{MODEL_NAME}"
    )
    print(
        f"LLM workers            : "
        f"{MAX_LLM_WORKERS}"
    )
    print(
        "Normalizer batch chars : "
        f"{NORMALIZER_MIN_BATCH_CHARACTERS:,}"
        " - "
        f"{NORMALIZER_MAX_BATCH_CHARACTERS:,}"
    )
    print(
        f"OCR input batches      : "
        f"{get_ocr_workspace(patient_name)['batches']}"
    )
    print(
        f"LLM temp               : "
        f"{get_patient_workspace(patient_name)['root']}"
    )
    print(
        "=" * 72
    )

    # ======================================================
    # LONG-LIVED EXECUTOR
    # ======================================================
    #
    # Unlike the previous design, we do NOT repeatedly create
    # and destroy a ThreadPoolExecutor every polling cycle.
    #
    # The executor lives for the entire LLM stage.
    #
    # ======================================================

    with ThreadPoolExecutor(
        max_workers=(
            MAX_LLM_WORKERS
        )
    ) as executor:

        while True:

            # ==================================================
            # 1. DISCOVER NEW OCR BATCHES
            # ==================================================

            submit_ready_ocr_batches(
                patient_name=(
                    patient_name
                ),
                executor=(
                    executor
                ),
                submitted_batch_numbers=(
                    submitted_batch_numbers
                ),
                completed_parts=(
                    completed_parts
                ),
                futures=(
                    futures
                ),
            )

            # ==================================================
            # 2. COLLECT FINISHED LLM REQUESTS
            # ==================================================

            newly_completed = (
                collect_completed_llm_futures(
                    futures=(
                        futures
                    ),
                    completed_parts=(
                        completed_parts
                    ),
                )
            )

            # ==================================================
            # 3. STREAM COMPLETED LLM TEXT TO NORMALIZER
            # ==================================================

            if newly_completed:

                emit_ready_normalizer_batches(
                    patient_name=(
                        patient_name
                    ),
                    completed_parts=(
                        completed_parts
                    ),
                    llm_stage_complete=False,
                )

            # ==================================================
            # 4. CHECK WHETHER OCR HAS COMPLETELY FINISHED
            # ==================================================

            ocr_completion_data = (
                load_ocr_completion_data(
                    patient_name
                )
            )

            if (
                ocr_completion_data
                is not None
            ):

                total_ocr_batches = int(
                    ocr_completion_data[
                        "total_batches"
                    ]
                )

                # There may be a very small timing window:
                #
                # OCR completion marker exists but this LLM
                # polling iteration has not yet discovered the
                # final .ready marker.
                #
                # Therefore discover one more time.

                submit_ready_ocr_batches(
                    patient_name=(
                        patient_name
                    ),
                    executor=(
                        executor
                    ),
                    submitted_batch_numbers=(
                        submitted_batch_numbers
                    ),
                    completed_parts=(
                        completed_parts
                    ),
                    futures=(
                        futures
                    ),
                )

                expected_numbers = set(
                    range(
                        1,
                        total_ocr_batches + 1,
                    )
                )

                discovered_numbers = (
                    set(
                        submitted_batch_numbers
                    )
                    | set(
                        completed_parts
                    )
                )

                # If we haven't even discovered every expected
                # OCR batch yet, keep polling.

                if not (
                    expected_numbers
                    <= discovered_numbers
                ):

                    time.sleep(
                        POLL_SECONDS
                    )

                    continue

                # If every OCR batch has been discovered, but
                # some model requests are still running, collect
                # completed ones and continue.

                if futures:

                    collect_completed_llm_futures(
                        futures=(
                            futures
                        ),
                        completed_parts=(
                            completed_parts
                        ),
                    )

                # Are all expected LLM results actually complete?

                if (
                    all_expected_llm_parts_complete(
                        completed_parts=(
                            completed_parts
                        ),
                        ocr_completion_data=(
                            ocr_completion_data
                        ),
                    )
                ):

                    break

            time.sleep(
                POLL_SECONDS
            )

        # ==================================================
        # WAIT FOR ANY REMAINING FUTURES
        # ==================================================

        finish_remaining_llm_futures(
            futures=(
                futures
            ),
            completed_parts=(
                completed_parts
            ),
        )

    # ======================================================
    # OCR MUST NOW HAVE COMPLETION DATA
    # ======================================================

    ocr_completion_data = (
        load_ocr_completion_data(
            patient_name
        )
    )

    if ocr_completion_data is None:

        raise RuntimeError(
            "LLM reached finalization but OCR "
            "completion marker is missing."
        )

    total_ocr_batches = int(
        ocr_completion_data[
            "total_batches"
        ]
    )

    # ======================================================
    # VALIDATE EXACTLY ONE LLM RESULT PER OCR BATCH
    # ======================================================

    validate_completed_llm_parts(
        patient_name=(
            patient_name
        ),
        completed_parts=(
            completed_parts
        ),
        expected_batch_count=(
            total_ocr_batches
        ),
    )

    # ======================================================
    # RELEASE FINAL NORMALIZER TAIL
    # ======================================================
    #
    # CRITICAL FOR THE PREVIOUS NORMALIZER PROBLEM.
    #
    # Even if the final remaining LLM output is below the
    # configured minimum character count, it MUST become a
    # Normalizer-ready TXT once LLM is fully complete.
    #
    # ======================================================

    emit_ready_normalizer_batches(
        patient_name=(
            patient_name
        ),
        completed_parts=(
            completed_parts
        ),
        llm_stage_complete=True,
    )

    # ======================================================
    # VALIDATE NORMALIZER MANIFEST
    # ======================================================

    manifest = (
        load_normalizer_manifest(
            patient_name
        )
    )

    emitted_parts = {
        int(
            value
        )
        for value in (
            manifest.get(
                "emitted_parts",
                [],
            )
        )
    }

    expected_part_numbers = set(
        range(
            1,
            total_ocr_batches + 1,
        )
    )

    if (
        emitted_parts
        != expected_part_numbers
    ):

        missing = sorted(
            expected_part_numbers
            - emitted_parts
        )

        raise RuntimeError(
            "Not every completed LLM part was "
            "released to Normalizer. Missing "
            "part(s): "
            + ", ".join(
                str(value)
                for value in missing
            )
        )

    normalizer_batches = list(
        manifest.get(
            "batches",
            [],
        )
    )

    total_normalizer_batches = len(
        normalizer_batches
    )

    if (
        total_normalizer_batches
        < 1
    ):

        raise RuntimeError(
            "LLM completed successfully but "
            "zero Normalizer-ready batches "
            "were created."
        )

    # ======================================================
    # VALIDATE EVERY NORMALIZER BATCH HAS TXT + READY
    # ======================================================

    llm_workspace = (
        get_patient_workspace(
            patient_name
        )
    )

    for batch_info in (
        normalizer_batches
    ):

        batch_file = str(
            batch_info[
                "batch_file"
            ]
        )

        batch_path = (
            llm_workspace[
                "batches"
            ]
            / batch_file
        )

        ready_path = (
            get_normalizer_ready_path(
                batch_path
            )
        )

        if not batch_path.exists():

            raise RuntimeError(
                "Normalizer clinical TXT "
                "is missing: "
                f"{batch_path}"
            )

        if not ready_path.exists():

            raise RuntimeError(
                "Normalizer ready marker "
                "is missing: "
                f"{ready_path}"
            )

        # Ensure TXT is readable and non-empty.

        read_text_file(
            batch_path
        )

    # ======================================================
    # CREATE FINAL LLM OUTPUT
    # ======================================================

    complete_path = (
        save_complete_llm_output(
            patient_name=(
                patient_name
            ),
            completed_parts=(
                completed_parts
            ),
        )
    )

    # ======================================================
    # IMPORTANT ORDERING
    # ======================================================
    #
    # llm_complete.json is created ONLY AFTER:
    #
    #   - every OCR batch has an LLM result
    #   - every LLM part has been assigned to Normalizer
    #   - every Normalizer batch TXT exists
    #   - every corresponding .ready exists
    #   - final LLM TXT exists
    #
    # This means Normalizer cannot legitimately see
    # "LLM completed" before all its input files exist.
    #
    # ======================================================

    completion_marker = (
        save_llm_completion_marker(
            patient_name=(
                patient_name
            ),
            total_ocr_batches=(
                total_ocr_batches
            ),
            total_normalizer_batches=(
                total_normalizer_batches
            ),
            complete_path=(
                complete_path
            ),
        )
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
        "LLM STAGE COMPLETED"
    )
    print(
        "=" * 72
    )
    print(
        f"Patient                : "
        f"{patient_name}"
    )
    print(
        f"OCR batches processed  : "
        f"{total_ocr_batches}"
    )
    print(
        f"Normalizer batches     : "
        f"{total_normalizer_batches}"
    )
    print(
        f"Final LLM output       : "
        f"{complete_path}"
    )
    print(
        f"Completion marker      : "
        f"{completion_marker}"
    )
    print(
        f"LLM temporary data     : "
        f"{llm_workspace['root']}"
    )
    print(
        f"Total LLM time         : "
        f"{elapsed:.2f} seconds"
    )
    print(
        "=" * 72
    )

    # ======================================================
    # DO NOT CLEAN ANYTHING HERE
    # ======================================================
    #
    # Specifically DO NOT DELETE:
    #
    #   ocr/temp/<patient>/
    #   llm/temp/<patient>/parts/
    #   llm/temp/<patient>/batches/
    #   *.ready
    #   llm_complete.json
    #
    # Normalizer may still be actively consuming these files.
    #
    # api/pipeline.py will perform cleanup only after:
    #
    # OCR -> LLM -> Normalizer -> Discharge
    #
    # have ALL succeeded.
    #
    # ======================================================

    return complete_path


# ==========================================================
# OLLAMA CONNECTION CHECK
# ==========================================================

def check_ollama_connection() -> bool:
    """
    Verify that Ollama is reachable and the configured model
    is installed.
    """

    print(
        "Checking Ollama connection...",
        flush=True,
    )

    try:

        client = (
            create_ollama_client()
        )

        response = (
            client.list()
        )

    except Exception as error:

        print(
            "Could not connect to Ollama: "
            f"{error}",
            flush=True,
        )

        return False

    if isinstance(
        response,
        dict,
    ):

        models = (
            response.get(
                "models",
                [],
            )
        )

    else:

        models = (
            getattr(
                response,
                "models",
                [],
            )
        )

    available_names: list[
        str
    ] = []

    for model in models:

        if isinstance(
            model,
            dict,
        ):

            model_name = (
                model.get(
                    "model"
                )
                or model.get(
                    "name"
                )
                or ""
            )

        else:

            model_name = (
                getattr(
                    model,
                    "model",
                    "",
                )
                or getattr(
                    model,
                    "name",
                    "",
                )
            )

        if model_name:

            available_names.append(
                str(
                    model_name
                )
            )

    requested_base = (
        MODEL_NAME.split(
            ":"
        )[0]
    )

    model_found = any(
        (
            available_name
            == MODEL_NAME
        )
        or (
            available_name.split(
                ":"
            )[0]
            == requested_base
        )
        for available_name
        in available_names
    )

    if not model_found:

        print(
            f"Model not installed: "
            f"{MODEL_NAME}",
            flush=True,
        )

        return False

    print(
        f"Ollama ready. Model: "
        f"{MODEL_NAME}",
        flush=True,
    )

    return True


# ==========================================================
# STANDALONE PATIENT DISCOVERY
# ==========================================================

def find_streaming_patients() -> list[str]:
    """
    Find patients currently represented inside OCR temporary
    streaming storage.

    Searches ONLY:

        ocr/temp/

    Never searches data/.
    """

    if not OCR_TEMP_FOLDER.exists():
        return []

    patients: list[str] = []

    for patient_folder in (
        OCR_TEMP_FOLDER.iterdir()
    ):

        if not patient_folder.is_dir():
            continue

        batch_folder = (
            patient_folder
            / "batches"
        )

        completion_marker = (
            patient_folder
            / "ocr_complete.json"
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
                safe_patient_name(
                    patient_folder.name
                )
            )

    return sorted(
        set(
            patients
        )
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Standalone LLM runner.

    The normal web pipeline calls:

        process_patient_stream(patient_name)

    directly.
    """

    overall_start = (
        time.perf_counter()
    )

    print()
    print(
        "=" * 72
    )
    print(
        "NICU STREAMING LLM STRUCTURER"
    )
    print(
        "=" * 72
    )
    print(
        f"OCR temp   : "
        f"{OCR_TEMP_FOLDER}"
    )
    print(
        f"LLM temp   : "
        f"{LLM_TEMP_FOLDER}"
    )
    print(
        f"LLM output : "
        f"{LLM_OUTPUT_FOLDER}"
    )
    print(
        f"Model      : "
        f"{MODEL_NAME}"
    )
    print(
        f"Workers    : "
        f"{MAX_LLM_WORKERS}"
    )
    print(
        "=" * 72
    )

    if not check_ollama_connection():

        raise SystemExit(
            1
        )

    patients = (
        find_streaming_patients()
    )

    if not patients:

        print(
            "No active OCR streaming "
            "patients found."
        )

        return

    successful = 0

    failed = 0

    for patient_name in patients:

        try:

            process_patient_stream(
                patient_name
            )

            successful += 1

        except KeyboardInterrupt:

            print(
                "LLM stage stopped by user."
            )

            raise

        except Exception as error:

            failed += 1

            print()
            print(
                "=" * 72
            )
            print(
                "LLM STAGE FAILED"
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
            print(
                traceback.format_exc()
            )

    elapsed = (
        time.perf_counter()
        - overall_start
    )

    print()
    print(
        "=" * 72
    )
    print(
        "LLM RUN FINISHED"
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
        f"{elapsed:.2f} sec"
    )
    print(
        "=" * 72
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()