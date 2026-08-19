"""
Streaming parallel NICU clinical normalizer.

The normalizer watches:

    data/<patient>/llm/batches/*.ready

Each completed LLM batch is normalized immediately using configurable
parallel workers.

Permanent output:

    data/<patient>/normalizer/
        <patient>_normalized_complete.txt

Temporary batch output:

    data/<patient>/normalizer/batches/

Temporary LLM and normalized batches are removed after the complete
normalized file is created successfully.
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
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import requests

from normalizer.clinical_terms import (
    normalize_clinical_terms,
)
from normalizer.normalizer_model import (
    DATA_FOLDER,
    ERROR_FOLDER,
    FILE_ENCODING,
    KEEP_ALIVE,
    MODEL_NAME,
    NORMALIZED_COMPLETE_SUFFIX,
    NORMALIZER_MAX_WORKERS,
    NORMALIZER_POLL_SECONDS,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    OPTIONS,
    RETRIES,
    RETRY_DELAY_SECONDS,
    TIMEOUT_SECONDS,
)
from normalizer.prompts import NORMALIZER_PROMPT


# ==========================================================
# CONSTANTS
# ==========================================================

LLM_BATCH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"_llm_batch_(\d+)_pages_(\d+)_to_(\d+)\.txt$",
    re.IGNORECASE,
)


# ==========================================================
# EXCEPTIONS
# ==========================================================

class NormalizerError(RuntimeError):
    """
    Base exception for normalizer failures.
    """


class OllamaConnectionError(NormalizerError):
    """
    Raised when Ollama cannot be reached.
    """


class OllamaModelError(NormalizerError):
    """
    Raised when the configured model is unavailable.
    """


class BatchNormalizationError(NormalizerError):
    """
    Raised when one LLM batch cannot be normalized.
    """


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass(frozen=True, slots=True)
class LLMBatch:
    """
    One LLM batch waiting for normalization.
    """

    patient_name: str
    batch_number: int
    first_page: int
    final_page: int
    total_pages: int
    text_path: Path
    ready_path: Path


@dataclass(frozen=True, slots=True)
class NormalizedBatchResult:
    """
    Result produced from one normalized batch.
    """

    batch: LLMBatch
    output_path: Path
    normalized_text: str
    elapsed_seconds: float


# ==========================================================
# GENERAL FILE HELPERS
# ==========================================================

def safe_patient_name(
    value: str,
) -> str:
    """
    Convert a patient identifier into a safe folder name.
    """

    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value.strip(),
    )

    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    ).strip("_")

    return cleaned or "patient"


def read_text_file(
    path: Path,
) -> str:
    """
    Read one UTF-8 text file.
    """

    text = path.read_text(
        encoding=FILE_ENCODING,
        errors="replace",
    ).strip()

    if not text:
        raise ValueError(
            f"Input file is empty: {path}"
        )

    return text


def atomic_write(
    path: Path,
    text: str,
) -> Path:
    """
    Write text through a temporary file and rename it safely.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        text.strip() + "\n",
        encoding=FILE_ENCODING,
    )

    temporary_path.replace(
        path
    )

    return path


def remove_code_fences(
    text: str,
) -> str:
    """
    Remove accidental Markdown fences from model output.
    """

    cleaned = text.strip()

    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()

    if (
        lines
        and lines[0].strip().startswith("```")
    ):
        lines = lines[1:]

    if (
        lines
        and lines[-1].strip() == "```"
    ):
        lines = lines[:-1]

    return "\n".join(
        lines
    ).strip()


# ==========================================================
# PATIENT FOLDERS
# ==========================================================

def get_patient_folders(
    patient_name: str,
) -> dict[str, Path]:
    """
    Create and return folders used by the normalizer.
    """

    patient_root = (
        DATA_FOLDER
        / safe_patient_name(
            patient_name
        )
    )

    folders = {
        "root": patient_root,

        "llm": (
            patient_root
            / "llm"
        ),

        "llm_batches": (
            patient_root
            / "llm"
            / "batches"
        ),

        "normalizer": (
            patient_root
            / "normalizer"
        ),

        "normalizer_batches": (
            patient_root
            / "normalizer"
            / "batches"
        ),

        "normalizer_errors": (
            patient_root
            / "normalizer"
            / "errors"
        ),
    }

    for folder in folders.values():
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    return folders


# ==========================================================
# LLM READY-MARKER DISCOVERY
# ==========================================================

def load_llm_ready_marker(
    ready_path: Path,
) -> LLMBatch:
    """
    Read one LLM-ready marker.
    """

    payload = json.loads(
        ready_path.read_text(
            encoding=FILE_ENCODING,
        )
    )

    patient_name = str(
        payload["patient_name"]
    )

    batch_number = int(
        payload["batch_number"]
    )

    first_page = int(
        payload["first_page"]
    )

    final_page = int(
        payload["final_page"]
    )

    total_pages = int(
        payload["total_pages"]
    )

    batch_filename = str(
        payload["batch_file"]
    )

    text_path = (
        ready_path.parent
        / batch_filename
    )

    if not text_path.exists():
        raise FileNotFoundError(
            f"LLM batch file not found: {text_path}"
        )

    return LLMBatch(
        patient_name=patient_name,
        batch_number=batch_number,
        first_page=first_page,
        final_page=final_page,
        total_pages=total_pages,
        text_path=text_path,
        ready_path=ready_path,
    )


def find_ready_llm_batches(
    patient_name: str,
) -> list[LLMBatch]:
    """
    Find all LLM batches currently ready for normalization.
    """

    folders = get_patient_folders(
        patient_name
    )

    ready_files = sorted(
        folders[
            "llm_batches"
        ].glob("*.ready")
    )

    batches: list[LLMBatch] = []

    for ready_file in ready_files:
        try:
            batch = load_llm_ready_marker(
                ready_file
            )

            batches.append(
                batch
            )

        except Exception as error:
            print(
                f"Could not read LLM marker "
                f"{ready_file}: {error}",
                flush=True,
            )

    return sorted(
        batches,
        key=lambda item: (
            item.batch_number
        ),
    )


def discover_patient_names() -> list[str]:
    """
    Find patient folders containing LLM-ready batches.
    """

    if not DATA_FOLDER.exists():
        return []

    patient_names: list[str] = []

    for patient_folder in (
        DATA_FOLDER.iterdir()
    ):
        if not patient_folder.is_dir():
            continue

        llm_folder = (
            patient_folder
            / "llm"
        )

        batch_folder = (
            llm_folder
            / "batches"
        )

        completion_marker = (
            llm_folder
            / "llm_complete.json"
        )

        has_ready_batches = (
            batch_folder.exists()
            and any(
                batch_folder.glob(
                    "*.ready"
                )
            )
        )

        if (
            has_ready_batches
            or completion_marker.exists()
        ):
            patient_names.append(
                patient_folder.name
            )

    return sorted(
        patient_names
    )


# ==========================================================
# NORMALIZED BATCH OUTPUT PATHS
# ==========================================================

def create_normalized_batch_path(
    batch: LLMBatch,
) -> Path:
    """
    Return the output path for one normalized batch.
    """

    folders = get_patient_folders(
        batch.patient_name
    )

    filename = (
        f"{safe_patient_name(batch.patient_name)}"
        f"_normalized_batch_{batch.batch_number:04}"
        f"_pages_{batch.first_page:04}_to_"
        f"{batch.final_page:04}.txt"
    )

    return (
        folders[
            "normalizer_batches"
        ]
        / filename
    )


def create_normalizer_ready_path(
    output_path: Path,
) -> Path:
    """
    Return the discharge-stage ready-marker path.
    """

    return output_path.with_suffix(
        ".ready"
    )


def normalized_batch_already_exists(
    batch: LLMBatch,
) -> bool:
    """
    Return True when one normalized batch already exists.
    """

    output_path = (
        create_normalized_batch_path(
            batch
        )
    )

    ready_path = (
        create_normalizer_ready_path(
            output_path
        )
    )

    return (
        output_path.exists()
        and ready_path.exists()
    )
# ==========================================================
# OLLAMA VALIDATION
# ==========================================================

def model_name_matches(
    configured_model: str,
    installed_model: str,
) -> bool:
    """
    Compare model names with or without the ':latest' suffix.
    """

    configured = configured_model.strip()
    installed = installed_model.strip()

    if configured == installed:
        return True

    configured_base = configured.removesuffix(
        ":latest"
    )

    installed_base = installed.removesuffix(
        ":latest"
    )

    return configured_base == installed_base


def get_installed_models() -> list[str]:
    """
    Return installed Ollama model names.
    """

    try:
        response = requests.get(
            OLLAMA_TAGS_ENDPOINT,
            timeout=30,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError as error:
        raise OllamaConnectionError(
            "Could not connect to Ollama."
        ) from error

    except requests.exceptions.Timeout as error:
        raise OllamaConnectionError(
            "Timed out while checking Ollama."
        ) from error

    except requests.exceptions.RequestException as error:
        raise OllamaConnectionError(
            f"Ollama connection check failed: {error}"
        ) from error

    try:
        payload = response.json()

    except ValueError as error:
        raise OllamaConnectionError(
            "Ollama returned invalid JSON."
        ) from error

    models = payload.get(
        "models",
        [],
    )

    names: list[str] = []

    if not isinstance(
        models,
        list,
    ):
        return names

    for model in models:
        if not isinstance(
            model,
            dict,
        ):
            continue

        model_name = (
            model.get("name")
            or model.get("model")
        )

        if (
            isinstance(model_name, str)
            and model_name.strip()
        ):
            names.append(
                model_name.strip()
            )

    return names


def verify_ollama() -> None:
    """
    Confirm Ollama is running and the selected model is installed.
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
            f"Model '{MODEL_NAME}' is not installed.\n"
            f"Installed models: {installed_text}\n"
            f"Run: ollama pull {MODEL_NAME}"
        )

    print(
        f"Ollama ready. Model: {MODEL_NAME}",
        flush=True,
    )


# ==========================================================
# PROMPT
# ==========================================================

def build_batch_prompt(
    batch: LLMBatch,
    batch_text: str,
) -> str:
    """
    Build the prompt for one LLM batch.
    """

    return (
        f"{NORMALIZER_PROMPT.strip()}\n\n"
        "============================================================\n"
        "STREAMING BATCH INFORMATION\n"
        "============================================================\n"
        f"Patient: {batch.patient_name}\n"
        f"Batch number: {batch.batch_number}\n"
        f"Source pages: "
        f"{batch.first_page} to {batch.final_page}\n"
        f"Total patient pages: {batch.total_pages}\n\n"
        "Normalize only the clinical text in this batch.\n"
        "Do not assume information from earlier or later batches.\n"
        "Do not create a discharge summary.\n"
        "Do not output JSON.\n"
        "Return only normalized clinical text.\n\n"
        "============================================================\n"
        "BEGIN STRUCTURED CLINICAL TEXT\n"
        "============================================================\n"
        f"{batch_text.strip()}\n"
        "============================================================\n"
        "END STRUCTURED CLINICAL TEXT\n"
        "============================================================\n"
    )


# ==========================================================
# OLLAMA RESPONSE
# ==========================================================

def extract_response_text(
    payload: dict[str, Any],
) -> str:
    """
    Extract generated text from Ollama.
    """

    response_text = payload.get(
        "response"
    )

    if not isinstance(
        response_text,
        str,
    ):
        raise NormalizerError(
            "Ollama response did not contain valid text."
        )

    response_text = response_text.strip()

    if not response_text:
        raise NormalizerError(
            "Ollama returned an empty response."
        )

    return response_text


def call_ollama(
    prompt: str,
) -> tuple[str, dict[str, Any]]:
    """
    Normalize one batch using the local Ollama server.
    """

    request_payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": OPTIONS,
    }

    try:
        response = requests.post(
            OLLAMA_GENERATE_ENDPOINT,
            json=request_payload,
            timeout=TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError as error:
        raise OllamaConnectionError(
            "Lost connection to Ollama during normalization."
        ) from error

    except requests.exceptions.Timeout as error:
        raise NormalizerError(
            f"Normalizer request exceeded "
            f"{TIMEOUT_SECONDS} seconds."
        ) from error

    except requests.exceptions.HTTPError as error:
        response_body = ""

        if error.response is not None:
            response_body = (
                error.response.text.strip()
            )

        message = (
            f"Ollama returned an HTTP error: {error}"
        )

        if response_body:
            message += (
                f"\nOllama response: {response_body}"
            )

        raise NormalizerError(
            message
        ) from error

    except requests.exceptions.RequestException as error:
        raise NormalizerError(
            f"Ollama request failed: {error}"
        ) from error

    try:
        payload = response.json()

    except ValueError as error:
        raise NormalizerError(
            "Ollama returned invalid JSON."
        ) from error

    if not isinstance(
        payload,
        dict,
    ):
        raise NormalizerError(
            "Ollama returned an unexpected response format."
        )

    generated_text = extract_response_text(
        payload
    )

    cleaned_text = remove_code_fences(
        generated_text
    )

    if not cleaned_text:
        raise NormalizerError(
            "The cleaned normalizer response is empty."
        )

    return (
        cleaned_text,
        payload,
    )


# ==========================================================
# VALIDATION
# ==========================================================

def validate_normalized_batch(
    input_text: str,
    output_text: str,
) -> None:
    """
    Detect empty or severely truncated output.
    """

    stripped_input = input_text.strip()
    stripped_output = output_text.strip()

    if not stripped_input:
        raise NormalizerError(
            "Normalizer input batch is empty."
        )

    if not stripped_output:
        raise NormalizerError(
            "Normalizer output batch is empty."
        )

    input_length = len(
        stripped_input
    )

    output_length = len(
        stripped_output
    )

    minimum_length = max(
        int(
            input_length * 0.10
        ),
        500,
    )

    if output_length < minimum_length:
        raise NormalizerError(
            "Normalized output appears severely truncated. "
            f"Input: {input_length:,} characters; "
            f"output: {output_length:,} characters; "
            f"minimum expected: {minimum_length:,}."
        )


# ==========================================================
# PERFORMANCE
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

    prompt_count = payload.get(
        "prompt_eval_count"
    )

    prompt_duration = payload.get(
        "prompt_eval_duration"
    )

    generation_count = payload.get(
        "eval_count"
    )

    generation_duration = payload.get(
        "eval_duration"
    )

    prompt_rate: float | None = None
    generation_rate: float | None = None

    if (
        isinstance(prompt_count, int)
        and isinstance(prompt_duration, int)
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
        isinstance(generation_count, int)
        and isinstance(generation_duration, int)
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
# ERROR FILES
# ==========================================================

def save_batch_error(
    *,
    batch: LLMBatch,
    error: Exception,
) -> Path:
    """
    Save an error report for one failed normalized batch.
    """

    folders = get_patient_folders(
        batch.patient_name
    )

    error_path = (
        folders[
            "normalizer_errors"
        ]
        / (
            f"{safe_patient_name(batch.patient_name)}"
            f"_normalized_batch_"
            f"{batch.batch_number:04}_error.txt"
        )
    )

    error_text = (
        f"Patient: {batch.patient_name}\n"
        f"Batch: {batch.batch_number}\n"
        f"Pages: {batch.first_page}-{batch.final_page}\n"
        f"Model: {MODEL_NAME}\n"
        f"Error type: {type(error).__name__}\n"
        f"Error: {error}\n\n"
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
    Normalize one LLM batch with retries.
    """

    started_at = time.perf_counter()

    batch_text = read_text_file(
        batch.text_path
    )

    prompt = build_batch_prompt(
        batch,
        batch_text,
    )

    total_attempts = max(
        1,
        RETRIES + 1,
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            model_output, response_payload = (
                call_ollama(
                    prompt
                )
            )

            validate_normalized_batch(
                batch_text,
                model_output,
            )

            (
                clinically_normalized_text,
                _,
            ) = normalize_clinical_terms(
                model_output
            )

            validate_normalized_batch(
                batch_text,
                clinically_normalized_text,
            )

            output_path = (
                create_normalized_batch_path(
                    batch
                )
            )

            atomic_write(
                output_path,
                clinically_normalized_text,
            )

            ready_path = (
                create_normalizer_ready_path(
                    output_path
                )
            )

            ready_payload = {
                "patient_name": batch.patient_name,
                "stage": "normalizer",
                "batch_number": batch.batch_number,
                "first_page": batch.first_page,
                "final_page": batch.final_page,
                "total_pages": batch.total_pages,
                "batch_file": output_path.name,
                "source_llm_batch": batch.text_path.name,
                "status": "ready",
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
            }

            atomic_write(
                ready_path,
                json.dumps(
                    ready_payload,
                    indent=2,
                ),
            )

            elapsed = (
                time.perf_counter()
                - started_at
            )

            (
                prompt_rate,
                generation_rate,
            ) = calculate_generation_statistics(
                response_payload
            )

            performance = [
                f"{elapsed:.1f}s",
                (
                    f"{len(clinically_normalized_text):,} "
                    "output characters"
                ),
            ]

            if prompt_rate is not None:
                performance.append(
                    f"{prompt_rate:.1f} prompt tokens/s"
                )

            if generation_rate is not None:
                performance.append(
                    (
                        f"{generation_rate:.1f} "
                        "generation tokens/s"
                    )
                )

            print(
                f"[NORMALIZER COMPLETE] "
                f"Batch {batch.batch_number:04}: "
                + " | ".join(
                    performance
                ),
                flush=True,
            )

            return NormalizedBatchResult(
                batch=batch,
                output_path=output_path,
                normalized_text=(
                    clinically_normalized_text
                ),
                elapsed_seconds=elapsed,
            )

        except Exception as error:
            last_error = error

            print(
                f"[NORMALIZER RETRY] "
                f"Batch {batch.batch_number:04} "
                f"attempt {attempt}/{total_attempts} "
                f"failed: {error}",
                flush=True,
            )

            if attempt < total_attempts:
                delay = (
                    RETRY_DELAY_SECONDS
                    * attempt
                )

                time.sleep(
                    delay
                )

    assert last_error is not None

    error_path = save_batch_error(
        batch=batch,
        error=last_error,
    )

    raise BatchNormalizationError(
        f"Normalizer batch {batch.batch_number} failed. "
        f"Error saved to: {error_path}"
    )
# ==========================================================
# LLM HANDOFF CLEANUP
# ==========================================================

def mark_llm_batch_consumed(
    batch: LLMBatch,
) -> None:
    """
    Remove the LLM-ready marker after successful normalization.

    The LLM batch TXT remains temporarily until the complete normalized
    file has been created successfully.
    """

    if batch.ready_path.exists():
        batch.ready_path.unlink()


# ==========================================================
# PARALLEL BATCH PROCESSING
# ==========================================================

def process_available_batches(
    patient_name: str,
    *,
    submitted_batches: set[int],
    completed_batches: dict[
        int,
        NormalizedBatchResult,
    ],
    failed_batches: dict[int, str],
) -> int:
    """
    Process all currently available LLM batches in parallel.

    Returns the number of newly discovered batches.
    """

    ready_batches = find_ready_llm_batches(
        patient_name
    )

    pending_batches = [
        batch
        for batch in ready_batches
        if (
            batch.batch_number
            not in submitted_batches
            and batch.batch_number
            not in completed_batches
            and batch.batch_number
            not in failed_batches
        )
    ]

    if not pending_batches:
        return 0

    print()
    print(
        f"Submitting {len(pending_batches)} "
        f"LLM batch(es) to "
        f"{NORMALIZER_MAX_WORKERS} "
        "normalizer worker(s)...",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=NORMALIZER_MAX_WORKERS,
    ) as executor:

        futures: dict[
            Future[NormalizedBatchResult],
            LLMBatch,
        ] = {}

        for batch in pending_batches:
            if normalized_batch_already_exists(
                batch
            ):
                output_path = (
                    create_normalized_batch_path(
                        batch
                    )
                )

                normalized_text = read_text_file(
                    output_path
                )

                completed_batches[
                    batch.batch_number
                ] = NormalizedBatchResult(
                    batch=batch,
                    output_path=output_path,
                    normalized_text=(
                        normalized_text
                    ),
                    elapsed_seconds=0.0,
                )

                submitted_batches.add(
                    batch.batch_number
                )

                mark_llm_batch_consumed(
                    batch
                )

                print(
                    f"[NORMALIZER SKIP] "
                    f"Batch {batch.batch_number:04} "
                    "already completed.",
                    flush=True,
                )

                continue

            submitted_batches.add(
                batch.batch_number
            )

            future = executor.submit(
                process_llm_batch,
                batch,
            )

            futures[
                future
            ] = batch

        for future in as_completed(
            futures
        ):
            batch = futures[
                future
            ]

            try:
                result = future.result()

                completed_batches[
                    batch.batch_number
                ] = result

                mark_llm_batch_consumed(
                    batch
                )

                print(
                    f"[NORMALIZER SAVED] "
                    f"Batch {batch.batch_number:04} "
                    f"(pages "
                    f"{batch.first_page}-"
                    f"{batch.final_page})",
                    flush=True,
                )

                print(
                    f"Saved: "
                    f"{result.output_path}",
                    flush=True,
                )

            except Exception as error:
                failed_batches[
                    batch.batch_number
                ] = str(
                    error
                )

                print(
                    f"[NORMALIZER FAILED] "
                    f"Batch {batch.batch_number:04}: "
                    f"{error}",
                    flush=True,
                )

    return len(
        pending_batches
    )


# ==========================================================
# LLM COMPLETION CHECK
# ==========================================================

def get_llm_completion_marker(
    patient_name: str,
) -> Path:
    """
    Return the LLM completion-marker path.
    """

    folders = get_patient_folders(
        patient_name
    )

    return (
        folders[
            "llm"
        ]
        / "llm_complete.json"
    )


def load_llm_completion_data(
    patient_name: str,
) -> dict[str, Any] | None:
    """
    Load LLM completion metadata after all LLM batches finish.
    """

    marker_path = (
        get_llm_completion_marker(
            patient_name
        )
    )

    if not marker_path.exists():
        return None

    try:
        payload = json.loads(
            marker_path.read_text(
                encoding=FILE_ENCODING,
            )
        )

    except Exception as error:
        raise RuntimeError(
            "Could not read LLM completion marker: "
            f"{marker_path}"
        ) from error

    if (
        payload.get("status")
        != "completed"
    ):
        return None

    return payload


def expected_batch_count(
    patient_name: str,
) -> int | None:
    """
    Return expected batch count after LLM completion.
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


def all_normalizer_batches_completed(
    patient_name: str,
    completed_batches: dict[
        int,
        NormalizedBatchResult,
    ],
) -> bool:
    """
    Return True when every expected normalized batch exists.
    """

    total_batches = expected_batch_count(
        patient_name
    )

    if total_batches is None:
        return False

    expected_numbers = set(
        range(
            1,
            total_batches + 1,
        )
    )

    completed_numbers = set(
        completed_batches.keys()
    )

    return (
        expected_numbers
        <= completed_numbers
    )


# ==========================================================
# LOAD EXISTING NORMALIZED BATCHES
# ==========================================================

NORMALIZED_BATCH_PATTERN: Final[
    re.Pattern[str]
] = re.compile(
    r"_normalized_batch_(\d+)_pages_"
    r"(\d+)_to_(\d+)\.txt$",
    re.IGNORECASE,
)


def load_existing_normalized_batches(
    patient_name: str,
) -> dict[
    int,
    NormalizedBatchResult,
]:
    """
    Reload normalized batches already stored on disk.
    """

    folders = get_patient_folders(
        patient_name
    )

    results: dict[
        int,
        NormalizedBatchResult,
    ] = {}

    completion_data = (
        load_llm_completion_data(
            patient_name
        )
    )

    for output_path in sorted(
        folders[
            "normalizer_batches"
        ].glob("*.txt")
    ):
        match = (
            NORMALIZED_BATCH_PATTERN.search(
                output_path.name
            )
        )

        if match is None:
            continue

        batch_number = int(
            match.group(1)
        )

        first_page = int(
            match.group(2)
        )

        final_page = int(
            match.group(3)
        )

        total_pages = final_page

        if completion_data is not None:
            total_pages = int(
                completion_data.get(
                    "total_pages",
                    final_page,
                )
            )

        placeholder_llm_path = (
            folders[
                "llm_batches"
            ]
            / (
                f"{safe_patient_name(patient_name)}"
                f"_llm_batch_{batch_number:04}"
                f"_pages_{first_page:04}_to_"
                f"{final_page:04}.txt"
            )
        )

        batch = LLMBatch(
            patient_name=patient_name,
            batch_number=batch_number,
            first_page=first_page,
            final_page=final_page,
            total_pages=total_pages,
            text_path=placeholder_llm_path,
            ready_path=(
                placeholder_llm_path.with_suffix(
                    ".ready"
                )
            ),
        )

        results[
            batch_number
        ] = NormalizedBatchResult(
            batch=batch,
            output_path=output_path,
            normalized_text=(
                read_text_file(
                    output_path
                )
            ),
            elapsed_seconds=0.0,
        )

        ready_path = (
            create_normalizer_ready_path(
                output_path
            )
        )

        if not ready_path.exists():
            ready_payload = {
                "patient_name": (
                    patient_name
                ),
                "stage": "normalizer",
                "batch_number": (
                    batch_number
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
                "batch_file": (
                    output_path.name
                ),
                "status": "ready",
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

    return results


# ==========================================================
# COMPLETE NORMALIZED OUTPUT
# ==========================================================

def build_complete_normalized_text(
    batch_results: dict[
        int,
        NormalizedBatchResult,
    ],
) -> str:
    """
    Merge normalized batches in original page order.
    """

    sections: list[str] = []

    for batch_number in sorted(
        batch_results
    ):
        result = batch_results[
            batch_number
        ]

        batch = result.batch

        sections.extend(
            [
                "#" * 72,
                (
                    f"NORMALIZED BATCH "
                    f"{batch.batch_number:04}"
                ),
                (
                    f"SOURCE PAGES: "
                    f"PAGE_{batch.first_page:04} "
                    f"TO "
                    f"PAGE_{batch.final_page:04}"
                ),
                "#" * 72,
                "",
                result.normalized_text.strip(),
                "",
                "",
            ]
        )

    combined_text = (
        "\n".join(
            sections
        ).rstrip()
        + "\n"
    )

    final_text, _ = (
        normalize_clinical_terms(
            combined_text
        )
    )

    return final_text.strip() + "\n"


def save_complete_normalized_output(
    *,
    patient_name: str,
    batch_results: dict[
        int,
        NormalizedBatchResult,
    ],
) -> Path:
    """
    Save the merged normalized patient record.
    """

    folders = get_patient_folders(
        patient_name
    )

    complete_path = (
        folders[
            "normalizer"
        ]
        / (
            f"{safe_patient_name(patient_name)}"
            f"{NORMALIZED_COMPLETE_SUFFIX}"
        )
    )

    complete_text = (
        build_complete_normalized_text(
            batch_results
        )
    )

    atomic_write(
        complete_path,
        complete_text,
    )

    return complete_path


def save_normalizer_completion_marker(
    *,
    patient_name: str,
    total_batches: int,
    complete_path: Path,
) -> Path:
    """
    Create the normalizer completion marker.
    """

    folders = get_patient_folders(
        patient_name
    )

    marker_path = (
        folders[
            "normalizer"
        ]
        / "normalizer_complete.json"
    )

    payload = {
        "patient_name": (
            patient_name
        ),
        "stage": "normalizer",
        "status": "completed",
        "model": MODEL_NAME,
        "workers": (
            NORMALIZER_MAX_WORKERS
        ),
        "total_batches": (
            total_batches
        ),
        "complete_file": (
            complete_path.name
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
# TEMPORARY BATCH CLEANUP
# ==========================================================

def cleanup_temporary_llm_batches(
    patient_name: str,
) -> None:
    """
    Remove temporary LLM batch files after normalization succeeds.

    The merged complete LLM file remains permanently stored.
    """

    folders = get_patient_folders(
        patient_name
    )

    batch_folder = folders[
        "llm_batches"
    ]

    if not batch_folder.exists():
        return

    for path in batch_folder.iterdir():
        if path.is_file():
            path.unlink()


def cleanup_temporary_normalizer_batches(
    patient_name: str,
) -> None:
    """
    Remove temporary normalized batch files after the merged file is saved.
    """

    folders = get_patient_folders(
        patient_name
    )

    batch_folder = folders[
        "normalizer_batches"
    ]

    if not batch_folder.exists():
        return

    for path in batch_folder.iterdir():
        if path.is_file():
            path.unlink()


# ==========================================================
# STREAMING PATIENT PROCESSING
# ==========================================================

def process_patient_stream(
    patient_name: str,
) -> Path:
    """
    Watch LLM-ready batches and normalize them immediately.

    Returns the complete merged normalized patient record.
    """

    started_at = time.perf_counter()

    submitted_batches: set[int] = set()

    completed_batches = (
        load_existing_normalized_batches(
            patient_name
        )
    )

    failed_batches: dict[int, str] = {}

    print()
    print("=" * 72)
    print("STREAMING CLINICAL NORMALIZER")
    print("=" * 72)
    print(f"Patient            : {patient_name}")
    print(f"Model              : {MODEL_NAME}")
    print(f"Normalizer workers : {NORMALIZER_MAX_WORKERS}")
    print("=" * 72)

    while True:
        process_available_batches(
            patient_name,
            submitted_batches=(
                submitted_batches
            ),
            completed_batches=(
                completed_batches
            ),
            failed_batches=(
                failed_batches
            ),
        )

        if failed_batches:
            details = ", ".join(
                (
                    f"batch {batch_number}: "
                    f"{message}"
                )
                for batch_number, message
                in sorted(
                    failed_batches.items()
                )
            )

            raise RuntimeError(
                "One or more normalizer batches failed: "
                + details
            )

        if all_normalizer_batches_completed(
            patient_name,
            completed_batches,
        ):
            break

        time.sleep(
            NORMALIZER_POLL_SECONDS
        )

    total_batches = expected_batch_count(
        patient_name
    )

    if total_batches is None:
        raise RuntimeError(
            "LLM completion marker was not found."
        )

    complete_path = (
        save_complete_normalized_output(
            patient_name=patient_name,
            batch_results=(
                completed_batches
            ),
        )
    )

    save_normalizer_completion_marker(
        patient_name=patient_name,
        total_batches=total_batches,
        complete_path=complete_path,
    )

    cleanup_temporary_llm_batches(
        patient_name
    )

    cleanup_temporary_normalizer_batches(
        patient_name
    )

    elapsed = (
        time.perf_counter()
        - started_at
    )

    print()
    print("=" * 72)
    print("NORMALIZER STAGE COMPLETED")
    print("=" * 72)
    print(
        f"Batches completed : "
        f"{len(completed_batches)}"
    )
    print(
        f"Complete file     : "
        f"{complete_path}"
    )
    print(
        f"Total time        : "
        f"{elapsed:.2f} seconds"
    )
    print("=" * 72)

    return complete_path


# ==========================================================
# PATIENT DISCOVERY
# ==========================================================

def find_streaming_patients() -> list[str]:
    """
    Find patients with LLM batches or a completed LLM record.
    """

    if not DATA_FOLDER.exists():
        return []

    patients: list[str] = []

    for patient_folder in (
        DATA_FOLDER.iterdir()
    ):
        if not patient_folder.is_dir():
            continue

        llm_folder = (
            patient_folder
            / "llm"
        )

        batch_folder = (
            llm_folder
            / "batches"
        )

        completion_marker = (
            llm_folder
            / "llm_complete.json"
        )

        has_ready_batches = (
            batch_folder.exists()
            and any(
                batch_folder.glob(
                    "*.ready"
                )
            )
        )

        if (
            has_ready_batches
            or completion_marker.exists()
        ):
            patients.append(
                patient_folder.name
            )

    return sorted(
        patients
    )


# ==========================================================
# HEADER
# ==========================================================

def print_header() -> None:
    """
    Display the active streaming normalizer configuration.
    """

    print("=" * 72)
    print("NICU STREAMING CLINICAL NORMALIZER")
    print("=" * 72)
    print(f"Data folder        : {DATA_FOLDER}")
    print(f"Model              : {MODEL_NAME}")
    print(
        f"Normalizer workers : "
        f"{NORMALIZER_MAX_WORKERS}"
    )
    print(
        f"Polling interval   : "
        f"{NORMALIZER_POLL_SECONDS}s"
    )
    print(f"Retries            : {RETRIES}")
    print("=" * 72)


# ==========================================================
# MAIN
# ==========================================================

def main() -> int:
    """
    Process all patients waiting for normalization.
    """

    overall_start = time.perf_counter()

    print_header()

    try:
        verify_ollama()

        patients = find_streaming_patients()

        if not patients:
            print()
            print(
                "No LLM-ready batches or completed "
                "LLM patient records were found."
            )

            return 0

        successful = 0
        failed = 0

        for patient_name in patients:
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
                print("=" * 72)
                print("NORMALIZER STAGE FAILED")
                print("=" * 72)
                print(
                    f"Patient: {patient_name}"
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
        print("=" * 72)
        print("NORMALIZER RUN FINISHED")
        print("=" * 72)
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
        print("=" * 72)

        if failed:
            return 1

        return 0

    except KeyboardInterrupt:
        return 130

    except (
        OllamaConnectionError,
        OllamaModelError,
        BatchNormalizationError,
        NormalizerError,
        FileNotFoundError,
        PermissionError,
        ValueError,
        TypeError,
    ) as error:
        print()
        print(
            f"ERROR [{type(error).__name__}]: "
            f"{error}"
        )

        return 1

    except Exception as error:
        print()
        print(
            "UNEXPECTED ERROR "
            f"[{type(error).__name__}]: "
            f"{error}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )