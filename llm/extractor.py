"""
Streaming LLM structurer for the NICU AI pipeline.

Flow:

OCR character batches
    data/<patient>/ocr/batches/*.ready
        ↓
LLM structuring
        ↓
temporary ordered LLM parts
    data/<patient>/llm/parts/
        ↓
character-range aggregation for the Normalizer
        ↓
15,000–20,000 character normalizer-ready LLM batches
    data/<patient>/llm/batches/*.ready
        ↓
Normalizer

Permanent LLM output:
    data/<patient>/llm/<patient>_llm_complete.txt

Important:
- OCR decides its own input batches by character range.
- This module processes each OCR batch through the LLM.
- LLM outputs are then accumulated in source order for the Normalizer.
- Normalizer-ready batches target a configurable character range.
- The final leftover batch is released when the LLM stage is complete.
- Medical data remains plain text. JSON files are workflow metadata only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import ollama

from llm import models as model_config
from llm.prompts import STRUCTURER_PROMPT


# ==========================================================
# ENVIRONMENT
# ==========================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_FOLDER: Final[Path] = PROJECT_ROOT / "data"
ERROR_FOLDER: Final[Path] = PROJECT_ROOT / "llm" / "errors"


# ==========================================================
# MODEL SETTINGS
# ==========================================================

MODEL_NAME: Final[str] = model_config.MODEL_NAME
STRUCTURER_OPTIONS: Final[dict[str, Any]] = model_config.STRUCTURER_OPTIONS

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

# These settings control ONLY how completed LLM outputs are grouped
# before they are handed to the Normalizer.
#
# They do not change the OCR -> LLM batch size.

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

if NORMALIZER_MIN_BATCH_CHARACTERS < 1:
    raise ValueError(
        "NORMALIZER_MIN_BATCH_CHARACTERS must be at least 1."
    )

if (
    NORMALIZER_MAX_BATCH_CHARACTERS
    < NORMALIZER_MIN_BATCH_CHARACTERS
):
    raise ValueError(
        "NORMALIZER_MAX_BATCH_CHARACTERS must be greater than "
        "or equal to NORMALIZER_MIN_BATCH_CHARACTERS."
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

if isinstance(STRUCTURER_PROMPT, str) and STRUCTURER_PROMPT.strip():
    ACTIVE_STRUCTURER_PROMPT = STRUCTURER_PROMPT.strip()
else:
    ACTIVE_STRUCTURER_PROMPT = DEFAULT_STRUCTURER_PROMPT


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass(frozen=True, slots=True)
class OCRBatch:
    patient_name: str
    batch_number: int
    first_page: int
    final_page: int
    total_pages: int
    text_path: Path
    ready_path: Path


@dataclass(frozen=True, slots=True)
class LLMPartResult:
    """
    One model result corresponding to one OCR character batch.
    """
    batch: OCRBatch
    output_path: Path
    structured_text: str
    elapsed_seconds: float

    @property
    def character_count(self) -> int:
        return len(self.structured_text)


@dataclass(frozen=True, slots=True)
class NormalizerReadyBatch:
    """
    One character-range batch emitted by the LLM stage for Normalizer.
    """
    batch_number: int
    first_page: int
    final_page: int
    total_pages: int
    output_path: Path
    character_count: int
    source_part_numbers: tuple[int, ...]


# ==========================================================
# FILE HELPERS
# ==========================================================

def safe_patient_name(value: str) -> str:
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


def read_text_file(path: Path) -> str:
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not text:
        raise ValueError(
            f"Input text file is empty: {path}"
        )

    return text


def atomic_write(
    path: Path,
    text: str,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        text.strip() + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)

    return path


def remove_code_fences(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines)

    return cleaned.strip()


def get_patient_folders(
    patient_name: str,
) -> dict[str, Path]:
    patient_root = (
        DATA_FOLDER
        / safe_patient_name(patient_name)
    )

    folders = {
        "root": patient_root,
        "ocr": patient_root / "ocr",
        "ocr_batches": (
            patient_root / "ocr" / "batches"
        ),
        "llm": patient_root / "llm",
        # Individual LLM outputs corresponding to OCR batches.
        # These are temporary and used to preserve order/resume.
        "llm_parts": (
            patient_root / "llm" / "parts"
        ),
        # Character-range files consumed by the Normalizer.
        "llm_batches": (
            patient_root / "llm" / "batches"
        ),
        "llm_errors": (
            patient_root / "llm" / "errors"
        ),
    }

    for folder in folders.values():
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    return folders


# ==========================================================
# OCR BATCH DISCOVERY
# ==========================================================

def load_ready_marker(
    ready_path: Path,
) -> OCRBatch:
    payload = json.loads(
        ready_path.read_text(
            encoding="utf-8",
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

    batch_file_name = str(
        payload["batch_file"]
    )

    text_path = (
        ready_path.parent
        / batch_file_name
    )

    if not text_path.exists():
        raise FileNotFoundError(
            f"OCR batch file not found: {text_path}"
        )

    return OCRBatch(
        patient_name=patient_name,
        batch_number=batch_number,
        first_page=first_page,
        final_page=final_page,
        total_pages=total_pages,
        text_path=text_path,
        ready_path=ready_path,
    )


def find_ready_ocr_batches(
    patient_name: str,
) -> list[OCRBatch]:
    folders = get_patient_folders(
        patient_name
    )

    ready_files = sorted(
        folders[
            "ocr_batches"
        ].glob("*.ready")
    )

    batches: list[OCRBatch] = []

    for ready_file in ready_files:
        try:
            batches.append(
                load_ready_marker(
                    ready_file
                )
            )

        except Exception as error:
            print(
                f"Could not read OCR marker "
                f"{ready_file}: {error}",
                flush=True,
            )

    return sorted(
        batches,
        key=lambda batch: batch.batch_number,
    )


# ==========================================================
# PER-OCR-BATCH LLM PART OUTPUT
# ==========================================================

def create_llm_part_output_path(
    batch: OCRBatch,
) -> Path:
    folders = get_patient_folders(
        batch.patient_name
    )

    output_name = (
        f"{safe_patient_name(batch.patient_name)}"
        f"_llm_part_{batch.batch_number:04}"
        f"_pages_{batch.first_page:04}_to_"
        f"{batch.final_page:04}.txt"
    )

    return (
        folders["llm_parts"]
        / output_name
    )


def llm_part_already_processed(
    batch: OCRBatch,
) -> bool:
    return create_llm_part_output_path(
        batch
    ).exists()


# ==========================================================
# NORMALIZER-READY OUTPUT PATHS
# ==========================================================

def create_normalizer_batch_output_path(
    *,
    patient_name: str,
    batch_number: int,
    first_page: int,
    final_page: int,
) -> Path:
    folders = get_patient_folders(
        patient_name
    )

    output_name = (
        f"{safe_patient_name(patient_name)}"
        f"_llm_batch_{batch_number:04}"
        f"_pages_{first_page:04}_to_"
        f"{final_page:04}.txt"
    )

    return (
        folders["llm_batches"]
        / output_name
    )


def create_llm_ready_path(
    output_path: Path,
) -> Path:
    return output_path.with_suffix(
        ".ready"
    )


# ==========================================================
# OLLAMA HELPERS
# ==========================================================

def create_ollama_client() -> ollama.Client:
    return ollama.Client(
        host=OLLAMA_HOST,
        timeout=TIMEOUT_SECONDS,
    )


def extract_response_content(
    response: object,
) -> str:
    if isinstance(response, dict):
        message = response.get(
            "message",
            {},
        )

        if isinstance(message, dict):
            content = message.get(
                "content",
                "",
            )
        else:
            content = getattr(
                message,
                "content",
                "",
            )

    else:
        message = getattr(
            response,
            "message",
            None,
        )

        if isinstance(message, dict):
            content = message.get(
                "content",
                "",
            )
        else:
            content = getattr(
                message,
                "content",
                "",
            )

    if not isinstance(content, str):
        content = str(content)

    return content.strip()


def build_batch_prompt(
    batch: OCRBatch,
    batch_text: str,
) -> str:
    return f"""
PATIENT FILE NAME:
{batch.patient_name}

OCR BATCH:
{batch.batch_number}

SOURCE PAGE RANGE:
PAGE_{batch.first_page:04}
TO
PAGE_{batch.final_page:04}

TOTAL PATIENT PAGES:
{batch.total_pages}

The patient filename is metadata only.

Do not treat the filename as a clinical fact unless the same information
appears in the OCR text.

The OCR text below contains consecutive pages from one NICU patient record.

Process every supplied page.

Keep reliable clinical facts connected to their source page.

OCR BATCH START
========================================================================

{batch_text}

========================================================================
OCR BATCH END

Structure all supplied pages according to the system instructions.

Return plain structured clinical text only.
""".strip()


def structure_batch_with_model(
    batch: OCRBatch,
) -> str:
    batch_text = read_text_file(
        batch.text_path
    )

    client = create_ollama_client()

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": ACTIVE_STRUCTURER_PROMPT,
            },
            {
                "role": "user",
                "content": build_batch_prompt(
                    batch,
                    batch_text,
                ),
            },
        ],
        options=STRUCTURER_OPTIONS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    structured_text = (
        extract_response_content(
            response
        )
    )

    structured_text = remove_code_fences(
        structured_text
    )

    if not structured_text:
        raise ValueError(
            "The LLM returned an empty structured response."
        )

    return structured_text


# ==========================================================
# ERROR HANDLING
# ==========================================================

def save_batch_error(
    *,
    batch: OCRBatch,
    error: Exception,
) -> Path:
    folders = get_patient_folders(
        batch.patient_name
    )

    error_path = (
        folders["llm_errors"]
        / (
            f"{safe_patient_name(batch.patient_name)}"
            f"_part_{batch.batch_number:04}"
            "_error.txt"
        )
    )

    error_text = (
        f"Patient: {batch.patient_name}\n"
        f"OCR batch: {batch.batch_number}\n"
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
# PROCESS ONE OCR BATCH THROUGH LLM
# ==========================================================

def process_ocr_batch(
    batch: OCRBatch,
) -> LLMPartResult:
    start_time = time.perf_counter()

    total_attempts = max(
        1,
        MAX_RETRIES + 1,
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            structured_text = (
                structure_batch_with_model(
                    batch
                )
            )

            output_path = (
                create_llm_part_output_path(
                    batch
                )
            )

            atomic_write(
                output_path,
                structured_text,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            return LLMPartResult(
                batch=batch,
                output_path=output_path,
                structured_text=structured_text,
                elapsed_seconds=elapsed,
            )

        except Exception as error:
            last_error = error

            print(
                f"[LLM RETRY] "
                f"OCR batch {batch.batch_number} "
                f"attempt {attempt}/{total_attempts} failed: "
                f"{error}",
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

    raise RuntimeError(
        f"LLM processing for OCR batch "
        f"{batch.batch_number} failed. "
        f"Error saved to: {error_path}"
    )


# ==========================================================
# OCR HANDOFF CLEANUP
# ==========================================================

def mark_ocr_batch_consumed(
    batch: OCRBatch,
) -> None:
    """
    Remove only the ready marker after successful LLM processing.
    OCR batch TXT remains until the complete LLM stage succeeds.
    """

    if batch.ready_path.exists():
        batch.ready_path.unlink()


# ==========================================================
# LOAD EXISTING LLM PARTS
# ==========================================================

LLM_PART_PATTERN = re.compile(
    r"_llm_part_(\d+)_pages_"
    r"(\d+)_to_(\d+)\.txt$",
    re.IGNORECASE,
)


def load_existing_llm_parts(
    patient_name: str,
) -> dict[int, LLMPartResult]:
    folders = get_patient_folders(
        patient_name
    )

    results: dict[
        int,
        LLMPartResult,
    ] = {}

    completion_data = load_ocr_completion_data(
        patient_name
    )

    for output_path in sorted(
        folders["llm_parts"].glob(
            "*.txt"
        )
    ):
        match = LLM_PART_PATTERN.search(
            output_path.name
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

        total_pages = (
            int(
                completion_data[
                    "total_pages"
                ]
            )
            if completion_data is not None
            else final_page
        )

        ocr_batch_path = (
            folders["ocr_batches"]
            / (
                f"{safe_patient_name(patient_name)}"
                f"_ocr_batch_{batch_number:04}"
                f"_pages_{first_page:04}_to_"
                f"{final_page:04}.txt"
            )
        )

        batch = OCRBatch(
            patient_name=patient_name,
            batch_number=batch_number,
            first_page=first_page,
            final_page=final_page,
            total_pages=total_pages,
            text_path=ocr_batch_path,
            ready_path=ocr_batch_path.with_suffix(
                ".ready"
            ),
        )

        results[
            batch_number
        ] = LLMPartResult(
            batch=batch,
            output_path=output_path,
            structured_text=read_text_file(
                output_path
            ),
            elapsed_seconds=0.0,
        )

    return results


# ==========================================================
# PARALLEL LLM PART PROCESSING
# ==========================================================

def process_available_batches(
    patient_name: str,
    *,
    submitted_batches: set[int],
    completed_parts: dict[int, LLMPartResult],
    failed_batches: dict[int, str],
) -> int:
    ready_batches = find_ready_ocr_batches(
        patient_name
    )

    pending_batches = [
        batch
        for batch in ready_batches
        if (
            batch.batch_number
            not in submitted_batches
            and batch.batch_number
            not in completed_parts
            and batch.batch_number
            not in failed_batches
        )
    ]

    if not pending_batches:
        return 0

    print()
    print(
        f"Submitting {len(pending_batches)} "
        f"OCR batch(es) to "
        f"{MAX_LLM_WORKERS} LLM worker(s)...",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=MAX_LLM_WORKERS,
    ) as executor:

        futures: dict[
            Future[LLMPartResult],
            OCRBatch,
        ] = {}

        for batch in pending_batches:
            if llm_part_already_processed(
                batch
            ):
                output_path = (
                    create_llm_part_output_path(
                        batch
                    )
                )

                completed_parts[
                    batch.batch_number
                ] = LLMPartResult(
                    batch=batch,
                    output_path=output_path,
                    structured_text=read_text_file(
                        output_path
                    ),
                    elapsed_seconds=0.0,
                )

                submitted_batches.add(
                    batch.batch_number
                )

                mark_ocr_batch_consumed(
                    batch
                )

                print(
                    f"[LLM SKIP] "
                    f"OCR batch {batch.batch_number:04} "
                    "already structured.",
                    flush=True,
                )

                continue

            submitted_batches.add(
                batch.batch_number
            )

            future = executor.submit(
                process_ocr_batch,
                batch,
            )

            futures[future] = batch

        for future in as_completed(
            futures
        ):
            batch = futures[
                future
            ]

            try:
                result = future.result()

                completed_parts[
                    batch.batch_number
                ] = result

                mark_ocr_batch_consumed(
                    batch
                )

                print(
                    f"[LLM COMPLETE] "
                    f"OCR batch {batch.batch_number:04} "
                    f"(pages "
                    f"{batch.first_page}-"
                    f"{batch.final_page}) "
                    f"=> "
                    f"{result.character_count:,} chars "
                    f"in "
                    f"{result.elapsed_seconds:.1f} sec",
                    flush=True,
                )

            except Exception as error:
                failed_batches[
                    batch.batch_number
                ] = str(
                    error
                )

                print(
                    f"[LLM FAILED] "
                    f"OCR batch {batch.batch_number:04}: "
                    f"{error}",
                    flush=True,
                )

    return len(
        pending_batches
    )


# ==========================================================
# OCR COMPLETION CHECK
# ==========================================================

def get_ocr_completion_marker(
    patient_name: str,
) -> Path:
    folders = get_patient_folders(
        patient_name
    )

    return (
        folders["ocr"]
        / "ocr_complete.json"
    )


def load_ocr_completion_data(
    patient_name: str,
) -> dict[str, Any] | None:
    marker_path = (
        get_ocr_completion_marker(
            patient_name
        )
    )

    if not marker_path.exists():
        return None

    try:
        payload = json.loads(
            marker_path.read_text(
                encoding="utf-8",
            )
        )

    except Exception as error:
        raise RuntimeError(
            f"Could not read OCR completion marker: "
            f"{marker_path}"
        ) from error

    if (
        payload.get("status")
        != "completed"
    ):
        return None

    return payload


def expected_ocr_batch_count(
    patient_name: str,
) -> int | None:
    completion_data = (
        load_ocr_completion_data(
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


def all_llm_parts_completed(
    patient_name: str,
    completed_parts: dict[
        int,
        LLMPartResult,
    ],
) -> bool:
    total_batches = expected_ocr_batch_count(
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
        completed_parts.keys()
    )

    return (
        expected_numbers
        <= completed_numbers
    )


# ==========================================================
# NORMALIZER BATCH MANIFEST
# ==========================================================

def get_normalizer_batch_manifest_path(
    patient_name: str,
) -> Path:
    folders = get_patient_folders(
        patient_name
    )

    return (
        folders["llm"]
        / "normalizer_batch_manifest.json"
    )


def load_normalizer_batch_manifest(
    patient_name: str,
) -> dict[str, Any]:
    path = get_normalizer_batch_manifest_path(
        patient_name
    )

    if not path.exists():
        return {
            "patient_name": patient_name,
            "emitted_parts": [],
            "batches": [],
        }

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

        if not isinstance(payload, dict):
            raise ValueError(
                "Manifest root must be an object."
            )

        payload.setdefault(
            "patient_name",
            patient_name,
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

    except Exception as error:
        raise RuntimeError(
            f"Could not read LLM normalizer-batch manifest: "
            f"{path}"
        ) from error


def save_normalizer_batch_manifest(
    patient_name: str,
    payload: dict[str, Any],
) -> None:
    path = get_normalizer_batch_manifest_path(
        patient_name
    )

    atomic_write(
        path,
        json.dumps(
            payload,
            indent=2,
        ),
    )


# ==========================================================
# CHARACTER-RANGE AGGREGATION
# ==========================================================

def build_normalizer_batch_text(
    parts: list[LLMPartResult],
) -> str:
    sections: list[str] = []

    for part in parts:
        sections.extend(
            [
                "#" * 72,
                (
                    f"STRUCTURED SOURCE PART "
                    f"{part.batch.batch_number:04}"
                ),
                (
                    f"SOURCE PAGES: "
                    f"PAGE_{part.batch.first_page:04} "
                    f"TO "
                    f"PAGE_{part.batch.final_page:04}"
                ),
                "#" * 72,
                "",
                part.structured_text.strip(),
                "",
            ]
        )

    return (
        "\n".join(
            sections
        ).strip()
        + "\n"
    )


def emit_normalizer_batch(
    *,
    patient_name: str,
    output_batch_number: int,
    parts: list[LLMPartResult],
) -> NormalizerReadyBatch:
    if not parts:
        raise ValueError(
            "Cannot emit an empty Normalizer batch."
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

    combined_text = (
        build_normalizer_batch_text(
            ordered_parts
        )
    )

    output_path = (
        create_normalizer_batch_output_path(
            patient_name=patient_name,
            batch_number=output_batch_number,
            first_page=first_page,
            final_page=final_page,
        )
    )

    atomic_write(
        output_path,
        combined_text,
    )

    ready_path = create_llm_ready_path(
        output_path
    )

    part_numbers = tuple(
        part.batch.batch_number
        for part in ordered_parts
    )

    ready_payload = {
        "patient_name": patient_name,
        "stage": "llm",
        "batch_number": output_batch_number,
        "first_page": first_page,
        "final_page": final_page,
        "total_pages": total_pages,
        "batch_file": output_path.name,
        "character_count": len(
            combined_text
        ),
        "source_llm_parts": list(
            part_numbers
        ),
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

    print()
    print(
        f"[NORMALIZER BATCH READY] "
        f"Batch {output_batch_number:04} "
        f"(pages {first_page}-{final_page}, "
        f"{len(combined_text):,} chars, "
        f"LLM parts "
        f"{part_numbers[0]}-{part_numbers[-1]})",
        flush=True,
    )
    print(
        f"Saved: {output_path}",
        flush=True,
    )

    return NormalizerReadyBatch(
        batch_number=output_batch_number,
        first_page=first_page,
        final_page=final_page,
        total_pages=total_pages,
        output_path=output_path,
        character_count=len(
            combined_text
        ),
        source_part_numbers=part_numbers,
    )


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
    Emit consecutive LLM output into 15k-20k character Normalizer batches.

    Rules:
    - Parts are always consumed in OCR/source order.
    - A batch becomes eligible after MIN chars.
    - More whole parts are added while staying <= MAX.
    - If the next whole part would exceed MAX, the current eligible batch
      is emitted first.
    - If the current batch is still below MIN, the next whole part is
      accepted even if this makes the batch slightly exceed MAX. We prefer
      not to split a structured LLM result in the middle.
    - At end-of-LLM, any remaining text is emitted even below MIN.
    """

    manifest = load_normalizer_batch_manifest(
        patient_name
    )

    emitted_parts = {
        int(value)
        for value in manifest.get(
            "emitted_parts",
            [],
        )
    }

    existing_batches = list(
        manifest.get(
            "batches",
            [],
        )
    )

    next_output_batch_number = (
        len(existing_batches)
        + 1
    )

    available_numbers = sorted(
        number
        for number in completed_parts
        if number not in emitted_parts
    )

    if not available_numbers:
        return []

    # Only consume a contiguous sequence after the last emitted part.
    expected_next = (
        max(emitted_parts) + 1
        if emitted_parts
        else 1
    )

    contiguous_numbers: list[int] = []

    for number in available_numbers:
        if number != expected_next:
            break

        contiguous_numbers.append(
            number
        )

        expected_next += 1

    if not contiguous_numbers:
        return []

    created: list[
        NormalizerReadyBatch
    ] = []

    current_parts: list[
        LLMPartResult
    ] = []

    current_characters = 0

    def flush_current() -> None:
        nonlocal current_parts
        nonlocal current_characters
        nonlocal next_output_batch_number

        if not current_parts:
            return

        emitted = emit_normalizer_batch(
            patient_name=patient_name,
            output_batch_number=(
                next_output_batch_number
            ),
            parts=current_parts,
        )

        created.append(
            emitted
        )

        part_numbers = [
            part.batch.batch_number
            for part in current_parts
        ]

        emitted_parts.update(
            part_numbers
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
                "source_llm_parts": (
                    list(
                        emitted.source_part_numbers
                    )
                ),
                "batch_file": (
                    emitted.output_path.name
                ),
            }
        )

        next_output_batch_number += 1
        current_parts = []
        current_characters = 0

        save_normalizer_batch_manifest(
            patient_name,
            {
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
                "batches": existing_batches,
                "updated_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
            },
        )

    for number in contiguous_numbers:
        part = completed_parts[
            number
        ]

        part_characters = len(
            part.structured_text
        )

        if not current_parts:
            current_parts.append(
                part
            )

            current_characters = (
                part_characters
            )

            # One single structured part may itself exceed MAX.
            # Keep it intact rather than cutting clinical text.
            if (
                current_characters
                >= NORMALIZER_MAX_BATCH_CHARACTERS
            ):
                flush_current()

            continue

        proposed_characters = (
            current_characters
            + part_characters
        )

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

    if current_parts:
        if (
            current_characters
            >= NORMALIZER_MIN_BATCH_CHARACTERS
        ):
            flush_current()

        elif llm_stage_complete:
            # Final leftover is always released.
            flush_current()

    return created


# ==========================================================
# COMPLETE LLM OUTPUT
# ==========================================================

def build_complete_llm_text(
    part_results: dict[
        int,
        LLMPartResult,
    ],
) -> str:
    sections: list[str] = []

    for part_number in sorted(
        part_results
    ):
        result = part_results[
            part_number
        ]

        batch = result.batch

        sections.extend(
            [
                "#" * 72,
                (
                    f"STRUCTURED PART "
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
                result.structured_text.strip(),
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


def save_complete_llm_output(
    *,
    patient_name: str,
    part_results: dict[
        int,
        LLMPartResult,
    ],
) -> Path:
    folders = get_patient_folders(
        patient_name
    )

    complete_path = (
        folders["llm"]
        / (
            f"{safe_patient_name(patient_name)}"
            "_llm_complete.txt"
        )
    )

    complete_text = (
        build_complete_llm_text(
            part_results
        )
    )

    atomic_write(
        complete_path,
        complete_text,
    )

    return complete_path


def save_llm_completion_marker(
    *,
    patient_name: str,
    total_ocr_batches: int,
    total_normalizer_batches: int,
    complete_path: Path,
) -> Path:
    folders = get_patient_folders(
        patient_name
    )

    marker_path = (
        folders["llm"]
        / "llm_complete.json"
    )

    payload = {
        "patient_name": patient_name,
        "stage": "llm",
        "status": "completed",
        "model": MODEL_NAME,
        "workers": MAX_LLM_WORKERS,
        "total_ocr_batches": (
            total_ocr_batches
        ),
        "normalizer_min_batch_characters": (
            NORMALIZER_MIN_BATCH_CHARACTERS
        ),
        "normalizer_max_batch_characters": (
            NORMALIZER_MAX_BATCH_CHARACTERS
        ),
        "total_batches": (
            total_normalizer_batches
        ),
        "complete_file": (
            complete_path.name
        ),
        "completed_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
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
# TEMPORARY CLEANUP
# ==========================================================

def cleanup_temporary_ocr_batches(
    patient_name: str,
) -> None:
    folders = get_patient_folders(
        patient_name
    )

    batch_folder = folders[
        "ocr_batches"
    ]

    if not batch_folder.exists():
        return

    for path in batch_folder.iterdir():
        if path.is_file():
            path.unlink()


def cleanup_temporary_llm_parts(
    patient_name: str,
) -> None:
    """
    Parts are no longer required after:
    - complete LLM text has been saved, and
    - all normalizer-ready batches have been emitted.
    """

    folders = get_patient_folders(
        patient_name
    )

    part_folder = folders[
        "llm_parts"
    ]

    if part_folder.exists():
        shutil.rmtree(
            part_folder,
            ignore_errors=True,
        )

        part_folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def cleanup_temporary_llm_batches(
    patient_name: str,
) -> None:
    """
    Called later by Normalizer after successful normalization.
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


# ==========================================================
# STREAMING PATIENT PROCESSING
# ==========================================================

def process_patient_stream(
    patient_name: str,
) -> Path:
    """
    Watch OCR character batches, structure them, and emit
    character-range batches for the Normalizer.
    """

    started_at = time.perf_counter()

    submitted_batches: set[int] = set()

    completed_parts = (
        load_existing_llm_parts(
            patient_name
        )
    )

    failed_batches: dict[
        int,
        str,
    ] = {}

    print()
    print("=" * 72)
    print("STREAMING LLM STRUCTURER")
    print("=" * 72)
    print(f"Patient                : {patient_name}")
    print(f"Model                  : {MODEL_NAME}")
    print(f"LLM workers            : {MAX_LLM_WORKERS}")
    print(
        "Normalizer batch chars : "
        f"{NORMALIZER_MIN_BATCH_CHARACTERS:,}"
        f" - "
        f"{NORMALIZER_MAX_BATCH_CHARACTERS:,}"
    )
    print("=" * 72)

    while True:
        process_available_batches(
            patient_name,
            submitted_batches=(
                submitted_batches
            ),
            completed_parts=(
                completed_parts
            ),
            failed_batches=(
                failed_batches
            ),
        )

        if failed_batches:
            details = ", ".join(
                (
                    f"batch {number}: "
                    f"{message}"
                )
                for number, message
                in sorted(
                    failed_batches.items()
                )
            )

            raise RuntimeError(
                "One or more LLM batches failed: "
                + details
            )

        llm_complete = (
            all_llm_parts_completed(
                patient_name,
                completed_parts,
            )
        )

        emit_ready_normalizer_batches(
            patient_name=patient_name,
            completed_parts=completed_parts,
            llm_stage_complete=(
                llm_complete
            ),
        )

        if llm_complete:
            break

        time.sleep(
            POLL_SECONDS
        )

    total_ocr_batches = (
        expected_ocr_batch_count(
            patient_name
        )
    )

    if total_ocr_batches is None:
        raise RuntimeError(
            "OCR completion marker was not found."
        )

    # One last call guarantees the final under-minimum
    # leftover batch is released.
    emit_ready_normalizer_batches(
        patient_name=patient_name,
        completed_parts=completed_parts,
        llm_stage_complete=True,
    )

    manifest = load_normalizer_batch_manifest(
        patient_name
    )

    total_normalizer_batches = len(
        manifest.get(
            "batches",
            [],
        )
    )

    if total_normalizer_batches < 1:
        raise RuntimeError(
            "LLM completed but no Normalizer-ready "
            "character batches were created."
        )

    complete_path = (
        save_complete_llm_output(
            patient_name=patient_name,
            part_results=(
                completed_parts
            ),
        )
    )

    save_llm_completion_marker(
        patient_name=patient_name,
        total_ocr_batches=(
            total_ocr_batches
        ),
        total_normalizer_batches=(
            total_normalizer_batches
        ),
        complete_path=complete_path,
    )

    cleanup_temporary_ocr_batches(
        patient_name
    )

    cleanup_temporary_llm_parts(
        patient_name
    )

    elapsed = (
        time.perf_counter()
        - started_at
    )

    print()
    print("=" * 72)
    print("LLM STAGE COMPLETED")
    print("=" * 72)
    print(
        f"OCR batches structured : "
        f"{len(completed_parts)}"
    )
    print(
        f"Normalizer batches     : "
        f"{total_normalizer_batches}"
    )
    print(
        f"Complete file          : "
        f"{complete_path}"
    )
    print(
        f"Total time             : "
        f"{elapsed:.2f} seconds"
    )
    print("=" * 72)

    return complete_path


# ==========================================================
# OLLAMA VALIDATION
# ==========================================================

def check_ollama_connection() -> bool:
    print(
        "Checking Ollama connection...",
        flush=True,
    )

    try:
        client = create_ollama_client()
        response = client.list()

    except Exception as error:
        print(
            f"Could not connect to Ollama: "
            f"{error}",
            flush=True,
        )

        return False

    if isinstance(response, dict):
        models = response.get(
            "models",
            [],
        )
    else:
        models = getattr(
            response,
            "models",
            [],
        )

    available_names: list[str] = []

    for model in models:
        if isinstance(model, dict):
            model_name = (
                model.get("model")
                or model.get("name")
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
                str(model_name)
            )

    requested_base = (
        MODEL_NAME.split(":")[0]
    )

    model_found = any(
        (
            available_name
            == MODEL_NAME
        )
        or (
            available_name.split(":")[0]
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
# PATIENT DISCOVERY
# ==========================================================

def find_streaming_patients() -> list[str]:
    if not DATA_FOLDER.exists():
        return []

    patients: list[str] = []

    for patient_folder in (
        DATA_FOLDER.iterdir()
    ):
        if not patient_folder.is_dir():
            continue

        ocr_folder = (
            patient_folder
            / "ocr"
        )

        batch_folder = (
            ocr_folder
            / "batches"
        )

        completion_marker = (
            ocr_folder
            / "ocr_complete.json"
        )

        has_batches = (
            batch_folder.exists()
            and any(
                batch_folder.glob(
                    "*.ready"
                )
            )
        )

        if (
            has_batches
            or completion_marker.exists()
        ):
            patients.append(
                patient_folder.name
            )

    return sorted(
        patients
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    overall_start = time.perf_counter()

    print("=" * 72)
    print(
        "NICU CHARACTER-STREAMING LLM STRUCTURER"
    )
    print("=" * 72)
    print(f"Data folder : {DATA_FOLDER}")
    print(f"Model       : {MODEL_NAME}")
    print(f"Workers     : {MAX_LLM_WORKERS}")
    print(
        "Normalizer character range: "
        f"{NORMALIZER_MIN_BATCH_CHARACTERS:,}"
        f" - "
        f"{NORMALIZER_MAX_BATCH_CHARACTERS:,}"
    )
    print("=" * 72)

    if not check_ollama_connection():
        raise SystemExit(1)

    patients = find_streaming_patients()

    if not patients:
        print(
            "No OCR batches or completed OCR "
            "records were found."
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
            print("=" * 72)
            print("LLM STAGE FAILED")
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
    print("LLM RUN FINISHED")
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


if __name__ == "__main__":
    main()