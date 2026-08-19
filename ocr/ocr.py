"""
Parallel streaming OCR for NICU patient PDFs.

Permanent storage structure:

data/<patient_name>/
    original/
    ocr/
        pages/
        batches/
        metadata/
        reports/
        <patient_name>_ocr_complete.txt

The OCR processes pages with configurable workers.

OCR text is accumulated in page order. Whenever the configured character
range is reached, one batch TXT file is created immediately for the next
LLM stage.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

import fitz

from ocr.config import (
    CREATE_BATCH_READY_MARKERS,
    DATA_FOLDER,
    DEBUG_CONFIDENCE_THRESHOLD,
    DEBUG_FOLDER,
    DEBUG_HEALTH_THRESHOLD,
    DELETE_SUCCESSFUL_TEMP_IMAGES,
    INPUT_FOLDER,
    MAX_OCR_WORKERS,
    OCR_DPI,
    PAGE_RETRIES,
    MAX_BATCH_CHARACTERS,
    MIN_BATCH_CHARACTERS,
    REPORT_FOLDER,
    RESUME_EXISTING_PAGES,
    SAVE_COMPLETE_TEXT,
    SAVE_CSV_REPORT,
    SAVE_DEBUG_IMAGES,
    SAVE_METADATA,
    SAVE_PAGE_TEXT,
    TEMP_FOLDER,
    USE_DOC_ORIENTATION,
    USE_DOC_UNWARPING,
    USE_TEXTLINE_ORIENTATION,
    WORKER_CPU_THREADS,
)
from ocr.page_analyzer import analyze_page_result
from ocr.report import (
    print_final_report,
    print_page_report,
    save_csv_report,
)


_WORKER_OCR = None


# ==========================================================
# PATIENT STORAGE
# ==========================================================

def safe_patient_name(value: str) -> str:
    """
    Convert a PDF filename into a safe patient-folder name.
    """

    cleaned = value.strip()

    cleaned = re.sub(
        r"\.pdf$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

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

    cleaned = cleaned.strip("_")

    return cleaned or "patient"


def get_patient_folders(
    patient_name: str,
) -> dict[str, Path]:
    """
    Create and return permanent folders for one patient.
    """

    patient_root = (
        DATA_FOLDER
        / safe_patient_name(patient_name)
    )

    folders = {
        "root": patient_root,
        "original": patient_root / "original",
        "ocr": patient_root / "ocr",
        "pages": patient_root / "ocr" / "pages",
        "batches": patient_root / "ocr" / "batches",
        "metadata": patient_root / "ocr" / "metadata",
        "reports": patient_root / "ocr" / "reports",
    }

    for folder in folders.values():
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    return folders


def store_original_pdf(
    pdf_path: Path,
    original_folder: Path,
) -> Path:
    """
    Save the original uploaded PDF inside the patient folder.
    """

    destination = (
        original_folder
        / pdf_path.name
    )

    if pdf_path.resolve() != destination.resolve():
        shutil.copy2(
            pdf_path,
            destination,
        )

    return destination


def clear_patient_ocr_batches(
    batch_folder: Path,
) -> None:
    """
    Remove old batch files and markers for this patient's new OCR run.
    """

    for path in batch_folder.iterdir():
        if path.is_file():
            path.unlink()


# ==========================================================
# WORKER CONFIGURATION
# ==========================================================

def _configure_worker_environment() -> None:
    value = str(
        WORKER_CPU_THREADS
    )

    os.environ[
        "OMP_NUM_THREADS"
    ] = value

    os.environ[
        "OPENBLAS_NUM_THREADS"
    ] = value

    os.environ[
        "MKL_NUM_THREADS"
    ] = value

    os.environ[
        "VECLIB_MAXIMUM_THREADS"
    ] = value

    os.environ[
        "NUMEXPR_NUM_THREADS"
    ] = value


def _initialize_worker() -> None:
    """
    Load one PaddleOCR model inside each worker process.
    """

    global _WORKER_OCR

    _configure_worker_environment()

    from paddleocr import PaddleOCR

    _WORKER_OCR = PaddleOCR(
        use_doc_orientation_classify=(
            USE_DOC_ORIENTATION
        ),
        use_doc_unwarping=(
            USE_DOC_UNWARPING
        ),
        use_textline_orientation=(
            USE_TEXTLINE_ORIENTATION
        ),
    )


# ==========================================================
# OCR RESULT EXTRACTION
# ==========================================================

def _extract_prediction_data(
    result: Any,
) -> tuple[list[str], list[float]]:
    """
    Extract recognized text and confidence scores.
    """

    if not result:
        return [], []

    page_result = result[0]

    if hasattr(
        page_result,
        "json",
    ):
        try:
            payload = page_result.json

            if callable(payload):
                payload = payload()

            if isinstance(
                payload,
                str,
            ):
                payload = json.loads(
                    payload
                )

            if isinstance(
                payload,
                dict,
            ):
                page_result = payload.get(
                    "res",
                    payload,
                )

        except Exception:
            pass

    if not isinstance(
        page_result,
        dict,
    ):
        try:
            page_result = dict(
                page_result
            )

        except Exception:
            return [], []

    texts = (
        page_result.get(
            "rec_texts",
            [],
        )
        or []
    )

    scores = (
        page_result.get(
            "rec_scores",
            [],
        )
        or []
    )

    return (
        [
            str(value)
            for value in texts
        ],
        [
            float(value)
            for value in scores
        ],
    )

# ==========================================================
# PAGE RENDERING
# ==========================================================

def _render_page(
    pdf_path: str,
    page_index: int,
    image_path: Path,
) -> None:
    """
    Render one PDF page as an image for OCR.
    """

    with fitz.open(pdf_path) as document:
        page = document.load_page(
            page_index
        )

        pixmap = page.get_pixmap(
            dpi=OCR_DPI,
            alpha=False,
        )

        image_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pixmap.save(
            str(image_path)
        )


# ==========================================================
# PAGE WORKER
# ==========================================================

def _process_page_job(
    job: dict[str, Any],
) -> dict[str, Any]:
    """
    OCR one page and save its page-level outputs.
    """

    global _WORKER_OCR

    pdf_path = job["pdf_path"]
    pdf_name = job["pdf_name"]

    page_number = int(
        job["page_number"]
    )

    page_index = (
        page_number - 1
    )

    page_folder = Path(
        job["page_folder"]
    )

    metadata_folder = Path(
        job["metadata_folder"]
    )

    temp_folder = Path(
        job["temp_folder"]
    )

    debug_folder = Path(
        job["debug_folder"]
    )

    text_path = (
        page_folder
        / f"page_{page_number:04}.txt"
    )

    metadata_path = (
        metadata_folder
        / f"page_{page_number:04}.json"
    )

    image_path = (
        temp_folder
        / pdf_name
        / f"page_{page_number:04}.png"
    )

    debug_image_path = (
        debug_folder
        / pdf_name
        / f"page_{page_number:04}.png"
    )

    start_time = time.perf_counter()
    last_error = ""

    for attempt in range(
        PAGE_RETRIES + 1
    ):
        try:
            if _WORKER_OCR is None:
                _initialize_worker()

            _render_page(
                pdf_path,
                page_index,
                image_path,
            )

            prediction = (
                _WORKER_OCR.predict(
                    str(image_path)
                )
            )

            texts, scores = (
                _extract_prediction_data(
                    prediction
                )
            )

            page_text = "\n".join(
                texts
            ).strip()

            elapsed = (
                time.perf_counter()
                - start_time
            )

            page_stats, metadata = (
                analyze_page_result(
                    pdf_name=pdf_name,
                    page_number=page_number,
                    texts=texts,
                    scores=scores,
                    processing_time=elapsed,
                )
            )

            if SAVE_PAGE_TEXT:
                text_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                text_path.write_text(
                    page_text,
                    encoding="utf-8",
                )

            if SAVE_METADATA:
                metadata_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                metadata_path.write_text(
                    json.dumps(
                        metadata,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            should_save_debug_image = (
                SAVE_DEBUG_IMAGES
                and (
                    page_stats[
                        "ocr_health_score"
                    ]
                    < DEBUG_HEALTH_THRESHOLD
                    or page_stats[
                        "average_confidence"
                    ]
                    < DEBUG_CONFIDENCE_THRESHOLD
                )
            )

            if should_save_debug_image:
                debug_image_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    image_path,
                    debug_image_path,
                )

            if (
                DELETE_SUCCESSFUL_TEMP_IMAGES
                and image_path.exists()
            ):
                image_path.unlink()

            return {
                "success": True,
                "page_number": page_number,
                "page_text": page_text,
                "page_stats": page_stats,
                "processing_time": elapsed,
                "attempts": attempt + 1,
                "error": "",
            }

        except Exception as error:
            last_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            if attempt < PAGE_RETRIES:
                time.sleep(1)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "success": False,
        "page_number": page_number,
        "page_text": "",
        "page_stats": None,
        "processing_time": elapsed,
        "attempts": PAGE_RETRIES + 1,
        "error": last_error,
        "traceback": traceback.format_exc(),
    }


# ==========================================================
# EXISTING PAGE RESUME
# ==========================================================

def _load_existing_page(
    *,
    page_number: int,
    page_folder: Path,
    metadata_folder: Path,
) -> dict[str, Any] | None:
    """
    Reload one previously completed OCR page.
    """

    text_path = (
        page_folder
        / f"page_{page_number:04}.txt"
    )

    metadata_path = (
        metadata_folder
        / f"page_{page_number:04}.json"
    )

    if (
        not text_path.exists()
        or not metadata_path.exists()
    ):
        return None

    try:
        page_text = text_path.read_text(
            encoding="utf-8"
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        statistics = metadata[
            "statistics"
        ]

        medical = metadata[
            "medical"
        ]

        confidence = metadata[
            "confidence"
        ]

        health = metadata[
            "health"
        ]

        page_stats = {
            "words": statistics[
                "words"
            ],
            "characters": statistics[
                "characters"
            ],
            "lines": statistics[
                "lines"
            ],
            "blocks": statistics[
                "blocks"
            ],
            "numbers_found": statistics[
                "numbers"
            ],
            "medical_keywords_found": medical[
                "count"
            ],
            "medical_fields": medical[
                "fields"
            ],
            "average_confidence": confidence[
                "average"
            ],
            "minimum_confidence": confidence[
                "minimum"
            ],
            "maximum_confidence": confidence[
                "maximum"
            ],
            "ocr_health_score": health[
                "score"
            ],
            "status": health[
                "status"
            ],
        }

        return {
            "success": True,
            "page_number": page_number,
            "page_text": page_text,
            "page_stats": page_stats,
            "processing_time": float(
                metadata.get(
                    "processing_time",
                    0,
                )
            ),
            "attempts": 0,
            "error": "",
            "resumed": True,
        }

    except Exception:
        return None


# ==========================================================
# REPORT ROW
# ==========================================================

def _row_from_stats(
    page_number: int,
    stats: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    """
    Convert page OCR statistics into a report row.
    """

    return {
        "Page": page_number,
        "Words": stats["words"],
        "Characters": stats[
            "characters"
        ],
        "Lines": stats["lines"],
        "Blocks": stats["blocks"],
        "Numbers": stats[
            "numbers_found"
        ],
        "Medical Fields": stats[
            "medical_keywords_found"
        ],
        "Confidence": stats[
            "average_confidence"
        ],
        "Health": stats[
            "ocr_health_score"
        ],
        "Time": round(
            elapsed,
            2,
        ),
        "Status": stats["status"],
    }


# ==========================================================
# CHARACTER-BASED BATCH HELPERS
# ==========================================================

def _batch_page_text(
    result: dict[str, Any],
) -> str:
    """Return the text used for one page inside a streaming batch."""

    if result["success"]:
        return str(
            result.get(
                "page_text",
                "",
            )
        )

    return (
        "[OCR FAILED: "
        f"{result.get('error', 'Unknown OCR error')}]"
    )


def _save_character_batch(
    *,
    patient_name: str,
    batch_number: int,
    page_numbers: list[int],
    total_pages: int,
    page_results: dict[int, dict[str, Any]],
    batch_folder: Path,
    batch_characters: int,
) -> Path:
    """Save one character-sized OCR batch for the LLM stage."""

    if not page_numbers:
        raise ValueError(
            "Cannot save an empty OCR batch."
        )

    first_page = page_numbers[0]
    final_page = page_numbers[-1]

    batch_sections: list[str] = []

    for page_number in page_numbers:
        page_text = _batch_page_text(
            page_results[page_number]
        )

        batch_sections.append(
            f"{'=' * 60}\n"
            f"PAGE {page_number}\n"
            f"{'=' * 60}\n\n"
            f"{page_text}\n"
        )

    batch_filename = (
        f"{safe_patient_name(patient_name)}"
        f"_ocr_batch_{batch_number:04}"
        f"_pages_{first_page:04}_to_"
        f"{final_page:04}.txt"
    )

    batch_path = (
        batch_folder
        / batch_filename
    )

    batch_path.write_text(
        "\n".join(batch_sections),
        encoding="utf-8",
    )

    if CREATE_BATCH_READY_MARKERS:
        ready_marker = batch_path.with_suffix(
            ".ready"
        )

        ready_marker.write_text(
            json.dumps(
                {
                    "patient_name": (
                        safe_patient_name(
                            patient_name
                        )
                    ),
                    "stage": "ocr",
                    "batch_number": batch_number,
                    "first_page": first_page,
                    "final_page": final_page,
                    "page_count": len(page_numbers),
                    "total_pages": total_pages,
                    "batch_characters": (
                        batch_characters
                    ),
                    "min_batch_characters": (
                        MIN_BATCH_CHARACTERS
                    ),
                    "max_batch_characters": (
                        MAX_BATCH_CHARACTERS
                    ),
                    "batch_file": batch_path.name,
                    "status": "ready",
                    "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return batch_path


def _emit_current_character_batch(
    *,
    patient_name: str,
    total_pages: int,
    page_results: dict[int, dict[str, Any]],
    batch_folder: Path,
    batch_state: dict[str, Any],
    saved_batches: set[int],
) -> Path | None:
    """Write the current in-memory character batch and reset it."""

    current_pages = list(
        batch_state["current_pages"]
    )

    if not current_pages:
        return None

    batch_number = int(
        batch_state["batch_number"]
    )

    batch_characters = int(
        batch_state["current_characters"]
    )

    batch_path = _save_character_batch(
        patient_name=patient_name,
        batch_number=batch_number,
        page_numbers=current_pages,
        total_pages=total_pages,
        page_results=page_results,
        batch_folder=batch_folder,
        batch_characters=batch_characters,
    )

    saved_batches.add(
        batch_number
    )

    first_page = current_pages[0]
    final_page = current_pages[-1]

    print(
        "\n"
        f"[BATCH READY] Batch {batch_number} "
        f"(pages {first_page}-{final_page}, "
        f"{batch_characters:,} characters)\n"
        f"Saved: {batch_path}"
    )

    batch_state["batch_number"] = (
        batch_number + 1
    )
    batch_state["current_pages"] = []
    batch_state["current_characters"] = 0

    return batch_path


def consume_ready_pages_into_batches(
    *,
    patient_name: str,
    total_pages: int,
    page_results: dict[int, dict[str, Any]],
    batch_folder: Path,
    batch_state: dict[str, Any],
    saved_batches: set[int],
    final: bool = False,
) -> list[Path]:
    """
    Consume completed OCR pages strictly in page order.

    Batching rules:

    - Keep collecting while below MIN_BATCH_CHARACTERS.
    - Once the current batch is at least MIN_BATCH_CHARACTERS, add the next
      whole page only when doing so keeps the batch at or below
      MAX_BATCH_CHARACTERS.
    - If the next whole page would cross MAX_BATCH_CHARACTERS, emit the
      current eligible batch and start a new one with that page.
    - If the current batch is still below the minimum, the next whole page is
      allowed to make the batch exceed the maximum. This avoids splitting a
      patient page purely to satisfy the character target.
    - A single page larger than MAX_BATCH_CHARACTERS is emitted by itself.
    - When final=True, any remaining smaller tail batch is emitted.

    Pages may finish OCR out of order because several OCR workers run in
    parallel. This function waits for the next required page number before
    advancing, so LLM batches always preserve clinical page order.
    """

    created_paths: list[Path] = []

    while True:
        next_page = int(
            batch_state["next_page"]
        )

        if next_page > total_pages:
            break

        if next_page not in page_results:
            break

        result = page_results[next_page]
        page_text = _batch_page_text(
            result
        )
        page_characters = len(
            page_text
        )

        current_pages: list[int] = (
            batch_state["current_pages"]
        )
        current_characters = int(
            batch_state["current_characters"]
        )

        if (
            current_pages
            and current_characters
            >= MIN_BATCH_CHARACTERS
            and current_characters
            + page_characters
            > MAX_BATCH_CHARACTERS
        ):
            batch_path = (
                _emit_current_character_batch(
                    patient_name=patient_name,
                    total_pages=total_pages,
                    page_results=page_results,
                    batch_folder=batch_folder,
                    batch_state=batch_state,
                    saved_batches=saved_batches,
                )
            )

            if batch_path is not None:
                created_paths.append(
                    batch_path
                )

            continue

        batch_state["current_pages"].append(
            next_page
        )
        batch_state[
            "current_characters"
        ] = (
            current_characters
            + page_characters
        )
        batch_state["next_page"] = (
            next_page + 1
        )

        new_character_count = int(
            batch_state[
                "current_characters"
            ]
        )

        # Emit immediately when a single page is already oversized, or when
        # adding a whole page pushed a below-minimum batch beyond the maximum.
        if (
            new_character_count
            >= MAX_BATCH_CHARACTERS
        ):
            batch_path = (
                _emit_current_character_batch(
                    patient_name=patient_name,
                    total_pages=total_pages,
                    page_results=page_results,
                    batch_folder=batch_folder,
                    batch_state=batch_state,
                    saved_batches=saved_batches,
                )
            )

            if batch_path is not None:
                created_paths.append(
                    batch_path
                )

    if final:
        if int(batch_state["next_page"]) <= total_pages:
            missing_for_batching = [
                page_number
                for page_number in range(
                    int(batch_state["next_page"]),
                    total_pages + 1,
                )
                if page_number
                not in page_results
            ]

            if missing_for_batching:
                raise RuntimeError(
                    "Cannot finalize OCR batches because page result(s) "
                    "are missing: "
                    + ", ".join(
                        str(page_number)
                        for page_number
                        in missing_for_batching
                    )
                )

            # If all pages exist, consume them now.
            created_paths.extend(
                consume_ready_pages_into_batches(
                    patient_name=patient_name,
                    total_pages=total_pages,
                    page_results=page_results,
                    batch_folder=batch_folder,
                    batch_state=batch_state,
                    saved_batches=saved_batches,
                    final=False,
                )
            )

        tail_path = (
            _emit_current_character_batch(
                patient_name=patient_name,
                total_pages=total_pages,
                page_results=page_results,
                batch_folder=batch_folder,
                batch_state=batch_state,
                saved_batches=saved_batches,
            )
        )

        if tail_path is not None:
            created_paths.append(
                tail_path
            )

    return created_paths


# ==========================================================
# COMPLETE OCR FILE
# ==========================================================

def save_complete_text(
    *,
    patient_name: str,
    page_results: dict[int, dict[str, Any]],
    ocr_folder: Path,
) -> Path:
    """
    Save the final complete OCR text after all pages finish.
    """

    sections: list[str] = []

    for page_number in sorted(
        page_results
    ):
        result = page_results[
            page_number
        ]

        if result["success"]:
            page_text = result[
                "page_text"
            ]
        else:
            page_text = (
                "[OCR FAILED: "
                f"{result['error']}]"
            )

        sections.append(
            f"{'=' * 60}\n"
            f"PAGE {page_number}\n"
            f"{'=' * 60}\n\n"
            f"{page_text}\n"
        )

    complete_path = (
        ocr_folder
        / (
            f"{safe_patient_name(patient_name)}"
            "_ocr_complete.txt"
        )
    )

    complete_path.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )

    return complete_path


# ==========================================================
# REPORT CREATION
# ==========================================================

def build_report_rows(
    page_results: dict[
        int,
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    """
    Build ordered OCR report rows.
    """

    report_rows: list[
        dict[str, Any]
    ] = []

    for page_number in sorted(
        page_results
    ):
        result = page_results[
            page_number
        ]

        if result["success"]:
            report_rows.append(
                _row_from_stats(
                    page_number,
                    result[
                        "page_stats"
                    ],
                    result[
                        "processing_time"
                    ],
                )
            )

        else:
            report_rows.append(
                {
                    "Page": (
                        page_number
                    ),
                    "Words": 0,
                    "Characters": 0,
                    "Lines": 0,
                    "Blocks": 0,
                    "Numbers": 0,
                    "Medical Fields": 0,
                    "Confidence": 0,
                    "Health": 0,
                    "Time": round(
                        result[
                            "processing_time"
                        ],
                        2,
                    ),
                    "Status": (
                        "FAILED: "
                        f"{result['error']}"
                    ),
                }
            )

    return report_rows


# ==========================================================
# PROCESS ONE PDF
# ==========================================================

def process_pdf(
    pdf_path: Path,
) -> Path:
    """
    Process one patient PDF with parallel page OCR.

    Completed page batches are saved immediately under:

        data/<patient>/ocr/batches/

    Returns the final complete OCR text path.
    """

    pdf_start = time.perf_counter()

    patient_name = safe_patient_name(
        pdf_path.stem
    )

    patient_folders = (
        get_patient_folders(
            patient_name
        )
    )

    original_pdf = store_original_pdf(
        pdf_path,
        patient_folders[
            "original"
        ],
    )

    page_folder = patient_folders[
        "pages"
    ]

    metadata_folder = (
        patient_folders[
            "metadata"
        ]
    )

    batch_folder = patient_folders[
        "batches"
    ]

    reports_folder = patient_folders[
        "reports"
    ]

    clear_patient_ocr_batches(
        batch_folder
    )

    with fitz.open(
        original_pdf
    ) as document:
        total_pages = len(
            document
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"Processing PDF : "
        f"{original_pdf.name}"
    )

    print(
        f"Patient        : "
        f"{patient_name}"
    )

    print(
        f"Pages          : "
        f"{total_pages}"
    )

    print(
        f"Batch chars    : "
        f"{MIN_BATCH_CHARACTERS:,}-"
        f"{MAX_BATCH_CHARACTERS:,}"
    )

    print(
        f"OCR workers    : "
        f"{MAX_OCR_WORKERS}"
    )

    print(
        f"DPI            : "
        f"{OCR_DPI}"
    )

    print(
        "=" * 72
    )

    page_results: dict[
        int,
        dict[str, Any],
    ] = {}

    saved_batches: set[int] = set()

    batch_state: dict[str, Any] = {
        "next_page": 1,
        "batch_number": 1,
        "current_pages": [],
        "current_characters": 0,
    }

    jobs: list[
        dict[str, Any]
    ] = []

    for page_number in range(
        1,
        total_pages + 1,
    ):
        existing_result = None

        if RESUME_EXISTING_PAGES:
            existing_result = (
                _load_existing_page(
                    page_number=(
                        page_number
                    ),
                    page_folder=(
                        page_folder
                    ),
                    metadata_folder=(
                        metadata_folder
                    ),
                )
            )

        if existing_result:
            page_results[
                page_number
            ] = existing_result

            print(
                f"[SKIP] Page "
                f"{page_number}/"
                f"{total_pages} "
                "already completed"
            )

        else:
            jobs.append(
                {
                    "pdf_path": str(
                        original_pdf
                    ),
                    "pdf_name": (
                        patient_name
                    ),
                    "page_number": (
                        page_number
                    ),
                    "page_folder": str(
                        page_folder
                    ),
                    "metadata_folder": str(
                        metadata_folder
                    ),
                    "temp_folder": str(
                        TEMP_FOLDER
                    ),
                    "debug_folder": str(
                        DEBUG_FOLDER
                    ),
                }
            )

    consume_ready_pages_into_batches(
        patient_name=patient_name,
        total_pages=total_pages,
        page_results=page_results,
        batch_folder=batch_folder,
        batch_state=batch_state,
        saved_batches=saved_batches,
        final=False,
    )

    if jobs:
        print()
        print(
            f"Running {len(jobs)} page(s) with "
            f"{MAX_OCR_WORKERS} OCR workers..."
        )
        print()

        with ProcessPoolExecutor(
            max_workers=MAX_OCR_WORKERS,
            initializer=_initialize_worker,
        ) as executor:

            futures = {
                executor.submit(
                    _process_page_job,
                    job,
                ): job
                for job in jobs
            }

            for future in as_completed(
                futures
            ):
                result = future.result()

                page_number = int(
                    result[
                        "page_number"
                    ]
                )

                page_results[
                    page_number
                ] = result

                if result[
                    "success"
                ]:
                    print_page_report(
                        page_number=(
                            page_number
                        ),
                        total_pages=(
                            total_pages
                        ),
                        page_stats=(
                            result[
                                "page_stats"
                            ]
                        ),
                        processing_time=(
                            result[
                                "processing_time"
                            ]
                        ),
                    )

                else:
                    print(
                        "\n"
                        f"[FAILED] Page "
                        f"{page_number}/"
                        f"{total_pages} "
                        f"after "
                        f"{result['attempts']} "
                        "attempt(s): "
                        f"{result['error']}"
                    )

                consume_ready_pages_into_batches(
                    patient_name=(
                        patient_name
                    ),
                    total_pages=(
                        total_pages
                    ),
                    page_results=(
                        page_results
                    ),
                    batch_folder=(
                        batch_folder
                    ),
                    batch_state=(
                        batch_state
                    ),
                    saved_batches=(
                        saved_batches
                    ),
                    final=False,
                )

    missing_pages = [
        page_number
        for page_number in range(
            1,
            total_pages + 1,
        )
        if page_number
        not in page_results
    ]

    if missing_pages:
        raise RuntimeError(
            "OCR did not return results "
            "for page(s): "
            + ", ".join(
                str(page_number)
                for page_number
                in missing_pages
            )
        )

    consume_ready_pages_into_batches(
        patient_name=patient_name,
        total_pages=total_pages,
        page_results=page_results,
        batch_folder=batch_folder,
        batch_state=batch_state,
        saved_batches=saved_batches,
        final=True,
    )

    total_batches = len(
        saved_batches
    )

    if total_pages > 0 and total_batches == 0:
        raise RuntimeError(
            "OCR completed but no character batch files were created."
        )

    complete_path = save_complete_text(
        patient_name=patient_name,
        page_results=page_results,
        ocr_folder=patient_folders[
            "ocr"
        ],
    )

    print()
    print(
        f"Complete OCR saved: "
        f"{complete_path}"
    )

    report_rows = build_report_rows(
        page_results
    )

    if SAVE_CSV_REPORT:
        report_path = (
            reports_folder
            / (
                f"{patient_name}"
                "_ocr_report.csv"
            )
        )

        save_csv_report(
            report_rows,
            report_path,
        )

        print(
            f"CSV report saved : "
            f"{report_path}"
        )

    total_time = (
        time.perf_counter()
        - pdf_start
    )

    print_final_report(
        pdf_name=original_pdf.name,
        total_pages=total_pages,
        report_rows=report_rows,
        total_time=total_time,
    )

    completion_marker = (
        patient_folders[
            "ocr"
        ]
        / "ocr_complete.json"
    )

    completion_marker.write_text(
        json.dumps(
            {
                "patient_name": (
                    patient_name
                ),
                "stage": "ocr",
                "status": "completed",
                "total_pages": (
                    total_pages
                ),
                "min_batch_characters": (
                    MIN_BATCH_CHARACTERS
                ),
                "max_batch_characters": (
                    MAX_BATCH_CHARACTERS
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"OCR data folder: "
        f"{patient_folders['ocr']}"
    )

    return complete_path


# ==========================================================
# INPUT CLEANUP
# ==========================================================

def remove_processed_input(
    pdf_path: Path,
) -> None:
    """
    Remove the temporary PDF from ocr/input after success.

    The original PDF remains safely stored under:

        data/<patient>/original/
    """

    try:
        if pdf_path.exists():
            pdf_path.unlink()

            print(
                f"Removed temporary input: "
                f"{pdf_path}"
            )

    except OSError as error:
        print(
            "Warning: could not remove "
            f"temporary input {pdf_path}: "
            f"{error}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Process all PDFs currently waiting inside ocr/input.

    The web pipeline is expected to place only one active
    patient PDF in this folder at a time.
    """

    pdf_files = sorted(
        INPUT_FOLDER.glob(
            "*.pdf"
        )
    )

    if not pdf_files:
        print(
            "No PDF files found "
            "inside ocr/input/"
        )
        return

    print(
        "=" * 72
    )

    print(
        "STREAMING PARALLEL "
        "PADDLEOCR"
    )

    print(
        "=" * 72
    )

    print(
        f"PDF files       : "
        f"{len(pdf_files)}"
    )

    print(
        f"OCR workers     : "
        f"{MAX_OCR_WORKERS}"
    )

    print(
        f"Batch characters: "
        f"{MIN_BATCH_CHARACTERS:,}-"
        f"{MAX_BATCH_CHARACTERS:,}"
    )

    print(
        f"DPI             : "
        f"{OCR_DPI}"
    )

    successful_files = 0
    failed_files = 0

    for pdf_path in pdf_files:
        try:
            complete_path = (
                process_pdf(
                    pdf_path
                )
            )

            successful_files += 1

            print()
            print(
                f"OCR completed: "
                f"{complete_path}"
            )

            remove_processed_input(
                pdf_path
            )

        except KeyboardInterrupt:
            print()
            print(
                "OCR stopped by user."
            )

            raise

        except Exception as error:
            failed_files += 1

            print()
            print(
                "=" * 72
            )

            print(
                "OCR FAILED"
            )

            print(
                "=" * 72
            )

            print(
                f"PDF   : "
                f"{pdf_path}"
            )

            print(
                f"Error : "
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                traceback.format_exc()
            )

    print()
    print(
        "=" * 72
    )

    print(
        "OCR RUN FINISHED"
    )

    print(
        "=" * 72
    )

    print(
        f"Successful PDFs : "
        f"{successful_files}"
    )

    print(
        f"Failed PDFs     : "
        f"{failed_files}"
    )

    print(
        f"Patient data    : "
        f"{DATA_FOLDER}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()