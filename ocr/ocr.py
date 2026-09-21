"""
Parallel streaming OCR for NICU patient PDFs.

NEW PIPELINE ARCHITECTURE
=========================

Input:
    ocr/input/<patient>.pdf

Temporary working files:
    ocr/temp/<patient>/

Streaming OCR batches:
    ocr/temp/<patient>/batches/

Final OCR output:
    ocr/output/OCR_<patient>.txt

The OCR stage DOES NOT write anything into data/.

The API pipeline is responsible for final archival after the complete
OCR -> LLM -> Normalizer -> Discharge pipeline succeeds.

Important:
- PDF pages are processed in parallel.
- OCR batches are emitted while OCR is still running.
- LLM can consume those batches immediately.
- Page order is preserved even when OCR workers finish out of order.
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

from ocr.page_analyzer import (
    analyze_page_result,
)

from ocr.report import (
    print_final_report,
    print_page_report,
    save_csv_report,
)


# ==========================================================
# GLOBAL PATHS
# ==========================================================

OCR_ROOT = Path(__file__).resolve().parent

OUTPUT_FOLDER = (
    OCR_ROOT
    / "output"
)

WORK_FOLDER = (
    OCR_ROOT
    / "temp"
)


# ==========================================================
# WORKER GLOBAL
# ==========================================================

_WORKER_OCR = None


# ==========================================================
# PATIENT NAME
# ==========================================================

def safe_patient_name(
    value: str,
) -> str:
    """
    Convert a PDF filename or patient name into one safe,
    consistent patient identifier.

    Examples:

        BO VAGDEVI.pdf
            -> BO_VAGDEVI

        BO_prathyusha
            -> BO_prathyusha

        pending_BO_VAGDEVI.pdf
            -> BO_VAGDEVI

    The temporary API prefix "pending_" is deliberately removed.
    """

    cleaned = value.strip()

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

    cleaned = cleaned.strip("_")

    return cleaned or "patient"


# ==========================================================
# PATIENT WORKSPACE
# ==========================================================

def get_patient_workspace(
    patient_name: str,
) -> dict[str, Path]:
    """
    Return the temporary OCR workspace for one patient.

    Nothing here is permanent.

    Structure:

        ocr/temp/<patient>/
            pages/
            batches/
            metadata/
            images/
            reports/
    """

    patient_name = safe_patient_name(
        patient_name
    )

    root = (
        WORK_FOLDER
        / patient_name
    )

    folders = {
        "root": root,
        "pages": (
            root
            / "pages"
        ),
        "batches": (
            root
            / "batches"
        ),
        "metadata": (
            root
            / "metadata"
        ),
        "images": (
            root
            / "images"
        ),
        "reports": (
            root
            / "reports"
        ),
    }

    for folder in folders.values():
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    return folders


# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

def ensure_output_folder() -> Path:
    """
    Ensure ocr/output exists.
    """

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    return OUTPUT_FOLDER


# ==========================================================
# FINAL OCR OUTPUT PATH
# ==========================================================

def get_final_output_path(
    patient_name: str,
) -> Path:
    """
    Return:

        ocr/output/OCR_<patient>.txt
    """

    patient_name = safe_patient_name(
        patient_name
    )

    ensure_output_folder()

    return (
        OUTPUT_FOLDER
        / f"OCR_{patient_name}.txt"
    )


# ==========================================================
# WORKSPACE CLEANUP
# ==========================================================

def clear_patient_workspace(
    patient_name: str,
) -> None:
    """
    Remove stale OCR temporary data for this patient.

    This should happen before a NEW OCR run.

    It prevents old:
        - batches
        - .ready markers
        - page files
        - metadata
        - temporary images
        - reports

    from being consumed by the new pipeline.
    """

    patient_name = safe_patient_name(
        patient_name
    )

    workspace = (
        WORK_FOLDER
        / patient_name
    )

    if workspace.exists():
        shutil.rmtree(
            workspace
        )


def clear_patient_output(
    patient_name: str,
) -> None:
    """
    Remove a stale final OCR output for this patient before
    beginning a completely new run.

    Final output from a successful previous pipeline should normally
    already have been moved into data/<patient>/ by api/pipeline.py.
    """

    output_path = get_final_output_path(
        patient_name
    )

    if output_path.exists():
        output_path.unlink()


# ==========================================================
# TEMPORARY BATCH PATH
# ==========================================================

def get_batch_folder(
    patient_name: str,
) -> Path:
    """
    Return the temporary streaming-batch directory.

        ocr/temp/<patient>/batches/
    """

    workspace = get_patient_workspace(
        patient_name
    )

    return workspace[
        "batches"
    ]


# ==========================================================
# OCR COMPLETION MARKER
# ==========================================================

def get_completion_marker_path(
    patient_name: str,
) -> Path:
    """
    Return temporary OCR completion marker.

        ocr/temp/<patient>/ocr_complete.json

    This marker is for pipeline coordination only.
    It is NOT permanent patient data.
    """

    patient_name = safe_patient_name(
        patient_name
    )

    workspace = get_patient_workspace(
        patient_name
    )

    return (
        workspace["root"]
        / "ocr_complete.json"
    )


def save_completion_marker(
    *,
    patient_name: str,
    total_pages: int,
    total_batches: int,
    complete_path: Path,
) -> Path:
    """
    Tell downstream pipeline stages that OCR has completely finished.

    LLM does NOT have to wait for this marker to start.
    LLM can start consuming individual .ready batches earlier.

    This marker only indicates that no more OCR batches will arrive.
    """

    patient_name = safe_patient_name(
        patient_name
    )

    marker_path = (
        get_completion_marker_path(
            patient_name
        )
    )

    payload = {
        "patient_name": (
            patient_name
        ),
        "stage": "ocr",
        "status": "completed",
        "total_pages": (
            total_pages
        ),
        "total_batches": (
            total_batches
        ),
        "complete_file": (
            complete_path.name
        ),
        "complete_path": (
            str(
                complete_path.resolve()
            )
        ),
        "completed_at": (
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        ),
    }

    marker_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return marker_path


# ==========================================================
# WORKER CONFIGURATION
# ==========================================================

def _configure_worker_environment() -> None:
    """
    Limit CPU threading inside each OCR worker.

    We parallelize at the PROCESS level, so allowing every PaddleOCR
    process to create many internal CPU threads can actually make
    processing slower.
    """

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
    Initialize one PaddleOCR instance per worker process.

    MAX_OCR_WORKERS controls how many page-processing workers
    can operate simultaneously.
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
) -> tuple[
    list[str],
    list[float],
]:
    """
    Extract recognized text and confidence scores from PaddleOCR.
    """

    if not result:
        return [], []

    page_result = result[0]

    if hasattr(
        page_result,
        "json",
    ):
        try:
            payload = (
                page_result.json
            )

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
                page_result = (
                    payload.get(
                        "res",
                        payload,
                    )
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
    Render one PDF page to an image before PaddleOCR processing.
    """

    with fitz.open(
        pdf_path
    ) as document:

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
            str(
                image_path
            )
        )


# ==========================================================
# SINGLE PAGE OCR WORKER
# ==========================================================

def _process_page_job(
    job: dict[str, Any],
) -> dict[str, Any]:
    """
    OCR exactly one PDF page.

    Multiple copies of this function run simultaneously through
    ProcessPoolExecutor.

    The function returns the page text to the parent process.
    """

    global _WORKER_OCR

    pdf_path = job[
        "pdf_path"
    ]

    pdf_name = job[
        "pdf_name"
    ]

    page_number = int(
        job[
            "page_number"
        ]
    )

    page_index = (
        page_number - 1
    )

    page_folder = Path(
        job[
            "page_folder"
        ]
    )

    metadata_folder = Path(
        job[
            "metadata_folder"
        ]
    )

    image_folder = Path(
        job[
            "image_folder"
        ]
    )

    debug_folder = Path(
        job[
            "debug_folder"
        ]
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
        image_folder
        / f"page_{page_number:04}.png"
    )

    debug_image_path = (
        debug_folder
        / pdf_name
        / f"page_{page_number:04}.png"
    )

    start_time = (
        time.perf_counter()
    )

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
                    str(
                        image_path
                    )
                )
            )

            texts, scores = (
                _extract_prediction_data(
                    prediction
                )
            )

            page_text = (
                "\n".join(
                    texts
                ).strip()
            )

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
                "page_number": (
                    page_number
                ),
                "page_text": (
                    page_text
                ),
                "page_stats": (
                    page_stats
                ),
                "processing_time": (
                    elapsed
                ),
                "attempts": (
                    attempt + 1
                ),
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
        "page_number": (
            page_number
        ),
        "page_text": "",
        "page_stats": None,
        "processing_time": (
            elapsed
        ),
        "attempts": (
            PAGE_RETRIES + 1
        ),
        "error": (
            last_error
        ),
        "traceback": (
            traceback.format_exc()
        ),
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

    This is only used when RESUME_EXISTING_PAGES is enabled.
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

        page_text = (
            text_path.read_text(
                encoding="utf-8"
            )
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
            "words": (
                statistics["words"]
            ),
            "characters": (
                statistics["characters"]
            ),
            "lines": (
                statistics["lines"]
            ),
            "blocks": (
                statistics["blocks"]
            ),
            "numbers_found": (
                statistics["numbers"]
            ),
            "medical_keywords_found": (
                medical["count"]
            ),
            "medical_fields": (
                medical["fields"]
            ),
            "average_confidence": (
                confidence["average"]
            ),
            "minimum_confidence": (
                confidence["minimum"]
            ),
            "maximum_confidence": (
                confidence["maximum"]
            ),
            "ocr_health_score": (
                health["score"]
            ),
            "status": (
                health["status"]
            ),
        }

        return {
            "success": True,
            "page_number": (
                page_number
            ),
            "page_text": (
                page_text
            ),
            "page_stats": (
                page_stats
            ),
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
    Convert page OCR statistics into one report row.
    """

    return {
        "Page": (
            page_number
        ),
        "Words": (
            stats["words"]
        ),
        "Characters": (
            stats["characters"]
        ),
        "Lines": (
            stats["lines"]
        ),
        "Blocks": (
            stats["blocks"]
        ),
        "Numbers": (
            stats["numbers_found"]
        ),
        "Medical Fields": (
            stats[
                "medical_keywords_found"
            ]
        ),
        "Confidence": (
            stats[
                "average_confidence"
            ]
        ),
        "Health": (
            stats[
                "ocr_health_score"
            ]
        ),
        "Time": round(
            elapsed,
            2,
        ),
        "Status": (
            stats["status"]
        ),
    }


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

        result = (
            page_results[
                page_number
            ]
        )

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
# STREAMING BATCH HELPERS
# ==========================================================

def _batch_page_text(
    result: dict[str, Any],
) -> str:
    """
    Return page text for a streaming OCR batch.
    """

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


# ==========================================================
# SAVE ONE STREAMING OCR BATCH
# ==========================================================

def _save_character_batch(
    *,
    patient_name: str,
    batch_number: int,
    page_numbers: list[int],
    total_pages: int,
    page_results: dict[
        int,
        dict[str, Any],
    ],
    batch_folder: Path,
    batch_characters: int,
) -> Path:
    """
    Save one TEMPORARY OCR batch.

    IMPORTANT:

    These files are NOT permanent OCR output.

    They exist only so the LLM can start working while OCR
    continues processing later pages.

    Location:

        ocr/temp/<patient>/batches/

    The final user-facing OCR file is separately stored at:

        ocr/output/OCR_<patient>.txt
    """

    if not page_numbers:
        raise ValueError(
            "Cannot save an empty OCR batch."
        )

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    first_page = (
        page_numbers[0]
    )

    final_page = (
        page_numbers[-1]
    )

    batch_sections: list[str] = []

    for page_number in page_numbers:

        page_text = (
            _batch_page_text(
                page_results[
                    page_number
                ]
            )
        )

        batch_sections.append(
            f"{'=' * 60}\n"
            f"PAGE {page_number}\n"
            f"{'=' * 60}\n\n"
            f"{page_text}\n"
        )

    batch_filename = (
        f"OCR_{patient_name}"
        f"_BATCH_{batch_number:04}"
        f"_PAGES_{first_page:04}_TO_"
        f"{final_page:04}.txt"
    )

    batch_path = (
        batch_folder
        / batch_filename
    )

    batch_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_path.write_text(
        "\n".join(
            batch_sections
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------
    # READY MARKER
    # ------------------------------------------------------
    #
    # The LLM watcher can detect this marker immediately.
    #
    # Therefore:
    #
    # OCR does NOT need to finish completely before LLM starts.
    #
    # ------------------------------------------------------

    if CREATE_BATCH_READY_MARKERS:

        ready_marker = (
            batch_path.with_suffix(
                ".ready"
            )
        )

        ready_payload = {
            "patient_name": (
                patient_name
            ),
            "stage": "ocr",
            "status": "ready",
            "batch_number": (
                batch_number
            ),
            "first_page": (
                first_page
            ),
            "final_page": (
                final_page
            ),
            "page_count": (
                len(
                    page_numbers
                )
            ),
            "total_pages": (
                total_pages
            ),
            "batch_characters": (
                batch_characters
            ),
            "min_batch_characters": (
                MIN_BATCH_CHARACTERS
            ),
            "max_batch_characters": (
                MAX_BATCH_CHARACTERS
            ),
            "batch_file": (
                batch_path.name
            ),
            "batch_path": str(
                batch_path.resolve()
            ),
            "created_at": (
                time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
            ),
        }

        # Write marker AFTER TXT is completely written.
        #
        # This prevents the LLM from seeing "ready"
        # before the actual batch is available.

        temporary_marker = (
            ready_marker.with_suffix(
                ".ready.tmp"
            )
        )

        temporary_marker.write_text(
            json.dumps(
                ready_payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_marker.replace(
            ready_marker
        )

    return batch_path


# ==========================================================
# EMIT CURRENT CHARACTER BATCH
# ==========================================================

def _emit_current_character_batch(
    *,
    patient_name: str,
    total_pages: int,
    page_results: dict[
        int,
        dict[str, Any],
    ],
    batch_folder: Path,
    batch_state: dict[
        str,
        Any,
    ],
    saved_batches: set[int],
) -> Path | None:
    """
    Write the currently accumulated OCR pages as one batch
    and reset the in-memory batch state.
    """

    current_pages = list(
        batch_state[
            "current_pages"
        ]
    )

    if not current_pages:
        return None

    batch_number = int(
        batch_state[
            "batch_number"
        ]
    )

    batch_characters = int(
        batch_state[
            "current_characters"
        ]
    )

    batch_path = (
        _save_character_batch(
            patient_name=(
                patient_name
            ),
            batch_number=(
                batch_number
            ),
            page_numbers=(
                current_pages
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
            batch_characters=(
                batch_characters
            ),
        )
    )

    saved_batches.add(
        batch_number
    )

    first_page = (
        current_pages[0]
    )

    final_page = (
        current_pages[-1]
    )

    print(
        "\n"
        f"[OCR BATCH READY] "
        f"Batch {batch_number} "
        f"(pages "
        f"{first_page}-"
        f"{final_page}, "
        f"{batch_characters:,} "
        f"characters)\n"
        f"Saved: "
        f"{batch_path}",
        flush=True,
    )

    batch_state[
        "batch_number"
    ] = (
        batch_number + 1
    )

    batch_state[
        "current_pages"
    ] = []

    batch_state[
        "current_characters"
    ] = 0

    return batch_path


# ==========================================================
# ORDERED STREAMING BATCH CREATION
# ==========================================================

def consume_ready_pages_into_batches(
    *,
    patient_name: str,
    total_pages: int,
    page_results: dict[
        int,
        dict[str, Any],
    ],
    batch_folder: Path,
    batch_state: dict[
        str,
        Any,
    ],
    saved_batches: set[int],
    final: bool = False,
) -> list[Path]:
    """
    Consume OCR pages in CLINICAL PAGE ORDER.

    OCR workers themselves run in parallel.

    Example:

        Worker finishes page 4
        Worker finishes page 2
        Worker finishes page 3
        Worker finishes page 1

    We DO NOT create the clinical text in that order.

    Once page 1 exists, the batching layer consumes:

        1 -> 2 -> 3 -> 4

    This allows:

        PARALLEL OCR
            +
        ORDERED CLINICAL OUTPUT
            +
        STREAMING INTO LLM


    Character batching rules
    ------------------------

    1. Keep accumulating pages while below
       MIN_BATCH_CHARACTERS.

    2. Once minimum size is reached, check whether adding the
       next whole page would exceed MAX_BATCH_CHARACTERS.

    3. If yes, emit the current batch first.

    4. Never split one clinical page just to satisfy the
       character target.

    5. When final=True, emit the remaining tail batch.
    """

    created_paths: list[
        Path
    ] = []

    while True:

        next_page = int(
            batch_state[
                "next_page"
            ]
        )

        if next_page > total_pages:
            break

        # We cannot advance until the next clinical page
        # has actually finished OCR.

        if next_page not in page_results:
            break

        result = (
            page_results[
                next_page
            ]
        )

        page_text = (
            _batch_page_text(
                result
            )
        )

        page_characters = len(
            page_text
        )

        current_pages: list[int] = (
            batch_state[
                "current_pages"
            ]
        )

        current_characters = int(
            batch_state[
                "current_characters"
            ]
        )

        # --------------------------------------------------
        # CURRENT BATCH IS ALREADY LARGE ENOUGH
        # --------------------------------------------------

        if (
            current_pages
            and current_characters
            >= MIN_BATCH_CHARACTERS
            and (
                current_characters
                + page_characters
            )
            > MAX_BATCH_CHARACTERS
        ):

            batch_path = (
                _emit_current_character_batch(
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
                )
            )

            if batch_path is not None:

                created_paths.append(
                    batch_path
                )

            # Reconsider next_page for the new empty batch.
            continue

        # --------------------------------------------------
        # ADD PAGE TO CURRENT BATCH
        # --------------------------------------------------

        batch_state[
            "current_pages"
        ].append(
            next_page
        )

        batch_state[
            "current_characters"
        ] = (
            current_characters
            + page_characters
        )

        batch_state[
            "next_page"
        ] = (
            next_page + 1
        )

        new_character_count = int(
            batch_state[
                "current_characters"
            ]
        )

        # --------------------------------------------------
        # BATCH HAS REACHED / EXCEEDED MAXIMUM
        # --------------------------------------------------

        if (
            new_character_count
            >= MAX_BATCH_CHARACTERS
        ):

            batch_path = (
                _emit_current_character_batch(
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
                )
            )

            if batch_path is not None:

                created_paths.append(
                    batch_path
                )

    # ======================================================
    # FINAL TAIL
    # ======================================================

    if final:

        if (
            int(
                batch_state[
                    "next_page"
                ]
            )
            <= total_pages
        ):

            missing_for_batching = [
                page_number
                for page_number in range(
                    int(
                        batch_state[
                            "next_page"
                        ]
                    ),
                    total_pages + 1,
                )
                if (
                    page_number
                    not in page_results
                )
            ]

            if missing_for_batching:

                raise RuntimeError(
                    "Cannot finalize OCR batches because "
                    "page result(s) are missing: "
                    + ", ".join(
                        str(
                            page_number
                        )
                        for page_number
                        in missing_for_batching
                    )
                )

            # Everything exists.
            # Consume the remaining ordered pages.

            created_paths.extend(
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
            )

        # Save whatever smaller tail remains.

        tail_path = (
            _emit_current_character_batch(
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
            )
        )

        if tail_path is not None:

            created_paths.append(
                tail_path
            )

    return created_paths


# ==========================================================
# FINAL COMPLETE OCR TEXT
# ==========================================================

def save_complete_text(
    *,
    patient_name: str,
    page_results: dict[
        int,
        dict[str, Any],
    ],
) -> Path:
    """
    Create ONE final permanent-stage OCR TXT:

        ocr/output/OCR_<patient>.txt

    This is the OCR file that will eventually be moved by
    api/pipeline.py into:

        data/<patient>/OCR_<patient>.txt

    after the ENTIRE pipeline succeeds.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    sections: list[str] = []

    for page_number in sorted(
        page_results
    ):

        result = (
            page_results[
                page_number
            ]
        )

        if result["success"]:

            page_text = str(
                result[
                    "page_text"
                ]
            )

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
        get_final_output_path(
            patient_name
        )
    )

    # Write to temporary path first.
    #
    # This is important because another pipeline stage must
    # never see half-written final output.

    temporary_path = (
        complete_path.with_suffix(
            ".txt.tmp"
        )
    )

    temporary_path.write_text(
        "\n".join(
            sections
        ),
        encoding="utf-8",
    )

    # Atomic rename once the entire OCR text exists.

    temporary_path.replace(
        complete_path
    )

    return complete_path
# ==========================================================
# PROCESS ONE PDF
# ==========================================================

def process_pdf(
    pdf_path: Path,
) -> Path:
    """
    Process one patient PDF.

    FLOW:

        PDF
          ↓
        Parallel OCR page workers
          ↓
        Ordered streaming OCR batches
          ↓
        LLM can consume ready batches immediately
          ↓
        Final OCR file:
            ocr/output/OCR_<patient>.txt

    IMPORTANT:

    OCR does NOT write to data/.

    api/pipeline.py will move the final OCR output into data/<patient>/
    only after the entire pipeline succeeds.
    """

    pdf_start = (
        time.perf_counter()
    )

    pdf_path = Path(
        pdf_path
    )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if (
        pdf_path.suffix.lower()
        != ".pdf"
    ):
        raise ValueError(
            "OCR input must be a PDF file."
        )

    # ======================================================
    # PATIENT NAME
    # ======================================================

    patient_name = (
        safe_patient_name(
            pdf_path.stem
        )
    )

    # ======================================================
    # CLEAN PREVIOUS OCR STATE FOR THIS PATIENT
    # ======================================================

    clear_patient_workspace(
        patient_name
    )

    clear_patient_output(
        patient_name
    )

    # Recreate clean workspace.

    workspace = (
        get_patient_workspace(
            patient_name
        )
    )

    page_folder = (
        workspace[
            "pages"
        ]
    )

    metadata_folder = (
        workspace[
            "metadata"
        ]
    )

    batch_folder = (
        workspace[
            "batches"
        ]
    )

    image_folder = (
        workspace[
            "images"
        ]
    )

    reports_folder = (
        workspace[
            "reports"
        ]
    )

    # ======================================================
    # OPEN PDF
    # ======================================================

    with fitz.open(
        pdf_path
    ) as document:

        total_pages = len(
            document
        )

    if total_pages <= 0:
        raise RuntimeError(
            "PDF contains no pages."
        )

    # ======================================================
    # RUN INFORMATION
    # ======================================================

    print()
    print(
        "=" * 72
    )

    print(
        "NICU PARALLEL STREAMING OCR"
    )

    print(
        "=" * 72
    )

    print(
        f"PDF            : "
        f"{pdf_path.name}"
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
        f"OCR workers    : "
        f"{MAX_OCR_WORKERS}"
    )

    print(
        f"DPI            : "
        f"{OCR_DPI}"
    )

    print(
        f"Batch chars    : "
        f"{MIN_BATCH_CHARACTERS:,}"
        f" - "
        f"{MAX_BATCH_CHARACTERS:,}"
    )

    print(
        f"OCR output     : "
        f"{get_final_output_path(patient_name)}"
    )

    print(
        f"Streaming temp : "
        f"{batch_folder}"
    )

    print(
        "=" * 72
    )

    # ======================================================
    # RESULT STORAGE
    # ======================================================

    page_results: dict[
        int,
        dict[str, Any],
    ] = {}

    saved_batches: set[int] = set()

    batch_state: dict[
        str,
        Any,
    ] = {
        "next_page": 1,
        "batch_number": 1,
        "current_pages": [],
        "current_characters": 0,
    }

    # ======================================================
    # BUILD PAGE JOBS
    # ======================================================

    jobs: list[
        dict[str, Any]
    ] = []

    for page_number in range(
        1,
        total_pages + 1,
    ):

        existing_result = None

        # Normally this should be False for web runs because
        # we intentionally start with a clean workspace.

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
            ] = (
                existing_result
            )

            print(
                f"[OCR RESUME] "
                f"Page {page_number}/"
                f"{total_pages}"
            )

        else:

            jobs.append(
                {
                    "pdf_path": (
                        str(
                            pdf_path.resolve()
                        )
                    ),
                    "pdf_name": (
                        patient_name
                    ),
                    "page_number": (
                        page_number
                    ),
                    "page_folder": (
                        str(
                            page_folder
                        )
                    ),
                    "metadata_folder": (
                        str(
                            metadata_folder
                        )
                    ),
                    "image_folder": (
                        str(
                            image_folder
                        )
                    ),
                    "debug_folder": (
                        str(
                            DEBUG_FOLDER
                        )
                    ),
                }
            )

    # ======================================================
    # CONSUME ANY RESUMED PAGES FIRST
    # ======================================================

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

    # ======================================================
    # PARALLEL OCR
    # ======================================================

    if jobs:

        print()
        print(
            f"Starting "
            f"{len(jobs)} page(s) "
            f"using "
            f"{MAX_OCR_WORKERS} "
            f"parallel OCR worker(s)..."
        )
        print()

        with ProcessPoolExecutor(
            max_workers=(
                MAX_OCR_WORKERS
            ),
            initializer=(
                _initialize_worker
            ),
        ) as executor:

            futures = {
                executor.submit(
                    _process_page_job,
                    job,
                ): job
                for job in jobs
            }

            # ------------------------------------------------
            # IMPORTANT
            # ------------------------------------------------
            #
            # as_completed() returns pages as soon as workers
            # finish them.
            #
            # So if page 8 finishes before page 7, we store it.
            #
            # consume_ready_pages_into_batches() then waits until
            # page 7 exists before advancing.
            #
            # This gives us:
            #
            # parallel computation
            #       +
            # clinical page ordering
            #       +
            # streaming LLM batches
            #
            # ------------------------------------------------

            for future in as_completed(
                futures
            ):

                try:

                    result = (
                        future.result()
                    )

                except Exception as error:

                    job = futures[
                        future
                    ]

                    page_number = int(
                        job[
                            "page_number"
                        ]
                    )

                    result = {
                        "success": False,
                        "page_number": (
                            page_number
                        ),
                        "page_text": "",
                        "page_stats": None,
                        "processing_time": 0.0,
                        "attempts": (
                            PAGE_RETRIES + 1
                        ),
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }

                page_number = int(
                    result[
                        "page_number"
                    ]
                )

                page_results[
                    page_number
                ] = (
                    result
                )

                # --------------------------------------------
                # PRINT PAGE RESULT
                # --------------------------------------------

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

                    print()
                    print(
                        f"[OCR FAILED] "
                        f"Page "
                        f"{page_number}/"
                        f"{total_pages} "
                        f"| "
                        f"{result['error']}"
                    )

                # --------------------------------------------
                # STREAM READY PAGES TO BATCHES
                # --------------------------------------------
                #
                # This is called EVERY TIME a page worker
                # completes.
                #
                # Therefore LLM-ready batches can appear before
                # OCR finishes the PDF.
                #
                # --------------------------------------------

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

    # ======================================================
    # VALIDATE ALL PAGES RETURNED
    # ======================================================

    missing_pages = [
        page_number
        for page_number in range(
            1,
            total_pages + 1,
        )
        if (
            page_number
            not in page_results
        )
    ]

    if missing_pages:

        raise RuntimeError(
            "OCR did not return results "
            "for page(s): "
            + ", ".join(
                str(
                    page_number
                )
                for page_number
                in missing_pages
            )
        )

    # ======================================================
    # EMIT FINAL TAIL BATCH
    # ======================================================

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
        final=True,
    )

    total_batches = len(
        saved_batches
    )

    if (
        total_pages > 0
        and total_batches == 0
    ):

        raise RuntimeError(
            "OCR completed but no streaming "
            "batch files were created."
        )

    # ======================================================
    # CREATE FINAL OCR OUTPUT
    # ======================================================

    if SAVE_COMPLETE_TEXT:

        complete_path = (
            save_complete_text(
                patient_name=(
                    patient_name
                ),
                page_results=(
                    page_results
                ),
            )
        )

    else:

        # Our pipeline expects final OCR text.
        #
        # Therefore even if the old config flag is disabled,
        # create it anyway so the architecture remains valid.

        complete_path = (
            save_complete_text(
                patient_name=(
                    patient_name
                ),
                page_results=(
                    page_results
                ),
            )
        )

    print()
    print(
        f"[OCR COMPLETE FILE] "
        f"{complete_path}"
    )

    # ======================================================
    # OPTIONAL OCR REPORT
    # ======================================================

    report_rows = (
        build_report_rows(
            page_results
        )
    )

    if SAVE_CSV_REPORT:

        report_path = (
            reports_folder
            / (
                f"OCR_{patient_name}"
                "_REPORT.csv"
            )
        )

        save_csv_report(
            report_rows,
            report_path,
        )

        print(
            f"[OCR TEMP REPORT] "
            f"{report_path}"
        )

    # ======================================================
    # FINAL REPORT
    # ======================================================

    total_time = (
        time.perf_counter()
        - pdf_start
    )

    print_final_report(
        pdf_name=(
            pdf_path.name
        ),
        total_pages=(
            total_pages
        ),
        report_rows=(
            report_rows
        ),
        total_time=(
            total_time
        ),
    )

    # ======================================================
    # COMPLETION MARKER
    # ======================================================
    #
    # LLM can already be processing .ready batches.
    #
    # This marker means:
    #
    #     "OCR is completely finished.
    #      There will be no more OCR batches."
    #
    # ======================================================

    completion_marker = (
        save_completion_marker(
            patient_name=(
                patient_name
            ),
            total_pages=(
                total_pages
            ),
            total_batches=(
                total_batches
            ),
            complete_path=(
                complete_path
            ),
        )
    )

    print()
    print(
        f"[OCR COMPLETE MARKER] "
        f"{completion_marker}"
    )

    print()
    print(
        "=" * 72
    )

    print(
        "OCR STAGE COMPLETED"
    )

    print(
        "=" * 72
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
        f"Batches        : "
        f"{total_batches}"
    )

    print(
        f"Final output   : "
        f"{complete_path}"
    )

    print(
        f"Temporary data : "
        f"{workspace['root']}"
    )

    print(
        "=" * 72
    )

    return complete_path


# ==========================================================
# TEMPORARY INPUT CLEANUP
# ==========================================================

def remove_processed_input(
    pdf_path: Path,
) -> None:
    """
    Remove the temporary PDF from ocr/input after successful OCR.

    IMPORTANT:

    This ONLY applies when running ocr.py directly via main().

    The API pipeline may manage its own uploaded PDF lifecycle.
    """

    pdf_path = Path(
        pdf_path
    )

    try:

        if pdf_path.exists():

            pdf_path.unlink()

            print(
                f"[OCR INPUT REMOVED] "
                f"{pdf_path}"
            )

    except OSError as error:

        print(
            "Warning: could not remove "
            f"temporary OCR input "
            f"{pdf_path}: "
            f"{error}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Standalone OCR runner.

    This is mainly useful for testing OCR independently.

    Input:

        ocr/input/*.pdf

    Output:

        ocr/output/OCR_<patient>.txt

    Temporary streaming data:

        ocr/temp/<patient>/

    The full web pipeline normally calls process_pdf()
    directly from api/pipeline.py.
    """

    INPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print()
    print(
        "=" * 72
    )

    print(
        "NICU OCR STANDALONE RUN"
    )

    print(
        "=" * 72
    )

    print(
        f"PDF files   : "
        f"{len(pdf_files)}"
    )

    print(
        f"OCR workers : "
        f"{MAX_OCR_WORKERS}"
    )

    print(
        "=" * 72
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
                f"[SUCCESS] "
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

            print()

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
        f"Output folder   : "
        f"{OUTPUT_FOLDER}"
    )

    print(
        f"Temporary root  : "
        f"{WORK_FOLDER}"
    )

    print(
        "=" * 72
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
    