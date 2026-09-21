"""
NICU AI pipeline coordinator.

NEW ARCHITECTURE
================

Processing work is NOT performed inside data/<patient>/.

Streaming pipeline:

    Upload
      ↓
    OCR
      ↓
    LLM
      ↓
    Normalizer
      ↓
    Discharge

OCR, LLM and Normalizer overlap while processing.

Discharge starts only after the complete Normalizer output exists
and the Normalizer completion marker confirms completion.

MODULE WORKSPACES
=================

OCR:
    ocr/input/
    ocr/temp/
    ocr/output/

LLM:
    llm/temp/
    llm/output/

Normalizer:
    normalizer/temp/
    normalizer/output/

Discharge:
    discharge_summary/temp/
    discharge_summary/output/

FINAL SUCCESSFUL STORAGE
========================

Only after the complete pipeline succeeds:

    data/<patient>/
        OCR_<patient>.txt
        LLM_<patient>.txt
        NORMALIZER_<patient>.txt
        DISCHARGE_<patient>.txt

The four module output files are MOVED into data/<patient>/.

After successful finalization:

    - temporary stage files are removed
    - module output folders are cleared
    - OCR input is removed

On failure:

    - temporary stage files are preserved
    - module outputs are preserved
    - debugging information remains available

IMPORTANT
=========

Temporary upload names such as:

    pending_BO_VAGDEVI.pdf

must NEVER become the patient identity.

The canonical patient name is:

    BO_VAGDEVI

and OCR receives:

    ocr/input/BO_VAGDEVI.pdf
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time

from dataclasses import (
    dataclass,
    field,
)

from datetime import (
    datetime,
)

from pathlib import (
    Path,
)

from time import (
    perf_counter,
)

from typing import (
    Final,
)


from discharge_summary.generator import (
    process_patient as run_discharge,
)

from llm.extractor import (
    process_patient_stream as run_llm_stream,
)

from normalizer.normalizer import (
    process_patient_stream as run_normalizer_stream,
)

from ocr.ocr import (
    process_pdf as run_ocr,
)


from .database import (
    database,
)

from .models import (
    Patient,
    PipelineResult,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT: Final[Path] = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIRECTORY: Final[Path] = (
    PROJECT_ROOT
    / "data"
)

UPLOAD_DIRECTORY: Final[Path] = (
    DATA_DIRECTORY
    / "uploads"
)

PIPELINE_ERROR_DIRECTORY: Final[Path] = (
    DATA_DIRECTORY
    / "pipeline_errors"
)


# =============================================================================
# SOURCE WORK AREA
# =============================================================================
#
# The route initially writes:
#
#     data/uploads/pending_<filename>.pdf
#
# pipeline.start() immediately copies that into this private canonical
# source workspace using the CLEAN patient name.
#
# This means the route may safely delete its pending upload after
# pipeline.start() returns.
#
# =============================================================================

PIPELINE_WORK_DIRECTORY: Final[Path] = (
    PROJECT_ROOT
    / "pipeline_work"
)

SOURCE_WORK_DIRECTORY: Final[Path] = (
    PIPELINE_WORK_DIRECTORY
    / "source"
)


# =============================================================================
# OCR PATHS
# =============================================================================

OCR_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "ocr"
)

OCR_INPUT_DIRECTORY: Final[Path] = (
    OCR_ROOT
    / "input"
)

OCR_TEMP_DIRECTORY: Final[Path] = (
    OCR_ROOT
    / "temp"
)

OCR_OUTPUT_DIRECTORY: Final[Path] = (
    OCR_ROOT
    / "output"
)


# =============================================================================
# LLM PATHS
# =============================================================================

LLM_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "llm"
)

LLM_TEMP_DIRECTORY: Final[Path] = (
    LLM_ROOT
    / "temp"
)

LLM_OUTPUT_DIRECTORY: Final[Path] = (
    LLM_ROOT
    / "output"
)


# =============================================================================
# NORMALIZER PATHS
# =============================================================================

NORMALIZER_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "normalizer"
)

NORMALIZER_TEMP_DIRECTORY: Final[Path] = (
    NORMALIZER_ROOT
    / "temp"
)

NORMALIZER_OUTPUT_DIRECTORY: Final[Path] = (
    NORMALIZER_ROOT
    / "output"
)


# =============================================================================
# DISCHARGE PATHS
# =============================================================================

DISCHARGE_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "discharge_summary"
)

DISCHARGE_TEMP_DIRECTORY: Final[Path] = (
    DISCHARGE_ROOT
    / "temp"
)

DISCHARGE_OUTPUT_DIRECTORY: Final[Path] = (
    DISCHARGE_ROOT
    / "output"
)


# =============================================================================
# CREATE REQUIRED ROOT DIRECTORIES
# =============================================================================

for directory in (
    DATA_DIRECTORY,
    UPLOAD_DIRECTORY,
    PIPELINE_ERROR_DIRECTORY,
    PIPELINE_WORK_DIRECTORY,
    SOURCE_WORK_DIRECTORY,
    OCR_INPUT_DIRECTORY,
    OCR_TEMP_DIRECTORY,
    OCR_OUTPUT_DIRECTORY,
    LLM_TEMP_DIRECTORY,
    LLM_OUTPUT_DIRECTORY,
    NORMALIZER_TEMP_DIRECTORY,
    NORMALIZER_OUTPUT_DIRECTORY,
    DISCHARGE_TEMP_DIRECTORY,
    DISCHARGE_OUTPUT_DIRECTORY,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# PROGRESS CONFIGURATION
# =============================================================================

STAGE_LABELS: Final[
    dict[
        str,
        str,
    ]
] = {
    "idle": (
        "Waiting"
    ),

    "upload": (
        "Uploading PDF"
    ),

    "ocr": (
        "OCR Processing"
    ),

    "llm": (
        "LLM Clinical Processing"
    ),

    "normalizer": (
        "Normalizing Patient Record"
    ),

    "discharge": (
        "Generating Discharge Summary"
    ),

    "finalizing": (
        "Saving Patient Results"
    ),

    "completed": (
        "Completed"
    ),

    "failed": (
        "Failed"
    ),
}


STAGE_PERCENTAGES: Final[
    dict[
        str,
        int,
    ]
] = {
    "idle": 0,

    "upload": 5,

    "ocr": 20,

    "llm": 50,

    "normalizer": 75,

    "discharge": 90,

    "finalizing": 97,

    "completed": 100,

    "failed": 0,
}


STAGE_ORDER: Final[
    tuple[
        str,
        ...,
    ]
] = (
    "upload",
    "ocr",
    "llm",
    "normalizer",
    "discharge",
    "finalizing",
)


# =============================================================================
# PROGRESS STATE
# =============================================================================

@dataclass(
    slots=True,
)
class ProgressState:

    running: bool = False

    patient_id: (
        int
        | None
    ) = None

    patient_name: str = ""

    stage: str = "idle"

    stage_label: str = (
        STAGE_LABELS[
            "idle"
        ]
    )

    percentage: int = 0

    message: str = (
        "Waiting for a patient PDF."
    )

    completed_stages: list[
        str
    ] = field(
        default_factory=list
    )

    active_stages: list[
        str
    ] = field(
        default_factory=list
    )

    error: str = ""

    output_path: str = ""

    started_at: str = ""

    completed_at: str = ""


_PROGRESS = (
    ProgressState()
)

_PROGRESS_LOCK = (
    threading.Lock()
)


# =============================================================================
# PROGRESS HELPERS
# =============================================================================

def reset_progress() -> None:
    """
    Reset frontend/API pipeline progress.
    """

    global _PROGRESS

    with _PROGRESS_LOCK:

        _PROGRESS = (
            ProgressState()
        )


def get_progress() -> dict[
    str,
    object,
]:
    """
    Return a thread-safe progress snapshot.
    """

    with _PROGRESS_LOCK:

        return {
            "running": (
                _PROGRESS.running
            ),

            "patient_id": (
                _PROGRESS.patient_id
            ),

            "patient_name": (
                _PROGRESS.patient_name
            ),

            "stage": (
                _PROGRESS.stage
            ),

            "stage_label": (
                _PROGRESS.stage_label
            ),

            "percentage": (
                _PROGRESS.percentage
            ),

            "message": (
                _PROGRESS.message
            ),

            "completed_stages": list(
                _PROGRESS.completed_stages
            ),

            "active_stages": list(
                _PROGRESS.active_stages
            ),

            "error": (
                _PROGRESS.error
            ),

            "output_path": (
                _PROGRESS.output_path
            ),

            "started_at": (
                _PROGRESS.started_at
            ),

            "completed_at": (
                _PROGRESS.completed_at
            ),
        }


def pipeline_is_running() -> bool:
    """
    Return whether one patient pipeline is currently active.
    """

    with _PROGRESS_LOCK:

        return (
            _PROGRESS.running
        )


def update_progress(
    *,
    stage: str | None = None,
    message: str | None = None,
    patient_id: int | None = None,
    patient_name: str | None = None,
    running: bool | None = None,
    error: str | None = None,
    output_path: str | None = None,
) -> None:
    """
    Update selected progress fields.
    """

    with _PROGRESS_LOCK:

        if (
            stage
            is not None
        ):

            _PROGRESS.stage = (
                stage
            )

            _PROGRESS.stage_label = (
                STAGE_LABELS.get(
                    stage,
                    stage
                    .replace(
                        "_",
                        " ",
                    )
                    .title(),
                )
            )

            _PROGRESS.percentage = (
                STAGE_PERCENTAGES.get(
                    stage,
                    _PROGRESS.percentage,
                )
            )

        if (
            message
            is not None
        ):

            _PROGRESS.message = (
                message
            )

        if (
            patient_id
            is not None
        ):

            _PROGRESS.patient_id = (
                patient_id
            )

        if (
            patient_name
            is not None
        ):

            _PROGRESS.patient_name = (
                patient_name
            )

        if (
            running
            is not None
        ):

            _PROGRESS.running = (
                running
            )

        if (
            error
            is not None
        ):

            _PROGRESS.error = (
                error
            )

        if (
            output_path
            is not None
        ):

            _PROGRESS.output_path = (
                output_path
            )


def mark_stage_active(
    stage: str,
    *,
    message: str,
) -> None:
    """
    Mark one processing stage active.
    """

    with _PROGRESS_LOCK:

        if (
            stage
            not in _PROGRESS.active_stages
        ):

            _PROGRESS.active_stages.append(
                stage
            )

        _PROGRESS.stage = (
            stage
        )

        _PROGRESS.stage_label = (
            STAGE_LABELS.get(
                stage,
                stage,
            )
        )

        _PROGRESS.percentage = max(
            _PROGRESS.percentage,
            STAGE_PERCENTAGES.get(
                stage,
                0,
            ),
        )

        _PROGRESS.message = (
            message
        )


def mark_stage_completed(
    stage: str,
) -> None:
    """
    Mark one stage completed.
    """

    with _PROGRESS_LOCK:

        if (
            stage
            in _PROGRESS.active_stages
        ):

            _PROGRESS.active_stages.remove(
                stage
            )

        if (
            stage
            not in _PROGRESS.completed_stages
        ):

            _PROGRESS.completed_stages.append(
                stage
            )

        _PROGRESS.percentage = max(
            _PROGRESS.percentage,
            STAGE_PERCENTAGES.get(
                stage,
                0,
            ),
        )


def set_progress_time(
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    """
    Update pipeline timestamps.
    """

    with _PROGRESS_LOCK:

        if (
            started_at
            is not None
        ):

            _PROGRESS.started_at = (
                started_at
            )

        if (
            completed_at
            is not None
        ):

            _PROGRESS.completed_at = (
                completed_at
            )


# =============================================================================
# PATIENT NAME HELPERS
# =============================================================================

def safe_patient_name(
    value: str,
) -> str:
    """
    Convert a patient/file name into the canonical patient name.

    Examples:

        pending_BO_VAGDEVI
            -> BO_VAGDEVI

        pending_pending_BO_VAGDEVI
            -> BO_VAGDEVI

        Baby of Vagdevi
            -> Baby_of_Vagdevi
    """

    value = (
        value.strip()
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


def safe_filename(
    filename: str,
) -> str:
    """
    Convert an uploaded filename into a safe filename.

    This function is still used by api/routes.py.
    """

    name = (
        Path(
            filename
        )
        .name
    )

    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name,
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
        or "patient.pdf"
    )


# =============================================================================
# FINAL PATIENT STORAGE
# =============================================================================

def get_patient_root(
    patient_name: str,
) -> Path:
    """
    Return:

        data/<patient>/

    Unlike the old pipeline, no OCR/LLM/Normalizer/Discharge
    subdirectories are created here.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    return (
        DATA_DIRECTORY
        / patient_name
    )


def get_final_patient_paths(
    patient_name: str,
) -> dict[
    str,
    Path,
]:
    """
    Return the final successful patient artifact paths.

    Nothing is created by this function.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    root = (
        get_patient_root(
            patient_name
        )
    )

    return {
        "root": (
            root
        ),

        "ocr": (
            root
            / (
                f"OCR_"
                f"{patient_name}"
                f".txt"
            )
        ),

        "llm": (
            root
            / (
                f"LLM_"
                f"{patient_name}"
                f".txt"
            )
        ),

        "normalizer": (
            root
            / (
                f"NORMALIZER_"
                f"{patient_name}"
                f".txt"
            )
        ),

        "discharge": (
            root
            / (
                f"DISCHARGE_"
                f"{patient_name}"
                f".txt"
            )
        ),
    }


# =============================================================================
# EXPECTED MODULE OUTPUT PATHS
# =============================================================================

def get_module_output_paths(
    patient_name: str,
) -> dict[
    str,
    Path,
]:
    """
    Return the four completed module outputs BEFORE finalization.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    return {
        "ocr": (
            OCR_OUTPUT_DIRECTORY
            / (
                f"OCR_"
                f"{patient_name}"
                f".txt"
            )
        ),

        "llm": (
            LLM_OUTPUT_DIRECTORY
            / (
                f"LLM_"
                f"{patient_name}"
                f".txt"
            )
        ),

        "normalizer": (
            NORMALIZER_OUTPUT_DIRECTORY
            / (
                f"NORMALIZER_"
                f"{patient_name}"
                f".txt"
            )
        ),

        "discharge": (
            DISCHARGE_OUTPUT_DIRECTORY
            / (
                f"DISCHARGE_"
                f"{patient_name}"
                f".txt"
            )
        ),
    }


# =============================================================================
# COMPLETION MARKER PATHS
# =============================================================================

def get_stage_marker_paths(
    patient_name: str,
) -> dict[
    str,
    Path,
]:
    """
    Return temporary completion markers for all stages.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    return {
        "ocr": (
            OCR_TEMP_DIRECTORY
            / patient_name
            / "ocr_complete.json"
        ),

        "llm": (
            LLM_TEMP_DIRECTORY
            / patient_name
            / "llm_complete.json"
        ),

        "normalizer": (
            NORMALIZER_TEMP_DIRECTORY
            / patient_name
            / "normalizer_complete.json"
        ),

        "discharge": (
            DISCHARGE_TEMP_DIRECTORY
            / patient_name
            / "discharge_complete.json"
        ),
    }


# =============================================================================
# GENERIC DIRECTORY CLEANER
# =============================================================================

def clear_directory_contents(
    directory: Path,
) -> None:
    """
    Remove every child inside one directory while keeping the
    directory itself.
    """

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for child in list(
        directory.iterdir()
    ):

        if (
            child.is_dir()
        ):

            shutil.rmtree(
                child
            )

        else:

            child.unlink(
                missing_ok=True
            )


# =============================================================================
# CLEAN ALL ACTIVE STAGE STATE
# =============================================================================

def clear_active_pipeline_state() -> None:
    """
    Clear previous temporary/output processing state BEFORE a
    new patient run.

    Because the backend processes only one patient at a time,
    globally clearing these active work folders is intentional.

    IMPORTANT:

    data/<patient>/ final completed records are NOT touched here.
    """

    directories_to_clear = (
        OCR_INPUT_DIRECTORY,
        OCR_TEMP_DIRECTORY,
        OCR_OUTPUT_DIRECTORY,

        LLM_TEMP_DIRECTORY,
        LLM_OUTPUT_DIRECTORY,

        NORMALIZER_TEMP_DIRECTORY,
        NORMALIZER_OUTPUT_DIRECTORY,

        DISCHARGE_TEMP_DIRECTORY,
        DISCHARGE_OUTPUT_DIRECTORY,

        SOURCE_WORK_DIRECTORY,
    )

    for directory in (
        directories_to_clear
    ):

        clear_directory_contents(
            directory
        )


# =============================================================================
# PREPARE FRESH PATIENT
# =============================================================================

def prepare_patient_source(
    uploaded_pdf: Path,
) -> tuple[
    str,
    Path,
]:
    """
    Prepare a fresh canonical PDF for one patient.

    Example route upload:

        data/uploads/pending_BO_VAGDEVI.pdf

    becomes:

        pipeline_work/source/BO_VAGDEVI.pdf

    The word 'pending' therefore cannot reach OCR.
    """

    uploaded_pdf = (
        uploaded_pdf
        .expanduser()
        .resolve()
    )

    if not uploaded_pdf.exists():

        raise FileNotFoundError(
            "Uploaded PDF was not found: "
            f"{uploaded_pdf}"
        )

    if (
        uploaded_pdf.suffix.lower()
        != ".pdf"
    ):

        raise ValueError(
            "Only PDF patient records "
            "are supported."
        )

    patient_name = (
        safe_patient_name(
            uploaded_pdf.stem
        )
    )

    if not patient_name:

        raise ValueError(
            "Could not determine a valid "
            "patient name."
        )

    # ------------------------------------------------------
    # CLEAR OLD ACTIVE PIPELINE STATE FIRST
    # ------------------------------------------------------

    clear_active_pipeline_state()

    # ------------------------------------------------------
    # SAME PATIENT RE-RUN
    # ------------------------------------------------------
    #
    # A fresh upload of the same patient replaces that
    # patient's previous finalized result.
    #
    # Other completed patients remain untouched.
    #
    # ------------------------------------------------------

    patient_root = (
        get_patient_root(
            patient_name
        )
    )

    if patient_root.exists():

        shutil.rmtree(
            patient_root
        )

    # ------------------------------------------------------
    # CANONICAL SOURCE PDF
    # ------------------------------------------------------

    canonical_pdf = (
        SOURCE_WORK_DIRECTORY
        / (
            f"{patient_name}"
            f".pdf"
        )
    )

    shutil.copy2(
        uploaded_pdf,
        canonical_pdf,
    )

    if not canonical_pdf.exists():

        raise RuntimeError(
            "Canonical patient PDF "
            "could not be created."
        )

    return (
        patient_name,
        canonical_pdf,
    )


# =============================================================================
# COPY CANONICAL PDF TO OCR INPUT
# =============================================================================

def copy_to_ocr_input(
    canonical_pdf: Path,
    patient_name: str,
) -> Path:
    """
    Copy the canonical source PDF into OCR input.

    OCR always receives:

        ocr/input/<patient>.pdf

    NEVER:

        ocr/input/pending_<patient>.pdf
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    OCR_INPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    active_pdf = (
        OCR_INPUT_DIRECTORY
        / (
            f"{patient_name}"
            f".pdf"
        )
    )

    if active_pdf.exists():

        active_pdf.unlink()

    shutil.copy2(
        canonical_pdf,
        active_pdf,
    )

    if not active_pdf.exists():

        raise RuntimeError(
            "Could not prepare OCR input "
            f"for patient {patient_name}."
        )

    if (
        active_pdf.stem
        != patient_name
    ):

        raise RuntimeError(
            "OCR input patient identity "
            "does not match the canonical "
            "patient name."
        )

    return active_pdf


# =============================================================================
# VERIFY TEXT FILE
# =============================================================================

def validate_text_output(
    path: Path,
    *,
    stage: str,
) -> Path:
    """
    Validate one completed clinical TXT output.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{stage} output was not "
            f"created: {path}"
        )

    if (
        path.suffix.lower()
        != ".txt"
    ):

        raise RuntimeError(
            f"{stage} output must be "
            f"a TXT file: {path}"
        )

    text = (
        path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        .strip()
    )

    if not text:

        raise RuntimeError(
            f"{stage} output is empty: "
            f"{path}"
        )

    return path


# =============================================================================
# LOAD COMPLETION MARKER
# =============================================================================

def load_completion_marker(
    marker_path: Path,
    *,
    patient_name: str,
    expected_stage: str,
) -> dict[
    str,
    object,
]:
    """
    Load and validate one temporary completion marker.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    if not marker_path.exists():

        raise FileNotFoundError(
            f"{expected_stage} completion "
            "marker was not created: "
            f"{marker_path}"
        )

    try:

        payload = json.loads(
            marker_path.read_text(
                encoding="utf-8",
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:

        raise RuntimeError(
            f"{expected_stage} completion "
            "marker is invalid."
        ) from error

    if not isinstance(
        payload,
        dict,
    ):

        raise RuntimeError(
            f"{expected_stage} completion "
            "marker must contain a JSON "
            "object."
        )

    if (
        payload.get(
            "status"
        )
        != "completed"
    ):

        raise RuntimeError(
            f"{expected_stage} stage is "
            "not marked completed."
        )

    marker_stage = str(
        payload.get(
            "stage",
            "",
        )
    ).strip()

    if (
        marker_stage
        != expected_stage
    ):

        raise RuntimeError(
            f"{expected_stage} completion "
            "marker has the wrong stage: "
            f"{marker_stage!r}"
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
            f"{expected_stage} completion "
            "marker belongs to another "
            f"patient. Expected "
            f"{patient_name}, found "
            f"{marker_patient}."
        )

    return payload


# =============================================================================
# PIPELINE ERROR PATH
# =============================================================================

def save_pipeline_error(
    *,
    patient_name: str,
    stage: str,
    error: Exception,
) -> Path:
    """
    Save a pipeline-level error report.

    Stage temp files are deliberately left untouched.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    safe_stage = (
        safe_patient_name(
            stage
        )
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    PIPELINE_ERROR_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_file = (
        PIPELINE_ERROR_DIRECTORY
        / (
            f"{patient_name}"
            f"_{safe_stage}"
            f"_{timestamp}"
            f"_ERROR.txt"
        )
    )

    error_text = (
        f"Patient: "
        f"{patient_name}\n"
        f"Stage: "
        f"{stage}\n"
        f"Time: "
        f"{datetime.now().isoformat()}\n"
        f"Error type: "
        f"{type(error).__name__}\n"
        f"Error: "
        f"{error}\n"
    )

    error_file.write_text(
        error_text,
        encoding="utf-8",
    )

    return error_file


# =============================================================================
# WORKER RESULT
# =============================================================================

@dataclass(
    slots=True,
)
class WorkerResult:
    """
    Result returned by one streaming stage worker.
    """

    stage: str

    success: bool

    output_path: (
        Path
        | None
    ) = None

    error: (
        Exception
        | None
    ) = None

# =============================================================================
# VERIFY OCR COMPLETION
# =============================================================================

def verify_ocr_completion(
    patient_name: str,
) -> Path:
    """
    Validate the completed OCR stage.

    Required:

        ocr/output/OCR_<patient>.txt

        ocr/temp/<patient>/ocr_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    outputs = (
        get_module_output_paths(
            patient_name
        )
    )

    markers = (
        get_stage_marker_paths(
            patient_name
        )
    )

    output_path = (
        validate_text_output(
            outputs["ocr"],
            stage="OCR",
        )
    )

    payload = (
        load_completion_marker(
            markers["ocr"],
            patient_name=(
                patient_name
            ),
            expected_stage="ocr",
        )
    )

    complete_file = str(
        payload.get(
            "complete_file",
            "",
        )
    ).strip()

    if (
        complete_file
        and complete_file
        != output_path.name
    ):

        raise RuntimeError(
            "OCR completion marker "
            "references an unexpected "
            "output file. "
            f"Expected {output_path.name}, "
            f"found {complete_file}."
        )

    return output_path


# =============================================================================
# VERIFY LLM COMPLETION
# =============================================================================

def verify_llm_completion(
    patient_name: str,
) -> Path:
    """
    Validate the completed LLM stage.

    Required:

        llm/output/LLM_<patient>.txt

        llm/temp/<patient>/llm_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    outputs = (
        get_module_output_paths(
            patient_name
        )
    )

    markers = (
        get_stage_marker_paths(
            patient_name
        )
    )

    output_path = (
        validate_text_output(
            outputs["llm"],
            stage="LLM",
        )
    )

    payload = (
        load_completion_marker(
            markers["llm"],
            patient_name=(
                patient_name
            ),
            expected_stage="llm",
        )
    )

    complete_file = str(
        payload.get(
            "complete_file",
            "",
        )
    ).strip()

    if (
        complete_file
        and complete_file
        != output_path.name
    ):

        raise RuntimeError(
            "LLM completion marker "
            "references an unexpected "
            "output file. "
            f"Expected {output_path.name}, "
            f"found {complete_file}."
        )

    return output_path


# =============================================================================
# VERIFY NORMALIZER COMPLETION
# =============================================================================

def verify_normalizer_completion(
    patient_name: str,
) -> Path:
    """
    Validate the strict Normalizer completion gate.

    Required:

        normalizer/output/NORMALIZER_<patient>.txt

        normalizer/temp/<patient>/normalizer_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    outputs = (
        get_module_output_paths(
            patient_name
        )
    )

    markers = (
        get_stage_marker_paths(
            patient_name
        )
    )

    output_path = (
        validate_text_output(
            outputs["normalizer"],
            stage="Normalizer",
        )
    )

    payload = (
        load_completion_marker(
            markers["normalizer"],
            patient_name=(
                patient_name
            ),
            expected_stage="normalizer",
        )
    )

    complete_file = str(
        payload.get(
            "complete_file",
            "",
        )
    ).strip()

    if (
        complete_file
        != output_path.name
    ):

        raise RuntimeError(
            "Normalizer completion marker "
            "does not reference the expected "
            "final Normalizer TXT. "
            f"Expected {output_path.name}, "
            f"found {complete_file!r}."
        )

    expected_character_count = (
        payload.get(
            "character_count"
        )
    )

    if isinstance(
        expected_character_count,
        int,
    ) and (
        expected_character_count
        > 0
    ):

        text = (
            output_path
            .read_text(
                encoding="utf-8",
                errors="replace",
            )
            .strip()
        )

        if (
            len(text)
            != expected_character_count
        ):

            raise RuntimeError(
                "Normalizer output character "
                "count does not match its "
                "completion marker."
            )

    return output_path


# =============================================================================
# VERIFY DISCHARGE COMPLETION
# =============================================================================

def verify_discharge_completion(
    patient_name: str,
) -> Path:
    """
    Validate the completed Discharge stage.

    Required:

        discharge_summary/output/DISCHARGE_<patient>.txt

        discharge_summary/temp/<patient>/discharge_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    outputs = (
        get_module_output_paths(
            patient_name
        )
    )

    markers = (
        get_stage_marker_paths(
            patient_name
        )
    )

    output_path = (
        validate_text_output(
            outputs["discharge"],
            stage="Discharge",
        )
    )

    payload = (
        load_completion_marker(
            markers["discharge"],
            patient_name=(
                patient_name
            ),
            expected_stage="discharge",
        )
    )

    summary_file = str(
        payload.get(
            "summary_file",
            "",
        )
    ).strip()

    if (
        summary_file
        != output_path.name
    ):

        raise RuntimeError(
            "Discharge completion marker "
            "does not reference the expected "
            "final discharge TXT. "
            f"Expected {output_path.name}, "
            f"found {summary_file!r}."
        )

    expected_character_count = (
        payload.get(
            "character_count"
        )
    )

    if isinstance(
        expected_character_count,
        int,
    ) and (
        expected_character_count
        > 0
    ):

        text = (
            output_path
            .read_text(
                encoding="utf-8",
                errors="replace",
            )
            .strip()
        )

        if (
            len(text)
            != expected_character_count
        ):

            raise RuntimeError(
                "Discharge output character "
                "count does not match its "
                "completion marker."
            )

    return output_path


# =============================================================================
# OCR WORKER
# =============================================================================

def run_ocr_worker(
    patient_name: str,
    active_pdf: Path,
) -> WorkerResult:
    """
    Run OCR and verify its final TXT + completion marker.
    """

    try:

        mark_stage_active(
            "ocr",
            message=(
                "OCR is processing "
                "patient pages."
            ),
        )

        returned_path = (
            run_ocr(
                active_pdf
            )
        )

        output_path = (
            verify_ocr_completion(
                patient_name
            )
        )

        if (
            returned_path
            is not None
        ):

            returned_path = (
                Path(
                    returned_path
                )
            )

            if (
                returned_path.resolve()
                != output_path.resolve()
            ):

                raise RuntimeError(
                    "OCR returned an unexpected "
                    "final output path. "
                    f"Returned: {returned_path}. "
                    f"Expected: {output_path}."
                )

        mark_stage_completed(
            "ocr"
        )

        patient_id = (
            get_progress()
            .get(
                "patient_id"
            )
        )

        if isinstance(
            patient_id,
            int,
        ):

            database.update_status(
                patient_id,
                "OCR Completed",
            )

            database.update_stage_path(
                patient_id,
                "ocr_path",
                str(
                    output_path
                ),
            )

        return WorkerResult(
            stage="ocr",
            success=True,
            output_path=(
                output_path
            ),
        )

    except Exception as error:

        return WorkerResult(
            stage="ocr",
            success=False,
            error=error,
        )


# =============================================================================
# LLM WORKER
# =============================================================================

def run_llm_worker(
    patient_name: str,
) -> WorkerResult:
    """
    Watch OCR streaming batches, run the LLM, then validate the
    complete LLM stage.
    """

    try:

        mark_stage_active(
            "llm",
            message=(
                "LLM is processing completed "
                "OCR batches while OCR continues."
            ),
        )

        returned_path = (
            run_llm_stream(
                patient_name
            )
        )

        output_path = (
            verify_llm_completion(
                patient_name
            )
        )

        if (
            returned_path
            is not None
        ):

            returned_path = (
                Path(
                    returned_path
                )
            )

            if (
                returned_path.resolve()
                != output_path.resolve()
            ):

                raise RuntimeError(
                    "LLM returned an unexpected "
                    "final output path. "
                    f"Returned: {returned_path}. "
                    f"Expected: {output_path}."
                )

        mark_stage_completed(
            "llm"
        )

        patient_id = (
            get_progress()
            .get(
                "patient_id"
            )
        )

        if isinstance(
            patient_id,
            int,
        ):

            database.update_status(
                patient_id,
                "LLM Completed",
            )

            database.update_stage_path(
                patient_id,
                "structured_path",
                str(
                    output_path
                ),
            )

        return WorkerResult(
            stage="llm",
            success=True,
            output_path=(
                output_path
            ),
        )

    except Exception as error:

        return WorkerResult(
            stage="llm",
            success=False,
            error=error,
        )


# =============================================================================
# NORMALIZER WORKER
# =============================================================================

def run_normalizer_worker(
    patient_name: str,
) -> WorkerResult:
    """
    Watch completed LLM batches while earlier stages continue.

    This worker returns only after the complete combined
    Normalizer TXT and completion marker are valid.
    """

    try:

        mark_stage_active(
            "normalizer",
            message=(
                "Normalizer is processing "
                "completed LLM batches while "
                "earlier stages continue."
            ),
        )

        returned_path = (
            run_normalizer_stream(
                patient_name
            )
        )

        output_path = (
            verify_normalizer_completion(
                patient_name
            )
        )

        if (
            returned_path
            is not None
        ):

            returned_path = (
                Path(
                    returned_path
                )
            )

            if (
                returned_path.resolve()
                != output_path.resolve()
            ):

                raise RuntimeError(
                    "Normalizer returned an "
                    "unexpected final output path. "
                    f"Returned: {returned_path}. "
                    f"Expected: {output_path}."
                )

        mark_stage_completed(
            "normalizer"
        )

        patient_id = (
            get_progress()
            .get(
                "patient_id"
            )
        )

        if isinstance(
            patient_id,
            int,
        ):

            database.update_status(
                patient_id,
                "Normalizer Completed",
            )

            database.update_stage_path(
                patient_id,
                "normalized_path",
                str(
                    output_path
                ),
            )

        return WorkerResult(
            stage="normalizer",
            success=True,
            output_path=(
                output_path
            ),
        )

    except Exception as error:

        return WorkerResult(
            stage="normalizer",
            success=False,
            error=error,
        )


# =============================================================================
# VERIFY ALL FOUR MODULE OUTPUTS
# =============================================================================

def verify_all_module_outputs(
    patient_name: str,
) -> dict[
    str,
    Path,
]:
    """
    Verify all four final stage outputs before ANY artifact is
    moved into data/<patient>/.

    This prevents partial finalization from starting when one
    stage is incomplete.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    verified = {
        "ocr": (
            verify_ocr_completion(
                patient_name
            )
        ),

        "llm": (
            verify_llm_completion(
                patient_name
            )
        ),

        "normalizer": (
            verify_normalizer_completion(
                patient_name
            )
        ),

        "discharge": (
            verify_discharge_completion(
                patient_name
            )
        ),
    }

    return verified


# =============================================================================
# FINALIZATION STAGING DIRECTORY
# =============================================================================

def get_finalization_staging_root(
    patient_name: str,
) -> Path:
    """
    Return a temporary directory inside data/.

    Example:

        data/.finalizing_BO_VAGDEVI/

    All four files are first MOVED here.

    Only after all four moves succeed is this directory renamed
    to:

        data/BO_VAGDEVI/

    This prevents the frontend from seeing a half-created final
    patient directory.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    return (
        DATA_DIRECTORY
        / (
            f".finalizing_"
            f"{patient_name}"
        )
    )


# =============================================================================
# MOVE OUTPUTS INTO FINALIZATION STAGING
# =============================================================================

def move_outputs_to_staging(
    patient_name: str,
    verified_outputs: dict[
        str,
        Path,
    ],
) -> Path:
    """
    Move all four verified module outputs into a temporary
    finalization directory.

    No copies are made.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    staging_root = (
        get_finalization_staging_root(
            patient_name
        )
    )

    if staging_root.exists():

        shutil.rmtree(
            staging_root
        )

    staging_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    expected_names = {
        "ocr": (
            f"OCR_"
            f"{patient_name}"
            f".txt"
        ),

        "llm": (
            f"LLM_"
            f"{patient_name}"
            f".txt"
        ),

        "normalizer": (
            f"NORMALIZER_"
            f"{patient_name}"
            f".txt"
        ),

        "discharge": (
            f"DISCHARGE_"
            f"{patient_name}"
            f".txt"
        ),
    }

    moved_paths: list[
        tuple[
            Path,
            Path,
        ]
    ] = []

    try:

        for stage in (
            "ocr",
            "llm",
            "normalizer",
            "discharge",
        ):

            source = (
                verified_outputs[
                    stage
                ]
            )

            destination = (
                staging_root
                / expected_names[
                    stage
                ]
            )

            if (
                source.name
                != expected_names[
                    stage
                ]
            ):

                raise RuntimeError(
                    f"{stage} output has "
                    "an unexpected filename: "
                    f"{source.name}"
                )

            if not source.exists():

                raise FileNotFoundError(
                    f"{stage} output disappeared "
                    "before finalization: "
                    f"{source}"
                )

            shutil.move(
                str(
                    source
                ),
                str(
                    destination
                ),
            )

            moved_paths.append(
                (
                    source,
                    destination,
                )
            )

            validate_text_output(
                destination,
                stage=(
                    stage.upper()
                ),
            )

    except Exception:

        # --------------------------------------------------
        # ROLLBACK ANY SUCCESSFUL MOVES
        # --------------------------------------------------
        #
        # If move 1/2 succeeds but a later move fails, restore
        # the files to their module output folders.
        #
        # --------------------------------------------------

        for (
            original_source,
            staging_destination,
        ) in reversed(
            moved_paths
        ):

            try:

                if (
                    staging_destination.exists()
                    and not original_source.exists()
                ):

                    original_source.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    shutil.move(
                        str(
                            staging_destination
                        ),
                        str(
                            original_source
                        ),
                    )

            except Exception:

                # Preserve the original finalization error.
                # Any remaining staging file is still available
                # for debugging.
                pass

        raise

    return staging_root


# =============================================================================
# VALIDATE STAGING DIRECTORY
# =============================================================================

def validate_finalization_staging(
    patient_name: str,
    staging_root: Path,
) -> None:
    """
    Confirm that the staging directory contains exactly the four
    required clinical TXT outputs.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    expected_names = {
        (
            f"OCR_"
            f"{patient_name}"
            f".txt"
        ),

        (
            f"LLM_"
            f"{patient_name}"
            f".txt"
        ),

        (
            f"NORMALIZER_"
            f"{patient_name}"
            f".txt"
        ),

        (
            f"DISCHARGE_"
            f"{patient_name}"
            f".txt"
        ),
    }

    actual_files = {
        path.name
        for path in (
            staging_root.iterdir()
        )
        if path.is_file()
    }

    if (
        actual_files
        != expected_names
    ):

        raise RuntimeError(
            "Finalization staging directory "
            "does not contain exactly the "
            "four required outputs. "
            f"Expected: "
            f"{sorted(expected_names)}. "
            f"Found: "
            f"{sorted(actual_files)}."
        )

    for filename in (
        expected_names
    ):

        validate_text_output(
            staging_root
            / filename,
            stage="Finalization",
        )


# =============================================================================
# COMMIT FINAL PATIENT DIRECTORY
# =============================================================================

def commit_final_patient_directory(
    patient_name: str,
    staging_root: Path,
) -> dict[
    str,
    Path,
]:
    """
    Atomically expose the completed patient directory.

    Before this function:

        data/.finalizing_<patient>/

    After this function:

        data/<patient>/

    containing exactly four TXT files.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    validate_finalization_staging(
        patient_name,
        staging_root,
    )

    final_paths = (
        get_final_patient_paths(
            patient_name
        )
    )

    final_root = (
        final_paths[
            "root"
        ]
    )

    if final_root.exists():

        shutil.rmtree(
            final_root
        )

    staging_root.rename(
        final_root
    )

    if not final_root.exists():

        raise RuntimeError(
            "Final patient directory "
            "could not be committed."
        )

    # ------------------------------------------------------
    # VERIFY ALL FOUR FINAL FILES AFTER DIRECTORY RENAME
    # ------------------------------------------------------

    for stage in (
        "ocr",
        "llm",
        "normalizer",
        "discharge",
    ):

        validate_text_output(
            final_paths[
                stage
            ],
            stage=(
                f"Final {stage}"
            ),
        )

    return final_paths


# =============================================================================
# CLEAN SUCCESSFUL TEMPORARY STATE
# =============================================================================

def cleanup_successful_pipeline(
    patient_name: str,
    canonical_pdf: Path,
    active_pdf: Path,
) -> None:
    """
    Delete temporary processing state ONLY after successful
    finalization.

    Final data/<patient>/ outputs are never touched.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    patient_temp_directories = (
        OCR_TEMP_DIRECTORY
        / patient_name,

        LLM_TEMP_DIRECTORY
        / patient_name,

        NORMALIZER_TEMP_DIRECTORY
        / patient_name,

        DISCHARGE_TEMP_DIRECTORY
        / patient_name,
    )

    for directory in (
        patient_temp_directories
    ):

        if directory.exists():

            shutil.rmtree(
                directory
            )

    # ------------------------------------------------------
    # REMOVE WORKING PDF COPIES
    # ------------------------------------------------------

    active_pdf.unlink(
        missing_ok=True
    )

    canonical_pdf.unlink(
        missing_ok=True
    )

    # ------------------------------------------------------
    # REMOVE ANY EMPTY SOURCE DIRECTORY CHILDREN ONLY
    # ------------------------------------------------------

    for directory in (
        OCR_INPUT_DIRECTORY,
        OCR_OUTPUT_DIRECTORY,
        LLM_OUTPUT_DIRECTORY,
        NORMALIZER_OUTPUT_DIRECTORY,
        DISCHARGE_OUTPUT_DIRECTORY,
        SOURCE_WORK_DIRECTORY,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# FINALIZE COMPLETE PATIENT
# =============================================================================

def finalize_patient_outputs(
    patient_name: str,
    canonical_pdf: Path,
    active_pdf: Path,
) -> dict[
    str,
    Path,
]:
    """
    Finalize one successful patient.

    Order:

        1. Verify ALL four module outputs + markers.
        2. Move all four outputs to hidden staging.
        3. Validate staging.
        4. Rename staging -> data/<patient>/.
        5. Validate final patient files.
        6. Delete temporary processing state.

    If verification fails before moving:
        nothing is moved.

    If a move fails:
        moved files are rolled back where possible.

    If finalization succeeds:
        module output folders no longer contain this patient's
        four final TXT files.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    mark_stage_active(
        "finalizing",
        message=(
            "All stages completed. "
            "Saving final patient results."
        ),
    )

    verified_outputs = (
        verify_all_module_outputs(
            patient_name
        )
    )

    staging_root = (
        move_outputs_to_staging(
            patient_name,
            verified_outputs,
        )
    )

    validate_finalization_staging(
        patient_name,
        staging_root,
    )

    final_paths = (
        commit_final_patient_directory(
            patient_name,
            staging_root,
        )
    )

    cleanup_successful_pipeline(
        patient_name,
        canonical_pdf,
        active_pdf,
    )

    mark_stage_completed(
        "finalizing"
    )

    return final_paths

# =============================================================================
# PIPELINE CLASS
# =============================================================================

class Pipeline:
    """
    Coordinate the complete NICU AI pipeline.

    Streaming stages:

        OCR
         ↓
        LLM
         ↓
        Normalizer

    These three stages overlap.

    Discharge does NOT stream.

    It starts only after the complete Normalizer TXT and its
    completion marker have been validated.

    After Discharge succeeds, all four final TXT outputs are
    verified and moved into:

        data/<patient>/
    """

    # =========================================================================
    # START BACKGROUND PIPELINE
    # =========================================================================

    def start(
        self,
        pdf_file: Path,
    ) -> Patient:
        """
        Start one patient pipeline in a background thread.

        Used by the FastAPI upload route.
        """

        if pipeline_is_running():

            raise RuntimeError(
                "Another patient is currently "
                "being processed. Please wait "
                "until it finishes."
            )

        pdf_file = (
            pdf_file
            .expanduser()
            .resolve()
        )

        if not pdf_file.exists():

            raise FileNotFoundError(
                "Uploaded PDF was not found: "
                f"{pdf_file}"
            )

        if (
            pdf_file.suffix.lower()
            != ".pdf"
        ):

            raise ValueError(
                "Only PDF patient records "
                "are supported."
            )

        # ---------------------------------------------------------------------
        # RESET FRONTEND STATE
        # ---------------------------------------------------------------------

        reset_progress()

        # ---------------------------------------------------------------------
        # PREPARE FRESH CANONICAL PATIENT SOURCE
        # ---------------------------------------------------------------------

        (
            patient_name,
            canonical_pdf,
        ) = (
            prepare_patient_source(
                pdf_file
            )
        )

        # ---------------------------------------------------------------------
        # DATABASE RECORD
        # ---------------------------------------------------------------------

        patient_id = (
            database.add_patient(
                patient_name=(
                    patient_name
                ),
                pdf_path=str(
                    canonical_pdf
                ),
            )
        )

        patient = Patient(
            id=(
                patient_id
            ),
            patient_name=(
                patient_name
            ),
            status=(
                "Uploaded"
            ),
            pdf_path=(
                canonical_pdf
            ),
        )

        # ---------------------------------------------------------------------
        # PROGRESS
        # ---------------------------------------------------------------------

        set_progress_time(
            started_at=(
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            )
        )

        update_progress(
            stage="upload",
            message=(
                "Patient PDF uploaded "
                "successfully."
            ),
            patient_id=(
                patient_id
            ),
            patient_name=(
                patient_name
            ),
            running=True,
            error="",
            output_path="",
        )

        mark_stage_completed(
            "upload"
        )

        # ---------------------------------------------------------------------
        # START BACKGROUND COORDINATOR
        # ---------------------------------------------------------------------

        worker = (
            threading.Thread(
                target=(
                    self._run_background
                ),
                args=(
                    patient,
                    canonical_pdf,
                ),
                daemon=True,
                name=(
                    "nicu-pipeline-"
                    f"{patient_id}"
                ),
            )
        )

        worker.start()

        return patient

    # =========================================================================
    # SYNCHRONOUS PIPELINE
    # =========================================================================

    def run_sync(
        self,
        pdf_file: Path,
    ) -> PipelineResult:
        """
        Run one complete patient pipeline synchronously.

        Useful later for terminal testing.
        """

        started = (
            perf_counter()
        )

        if pipeline_is_running():

            raise RuntimeError(
                "Another patient is currently "
                "being processed."
            )

        pdf_file = (
            pdf_file
            .expanduser()
            .resolve()
        )

        if not pdf_file.exists():

            raise FileNotFoundError(
                f"PDF was not found: "
                f"{pdf_file}"
            )

        if (
            pdf_file.suffix.lower()
            != ".pdf"
        ):

            raise ValueError(
                "Only PDF patient records "
                "are supported."
            )

        reset_progress()

        (
            patient_name,
            canonical_pdf,
        ) = (
            prepare_patient_source(
                pdf_file
            )
        )

        patient_id = (
            database.add_patient(
                patient_name=(
                    patient_name
                ),
                pdf_path=str(
                    canonical_pdf
                ),
            )
        )

        patient = Patient(
            id=(
                patient_id
            ),
            patient_name=(
                patient_name
            ),
            status=(
                "Uploaded"
            ),
            pdf_path=(
                canonical_pdf
            ),
        )

        set_progress_time(
            started_at=(
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            )
        )

        update_progress(
            stage="upload",
            message=(
                "Patient PDF uploaded "
                "successfully."
            ),
            patient_id=(
                patient_id
            ),
            patient_name=(
                patient_name
            ),
            running=True,
            error="",
            output_path="",
        )

        mark_stage_completed(
            "upload"
        )

        try:

            self._process_patient(
                patient,
                canonical_pdf,
            )

            return PipelineResult(
                patient=(
                    patient
                ),
                success=True,
                message=(
                    "Pipeline completed "
                    "successfully."
                ),
                elapsed_seconds=(
                    perf_counter()
                    - started
                ),
            )

        except Exception as error:

            self._handle_failure(
                patient=(
                    patient
                ),
                stage=str(
                    get_progress()
                    .get(
                        "stage",
                        "unknown",
                    )
                ),
                error=(
                    error
                ),
            )

            return PipelineResult(
                patient=(
                    patient
                ),
                success=False,
                message=str(
                    error
                ),
                elapsed_seconds=(
                    perf_counter()
                    - started
                ),
            )

    # =========================================================================
    # BACKGROUND ENTRY POINT
    # =========================================================================

    def _run_background(
        self,
        patient: Patient,
        canonical_pdf: Path,
    ) -> None:
        """
        Background pipeline entry point.
        """

        try:

            self._process_patient(
                patient,
                canonical_pdf,
            )

        except Exception as error:

            self._handle_failure(
                patient=(
                    patient
                ),
                stage=str(
                    get_progress()
                    .get(
                        "stage",
                        "unknown",
                    )
                ),
                error=(
                    error
                ),
            )

    # =========================================================================
    # PROCESS PATIENT
    # =========================================================================

    def _process_patient(
        self,
        patient: Patient,
        canonical_pdf: Path,
    ) -> Path:
        """
        Execute the complete pipeline.

        OCR + LLM + Normalizer overlap.

        Discharge starts only after all three streaming workers
        have successfully completed and the final Normalizer gate
        has been validated.
        """

        patient_name = (
            safe_patient_name(
                patient.patient_name
            )
        )

        # ---------------------------------------------------------------------
        # CANONICAL OCR INPUT
        # ---------------------------------------------------------------------

        active_pdf = (
            copy_to_ocr_input(
                canonical_pdf,
                patient_name,
            )
        )

        database.update_status(
            patient.id,
            "Streaming Processing",
        )

        update_progress(
            stage="ocr",
            message=(
                "OCR, LLM and Normalizer "
                "streaming workers are starting."
            ),
            running=True,
        )

        # ---------------------------------------------------------------------
        # WORKER RESULTS
        # ---------------------------------------------------------------------

        worker_results: dict[
            str,
            WorkerResult,
        ] = {}

        worker_lock = (
            threading.Lock()
        )

        def execute_worker(
            stage: str,
            function,
            *args,
        ) -> None:
            """
            Execute one stage and safely store its WorkerResult.
            """

            result = (
                function(
                    *args
                )
            )

            with worker_lock:

                worker_results[
                    stage
                ] = (
                    result
                )

        # ---------------------------------------------------------------------
        # THREADS
        # ---------------------------------------------------------------------

        ocr_thread = (
            threading.Thread(
                target=(
                    execute_worker
                ),
                args=(
                    "ocr",
                    run_ocr_worker,
                    patient_name,
                    active_pdf,
                ),
                daemon=True,
                name=(
                    f"ocr-worker-"
                    f"{patient.id}"
                ),
            )
        )

        llm_thread = (
            threading.Thread(
                target=(
                    execute_worker
                ),
                args=(
                    "llm",
                    run_llm_worker,
                    patient_name,
                ),
                daemon=True,
                name=(
                    f"llm-worker-"
                    f"{patient.id}"
                ),
            )
        )

        normalizer_thread = (
            threading.Thread(
                target=(
                    execute_worker
                ),
                args=(
                    "normalizer",
                    run_normalizer_worker,
                    patient_name,
                ),
                daemon=True,
                name=(
                    f"normalizer-worker-"
                    f"{patient.id}"
                ),
            )
        )

        # ---------------------------------------------------------------------
        # START WATCHERS FIRST
        # ---------------------------------------------------------------------
        #
        # LLM waits for OCR *.ready markers.
        #
        # Normalizer waits for LLM *.ready markers.
        #
        # Starting both watchers before OCR prevents us from losing
        # streaming time while the first OCR pages are being processed.
        #
        # ---------------------------------------------------------------------

        llm_thread.start()

        normalizer_thread.start()

        ocr_thread.start()

        threads = (
            ocr_thread,
            llm_thread,
            normalizer_thread,
        )

        # ---------------------------------------------------------------------
        # WATCH STREAMING WORKERS
        # ---------------------------------------------------------------------

        while any(
            thread.is_alive()
            for thread in (
                threads
            )
        ):

            self._raise_if_worker_failed(
                worker_results
            )

            self._update_streaming_progress()

            time.sleep(
                1
            )

        # ---------------------------------------------------------------------
        # JOIN ALL THREE
        # ---------------------------------------------------------------------

        for thread in (
            threads
        ):

            thread.join()

        self._raise_if_worker_failed(
            worker_results
        )

        # ---------------------------------------------------------------------
        # ENSURE ALL WORKERS RETURNED
        # ---------------------------------------------------------------------

        expected_stages = {
            "ocr",
            "llm",
            "normalizer",
        }

        missing_results = (
            expected_stages
            - set(
                worker_results
            )
        )

        if missing_results:

            raise RuntimeError(
                "Streaming stage result was "
                "not returned for: "
                + ", ".join(
                    sorted(
                        missing_results
                    )
                )
            )

        # ---------------------------------------------------------------------
        # ENSURE ALL WORKERS SUCCEEDED
        # ---------------------------------------------------------------------

        for stage in (
            "ocr",
            "llm",
            "normalizer",
        ):

            result = (
                worker_results[
                    stage
                ]
            )

            if not result.success:

                error = (
                    result.error
                    or RuntimeError(
                        f"{stage} failed."
                    )
                )

                raise RuntimeError(
                    f"{STAGE_LABELS.get(stage, stage)} "
                    f"failed: {error}"
                ) from error

        # ---------------------------------------------------------------------
        # STRICTLY REVERIFY STREAMING OUTPUTS
        # ---------------------------------------------------------------------

        ocr_path = (
            verify_ocr_completion(
                patient_name
            )
        )

        llm_path = (
            verify_llm_completion(
                patient_name
            )
        )

        normalized_path = (
            verify_normalizer_completion(
                patient_name
            )
        )

        # ---------------------------------------------------------------------
        # UPDATE PATIENT OBJECT
        # ---------------------------------------------------------------------

        patient.normalized_path = (
            normalized_path
        )

        # ---------------------------------------------------------------------
        # DATABASE TEMPORARY STAGE PATHS
        # ---------------------------------------------------------------------
        #
        # These point to module outputs during processing.
        #
        # After finalization below they are updated to data/<patient>/.
        #
        # ---------------------------------------------------------------------

        database.update_stage_path(
            patient.id,
            "ocr_path",
            str(
                ocr_path
            ),
        )

        database.update_stage_path(
            patient.id,
            "structured_path",
            str(
                llm_path
            ),
        )

        database.update_stage_path(
            patient.id,
            "normalized_path",
            str(
                normalized_path
            ),
        )

        # ---------------------------------------------------------------------
        # DISCHARGE MAY START ONLY NOW
        # ---------------------------------------------------------------------

        mark_stage_active(
            "discharge",
            message=(
                "Complete normalized record "
                "is ready. Generating the "
                "discharge summary."
            ),
        )

        database.update_status(
            patient.id,
            "Generating Discharge Summary",
        )

        discharge_path = (
            run_discharge(
                patient_name
            )
        )

        # ---------------------------------------------------------------------
        # STRICT DISCHARGE VERIFICATION
        # ---------------------------------------------------------------------

        verified_discharge_path = (
            verify_discharge_completion(
                patient_name
            )
        )

        if (
            discharge_path
            is not None
        ):

            discharge_path = (
                Path(
                    discharge_path
                )
            )

            if (
                discharge_path.resolve()
                != verified_discharge_path.resolve()
            ):

                raise RuntimeError(
                    "Discharge generator returned "
                    "an unexpected output path. "
                    f"Returned: {discharge_path}. "
                    f"Expected: "
                    f"{verified_discharge_path}."
                )

        discharge_path = (
            verified_discharge_path
        )

        mark_stage_completed(
            "discharge"
        )

        database.update_stage_path(
            patient.id,
            "discharge_path",
            str(
                discharge_path
            ),
        )

        # ---------------------------------------------------------------------
        # FINAL VERIFICATION BEFORE MOVING ANYTHING
        # ---------------------------------------------------------------------

        verify_all_module_outputs(
            patient_name
        )

        # ---------------------------------------------------------------------
        # FINALIZE
        # ---------------------------------------------------------------------
        #
        # This MOVES:
        #
        #   OCR_<patient>.txt
        #   LLM_<patient>.txt
        #   NORMALIZER_<patient>.txt
        #   DISCHARGE_<patient>.txt
        #
        # into:
        #
        #   data/<patient>/
        #
        # ---------------------------------------------------------------------

        final_paths = (
            finalize_patient_outputs(
                patient_name,
                canonical_pdf,
                active_pdf,
            )
        )

        # ---------------------------------------------------------------------
        # FINAL DATABASE PATHS
        # ---------------------------------------------------------------------

        database.update_stage_path(
            patient.id,
            "ocr_path",
            str(
                final_paths[
                    "ocr"
                ]
            ),
        )

        database.update_stage_path(
            patient.id,
            "structured_path",
            str(
                final_paths[
                    "llm"
                ]
            ),
        )

        database.update_stage_path(
            patient.id,
            "normalized_path",
            str(
                final_paths[
                    "normalizer"
                ]
            ),
        )

        database.update_stage_path(
            patient.id,
            "discharge_path",
            str(
                final_paths[
                    "discharge"
                ]
            ),
        )

        # ---------------------------------------------------------------------
        # PATIENT OBJECT FINAL PATHS
        # ---------------------------------------------------------------------

        patient.normalized_path = (
            final_paths[
                "normalizer"
            ]
        )

        patient.discharge_path = (
            final_paths[
                "discharge"
            ]
        )

        patient.status = (
            "Completed"
        )

        # ---------------------------------------------------------------------
        # DATABASE COMPLETED
        # ---------------------------------------------------------------------

        database.update_status(
            patient.id,
            "Completed",
        )

        # ---------------------------------------------------------------------
        # FINAL PROGRESS
        # ---------------------------------------------------------------------

        set_progress_time(
            completed_at=(
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            )
        )

        update_progress(
            stage="completed",
            message=(
                "The NICU discharge summary "
                "was generated successfully."
            ),
            running=False,
            error="",
            output_path=str(
                final_paths[
                    "discharge"
                ]
            ),
        )

        return (
            final_paths[
                "discharge"
            ]
        )

    # =========================================================================
    # WORKER FAILURE CHECK
    # =========================================================================

    def _raise_if_worker_failed(
        self,
        worker_results: dict[
            str,
            WorkerResult,
        ],
    ) -> None:
        """
        Raise as soon as a completed streaming worker reports failure.
        """

        for (
            stage,
            result,
        ) in list(
            worker_results.items()
        ):

            if result.success:

                continue

            error = (
                result.error
                or RuntimeError(
                    f"{stage} failed."
                )
            )

            raise RuntimeError(
                f"{STAGE_LABELS.get(stage, stage)} "
                f"failed: {error}"
            ) from error

    # =========================================================================
    # STREAMING FRONTEND PROGRESS
    # =========================================================================

    def _update_streaming_progress(
        self,
    ) -> None:
        """
        Update frontend progress while OCR, LLM and Normalizer
        overlap.
        """

        state = (
            get_progress()
        )

        completed = set(
            state.get(
                "completed_stages",
                [],
            )
        )

        active = set(
            state.get(
                "active_stages",
                [],
            )
        )

        if (
            "normalizer"
            in completed
        ):

            stage = (
                "normalizer"
            )

            percentage = (
                85
            )

            message = (
                "Normalization completed. "
                "Preparing the discharge stage."
            )

        elif (
            "normalizer"
            in active
        ):

            stage = (
                "normalizer"
            )

            percentage = (
                75
            )

            message = (
                "Normalizer is processing "
                "completed LLM batches while "
                "OCR and LLM continue."
            )

        elif (
            "llm"
            in active
        ):

            stage = (
                "llm"
            )

            percentage = (
                50
            )

            message = (
                "LLM is processing completed "
                "OCR batches while OCR continues."
            )

        else:

            stage = (
                "ocr"
            )

            percentage = (
                20
            )

            message = (
                "OCR is reading the "
                "patient record."
            )

        with _PROGRESS_LOCK:

            _PROGRESS.stage = (
                stage
            )

            _PROGRESS.stage_label = (
                STAGE_LABELS[
                    stage
                ]
            )

            _PROGRESS.percentage = max(
                _PROGRESS.percentage,
                percentage,
            )

            _PROGRESS.message = (
                message
            )

    # =========================================================================
    # FAILURE HANDLER
    # =========================================================================

    def _handle_failure(
        self,
        *,
        patient: Patient,
        stage: str,
        error: Exception,
    ) -> None:
        """
        Record one complete-pipeline failure.

        IMPORTANT:

        Temporary module files are NOT cleaned here.

        This allows us to inspect:

            ocr/temp/
            ocr/output/

            llm/temp/
            llm/output/

            normalizer/temp/
            normalizer/output/

            discharge_summary/temp/
            discharge_summary/output/

        after a failure.
        """

        try:

            database.update_status(
                patient.id,
                "Failed",
            )

        except Exception:

            pass

        patient.status = (
            "Failed"
        )

        try:

            error_file = (
                save_pipeline_error(
                    patient_name=(
                        patient.patient_name
                    ),
                    stage=(
                        stage
                    ),
                    error=(
                        error
                    ),
                )
            )

            error_path = str(
                error_file
            )

        except Exception:

            error_path = ""

        set_progress_time(
            completed_at=(
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            )
        )

        update_progress(
            stage="failed",
            message=(
                "Pipeline failed during "
                f"{STAGE_LABELS.get(stage, stage)}."
            ),
            running=False,
            error=str(
                error
            ),
            output_path=(
                error_path
            ),
        )


# =============================================================================
# PIPELINE INSTANCE
# =============================================================================

pipeline = (
    Pipeline()
)


# =============================================================================
# TERMINAL INFORMATION
# =============================================================================

if __name__ == "__main__":

    print(
        "NICU AI pipeline coordinator ready."
    )

    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Data folder  : "
        f"{DATA_DIRECTORY}"
    )

    print(
        f"OCR output   : "
        f"{OCR_OUTPUT_DIRECTORY}"
    )

    print(
        f"LLM output   : "
        f"{LLM_OUTPUT_DIRECTORY}"
    )

    print(
        f"Normalizer   : "
        f"{NORMALIZER_OUTPUT_DIRECTORY}"
    )

    print(
        f"Discharge    : "
        f"{DISCHARGE_OUTPUT_DIRECTORY}"
    )

    print(
        f"Progress     : "
        f"{get_progress()}"
    )