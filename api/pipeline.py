"""
Streaming NICU AI backend coordinator.

Pipeline:

Upload PDF
    ↓
OCR starts with configurable page workers
    ↓
LLM watcher processes completed OCR batches immediately
    ↓
Normalizer watcher processes completed LLM batches immediately
    ↓
Wait for complete normalized record
    ↓
Generate final discharge summary

Permanent storage:

data/<patient>/
    original/
    ocr/
    llm/
    normalizer/
    discharge/
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Final

from discharge_summary.generator import process_patient as run_discharge
from llm.extractor import process_patient_stream as run_llm_stream
from normalizer.normalizer import process_patient_stream as run_normalizer_stream
from ocr.ocr import process_pdf as run_ocr

from .database import database
from .models import Patient, PipelineResult


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT: Final[Path] = (
    Path(__file__).resolve().parent.parent
)

DATA_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "data"
)

UPLOAD_DIRECTORY: Final[Path] = (
    DATA_DIRECTORY / "uploads"
)

PIPELINE_ERROR_DIRECTORY: Final[Path] = (
    DATA_DIRECTORY / "pipeline_errors"
)

for directory in (
    DATA_DIRECTORY,
    UPLOAD_DIRECTORY,
    PIPELINE_ERROR_DIRECTORY,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# PROGRESS
# =============================================================================

STAGE_LABELS: Final[dict[str, str]] = {
    "idle": "Waiting",
    "upload": "Uploading PDF",
    "ocr": "OCR Processing",
    "llm": "LLM Clinical Processing",
    "normalizer": "Normalizing Patient Record",
    "discharge": "Generating Discharge Summary",
    "completed": "Completed",
    "failed": "Failed",
}


STAGE_PERCENTAGES: Final[dict[str, int]] = {
    "idle": 0,
    "upload": 5,
    "ocr": 20,
    "llm": 50,
    "normalizer": 75,
    "discharge": 90,
    "completed": 100,
    "failed": 0,
}


STAGE_ORDER: Final[tuple[str, ...]] = (
    "upload",
    "ocr",
    "llm",
    "normalizer",
    "discharge",
)


@dataclass(slots=True)
class ProgressState:
    running: bool = False

    patient_id: int | None = None

    patient_name: str = ""

    stage: str = "idle"

    stage_label: str = STAGE_LABELS["idle"]

    percentage: int = 0

    message: str = "Waiting for a patient PDF."

    completed_stages: list[str] = field(
        default_factory=list
    )

    active_stages: list[str] = field(
        default_factory=list
    )

    error: str = ""

    output_path: str = ""

    started_at: str = ""

    completed_at: str = ""


_PROGRESS = ProgressState()

_PROGRESS_LOCK = threading.Lock()


def reset_progress() -> None:
    global _PROGRESS

    with _PROGRESS_LOCK:
        _PROGRESS = ProgressState()


def get_progress() -> dict[str, object]:
    with _PROGRESS_LOCK:
        return {
            "running": _PROGRESS.running,
            "patient_id": _PROGRESS.patient_id,
            "patient_name": _PROGRESS.patient_name,
            "stage": _PROGRESS.stage,
            "stage_label": _PROGRESS.stage_label,
            "percentage": _PROGRESS.percentage,
            "message": _PROGRESS.message,
            "completed_stages": list(
                _PROGRESS.completed_stages
            ),
            "active_stages": list(
                _PROGRESS.active_stages
            ),
            "error": _PROGRESS.error,
            "output_path": _PROGRESS.output_path,
            "started_at": _PROGRESS.started_at,
            "completed_at": _PROGRESS.completed_at,
        }


def pipeline_is_running() -> bool:
    with _PROGRESS_LOCK:
        return _PROGRESS.running


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
    with _PROGRESS_LOCK:
        if stage is not None:
            _PROGRESS.stage = stage
            _PROGRESS.stage_label = STAGE_LABELS.get(
                stage,
                stage.replace("_", " ").title(),
            )
            _PROGRESS.percentage = STAGE_PERCENTAGES.get(
                stage,
                _PROGRESS.percentage,
            )

        if message is not None:
            _PROGRESS.message = message

        if patient_id is not None:
            _PROGRESS.patient_id = patient_id

        if patient_name is not None:
            _PROGRESS.patient_name = patient_name

        if running is not None:
            _PROGRESS.running = running

        if error is not None:
            _PROGRESS.error = error

        if output_path is not None:
            _PROGRESS.output_path = output_path


def mark_stage_active(
    stage: str,
    *,
    message: str,
) -> None:
    with _PROGRESS_LOCK:
        if stage not in _PROGRESS.active_stages:
            _PROGRESS.active_stages.append(
                stage
            )

        _PROGRESS.stage = stage
        _PROGRESS.stage_label = STAGE_LABELS.get(
            stage,
            stage,
        )

        _PROGRESS.percentage = max(
            _PROGRESS.percentage,
            STAGE_PERCENTAGES.get(
                stage,
                0,
            ),
        )

        _PROGRESS.message = message


def mark_stage_completed(
    stage: str,
) -> None:
    with _PROGRESS_LOCK:
        if stage in _PROGRESS.active_stages:
            _PROGRESS.active_stages.remove(
                stage
            )

        if stage not in _PROGRESS.completed_stages:
            _PROGRESS.completed_stages.append(
                stage
            )

        next_percentage = STAGE_PERCENTAGES.get(
            stage,
            0,
        )

        _PROGRESS.percentage = max(
            _PROGRESS.percentage,
            next_percentage,
        )


def set_progress_time(
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    with _PROGRESS_LOCK:
        if started_at is not None:
            _PROGRESS.started_at = started_at

        if completed_at is not None:
            _PROGRESS.completed_at = completed_at

# =============================================================================
# PATIENT STORAGE HELPERS
# =============================================================================

def safe_patient_name(
    value: str,
) -> str:
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


def safe_filename(
    filename: str,
) -> str:
    name = Path(
        filename
    ).name

    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name,
    )

    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    ).strip("_")

    return cleaned or "patient.pdf"


def get_patient_root(
    patient_name: str,
) -> Path:
    patient_root = (
        DATA_DIRECTORY
        / safe_patient_name(
            patient_name
        )
    )

    patient_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return patient_root


def get_patient_paths(
    patient_name: str,
) -> dict[str, Path]:
    patient_root = get_patient_root(
        patient_name
    )

    paths = {
        "root": patient_root,

        "original": (
            patient_root
            / "original"
        ),

        "ocr": (
            patient_root
            / "ocr"
        ),

        "llm": (
            patient_root
            / "llm"
        ),

        "normalizer": (
            patient_root
            / "normalizer"
        ),

        "discharge": (
            patient_root
            / "discharge"
        ),

        "pipeline_errors": (
            patient_root
            / "pipeline_errors"
        ),
    }

    for path in paths.values():
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    return paths


def store_uploaded_pdf(
    pdf_file: Path,
) -> tuple[str, Path]:
    """
    Store the uploaded file in the patient-first data folder.

    The timestamp is removed from the patient name so the folder remains
    readable.
    """

    original_filename = safe_filename(
        pdf_file.name
    )

    patient_name = safe_patient_name(
        pdf_file.stem
    )

    paths = get_patient_paths(
        patient_name
    )

    destination = (
        paths[
            "original"
        ]
        / original_filename
    )

    if pdf_file.resolve() != destination.resolve():
        shutil.copy2(
            pdf_file,
            destination,
        )

    return (
        patient_name,
        destination,
    )


def copy_to_ocr_input(
    original_pdf: Path,
) -> Path:
    """
    Copy the patient PDF into the active OCR input folder.
    """

    ocr_input = (
        PROJECT_ROOT
        / "ocr"
        / "input"
    )

    ocr_input.mkdir(
        parents=True,
        exist_ok=True,
    )

    for existing_file in (
        ocr_input.glob("*.pdf")
    ):
        existing_file.unlink()

    active_pdf = (
        ocr_input
        / original_pdf.name
    )

    shutil.copy2(
        original_pdf,
        active_pdf,
    )

    return active_pdf


def save_pipeline_error(
    *,
    patient_name: str,
    stage: str,
    error: Exception,
) -> Path:
    paths = get_patient_paths(
        patient_name
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_stage = safe_patient_name(
        stage
    )

    error_file = (
        paths[
            "pipeline_errors"
        ]
        / (
            f"{safe_patient_name(patient_name)}"
            f"_{safe_stage}"
            f"_{timestamp}_error.txt"
        )
    )

    error_text = (
        f"Patient: {patient_name}\n"
        f"Stage: {stage}\n"
        f"Time: {datetime.now().isoformat()}\n"
        f"Error type: {type(error).__name__}\n"
        f"Error: {error}\n"
    )

    error_file.write_text(
        error_text,
        encoding="utf-8",
    )

    return error_file


# =============================================================================
# STAGE RESULT HELPERS
# =============================================================================

def get_ocr_complete_path(
    patient_name: str,
) -> Path:
    paths = get_patient_paths(
        patient_name
    )

    complete_file = (
        paths[
            "ocr"
        ]
        / (
            f"{safe_patient_name(patient_name)}"
            "_ocr_complete.txt"
        )
    )

    if not complete_file.exists():
        raise FileNotFoundError(
            f"OCR complete file was not created: {complete_file}"
        )

    return complete_file


def get_llm_complete_path(
    patient_name: str,
) -> Path:
    paths = get_patient_paths(
        patient_name
    )

    complete_file = (
        paths[
            "llm"
        ]
        / (
            f"{safe_patient_name(patient_name)}"
            "_llm_complete.txt"
        )
    )

    if not complete_file.exists():
        raise FileNotFoundError(
            f"LLM complete file was not created: {complete_file}"
        )

    return complete_file


def get_normalized_complete_path(
    patient_name: str,
) -> Path:
    paths = get_patient_paths(
        patient_name
    )

    marker_path = (
        paths[
            "normalizer"
        ]
        / "normalizer_complete.json"
    )

    if not marker_path.exists():
        raise FileNotFoundError(
            f"Normalizer completion marker was not created: {marker_path}"
        )

    import json

    payload = json.loads(
        marker_path.read_text(
            encoding="utf-8",
        )
    )

    complete_filename = str(
        payload.get(
            "complete_file",
            "",
        )
    ).strip()

    if not complete_filename:
        raise RuntimeError(
            "Normalizer completion marker does not contain complete_file."
        )

    complete_file = (
        paths[
            "normalizer"
        ]
        / complete_filename
    )

    if not complete_file.exists():
        raise FileNotFoundError(
            f"Normalized complete file was not created: {complete_file}"
        )

    return complete_file


def get_discharge_complete_path(
    patient_name: str,
) -> Path:
    paths = get_patient_paths(
        patient_name
    )

    marker_path = (
        paths[
            "discharge"
        ]
        / "discharge_complete.json"
    )

    if not marker_path.exists():
        raise FileNotFoundError(
            f"Discharge completion marker was not created: {marker_path}"
        )

    import json

    payload = json.loads(
        marker_path.read_text(
            encoding="utf-8",
        )
    )

    summary_filename = str(
        payload.get(
            "summary_file",
            "",
        )
    ).strip()

    if not summary_filename:
        raise RuntimeError(
            "Discharge completion marker does not contain summary_file."
        )

    summary_path = (
        paths[
            "discharge"
        ]
        / summary_filename
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Discharge summary was not created: {summary_path}"
        )

    return summary_path


# =============================================================================
# WORKER RESULT
# =============================================================================

@dataclass(slots=True)
class WorkerResult:
    stage: str

    success: bool

    output_path: Path | None = None

    error: Exception | None = None


# =============================================================================
# STAGE WORKERS
# =============================================================================

def run_ocr_worker(
    patient_name: str,
    active_pdf: Path,
) -> WorkerResult:
    try:
        mark_stage_active(
            "ocr",
            message="OCR is processing patient pages.",
        )

        output_path = run_ocr(
            active_pdf
        )

        if output_path is None:
            output_path = get_ocr_complete_path(
                patient_name
            )

        mark_stage_completed(
            "ocr"
        )

        database.update_status(
            _PROGRESS.patient_id,
            "OCR Completed",
        )

        if _PROGRESS.patient_id is not None:
            database.update_stage_path(
                _PROGRESS.patient_id,
                "ocr_path",
                str(output_path),
            )

        return WorkerResult(
            stage="ocr",
            success=True,
            output_path=output_path,
        )

    except Exception as error:
        return WorkerResult(
            stage="ocr",
            success=False,
            error=error,
        )


def run_llm_worker(
    patient_name: str,
) -> WorkerResult:
    try:
        mark_stage_active(
            "llm",
            message=(
                "LLM is structuring completed OCR batches "
                "while OCR continues."
            ),
        )

        output_path = run_llm_stream(
            patient_name
        )

        mark_stage_completed(
            "llm"
        )

        if _PROGRESS.patient_id is not None:
            database.update_status(
                _PROGRESS.patient_id,
                "LLM Completed",
            )

            database.update_stage_path(
                _PROGRESS.patient_id,
                "structured_path",
                str(output_path),
            )

        return WorkerResult(
            stage="llm",
            success=True,
            output_path=output_path,
        )

    except Exception as error:
        return WorkerResult(
            stage="llm",
            success=False,
            error=error,
        )


def run_normalizer_worker(
    patient_name: str,
) -> WorkerResult:
    try:
        mark_stage_active(
            "normalizer",
            message=(
                "Normalizer is processing completed LLM batches "
                "while earlier stages continue."
            ),
        )

        output_path = run_normalizer_stream(
            patient_name
        )

        mark_stage_completed(
            "normalizer"
        )

        if _PROGRESS.patient_id is not None:
            database.update_status(
                _PROGRESS.patient_id,
                "Normalizer Completed",
            )

            database.update_stage_path(
                _PROGRESS.patient_id,
                "normalized_path",
                str(output_path),
            )

        return WorkerResult(
            stage="normalizer",
            success=True,
            output_path=output_path,
        )

    except Exception as error:
        return WorkerResult(
            stage="normalizer",
            success=False,
            error=error,
        )
# =============================================================================
# PIPELINE CLASS
# =============================================================================

class Pipeline:
    """
    Coordinate the complete streaming NICU AI pipeline.

    OCR, LLM and Normalizer are launched together.

    The discharge generator starts only after the normalizer has produced
    the complete normalized patient record.
    """

    def start(
        self,
        pdf_file: Path,
    ) -> Patient:
        """
        Start one patient pipeline in a background thread.
        """

        if pipeline_is_running():
            raise RuntimeError(
                "Another patient is currently being processed. "
                "Please wait until it finishes."
            )

        pdf_file = pdf_file.expanduser().resolve()

        if not pdf_file.exists():
            raise FileNotFoundError(
                f"Uploaded PDF was not found: {pdf_file}"
            )

        if pdf_file.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF patient records are supported."
            )

        reset_progress()

        patient_name, original_pdf = (
            store_uploaded_pdf(
                pdf_file
            )
        )

        patient_id = database.add_patient(
            patient_name=patient_name,
            pdf_path=str(original_pdf),
        )

        patient = Patient(
            id=patient_id,
            patient_name=patient_name,
            status="Uploaded",
            pdf_path=original_pdf,
        )

        set_progress_time(
            started_at=datetime.now().isoformat(
                timespec="seconds"
            )
        )

        update_progress(
            stage="upload",
            message="Patient PDF uploaded successfully.",
            patient_id=patient_id,
            patient_name=patient_name,
            running=True,
            error="",
            output_path="",
        )

        mark_stage_completed(
            "upload"
        )

        worker = threading.Thread(
            target=self._run_background,
            args=(
                patient,
                original_pdf,
            ),
            daemon=True,
            name=f"nicu-streaming-pipeline-{patient_id}",
        )

        worker.start()

        return patient

    def run_sync(
        self,
        pdf_file: Path,
    ) -> PipelineResult:
        """
        Run one complete patient pipeline synchronously.

        Useful for terminal testing.
        """

        start_time = perf_counter()

        if pipeline_is_running():
            raise RuntimeError(
                "Another patient is currently being processed."
            )

        pdf_file = pdf_file.expanduser().resolve()

        if not pdf_file.exists():
            raise FileNotFoundError(
                f"PDF was not found: {pdf_file}"
            )

        if pdf_file.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF patient records are supported."
            )

        reset_progress()

        patient_name, original_pdf = (
            store_uploaded_pdf(
                pdf_file
            )
        )

        patient_id = database.add_patient(
            patient_name=patient_name,
            pdf_path=str(original_pdf),
        )

        patient = Patient(
            id=patient_id,
            patient_name=patient_name,
            status="Uploaded",
            pdf_path=original_pdf,
        )

        set_progress_time(
            started_at=datetime.now().isoformat(
                timespec="seconds"
            )
        )

        update_progress(
            stage="upload",
            message="Patient PDF uploaded successfully.",
            patient_id=patient_id,
            patient_name=patient_name,
            running=True,
            error="",
            output_path="",
        )

        mark_stage_completed(
            "upload"
        )

        try:
            output_path = self._process_patient(
                patient,
                original_pdf,
            )

            return PipelineResult(
                patient=patient,
                success=True,
                message="Pipeline completed successfully.",
                elapsed_seconds=(
                    perf_counter()
                    - start_time
                ),
            )

        except Exception as error:
            self._handle_failure(
                patient=patient,
                stage=str(
                    get_progress().get(
                        "stage",
                        "unknown",
                    )
                ),
                error=error,
            )

            return PipelineResult(
                patient=patient,
                success=False,
                message=str(error),
                elapsed_seconds=(
                    perf_counter()
                    - start_time
                ),
            )

    def _run_background(
        self,
        patient: Patient,
        original_pdf: Path,
    ) -> None:
        """
        Background pipeline entry point.
        """

        try:
            self._process_patient(
                patient,
                original_pdf,
            )

        except Exception as error:
            self._handle_failure(
                patient=patient,
                stage=str(
                    get_progress().get(
                        "stage",
                        "unknown",
                    )
                ),
                error=error,
            )

    def _process_patient(
        self,
        patient: Patient,
        original_pdf: Path,
    ) -> Path:
        """
        Run OCR, LLM and Normalizer as overlapping streaming stages.

        The discharge generator runs only after all normalizer batches
        are merged successfully.
        """

        active_pdf = copy_to_ocr_input(
            original_pdf
        )

        patient_name = patient.patient_name

        database.update_status(
            patient.id,
            "Streaming Processing",
        )

        update_progress(
            stage="ocr",
            message=(
                "OCR, LLM and Normalizer streaming "
                "workers are starting."
            ),
            running=True,
        )

        worker_results: dict[
            str,
            WorkerResult,
        ] = {}

        worker_lock = threading.Lock()

        def execute_worker(
            stage: str,
            function,
            *args,
        ) -> None:
            result = function(
                *args
            )

            with worker_lock:
                worker_results[
                    stage
                ] = result

        ocr_thread = threading.Thread(
            target=execute_worker,
            args=(
                "ocr",
                run_ocr_worker,
                patient_name,
                active_pdf,
            ),
            daemon=True,
            name=f"ocr-worker-{patient.id}",
        )

        llm_thread = threading.Thread(
            target=execute_worker,
            args=(
                "llm",
                run_llm_worker,
                patient_name,
            ),
            daemon=True,
            name=f"llm-worker-{patient.id}",
        )

        normalizer_thread = threading.Thread(
            target=execute_worker,
            args=(
                "normalizer",
                run_normalizer_worker,
                patient_name,
            ),
            daemon=True,
            name=f"normalizer-worker-{patient.id}",
        )

        # Start watchers first so they are already waiting when OCR batches
        # begin to appear.
        llm_thread.start()
        normalizer_thread.start()
        ocr_thread.start()

        threads = (
            ocr_thread,
            llm_thread,
            normalizer_thread,
        )

        while any(
            thread.is_alive()
            for thread in threads
        ):
            self._raise_if_worker_failed(
                worker_results
            )

            self._update_streaming_progress()

            time.sleep(
                1
            )

        for thread in threads:
            thread.join()

        self._raise_if_worker_failed(
            worker_results
        )

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
                "Streaming stage result was not returned for: "
                + ", ".join(
                    sorted(
                        missing_results
                    )
                )
            )

        normalized_path = (
            get_normalized_complete_path(
                patient_name
            )
        )

        patient.normalized_path = (
            normalized_path
        )

        database.update_stage_path(
            patient.id,
            "normalized_path",
            str(normalized_path),
        )

        mark_stage_active(
            "discharge",
            message=(
                "Complete normalized record is ready. "
                "Generating the discharge summary."
            ),
        )

        database.update_status(
            patient.id,
            "Generating Discharge Summary",
        )

        discharge_path = run_discharge(
            patient_name
        )

        verified_discharge_path = (
            get_discharge_complete_path(
                patient_name
            )
        )

        if (
            discharge_path.resolve()
            != verified_discharge_path.resolve()
        ):
            discharge_path = (
                verified_discharge_path
            )

        mark_stage_completed(
            "discharge"
        )

        patient.discharge_path = (
            discharge_path
        )

        patient.status = "Completed"

        database.update_stage_path(
            patient.id,
            "discharge_path",
            str(discharge_path),
        )

        database.update_status(
            patient.id,
            "Completed",
        )

        set_progress_time(
            completed_at=datetime.now().isoformat(
                timespec="seconds"
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
                discharge_path
            ),
        )

        return discharge_path

    def _raise_if_worker_failed(
        self,
        worker_results: dict[
            str,
            WorkerResult,
        ],
    ) -> None:
        """
        Raise immediately after any completed streaming worker fails.
        """

        for stage, result in (
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
                f"{STAGE_LABELS.get(stage, stage)} failed: "
                f"{error}"
            ) from error

    def _update_streaming_progress(
        self,
    ) -> None:
        """
        Update the frontend while several stages are active together.
        """

        state = get_progress()

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

        if "normalizer" in completed:
            stage = "normalizer"
            percentage = 85
            message = (
                "Normalization completed. "
                "Preparing the final discharge stage."
            )

        elif "normalizer" in active:
            stage = "normalizer"
            percentage = 75
            message = (
                "Normalizer is processing LLM batches "
                "while OCR and LLM continue."
            )

        elif "llm" in active:
            stage = "llm"
            percentage = 50
            message = (
                "LLM is processing completed OCR batches "
                "while OCR continues."
            )

        else:
            stage = "ocr"
            percentage = 20
            message = (
                "OCR is reading the patient record."
            )

        with _PROGRESS_LOCK:
            _PROGRESS.stage = stage

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

    def _handle_failure(
        self,
        *,
        patient: Patient,
        stage: str,
        error: Exception,
    ) -> None:
        """
        Save and expose one complete-pipeline failure.
        """

        database.update_status(
            patient.id,
            "Failed",
        )

        patient.status = "Failed"

        error_file = save_pipeline_error(
            patient_name=(
                patient.patient_name
            ),
            stage=stage,
            error=error,
        )

        set_progress_time(
            completed_at=datetime.now().isoformat(
                timespec="seconds"
            )
        )

        update_progress(
            stage="failed",
            message=(
                f"Pipeline failed during "
                f"{STAGE_LABELS.get(stage, stage)}."
            ),
            running=False,
            error=str(error),
            output_path=str(
                error_file
            ),
        )


# =============================================================================
# PIPELINE INSTANCE
# =============================================================================

pipeline = Pipeline()


# =============================================================================
# TERMINAL TEST
# =============================================================================

if __name__ == "__main__":
    print("NICU AI streaming pipeline ready.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data folder : {DATA_DIRECTORY}")
    print(f"Progress    : {get_progress()}")