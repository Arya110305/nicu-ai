"""
FastAPI routes for the NICU AI backend.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from .database import database
from .pipeline import (
    UPLOAD_DIRECTORY,
    get_progress,
    pipeline,
    pipeline_is_running,
    safe_filename,
)


router = APIRouter()


@router.get("/")
def home() -> dict[str, str]:
    return {
        "message": "NICU AI backend is running."
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok"
    }


@router.get("/progress")
def progress() -> dict[str, object]:
    return get_progress()


@router.post("/upload")
async def upload_patient_pdf(
    file: UploadFile = File(...),
) -> dict[str, object]:
    """
    Upload one patient PDF and start the pipeline.
    """

    if pipeline_is_running():
        raise HTTPException(
            status_code=409,
            detail=(
                "Another patient is currently being processed. "
                "Please wait until it finishes."
            ),
        )

    filename = file.filename or "patient.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    safe_name = safe_filename(
        filename
    )

    temporary_file = (
        UPLOAD_DIRECTORY
        / f"pending_{safe_name}"
    )

    try:
        with temporary_file.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        patient = pipeline.start(
            temporary_file
        )

    except Exception as error:
        temporary_file.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    finally:
        await file.close()

    temporary_file.unlink(
        missing_ok=True
    )

    return {
        "success": True,
        "patient_id": patient.id,
        "patient_name": patient.patient_name,
        "status": patient.status,
        "message": (
            "Patient PDF uploaded. "
            "Processing has started."
        ),
    }


@router.get("/patients")
def list_patients() -> list[dict[str, object]]:
    rows = database.list_patients()

    return [
        dict(row)
        for row in rows
    ]


@router.get("/patients/{patient_id}")
def get_patient(
    patient_id: int,
) -> dict[str, object]:
    patient = database.get_patient(
        patient_id
    )

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient record not found.",
        )

    return dict(patient)


@router.get("/summary")
def get_current_summary() -> FileResponse:
    progress_state = get_progress()

    output_path = str(
        progress_state.get(
            "output_path",
            "",
        )
    ).strip()

    stage = str(
        progress_state.get(
            "stage",
            "",
        )
    )

    if stage != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                "The discharge summary "
                "is not ready yet."
            ),
        )

    if not output_path:
        raise HTTPException(
            status_code=404,
            detail=(
                "No discharge-summary "
                "output path was recorded."
            ),
        )

    summary_path = Path(
        output_path
    )

    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "The discharge-summary "
                "file could not be found."
            ),
        )

    return FileResponse(
        path=summary_path,
        media_type="text/plain",
        filename=summary_path.name,
    )


@router.get("/summary/text")
def get_current_summary_text() -> dict[str, str]:
    progress_state = get_progress()

    output_path = str(
        progress_state.get(
            "output_path",
            "",
        )
    ).strip()

    stage = str(
        progress_state.get(
            "stage",
            "",
        )
    )

    if stage != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                "The discharge summary "
                "is not ready yet."
            ),
        )

    summary_path = Path(
        output_path
    )

    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "The discharge-summary "
                "file could not be found."
            ),
        )

    summary_text = summary_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return {
        "filename": summary_path.name,
        "summary": summary_text,
    }