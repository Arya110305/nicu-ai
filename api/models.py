"""
Data models used throughout the NICU AI backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class Patient:

    id: Optional[int] = None

    patient_name: str = ""

    uhid: str = ""

    status: str = "Uploaded"

    pdf_path: Optional[Path] = None

    ocr_path: Optional[Path] = None

    structured_path: Optional[Path] = None

    normalized_path: Optional[Path] = None

    discharge_path: Optional[Path] = None


@dataclass(slots=True)
class PipelineResult:

    patient: Patient

    success: bool

    message: str = ""

    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class StageResult:

    stage: str

    success: bool

    output_path: Optional[Path] = None

    message: str = ""