"""
Utility functions for the NICU discharge-summary pipeline.

This module handles:

1. Input-file discovery.
2. Reading normalized patient text.
3. Text chunking.
4. Safe patient and file names.
5. Patient-specific output directories.
6. Evidence-report TXT output.
7. Discharge-summary TXT output.
8. Discharge-summary DOCX creation.
9. DOCX-to-PDF conversion.
10. Error and intermediate-file saving.

The clinical LLM logic does not belong in this file.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Final, Iterable

try:
    from docx import Document
    from docx.document import Document as DocumentObject
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt
except ImportError as exc:
    raise ImportError(
        "python-docx is required. Install it using:\n"
        "python3 -m pip install python-docx"
    ) from exc


# =============================================================================
# PROJECT PATHS
# =============================================================================

DISCHARGE_SUMMARY_DIRECTORY: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = DISCHARGE_SUMMARY_DIRECTORY.parent

NORMALIZER_OUTPUT_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "normalizer" / "output"
)

OUTPUT_DIRECTORY: Final[Path] = (
    DISCHARGE_SUMMARY_DIRECTORY / "output"
)

ERROR_DIRECTORY: Final[Path] = (
    DISCHARGE_SUMMARY_DIRECTORY / "errors"
)

INTERMEDIATE_DIRECTORY: Final[Path] = (
    DISCHARGE_SUMMARY_DIRECTORY / "intermediate"
)


# =============================================================================
# FILE-NAME SUFFIXES
# =============================================================================

NORMALIZED_FILE_SUFFIX: Final[str] = "_normalized_complete.txt"

EVIDENCE_REPORT_SUFFIX: Final[str] = "_evidence_report.txt"

DISCHARGE_TEXT_SUFFIX: Final[str] = "_discharge_summary.txt"

DISCHARGE_DOCX_SUFFIX: Final[str] = "_discharge_summary.docx"

DISCHARGE_PDF_SUFFIX: Final[str] = "_discharge_summary.pdf"

CONSOLIDATED_EVIDENCE_SUFFIX: Final[str] = (
    "_consolidated_evidence.txt"
)

RAW_EXTRACTION_SUFFIX: Final[str] = "_raw_extracted_evidence.txt"

REVIEW_SUFFIX: Final[str] = "_doctor_review.txt"


# =============================================================================
# GENERAL DIRECTORY FUNCTIONS
# =============================================================================

def ensure_directories() -> None:
    """
    Create all directories required by the discharge-summary pipeline.
    """

    directories = (
        NORMALIZER_OUTPUT_DIRECTORY,
        OUTPUT_DIRECTORY,
        ERROR_DIRECTORY,
        INTERMEDIATE_DIRECTORY,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def create_directory(directory: Path) -> Path:
    """
    Create a directory if it does not already exist.

    Args:
        directory:
            Directory path.

    Returns:
        The same directory path.
    """

    directory.mkdir(parents=True, exist_ok=True)
    return directory


# =============================================================================
# INPUT-FILE FUNCTIONS
# =============================================================================

def find_normalized_files(
    input_directory: Path | None = None,
) -> list[Path]:
    """
    Find normalized patient files ready for discharge processing.

    Expected file format:

        <patient_name>_normalized_complete.txt

    Args:
        input_directory:
            Optional custom input directory. Defaults to
            normalizer/output.

    Returns:
        Sorted list of normalized text files.
    """

    directory = input_directory or NORMALIZER_OUTPUT_DIRECTORY

    if not directory.exists():
        logging.warning(
            "Normalized input directory does not exist: %s",
            directory,
        )
        return []

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Normalized input path is not a directory: {directory}"
        )

    normalized_files = sorted(
        path
        for path in directory.glob(
            f"*{NORMALIZED_FILE_SUFFIX}"
        )
        if path.is_file()
    )

    return normalized_files


def read_text_file(
    file_path: Path,
    *,
    encoding: str = "utf-8",
) -> str:
    """
    Read a UTF-8 text file.

    Args:
        file_path:
            Path to the text file.

        encoding:
            Text encoding. UTF-8 is used by default.

    Returns:
        File contents without leading or trailing whitespace.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Input path is not a file: {file_path}"
        )

    try:
        content = file_path.read_text(
            encoding=encoding,
            errors="replace",
        )
    except OSError as exc:
        raise OSError(
            f"Could not read text file: {file_path}"
        ) from exc

    content = content.strip()

    if not content:
        raise ValueError(
            f"Input file is empty: {file_path}"
        )

    return content


# =============================================================================
# PATIENT-NAME AND FILE-NAME FUNCTIONS
# =============================================================================

def extract_patient_name_from_filename(
    file_path: Path,
) -> str:
    """
    Extract a patient name from a normalized filename.

    Example:

        BO VAGDEVI_normalized_complete.txt

    becomes:

        BO VAGDEVI
    """

    filename = file_path.name

    if filename.endswith(NORMALIZED_FILE_SUFFIX):
        patient_name = filename[
            :-len(NORMALIZED_FILE_SUFFIX)
        ]
    else:
        patient_name = file_path.stem

    patient_name = patient_name.replace("_", " ")
    patient_name = " ".join(patient_name.split()).strip()

    if not patient_name:
        return "UNKNOWN PATIENT"

    return patient_name


def safe_filename(
    value: str,
    *,
    default: str = "unknown_patient",
    maximum_length: int = 100,
) -> str:
    """
    Convert a patient name or label into a safe filename.

    Example:

        Baby of Vagdevi

    becomes:

        baby_of_vagdevi
    """

    cleaned = value.strip().lower()

    cleaned = re.sub(
        r"[^a-z0-9\s_-]",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"[\s-]+",
        "_",
        cleaned,
    )

    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    )

    cleaned = cleaned.strip("_")

    if not cleaned:
        cleaned = default

    return cleaned[:maximum_length].rstrip("_")


def get_patient_output_directory(
    patient_name: str,
) -> Path:
    """
    Create and return a patient-specific output directory.

    Example:

        discharge_summary/output/baby_of_vagdevi/
    """

    patient_folder_name = safe_filename(patient_name)

    patient_directory = (
        OUTPUT_DIRECTORY / patient_folder_name
    )

    patient_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return patient_directory


def get_patient_intermediate_directory(
    patient_name: str,
) -> Path:
    """
    Create and return a patient-specific intermediate directory.
    """

    patient_folder_name = safe_filename(patient_name)

    patient_directory = (
        INTERMEDIATE_DIRECTORY / patient_folder_name
    )

    patient_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return patient_directory


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_model_output(text: str) -> str:
    """
    Clean basic unwanted wrapping from model output.

    This function does not rewrite clinical content.
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:text|markdown|plaintext)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def normalize_line_endings(text: str) -> str:
    """
    Convert mixed line endings into standard newline characters.
    """

    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def remove_excess_blank_lines(text: str) -> str:
    """
    Reduce three or more blank lines to two blank lines.
    """

    normalized = normalize_line_endings(text)

    return re.sub(
        r"\n{3,}",
        "\n\n",
        normalized,
    ).strip()


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def split_text_into_chunks(
    text: str,
    *,
    maximum_characters: int = 14_000,
    overlap_characters: int = 800,
) -> list[str]:
    """
    Split normalized clinical text into manageable chunks.

    The function prefers splitting at paragraph boundaries. A small overlap
    is retained to avoid losing clinical context between chunks.

    Args:
        text:
            Complete normalized patient text.

        maximum_characters:
            Approximate maximum characters in each chunk.

        overlap_characters:
            Number of characters copied from the end of the previous chunk
            into the next chunk.

    Returns:
        List of non-empty text chunks.
    """

    cleaned_text = remove_excess_blank_lines(text)

    if not cleaned_text:
        raise ValueError(
            "Cannot split empty clinical text."
        )

    if maximum_characters < 1_000:
        raise ValueError(
            "maximum_characters must be at least 1000."
        )

    if overlap_characters < 0:
        raise ValueError(
            "overlap_characters cannot be negative."
        )

    if overlap_characters >= maximum_characters:
        raise ValueError(
            "overlap_characters must be smaller than "
            "maximum_characters."
        )

    if len(cleaned_text) <= maximum_characters:
        return [cleaned_text]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            cleaned_text,
        )
        if paragraph.strip()
    ]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        if paragraph_length > maximum_characters:
            if current_parts:
                chunks.append(
                    "\n\n".join(current_parts).strip()
                )
                current_parts = []
                current_length = 0

            large_parts = _split_large_paragraph(
                paragraph,
                maximum_characters=maximum_characters,
                overlap_characters=overlap_characters,
            )

            chunks.extend(large_parts)
            continue

        extra_length = (
            paragraph_length
            if not current_parts
            else paragraph_length + 2
        )

        if (
            current_parts
            and current_length + extra_length
            > maximum_characters
        ):
            completed_chunk = "\n\n".join(
                current_parts
            ).strip()

            chunks.append(completed_chunk)

            overlap_text = _get_chunk_overlap(
                completed_chunk,
                overlap_characters,
            )

            current_parts = (
                [overlap_text, paragraph]
                if overlap_text
                else [paragraph]
            )

            current_length = len(
                "\n\n".join(current_parts)
            )

        else:
            current_parts.append(paragraph)
            current_length += extra_length

    if current_parts:
        final_chunk = "\n\n".join(
            current_parts
        ).strip()

        if final_chunk:
            chunks.append(final_chunk)

    return [
        chunk
        for chunk in chunks
        if chunk.strip()
    ]


def _split_large_paragraph(
    paragraph: str,
    *,
    maximum_characters: int,
    overlap_characters: int,
) -> list[str]:
    """
    Split a paragraph that is larger than the normal chunk limit.
    """

    parts: list[str] = []
    start = 0
    paragraph_length = len(paragraph)

    while start < paragraph_length:
        end = min(
            start + maximum_characters,
            paragraph_length,
        )

        if end < paragraph_length:
            preferred_break = paragraph.rfind(
                "\n",
                start,
                end,
            )

            if preferred_break <= start:
                preferred_break = paragraph.rfind(
                    ". ",
                    start,
                    end,
                )

                if preferred_break > start:
                    preferred_break += 1

            if preferred_break > start:
                end = preferred_break

        part = paragraph[start:end].strip()

        if part:
            parts.append(part)

        if end >= paragraph_length:
            break

        next_start = end - overlap_characters

        if next_start <= start:
            next_start = end

        start = next_start

    return parts


def _get_chunk_overlap(
    chunk: str,
    overlap_characters: int,
) -> str:
    """
    Return a readable overlap from the end of a chunk.
    """

    if overlap_characters <= 0:
        return ""

    if len(chunk) <= overlap_characters:
        return chunk

    overlap_start = len(chunk) - overlap_characters

    paragraph_boundary = chunk.find(
        "\n\n",
        overlap_start,
    )

    if paragraph_boundary != -1:
        overlap_start = paragraph_boundary + 2

    return chunk[overlap_start:].strip()


# =============================================================================
# TEXT OUTPUT
# =============================================================================

def save_text(
    text: str,
    output_path: Path,
    *,
    encoding: str = "utf-8",
) -> Path:
    """
    Save non-empty text to a file.
    """

    cleaned_text = clean_model_output(text)

    if not cleaned_text:
        raise ValueError(
            f"Cannot save empty text to: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        output_path.write_text(
            cleaned_text + "\n",
            encoding=encoding,
        )
    except OSError as exc:
        raise OSError(
            f"Could not save text file: {output_path}"
        ) from exc

    return output_path


def save_evidence_report(
    patient_name: str,
    evidence_report: str,
) -> Path:
    """
    Save the doctor-friendly evidence report as a TXT file.
    """

    patient_directory = get_patient_output_directory(
        patient_name
    )

    filename_base = safe_filename(patient_name)

    output_path = (
        patient_directory
        / f"{filename_base}{EVIDENCE_REPORT_SUFFIX}"
    )

    return save_text(
        evidence_report,
        output_path,
    )


def save_discharge_summary_text(
    patient_name: str,
    discharge_summary: str,
) -> Path:
    """
    Save the completed hospital discharge summary as plain text.
    """

    patient_directory = get_patient_output_directory(
        patient_name
    )

    filename_base = safe_filename(patient_name)

    output_path = (
        patient_directory
        / f"{filename_base}{DISCHARGE_TEXT_SUFFIX}"
    )

    return save_text(
        discharge_summary,
        output_path,
    )


def save_raw_extracted_evidence(
    patient_name: str,
    extracted_evidence: str,
) -> Path:
    """
    Save evidence extracted from all chunks for debugging.
    """

    patient_directory = (
        get_patient_intermediate_directory(
            patient_name
        )
    )

    filename_base = safe_filename(patient_name)

    output_path = (
        patient_directory
        / f"{filename_base}{RAW_EXTRACTION_SUFFIX}"
    )

    return save_text(
        extracted_evidence,
        output_path,
    )


def save_consolidated_evidence(
    patient_name: str,
    consolidated_evidence: str,
) -> Path:
    """
    Save consolidated clinical evidence for debugging and auditing.
    """

    patient_directory = (
        get_patient_intermediate_directory(
            patient_name
        )
    )

    filename_base = safe_filename(patient_name)

    output_path = (
        patient_directory
        / f"{filename_base}{CONSOLIDATED_EVIDENCE_SUFFIX}"
    )

    return save_text(
        consolidated_evidence,
        output_path,
    )


# =============================================================================
# DOCX GENERATION
# =============================================================================

def save_discharge_summary_docx(
    patient_name: str,
    discharge_summary: str,
) -> Path:
    """
    Create a readable DOCX from the generated discharge-summary text.

    The model output is expected to be plain text. Basic heading detection and
    bullet formatting are applied without altering clinical wording.
    """

    cleaned_summary = clean_model_output(
        discharge_summary
    )

    if not cleaned_summary:
        raise ValueError(
            "Discharge summary cannot be empty."
        )

    patient_directory = get_patient_output_directory(
        patient_name
    )

    filename_base = safe_filename(patient_name)

    output_path = (
        patient_directory
        / f"{filename_base}{DISCHARGE_DOCX_SUFFIX}"
    )

    document = Document()

    _configure_document_layout(document)
    _add_discharge_text_to_document(
        document,
        cleaned_summary,
    )

    try:
        document.save(str(output_path))
    except Exception as exc:
        raise OSError(
            f"Could not save DOCX file: {output_path}"
        ) from exc

    return output_path


def _configure_document_layout(
    document: DocumentObject,
) -> None:
    """
    Apply basic page and font settings to a DOCX document.
    """

    for section in document.sections:
        section.top_margin = Inches(0.65)
        section.bottom_margin = Inches(0.65)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)

    normal_style = document.styles["Normal"]
    normal_style.font.name = "Arial"
    normal_style.font.size = Pt(10.5)


def _add_discharge_text_to_document(
    document: DocumentObject,
    text: str,
) -> None:
    """
    Add generated plain text to a DOCX with simple formatting.
    """

    lines = normalize_line_endings(text).split("\n")

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            document.add_paragraph()
            continue

        if _is_main_title(line):
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

            run = paragraph.add_run(
                _remove_markdown_markers(line)
            )

            run.bold = True
            run.font.size = Pt(14)
            continue

        if _is_section_heading(line):
            paragraph = document.add_paragraph()

            run = paragraph.add_run(
                _remove_markdown_markers(line)
            )

            run.bold = True
            run.font.size = Pt(11.5)
            continue

        if line.startswith(("* ", "- ", "• ")):
            bullet_text = line[2:].strip()

            paragraph = document.add_paragraph(
                style="List Bullet"
            )

            _add_inline_bold_text(
                paragraph,
                bullet_text,
            )

            continue

        paragraph = document.add_paragraph()
        _add_inline_bold_text(paragraph, line)


def _is_main_title(line: str) -> bool:
    """
    Check whether a line is the primary document title.
    """

    cleaned = _remove_markdown_markers(
        line
    ).strip().upper()

    return cleaned in {
        "DISCHARGE SUMMARY",
        "NICU DISCHARGE SUMMARY",
    }


def _is_section_heading(line: str) -> bool:
    """
    Detect likely discharge-summary section headings.
    """

    cleaned = _remove_markdown_markers(
        line
    ).strip()

    if not cleaned:
        return False

    known_headings = {
        "Demographics",
        "Antenatal and Perinatal History",
        "History of Present Illness / Reason for Admission",
        "Admission Examination",
        "Morbidities and Course during NICU Stay",
        "Investigations and Interpretation",
        "Procedures and Interventions",
        "Treatments Given",
        "Condition at Discharge",
        "Discharge Medications",
        "Discharge Advice",
        "Follow-up Plan",
        "Items Requiring Doctor Review",
        "Course in Hospital",
        "Diagnosis",
        "Final Diagnosis",
        "Advice at Discharge",
        "Follow Up",
        "Investigations",
        "Treatment",
        "Screening",
    }

    if cleaned in known_headings:
        return True

    if (
        cleaned.isupper()
        and len(cleaned) <= 80
        and not cleaned.endswith(".")
    ):
        return True

    return (
        line.startswith("**")
        and line.endswith("**")
        and len(cleaned) <= 100
    )


def _remove_markdown_markers(text: str) -> str:
    """
    Remove simple Markdown bold markers.
    """

    return (
        text
        .replace("**", "")
        .replace("__", "")
        .strip()
    )


def _add_inline_bold_text(
    paragraph,
    text: str,
) -> None:
    """
    Convert simple **bold** Markdown into DOCX bold runs.
    """

    parts = re.split(
        r"(\*\*.*?\*\*)",
        text,
    )

    for part in parts:
        if not part:
            continue

        if (
            part.startswith("**")
            and part.endswith("**")
        ):
            run = paragraph.add_run(
                part[2:-2]
            )
            run.bold = True
        else:
            paragraph.add_run(part)


# =============================================================================
# PDF CONVERSION
# =============================================================================

def convert_docx_to_pdf(
    docx_path: Path,
    *,
    output_directory: Path | None = None,
) -> Path | None:
    """
    Convert a DOCX file to PDF using LibreOffice.

    Returns:
        Path to the generated PDF.

        Returns None when LibreOffice is not installed or PDF conversion
        fails. The DOCX remains available.

    Installation on macOS:

        brew install --cask libreoffice
    """

    if not docx_path.exists():
        raise FileNotFoundError(
            f"DOCX file was not found: {docx_path}"
        )

    target_directory = (
        output_directory
        or docx_path.parent
    )

    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    libreoffice_command = _find_libreoffice_command()

    if libreoffice_command is None:
        logging.warning(
            "LibreOffice was not found. "
            "DOCX created, but PDF conversion was skipped."
        )
        return None

    command = [
        libreoffice_command,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(target_directory),
        str(docx_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        logging.error(
            "PDF conversion failed for %s: %s",
            docx_path,
            exc,
        )
        return None

    expected_pdf_path = (
        target_directory
        / f"{docx_path.stem}.pdf"
    )

    if (
        result.returncode != 0
        or not expected_pdf_path.exists()
    ):
        logging.error(
            "LibreOffice PDF conversion failed.\n"
            "Return code: %s\n"
            "Output: %s\n"
            "Error: %s",
            result.returncode,
            result.stdout,
            result.stderr,
        )
        return None

    return expected_pdf_path


def _find_libreoffice_command() -> str | None:
    """
    Locate the LibreOffice executable on macOS, Linux or Windows.
    """

    command_candidates = (
        "libreoffice",
        "soffice",
    )

    for command in command_candidates:
        command_path = shutil.which(command)

        if command_path:
            return command_path

    macos_paths = (
        Path(
            "/Applications/LibreOffice.app/"
            "Contents/MacOS/soffice"
        ),
        Path(
            "/Applications/LibreOffice.app/"
            "Contents/MacOS/LibreOffice"
        ),
    )

    for path in macos_paths:
        if path.exists():
            return str(path)

    return None


def save_discharge_summary_documents(
    patient_name: str,
    discharge_summary: str,
    *,
    create_pdf: bool = True,
) -> dict[str, Path | None]:
    """
    Save the final discharge summary as TXT, DOCX and optionally PDF.

    The TXT, DOCX and PDF represent the same clinical output.

    Returns:
        Dictionary containing generated file paths.
    """

    text_path = save_discharge_summary_text(
        patient_name,
        discharge_summary,
    )

    docx_path = save_discharge_summary_docx(
        patient_name,
        discharge_summary,
    )

    pdf_path: Path | None = None

    if create_pdf:
        pdf_path = convert_docx_to_pdf(
            docx_path
        )

    return {
        "text": text_path,
        "docx": docx_path,
        "pdf": pdf_path,
    }


# =============================================================================
# ERROR LOGGING
# =============================================================================

def save_error_report(
    *,
    patient_name: str,
    stage: str,
    error: Exception | str,
    additional_details: str | None = None,
) -> Path:
    """
    Save a patient-processing error as a text file.
    """

    ERROR_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    patient_part = safe_filename(
        patient_name
    )

    stage_part = safe_filename(
        stage,
        default="unknown_stage",
    )

    output_path = (
        ERROR_DIRECTORY
        / (
            f"{timestamp}_{patient_part}_"
            f"{stage_part}_error.txt"
        )
    )

    error_message = str(error)

    report_lines = [
        "NICU DISCHARGE SUMMARY PROCESSING ERROR",
        "",
        f"Patient: {patient_name}",
        f"Stage: {stage}",
        f"Time: {datetime.now().isoformat()}",
        "",
        "Error:",
        error_message,
    ]

    if additional_details:
        report_lines.extend(
            [
                "",
                "Additional Details:",
                additional_details.strip(),
            ]
        )

    output_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    return output_path


# =============================================================================
# STATISTICS
# =============================================================================

def text_statistics(text: str) -> dict[str, int]:
    """
    Calculate simple text statistics.
    """

    cleaned = text.strip()

    return {
        "characters": len(cleaned),
        "words": len(cleaned.split()),
        "lines": (
            len(cleaned.splitlines())
            if cleaned
            else 0
        ),
    }


def total_character_count(
    chunks: Iterable[str],
) -> int:
    """
    Calculate total characters across multiple chunks.
    """

    return sum(
        len(chunk)
        for chunk in chunks
    )


# =============================================================================
# STARTUP TEST
# =============================================================================

if __name__ == "__main__":
    ensure_directories()

    print("Discharge-summary utility check")
    print("=" * 50)
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        "Normalizer output: "
        f"{NORMALIZER_OUTPUT_DIRECTORY}"
    )
    print(
        f"Output directory: {OUTPUT_DIRECTORY}"
    )
    print(
        f"Error directory: {ERROR_DIRECTORY}"
    )

    files = find_normalized_files()

    print(
        f"Normalized files found: {len(files)}"
    )

    for file_path in files:
        patient_name = (
            extract_patient_name_from_filename(
                file_path
            )
        )

        print(
            f"- {patient_name}: {file_path.name}"
        )

    libreoffice = _find_libreoffice_command()

    if libreoffice:
        print(
            f"LibreOffice detected: {libreoffice}"
        )
    else:
        print(
            "LibreOffice not detected. "
            "DOCX generation will work, but PDF "
            "conversion will be skipped."
        )