"""
Utility functions for the NICU Medical Parser.

Handles:

- finding patient folders
- loading the complete structured document
- creating parser output folders
- creating patient_record.txt
- appending extracted sections
- reading the accumulated patient record
"""

from __future__ import annotations

from pathlib import Path
import re


# ============================================================
# GET PATIENT FOLDERS
# ============================================================

def get_patients(
    input_root: Path,
) -> list[Path]:
    """
    Returns all patient folders inside structurer/output.
    """

    if not input_root.exists():
        print(
            f"Input folder does not exist: {input_root}"
        )
        return []

    if not input_root.is_dir():
        print(
            f"Input path is not a folder: {input_root}"
        )
        return []

    patients = [
        path
        for path in input_root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
    ]

    return sorted(
        patients,
        key=lambda path: path.name.lower(),
    )


# ============================================================
# NATURAL FILE SORTING
# ============================================================

def natural_sort_key(
    path: Path,
) -> list:
    """
    Sorts file names naturally.

    Example:
    page_2.md comes before page_10.md.
    """

    parts = re.split(
        r"(\d+)",
        str(path).lower(),
    )

    return [
        int(part)
        if part.isdigit()
        else part
        for part in parts
    ]


# ============================================================
# READ TEXT FILE
# ============================================================

def read_text_file(
    file_path: Path,
) -> str:
    """
    Reads a text file safely using UTF-8.
    """

    try:

        return file_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

    except OSError as error:

        print(
            f"Unable to read {file_path}: {error}"
        )

        return ""


# ============================================================
# FIND COMPLETE STRUCTURED DOCUMENT
# ============================================================

def find_complete_document(
    patient_folder: Path,
) -> Path | None:
    """
    Searches for a complete structured patient document.
    """

    preferred_patterns = [
        "*_structured_complete.md",
        "*structured_complete*.md",
        "*complete_structured*.md",
        "*_complete.md",
        "*complete*.md",
        "*_structured_complete.txt",
        "*structured_complete*.txt",
    ]

    for pattern in preferred_patterns:

        matches = [
            path
            for path in patient_folder.glob(pattern)
            if path.is_file()
            and not path.name.startswith(".")
        ]

        if matches:

            matches.sort(
                key=natural_sort_key
            )

            return matches[0]

    return None


# ============================================================
# FIND INDIVIDUAL STRUCTURED FILES
# ============================================================

def find_structured_files(
    patient_folder: Path,
) -> list[Path]:
    """
    Finds individual markdown or text files if no single complete
    document exists.
    """

    markdown_files = [
        path
        for path in patient_folder.rglob("*.md")
        if path.is_file()
        and not path.name.startswith(".")
    ]

    if markdown_files:

        return sorted(
            markdown_files,
            key=natural_sort_key,
        )

    text_files = [
        path
        for path in patient_folder.rglob("*.txt")
        if path.is_file()
        and not path.name.startswith(".")
    ]

    return sorted(
        text_files,
        key=natural_sort_key,
    )


# ============================================================
# COMBINE STRUCTURED FILES
# ============================================================

def combine_structured_files(
    files: list[Path],
    patient_folder: Path,
) -> str:
    """
    Combines multiple structured files into one complete patient
    document while preserving file order.
    """

    document_parts: list[str] = []

    for file_path in files:

        content = read_text_file(
            file_path
        )

        if not content:
            continue

        try:

            relative_name = file_path.relative_to(
                patient_folder
            )

        except ValueError:

            relative_name = file_path.name

        document_part = f"""
============================================================
SOURCE DOCUMENT: {relative_name}
============================================================

{content}
""".strip()

        document_parts.append(
            document_part
        )

    return "\n\n".join(
        document_parts
    ).strip()


# ============================================================
# LOAD COMPLETE PATIENT DOCUMENT
# ============================================================

def load_patient_document(
    patient_folder: Path,
) -> str:
    """
    Loads the complete structured document for one patient.

    Preference:

    1. One complete structured markdown file
    2. Combined markdown files
    3. Combined text files
    """

    if not patient_folder.exists():
        print(
            f"Patient folder does not exist: "
            f"{patient_folder}"
        )
        return ""

    complete_document = find_complete_document(
        patient_folder
    )

    if complete_document is not None:

        print(
            f"Using complete structured document: "
            f"{complete_document.name}"
        )

        return read_text_file(
            complete_document
        )

    structured_files = find_structured_files(
        patient_folder
    )

    if not structured_files:

        print(
            "No structured markdown or text files found."
        )

        return ""

    print(
        f"Combining {len(structured_files)} "
        f"structured files."
    )

    return combine_structured_files(
        files=structured_files,
        patient_folder=patient_folder,
    )


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

def create_output_folder(
    output_root: Path,
    patient_name: str,
) -> Path:
    """
    Creates the parser output folder for one patient.
    """

    output_folder = (
        output_root
        / patient_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_folder


# ============================================================
# INITIALIZE PATIENT RECORD
# ============================================================

def initialize_patient_record(
    output_folder: Path,
) -> Path:
    """
    Creates a new patient_record.txt.

    The old patient_record.txt is replaced on every run.
    """

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_record = (
        output_folder
        / "patient_record.txt"
    )

    patient_record.write_text(
        "",
        encoding="utf-8",
    )

    return patient_record


# ============================================================
# CLEAN MODEL RESPONSE
# ============================================================

def clean_model_output(
    content: str | None,
) -> str:
    """
    Cleans an LLM response before it is saved.
    """

    if content is None:
        return "Not Found"

    cleaned = content.strip()

    if not cleaned:
        return "Not Found"

    if cleaned.startswith("```markdown"):
        cleaned = cleaned[len("```markdown"):].strip()

    elif cleaned.startswith("```text"):
        cleaned = cleaned[len("```text"):].strip()

    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    unwanted_openings = [
        "Here is the extracted information:",
        "Here is the requested information:",
        "Based on the provided document:",
        "According to the provided document:",
    ]

    for opening in unwanted_openings:

        if cleaned.lower().startswith(
            opening.lower()
        ):
            cleaned = cleaned[
                len(opening):
            ].strip()

    if not cleaned:
        return "Not Found"

    return cleaned


# ============================================================
# APPEND EXTRACTED SECTION
# ============================================================

def append_section(
    patient_record: Path,
    section_name: str,
    content: str | None,
) -> None:
    """
    Appends one extracted clinical section to patient_record.txt.
    """

    cleaned_content = clean_model_output(
        content
    )

    section = f"""
============================================================
{section_name.upper()}
============================================================

{cleaned_content}

""".lstrip()

    with patient_record.open(
        mode="a",
        encoding="utf-8",
    ) as file:

        file.write(
            section
        )


# ============================================================
# READ PATIENT RECORD
# ============================================================

def read_patient_record(
    patient_record: Path,
) -> str:
    """
    Reads all sections already saved in patient_record.txt.
    """

    if not patient_record.exists():
        return ""

    return read_text_file(
        patient_record
    )