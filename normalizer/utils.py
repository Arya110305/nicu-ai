"""
Utility functions for the NICU clinical normalizer.

This module handles:

- folder creation
- input-file discovery
- safe text reading
- internal chunking
- output filename generation
- atomic file writing
- error-file writing

Chunk files are never created.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


PAGE_MARKER_PATTERN = re.compile(
    r"(?im)^(?:={2,}\s*)?"
    r"(?:PAGE|SOURCE PAGE|PDF PAGE|DOCUMENT PAGE)"
    r"[\s:_-]*(?:\d+|[A-Z0-9_-]+)"
    r"(?:\s*={2,})?\s*$"
)

HEADING_PATTERN = re.compile(
    r"(?m)^(?:"
    r"[A-Z][A-Z0-9 /&(),:'-]{3,}"
    r"|#{1,6}\s+.+"
    r"|\[[A-Z][A-Z0-9 _/-]{2,}\]"
    r")\s*$"
)


def ensure_directories(*directories: str | Path) -> None:
    """
    Create required folders when they do not already exist.
    """

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def discover_input_files(
    input_folder: str | Path,
    input_suffix: str,
) -> list[Path]:
    """
    Find structured input files inside the input folder.
    """

    folder = Path(input_folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Input path is not a folder: {folder}"
        )

    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.name.endswith(input_suffix)
        and not path.name.startswith(".")
    )

    return files


def read_text_file(
    file_path: str | Path,
    encoding: str = "utf-8",
) -> str:
    """
    Read a text file and validate that it contains usable content.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        text = path.read_text(
            encoding=encoding,
            errors="replace",
        )

    text = normalize_line_endings(text)

    if not text.strip():
        raise ValueError(f"Input file is empty: {path}")

    return text


def normalize_line_endings(text: str) -> str:
    """
    Convert all line endings to Unix-style newlines.
    """

    return text.replace("\r\n", "\n").replace("\r", "\n")


def create_output_filename(
    input_filename: str,
    input_suffix: str,
    output_suffix: str,
) -> str:
    """
    Convert the structured input filename into the normalized filename.

    Example:
        BO_Vagdevi_structured_complete.txt
        becomes
        BO_Vagdevi_normalized_complete.txt
    """

    if not input_filename.endswith(input_suffix):
        raise ValueError(
            f"Input filename does not end with {input_suffix}: "
            f"{input_filename}"
        )

    base_name = input_filename[: -len(input_suffix)]

    return f"{base_name}{output_suffix}"


def find_preferred_split_position(
    text: str,
    start: int,
    target_end: int,
    minimum_end: int,
) -> int:
    """
    Find a safe location near the target chunk ending.

    Split priority:

    1. Page marker
    2. Section heading
    3. Blank line
    4. Sentence ending
    5. Line ending
    6. Hard character limit
    """

    search_text = text[minimum_end:target_end]

    page_matches = list(PAGE_MARKER_PATTERN.finditer(search_text))
    if page_matches:
        return minimum_end + page_matches[-1].start()

    heading_matches = list(HEADING_PATTERN.finditer(search_text))
    if heading_matches:
        return minimum_end + heading_matches[-1].start()

    blank_line_position = search_text.rfind("\n\n")
    if blank_line_position != -1:
        return minimum_end + blank_line_position + 2

    sentence_positions = [
        search_text.rfind(". "),
        search_text.rfind(".\n"),
        search_text.rfind("? "),
        search_text.rfind("?\n"),
        search_text.rfind("! "),
        search_text.rfind("!\n"),
    ]

    sentence_position = max(sentence_positions)

    if sentence_position != -1:
        return minimum_end + sentence_position + 1

    line_position = search_text.rfind("\n")

    if line_position != -1:
        return minimum_end + line_position + 1

    return target_end


def split_text_into_chunks(
    text: str,
    max_characters: int,
) -> list[str]:
    """
    Split the complete structured record into internal chunks.

    The function tries to split near page markers, headings, blank lines,
    or sentence boundaries.

    No overlap is used because overlapping text could produce duplicate
    clinical content when chunk responses are joined.
    """

    if max_characters <= 0:
        raise ValueError(
            "max_characters must be greater than zero"
        )

    text = normalize_line_endings(text).strip()

    if not text:
        return []

    if len(text) <= max_characters:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(
            start + max_characters,
            text_length,
        )

        if target_end >= text_length:
            final_chunk = text[start:].strip()

            if final_chunk:
                chunks.append(final_chunk)

            break

        minimum_chunk_size = max(
            int(max_characters * 0.60),
            1,
        )

        minimum_end = min(
            start + minimum_chunk_size,
            target_end,
        )

        split_position = find_preferred_split_position(
            text=text,
            start=start,
            target_end=target_end,
            minimum_end=minimum_end,
        )

        if split_position <= start:
            split_position = target_end

        chunk = text[start:split_position].strip()

        if chunk:
            chunks.append(chunk)

        start = split_position

        while start < text_length and text[start] in " \t\n":
            start += 1

    return chunks


def combine_normalized_chunks(
    chunks: list[str],
) -> str:
    """
    Combine normalized chunk responses into one final document.

    Empty chunks are ignored. The order remains unchanged.
    """

    cleaned_chunks = [
        normalize_line_endings(chunk).strip()
        for chunk in chunks
        if chunk and chunk.strip()
    ]

    if not cleaned_chunks:
        raise ValueError(
            "No normalized chunk content was produced."
        )

    return "\n\n".join(cleaned_chunks).strip() + "\n"


def atomic_write_text(
    file_path: str | Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """
    Write text safely using a temporary file and atomic replacement.

    This prevents an incomplete final output if the program stops while
    writing.
    """

    destination = Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, destination)

    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

        raise


def write_error_file(
    error_folder: str | Path,
    patient_name: str,
    chunk_number: int,
    chunk_text: str,
    error_message: str,
    encoding: str = "utf-8",
) -> Path:
    """
    Write an error file only when a chunk fails completely.
    """

    folder = Path(error_folder)
    folder.mkdir(parents=True, exist_ok=True)

    safe_patient_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        patient_name,
    ).strip("_")

    if not safe_patient_name:
        safe_patient_name = "unknown_patient"

    error_path = folder / (
        f"{safe_patient_name}_chunk_{chunk_number:03d}_error.txt"
    )

    error_content = (
        "NICU NORMALIZER CHUNK FAILURE\n"
        "========================================\n"
        f"Patient/File : {patient_name}\n"
        f"Chunk number : {chunk_number}\n"
        f"Error        : {error_message}\n"
        "========================================\n\n"
        "FAILED CHUNK INPUT\n"
        "========================================\n"
        f"{chunk_text}\n"
    )

    atomic_write_text(
        file_path=error_path,
        content=error_content,
        encoding=encoding,
    )

    return error_path


def count_words(text: str) -> int:
    """
    Return a simple word count for progress reporting.
    """

    return len(text.split())


def format_number(value: int) -> str:
    """
    Format a number with commas for terminal output.
    """

    return f"{value:,}"