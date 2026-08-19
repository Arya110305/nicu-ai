from __future__ import annotations

import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import ollama

from llm.models import (
    MODEL_NAME,
    CLEANER_OPTIONS,
    OLLAMA_KEEP_ALIVE,
    CLEANER_MAX_WORKERS,
    CLEANER_TIMEOUT_SECONDS,
    CLEANER_RETRIES,
)
from llm.prompts import OCR_CLEANER_PROMPT


# ==========================================================
# CPU THREAD LIMITING
# ==========================================================

# Ollama controls the model execution.
# These settings prevent Python libraries from creating
# unnecessary additional CPU threads for every worker.

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

INPUT_FOLDER = Path("ocr/output")
OUTPUT_FOLDER = Path("llm/output")
ERROR_FOLDER = Path("llm/errors")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
ERROR_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================================
# CLEANER CONFIGURATION
# ==========================================================

# Skip a page when its cleaned output already exists.
RESUME_EXISTING_PAGES = True

# Reprocess an existing file when it is empty.
REPROCESS_EMPTY_OUTPUTS = True

# Save the original model response when cleaning fails.
SAVE_FAILED_RESPONSES = True

# Pause between retry attempts.
RETRY_DELAY_SECONDS = 5

# Ollama server address.
OLLAMA_HOST = "http://localhost:11434"


# ==========================================================
# CONSOLE PRINT LOCK
# ==========================================================

PRINT_LOCK = threading.Lock()


def safe_print(message: str = "") -> None:
    """Print without mixing messages from parallel workers."""

    with PRINT_LOCK:
        print(message, flush=True)


# ==========================================================
# PAGE RESULT
# ==========================================================

class PageResult:
    """Stores the result of cleaning one OCR page."""

    def __init__(
        self,
        page_file: Path,
        output_file: Path,
        success: bool,
        skipped: bool = False,
        error: Optional[str] = None,
        duration: float = 0.0,
    ) -> None:
        self.page_file = page_file
        self.output_file = output_file
        self.success = success
        self.skipped = skipped
        self.error = error
        self.duration = duration


# ==========================================================
# FILE HELPERS
# ==========================================================

def natural_page_number(path: Path) -> int:
    """
    Extract the page number from filenames such as:

    page_001.txt
    page_12.txt
    """

    match = re.search(r"page[_\-\s]*(\d+)", path.stem, re.IGNORECASE)

    if match:
        return int(match.group(1))

    numbers = re.findall(r"\d+", path.stem)

    if numbers:
        return int(numbers[-1])

    return 0


def read_text_file(file_path: Path) -> str:
    """Read a UTF-8 text file safely."""

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )


def atomic_write(file_path: Path, content: str) -> None:
    """
    Write using a temporary file first.

    This prevents partially written files if the program stops
    while saving.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = file_path.with_suffix(
        file_path.suffix + ".tmp"
    )

    temporary_file.write_text(
        content,
        encoding="utf-8",
    )

    temporary_file.replace(file_path)


def output_is_valid(output_file: Path) -> bool:
    """Check whether an existing cleaned output can be reused."""

    if not output_file.exists():
        return False

    if not REPROCESS_EMPTY_OUTPUTS:
        return True

    try:
        return bool(output_file.read_text(
            encoding="utf-8"
        ).strip())
    except Exception:
        return False


# ==========================================================
# TEXT PREPARATION
# ==========================================================

def prepare_page_text(raw_text: str, page_name: str) -> str:
    """
    Prepare one OCR page for the cleaner model.

    The page filename is supplied only as context.
    The cleaner is instructed not to add the filename itself
    to the output.
    """

    raw_text = raw_text.replace("\x00", "")
    raw_text = raw_text.replace("\r\n", "\n")
    raw_text = raw_text.replace("\r", "\n")

    return f"""
PAGE FILE: {page_name}

The page filename above is metadata only.
Do not include it in the corrected output.

OCR TEXT START
------------------------------------------------------------
{raw_text}
------------------------------------------------------------
OCR TEXT END

Correct this OCR page according to the system instructions.
Return only the complete corrected page text.
""".strip()


# ==========================================================
# OLLAMA REQUEST
# ==========================================================

def create_ollama_client() -> ollama.Client:
    """
    Create a separate client for each worker request.

    The model itself remains managed by the same local
    Ollama server.
    """

    return ollama.Client(
        host=OLLAMA_HOST,
        timeout=CLEANER_TIMEOUT_SECONDS,
    )


def clean_text_with_model(
    raw_text: str,
    page_name: str,
) -> str:
    """Send one OCR page to the local LLM."""

    client = create_ollama_client()

    prepared_text = prepare_page_text(
        raw_text=raw_text,
        page_name=page_name,
    )

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": OCR_CLEANER_PROMPT,
            },
            {
                "role": "user",
                "content": prepared_text,
            },
        ],
        options=CLEANER_OPTIONS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    message = response.get("message", {})
    cleaned_text = message.get("content", "")

    if not isinstance(cleaned_text, str):
        cleaned_text = str(cleaned_text)

    cleaned_text = cleaned_text.strip()

    # Remove accidental Markdown code fences.
    if cleaned_text.startswith("```"):
        lines = cleaned_text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_text = "\n".join(lines).strip()

    if not cleaned_text:
        raise ValueError(
            "The model returned an empty response."
        )

    return cleaned_text


# ==========================================================
# ERROR SAVING
# ==========================================================

def save_error(
    patient_name: str,
    page_file: Path,
    error_message: str,
    raw_text: str,
) -> None:
    """Save details for pages that could not be cleaned."""

    patient_error_folder = ERROR_FOLDER / patient_name
    patient_error_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_file = (
        patient_error_folder
        / f"{page_file.stem}_error.txt"
    )

    content = (
        f"PATIENT: {patient_name}\n"
        f"PAGE: {page_file.name}\n"
        f"MODEL: {MODEL_NAME}\n"
        f"ERROR: {error_message}\n"
        f"\n"
        f"{'=' * 70}\n"
        f"ORIGINAL OCR TEXT\n"
        f"{'=' * 70}\n\n"
        f"{raw_text}\n"
    )

    atomic_write(error_file, content)


# ==========================================================
# CLEAN ONE PAGE
# ==========================================================

def clean_page(
    page_file: Path,
    output_patient_folder: Path,
    patient_name: str,
) -> PageResult:
    """Clean one OCR page with retries and resume support."""

    start_time = time.perf_counter()

    output_file = (
        output_patient_folder
        / f"{page_file.stem}_clean.txt"
    )

    if (
        RESUME_EXISTING_PAGES
        and output_is_valid(output_file)
    ):
        return PageResult(
            page_file=page_file,
            output_file=output_file,
            success=True,
            skipped=True,
            duration=time.perf_counter() - start_time,
        )

    try:
        raw_text = read_text_file(page_file)
    except Exception as exc:
        return PageResult(
            page_file=page_file,
            output_file=output_file,
            success=False,
            error=f"Could not read input file: {exc}",
            duration=time.perf_counter() - start_time,
        )

    # Do not call the LLM for blank pages.
    if not raw_text.strip():
        atomic_write(
            output_file,
            "[EMPTY OCR PAGE]",
        )

        return PageResult(
            page_file=page_file,
            output_file=output_file,
            success=True,
            duration=time.perf_counter() - start_time,
        )

    last_error = None

    total_attempts = max(1, CLEANER_RETRIES + 1)

    for attempt in range(1, total_attempts + 1):
        try:
            cleaned_text = clean_text_with_model(
                raw_text=raw_text,
                page_name=page_file.name,
            )

            atomic_write(
                output_file,
                cleaned_text + "\n",
            )

            return PageResult(
                page_file=page_file,
                output_file=output_file,
                success=True,
                duration=time.perf_counter() - start_time,
            )

        except Exception as exc:
            last_error = str(exc)

            safe_print(
                f"    Retry {attempt}/{total_attempts} failed "
                f"for {page_file.name}: {last_error}"
            )

            if attempt < total_attempts:
                delay = RETRY_DELAY_SECONDS * attempt
                time.sleep(delay)

    if SAVE_FAILED_RESPONSES:
        save_error(
            patient_name=patient_name,
            page_file=page_file,
            error_message=last_error or "Unknown error",
            raw_text=raw_text,
        )

    return PageResult(
        page_file=page_file,
        output_file=output_file,
        success=False,
        error=last_error or "Unknown error",
        duration=time.perf_counter() - start_time,
    )


# ==========================================================
# COMPLETE PATIENT FILE
# ==========================================================

def create_complete_patient_file(
    patient_name: str,
    page_files: list[Path],
    output_patient_folder: Path,
) -> Path:
    """
    Combine all successfully cleaned pages in their original
    page order.
    """

    complete_file = (
        output_patient_folder
        / f"{patient_name}_clean_complete.txt"
    )

    complete_parts: list[str] = []

    sorted_pages = sorted(
        page_files,
        key=natural_page_number,
    )

    for page_file in sorted_pages:
        cleaned_page_file = (
            output_patient_folder
            / f"{page_file.stem}_clean.txt"
        )

        complete_parts.append("=" * 70)
        complete_parts.append(page_file.stem.upper())
        complete_parts.append("=" * 70)
        complete_parts.append("")

        if cleaned_page_file.exists():
            cleaned_text = read_text_file(
                cleaned_page_file
            ).strip()

            complete_parts.append(
                cleaned_text or "[EMPTY CLEANED PAGE]"
            )
        else:
            complete_parts.append(
                "[CLEANING FAILED — PAGE OUTPUT NOT AVAILABLE]"
            )

        complete_parts.append("")
        complete_parts.append("")

    complete_text = "\n".join(complete_parts).rstrip()
    complete_text += "\n"

    atomic_write(
        complete_file,
        complete_text,
    )

    return complete_file


# ==========================================================
# PROCESS ONE PATIENT
# ==========================================================

def process_patient(patient_folder: Path) -> bool:
    """Clean every OCR page belonging to one patient."""

    patient_start = time.perf_counter()
    patient_name = patient_folder.name

    safe_print("")
    safe_print("=" * 70)
    safe_print(f"PATIENT: {patient_name}")
    safe_print("=" * 70)

    page_files = sorted(
        patient_folder.glob("page_*.txt"),
        key=natural_page_number,
    )

    if not page_files:
        safe_print("No page_*.txt files found.")
        return False

    output_patient_folder = OUTPUT_FOLDER / patient_name
    output_patient_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_pages = len(page_files)

    worker_count = max(
        1,
        min(
            CLEANER_MAX_WORKERS,
            total_pages,
        ),
    )

    safe_print(f"Pages   : {total_pages}")
    safe_print(f"Model   : {MODEL_NAME}")
    safe_print(f"Workers : {worker_count}")
    safe_print("")

    results: list[PageResult] = []
    completed_count = 0

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="llm-cleaner",
    ) as executor:

        future_to_page = {
            executor.submit(
                clean_page,
                page_file,
                output_patient_folder,
                patient_name,
            ): page_file
            for page_file in page_files
        }

        for future in as_completed(future_to_page):
            page_file = future_to_page[future]
            completed_count += 1

            try:
                result = future.result()
            except Exception as exc:
                result = PageResult(
                    page_file=page_file,
                    output_file=(
                        output_patient_folder
                        / f"{page_file.stem}_clean.txt"
                    ),
                    success=False,
                    error=str(exc),
                )

            results.append(result)

            if result.skipped:
                status = "SKIPPED"
            elif result.success:
                status = "CLEANED"
            else:
                status = "FAILED"

            safe_print(
                f"[{completed_count:03d}/{total_pages:03d}] "
                f"{status:<7} "
                f"{page_file.name:<18} "
                f"{result.duration:.1f}s"
            )

            if result.error:
                safe_print(
                    f"    Error: {result.error}"
                )

    complete_file = create_complete_patient_file(
        patient_name=patient_name,
        page_files=page_files,
        output_patient_folder=output_patient_folder,
    )

    successful = sum(
        1
        for result in results
        if result.success and not result.skipped
    )

    skipped = sum(
        1
        for result in results
        if result.skipped
    )

    failed = sum(
        1
        for result in results
        if not result.success
    )

    patient_duration = time.perf_counter() - patient_start

    safe_print("")
    safe_print("-" * 70)
    safe_print(f"Patient completed : {patient_name}")
    safe_print(f"Cleaned           : {successful}")
    safe_print(f"Skipped           : {skipped}")
    safe_print(f"Failed            : {failed}")
    safe_print(f"Time              : {patient_duration:.2f} sec")
    safe_print(f"Complete file     : {complete_file}")
    safe_print("-" * 70)

    return failed == 0


# ==========================================================
# DISCOVER PATIENT FOLDERS
# ==========================================================

def find_patient_folders() -> list[Path]:
    """Find OCR output folders containing page text files."""

    if not INPUT_FOLDER.exists():
        return []

    patient_folders = [
        folder
        for folder in INPUT_FOLDER.iterdir()
        if folder.is_dir()
        and any(folder.glob("page_*.txt"))
    ]

    return sorted(
        patient_folders,
        key=lambda folder: folder.name.lower(),
    )


# ==========================================================
# OLLAMA CONNECTION CHECK
# ==========================================================

def check_ollama_connection() -> bool:
    """Verify that Ollama is running and the model is available."""

    safe_print("Checking Ollama connection...")

    try:
        client = create_ollama_client()
        available_models = client.list()

    except Exception as exc:
        safe_print("")
        safe_print("Could not connect to Ollama.")
        safe_print(f"Error: {exc}")
        safe_print("")
        safe_print("Start Ollama and try again:")
        safe_print("    ollama serve")
        return False

    model_names: list[str] = []

    for model in available_models.get("models", []):
        model_name = (
            model.get("model")
            or model.get("name")
            or ""
        )

        if model_name:
            model_names.append(model_name)

    requested_base = MODEL_NAME.split(":")[0]

    model_found = any(
        name == MODEL_NAME
        or name.split(":")[0] == requested_base
        for name in model_names
    )

    if not model_found:
        safe_print("")
        safe_print(
            f"Model '{MODEL_NAME}' was not found in Ollama."
        )
        safe_print("")
        safe_print("Install it using:")
        safe_print(f"    ollama pull {MODEL_NAME}")
        return False

    safe_print(f"Ollama ready. Model: {MODEL_NAME}")
    return True


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Run the LLM cleaner for every OCR patient folder."""

    overall_start = time.perf_counter()

    safe_print("=" * 70)
    safe_print("NICU MEDICAL OCR CLEANER")
    safe_print("=" * 70)
    safe_print(f"Input folder : {INPUT_FOLDER}")
    safe_print(f"Output folder: {OUTPUT_FOLDER}")
    safe_print(f"Model        : {MODEL_NAME}")
    safe_print(f"Max workers  : {CLEANER_MAX_WORKERS}")
    safe_print("=" * 70)

    if not check_ollama_connection():
        return

    patient_folders = find_patient_folders()

    if not patient_folders:
        safe_print("")
        safe_print(
            f"No OCR patient folders were found in "
            f"'{INPUT_FOLDER}'."
        )
        safe_print("")
        safe_print(
            "Expected structure:"
        )
        safe_print(
            "ocr/output/<patient_name>/page_001.txt"
        )
        return

    safe_print("")
    safe_print(
        f"Patient folders found: {len(patient_folders)}"
    )

    completed_patients = 0
    failed_patients = 0

    # Patients are processed one at a time.
    # Pages inside each patient are processed concurrently.
    for patient_folder in patient_folders:
        success = process_patient(patient_folder)

        if success:
            completed_patients += 1
        else:
            failed_patients += 1

    overall_duration = time.perf_counter() - overall_start

    safe_print("")
    safe_print("=" * 70)
    safe_print("CLEANING FINISHED")
    safe_print("=" * 70)
    safe_print(f"Patients completed : {completed_patients}")
    safe_print(f"Patients with errors: {failed_patients}")
    safe_print(f"Total time         : {overall_duration:.2f} sec")
    safe_print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()