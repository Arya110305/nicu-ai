"""
NICU discharge-summary generator.

This stage does not stream.

It waits until the normalizer has fully completed and created:

    data/<patient>/normalizer/normalizer_complete.json

It then reads only the complete normalized file:

    data/<patient>/normalizer/
        <patient>_normalized_complete.txt

The generator produces:

    data/<patient>/discharge/
        <patient>_discharge_summary.txt
        discharge_complete.json
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from time import perf_counter
from typing import Callable, Final

from .generator_model import create_model
from .prompt import (
    build_admission_and_diagnosis_prompt,
    build_discharge_and_followup_prompt,
    build_final_polish_prompt,
    build_hospital_course_prompt,
    build_investigation_and_treatment_prompt,
    build_patient_and_birth_prompt,
)
from .utils import (
    clean_clinical_text,
    clean_model_output,
    read_text_file,
    write_text_file,
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT: Final[Path] = (
    Path(__file__).resolve().parent.parent
)

DATA_FOLDER: Final[Path] = (
    PROJECT_ROOT / "data"
)

GLOBAL_ERROR_FOLDER: Final[Path] = (
    PROJECT_ROOT
    / "discharge_summary"
    / "errors"
)

GLOBAL_ERROR_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# SETTINGS
# ==========================================================

MAX_WORKERS: Final[int] = 4

POLL_SECONDS: Final[float] = 1.0


# ==========================================================
# TYPE DEFINITIONS
# ==========================================================

PromptBuilder = Callable[
    [str],
    tuple[str, str],
]


# ==========================================================
# SECTION TASKS
# ==========================================================

TASKS: tuple[
    tuple[str, str, PromptBuilder],
    ...,
] = (
    (
        "patient_and_birth",
        "PATIENT AND BIRTH DETAILS",
        build_patient_and_birth_prompt,
    ),
    (
        "admission_and_diagnosis",
        "ADMISSION, ANTENATAL HISTORY AND DIAGNOSIS",
        build_admission_and_diagnosis_prompt,
    ),
    (
        "hospital_course",
        "MORBIDITIES AND COURSE DURING NICU STAY",
        build_hospital_course_prompt,
    ),
    (
        "investigations_and_treatment",
        "INVESTIGATIONS, PROCEDURES AND TREATMENT",
        build_investigation_and_treatment_prompt,
    ),
    (
        "discharge_and_followup",
        (
            "CONDITION AT DISCHARGE, MEDICATIONS, "
            "ADVICE AND FOLLOW-UP"
        ),
        build_discharge_and_followup_prompt,
    ),
)


# ==========================================================
# PATIENT HELPERS
# ==========================================================

def safe_patient_name(
    value: str,
) -> str:
    """
    Convert a patient name into a safe folder and filename.
    """

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


def get_patient_folders(
    patient_name: str,
) -> dict[str, Path]:
    """
    Create and return discharge-stage patient folders.
    """

    patient_root = (
        DATA_FOLDER
        / safe_patient_name(
            patient_name
        )
    )

    folders = {
        "root": patient_root,

        "normalizer": (
            patient_root
            / "normalizer"
        ),

        "discharge": (
            patient_root
            / "discharge"
        ),

        "discharge_errors": (
            patient_root
            / "discharge"
            / "errors"
        ),
    }

    for folder in folders.values():
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    return folders


# ==========================================================
# NORMALIZER COMPLETION
# ==========================================================

def get_normalizer_marker_path(
    patient_name: str,
) -> Path:
    """
    Return the normalizer completion-marker path.
    """

    folders = get_patient_folders(
        patient_name
    )

    return (
        folders[
            "normalizer"
        ]
        / "normalizer_complete.json"
    )


def load_normalizer_completion_data(
    patient_name: str,
) -> dict[str, object]:
    """
    Load and validate normalizer completion metadata.
    """

    marker_path = (
        get_normalizer_marker_path(
            patient_name
        )
    )

    if not marker_path.exists():
        raise FileNotFoundError(
            "Normalizer completion marker was not found:\n"
            f"{marker_path}"
        )

    try:
        payload = json.loads(
            marker_path.read_text(
                encoding="utf-8",
            )
        )

    except Exception as error:
        raise RuntimeError(
            "Could not read normalizer completion marker:\n"
            f"{marker_path}"
        ) from error

    if (
        payload.get("status")
        != "completed"
    ):
        raise RuntimeError(
            "Normalizer completion marker does not show "
            "a completed status."
        )

    return payload


def get_complete_normalized_file(
    patient_name: str,
) -> Path:
    """
    Return the complete normalized file recorded by the normalizer.
    """

    folders = get_patient_folders(
        patient_name
    )

    payload = load_normalizer_completion_data(
        patient_name
    )

    complete_filename = str(
        payload.get(
            "complete_file",
            "",
        )
    ).strip()

    if not complete_filename:
        raise RuntimeError(
            "Normalizer completion marker does not contain "
            "the complete normalized filename."
        )

    normalized_file = (
        folders[
            "normalizer"
        ]
        / complete_filename
    )

    if not normalized_file.exists():
        raise FileNotFoundError(
            "Complete normalized file was not found:\n"
            f"{normalized_file}"
        )

    return normalized_file


def patient_is_ready(
    patient_name: str,
) -> bool:
    """
    Return True only after the normalizer has fully completed.
    """

    try:
        get_complete_normalized_file(
            patient_name
        )

    except Exception:
        return False

    return True


# ==========================================================
# PATIENT DISCOVERY
# ==========================================================

def find_ready_patients() -> list[str]:
    """
    Find patients whose normalizer stage is complete and whose
    discharge stage is not yet complete.
    """

    if not DATA_FOLDER.exists():
        return []

    ready_patients: list[str] = []

    for patient_folder in sorted(
        DATA_FOLDER.iterdir(),
        key=lambda path: path.name.lower(),
    ):
        if not patient_folder.is_dir():
            continue

        patient_name = patient_folder.name

        normalizer_marker = (
            patient_folder
            / "normalizer"
            / "normalizer_complete.json"
        )

        discharge_marker = (
            patient_folder
            / "discharge"
            / "discharge_complete.json"
        )

        if not normalizer_marker.exists():
            continue

        if discharge_marker.exists():
            continue

        if patient_is_ready(
            patient_name
        ):
            ready_patients.append(
                patient_name
            )

    return ready_patients

# ==========================================================
# ERROR HANDLING
# ==========================================================

def save_error(
    *,
    patient_name: str,
    task_name: str,
    error: Exception,
) -> Path:
    """
    Save a readable discharge-generation error report.
    """

    folders = get_patient_folders(
        patient_name
    )

    safe_task_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        task_name.strip(),
    ).strip("_")

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    error_path = (
        folders[
            "discharge_errors"
        ]
        / (
            f"{safe_patient_name(patient_name)}"
            f"_{safe_task_name}"
            f"_{timestamp}_error.txt"
        )
    )

    error_text = (
        f"Patient: {patient_name}\n"
        f"Task: {task_name}\n"
        f"Error type: {type(error).__name__}\n"
        f"Error: {error}\n\n"
        "TRACEBACK\n"
        "=========\n"
        f"{traceback.format_exc()}\n"
    )

    write_text_file(
        error_path,
        error_text,
    )

    return error_path


# ==========================================================
# SECTION GENERATION
# ==========================================================

def generate_task(
    *,
    task_key: str,
    heading: str,
    prompt_builder: PromptBuilder,
    complete_record: str,
) -> tuple[str, str, str]:
    """
    Generate one discharge-summary section.

    Every section receives the full normalized patient record.
    """

    model = create_model()

    system_prompt, user_prompt = (
        prompt_builder(
            complete_record
        )
    )

    generated_text = model.generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    cleaned_text = clean_model_output(
        generated_text
    ).strip()

    if cleaned_text.upper() == "EMPTY":
        cleaned_text = ""

    return (
        task_key,
        heading,
        cleaned_text,
    )


def run_parallel_tasks(
    *,
    complete_record: str,
    patient_name: str,
) -> dict[str, tuple[str, str]]:
    """
    Run five discharge-summary section tasks using four workers.
    """

    completed_results: dict[
        str,
        tuple[str, str],
    ] = {}

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
    ) as executor:

        futures: dict[
            Future[tuple[str, str, str]],
            tuple[str, str],
        ] = {}

        for (
            task_key,
            heading,
            prompt_builder,
        ) in TASKS:

            future = executor.submit(
                generate_task,
                task_key=task_key,
                heading=heading,
                prompt_builder=prompt_builder,
                complete_record=complete_record,
            )

            futures[
                future
            ] = (
                task_key,
                heading,
            )

        for future in as_completed(
            futures
        ):
            task_key, heading = futures[
                future
            ]

            try:
                (
                    result_key,
                    result_heading,
                    content,
                ) = future.result()

                completed_results[
                    result_key
                ] = (
                    result_heading,
                    content,
                )

                print(
                    f"[DISCHARGE COMPLETE] "
                    f"{result_heading}",
                    flush=True,
                )

            except Exception as error:
                error_path = save_error(
                    patient_name=patient_name,
                    task_name=task_key,
                    error=error,
                )

                print(
                    f"[DISCHARGE FAILED] "
                    f"{heading}: {error}",
                    flush=True,
                )

                print(
                    f"Error saved: {error_path}",
                    flush=True,
                )

                completed_results[
                    task_key
                ] = (
                    heading,
                    "",
                )

    return completed_results


# ==========================================================
# DRAFT SUMMARY
# ==========================================================

def build_draft_summary(
    results: dict[
        str,
        tuple[str, str],
    ],
) -> str:
    """
    Combine generated sections in the required order.
    """

    sections: list[str] = []

    for (
        task_key,
        default_heading,
        _,
    ) in TASKS:

        heading, content = results.get(
            task_key,
            (
                default_heading,
                "",
            ),
        )

        content = content.strip()

        if not content:
            continue

        sections.append(
            (
                f"{heading}\n"
                f"{'-' * len(heading)}\n"
                f"{content}"
            )
        )

    if not sections:
        raise RuntimeError(
            "All discharge-summary section tasks returned empty output."
        )

    return (
        "NICU DISCHARGE SUMMARY\n"
        "======================\n\n"
        + "\n\n".join(
            sections
        )
    ).strip()


# ==========================================================
# FINAL POLISH
# ==========================================================

def polish_final_summary(
    draft_summary: str,
) -> str:
    """
    Run one final model pass over all generated sections.
    """

    model = create_model()

    system_prompt, user_prompt = (
        build_final_polish_prompt(
            draft_summary
        )
    )

    generated_text = model.generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    polished_summary = clean_model_output(
        generated_text
    ).strip()

    if not polished_summary:
        return draft_summary

    return polished_summary


# ==========================================================
# OUTPUT HELPERS
# ==========================================================

def get_discharge_output_path(
    patient_name: str,
) -> Path:
    """
    Return the final discharge-summary TXT path.
    """

    folders = get_patient_folders(
        patient_name
    )

    return (
        folders[
            "discharge"
        ]
        / (
            f"{safe_patient_name(patient_name)}"
            "_discharge_summary.txt"
        )
    )


def save_discharge_completion_marker(
    *,
    patient_name: str,
    normalized_file: Path,
    output_file: Path,
    elapsed_seconds: float,
) -> Path:
    """
    Save a completion marker for the API and frontend.
    """

    folders = get_patient_folders(
        patient_name
    )

    marker_path = (
        folders[
            "discharge"
        ]
        / "discharge_complete.json"
    )

    payload = {
        "patient_name": (
            patient_name
        ),
        "stage": "discharge",
        "status": "completed",
        "workers": MAX_WORKERS,
        "normalized_file": (
            normalized_file.name
        ),
        "summary_file": (
            output_file.name
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
        "completed_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    }

    write_text_file(
        marker_path,
        json.dumps(
            payload,
            indent=2,
        ),
    )

    return marker_path


# ==========================================================
# PROCESS ONE PATIENT
# ==========================================================

def process_patient(
    patient_name: str,
) -> Path:
    """
    Generate one discharge summary only after normalization is complete.
    """

    start_time = perf_counter()

    normalized_file = (
        get_complete_normalized_file(
            patient_name
        )
    )

    print()
    print("=" * 72)
    print("NICU DISCHARGE SUMMARY GENERATOR")
    print("=" * 72)
    print(
        f"Patient         : "
        f"{patient_name}"
    )
    print(
        f"Normalized file : "
        f"{normalized_file}"
    )
    print(
        f"Workers         : "
        f"{MAX_WORKERS}"
    )
    print("=" * 72)

    patient_record = read_text_file(
        normalized_file
    )

    cleaned_record = clean_clinical_text(
        patient_record
    )

    if not cleaned_record.strip():
        raise RuntimeError(
            "The normalized clinical record became empty after cleaning."
        )

    print(
        f"Normalized size : "
        f"{len(patient_record):,} characters"
    )

    print(
        f"Cleaned size    : "
        f"{len(cleaned_record):,} characters"
    )

    print()
    print(
        "Generating five discharge-summary sections..."
    )

    results = run_parallel_tasks(
        complete_record=cleaned_record,
        patient_name=patient_name,
    )

    print()
    print(
        "Combining generated sections..."
    )

    draft_summary = build_draft_summary(
        results
    )

    print(
        "Running final clinical polish..."
    )

    try:
        final_summary = polish_final_summary(
            draft_summary
        )

    except Exception as error:
        error_path = save_error(
            patient_name=patient_name,
            task_name="final_polish",
            error=error,
        )

        print(
            "Final polish failed. "
            "Saving the draft summary instead."
        )

        print(
            f"Error saved: {error_path}"
        )

        final_summary = draft_summary

    output_file = get_discharge_output_path(
        patient_name
    )

    write_text_file(
        output_file,
        final_summary,
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    marker_path = (
        save_discharge_completion_marker(
            patient_name=patient_name,
            normalized_file=normalized_file,
            output_file=output_file,
            elapsed_seconds=elapsed,
        )
    )

    print()
    print("=" * 72)
    print("DISCHARGE GENERATION COMPLETED")
    print("=" * 72)
    print(
        f"Summary file : "
        f"{output_file}"
    )
    print(
        f"Marker file  : "
        f"{marker_path}"
    )
    print(
        f"Total time   : "
        f"{elapsed:.2f} seconds"
    )
    print("=" * 72)

    return output_file

# ==========================================================
# WAIT FOR NORMALIZER
# ==========================================================

def wait_for_normalizer(
    patient_name: str,
) -> Path:
    """
    Wait until the normalizer has completely finished.

    The discharge generator starts only after:

        data/<patient>/normalizer/normalizer_complete.json

    exists and points to a valid complete normalized TXT file.
    """

    print()
    print(
        "Waiting for complete normalized patient record...",
        flush=True,
    )

    while True:
        try:
            normalized_file = (
                get_complete_normalized_file(
                    patient_name
                )
            )

            print(
                f"Normalizer completed: "
                f"{normalized_file}",
                flush=True,
            )

            return normalized_file

        except (
            FileNotFoundError,
            RuntimeError,
        ):
            time.sleep(
                POLL_SECONDS
            )


# ==========================================================
# FIND PATIENTS WAITING FOR DISCHARGE
# ==========================================================

def find_waiting_patients() -> list[str]:
    """
    Find patients that have started processing but do not yet
    have a completed discharge summary.

    Patients are included when they have either:

    - a normalizer folder
    - a normalizer completion marker

    Existing completed discharge patients are skipped.
    """

    if not DATA_FOLDER.exists():
        return []

    patients: list[str] = []

    for patient_folder in sorted(
        DATA_FOLDER.iterdir(),
        key=lambda path: path.name.lower(),
    ):
        if not patient_folder.is_dir():
            continue

        normalizer_folder = (
            patient_folder
            / "normalizer"
        )

        discharge_marker = (
            patient_folder
            / "discharge"
            / "discharge_complete.json"
        )

        if discharge_marker.exists():
            continue

        if not normalizer_folder.exists():
            continue

        patients.append(
            patient_folder.name
        )

    return patients


# ==========================================================
# PROCESS PATIENT AFTER WAITING
# ==========================================================

def wait_and_process_patient(
    patient_name: str,
) -> Path:
    """
    Wait for one patient's complete normalized file and then
    generate the discharge summary.
    """

    wait_for_normalizer(
        patient_name
    )

    return process_patient(
        patient_name
    )


# ==========================================================
# TERMINAL HEADER
# ==========================================================

def print_header() -> None:
    """
    Display the discharge-generator configuration.
    """

    print("=" * 72)
    print("NICU COMPLETE-RECORD DISCHARGE GENERATOR")
    print("=" * 72)
    print(
        f"Data folder     : "
        f"{DATA_FOLDER}"
    )
    print(
        f"Section workers : "
        f"{MAX_WORKERS}"
    )
    print(
        f"Polling interval: "
        f"{POLL_SECONDS}s"
    )
    print(
        "Input policy    : "
        "Complete normalized record only"
    )
    print("=" * 72)


# ==========================================================
# PROCESS READY PATIENTS
# ==========================================================

def process_ready_patients() -> tuple[
    list[Path],
    list[tuple[str, Exception]],
]:
    """
    Generate discharge summaries for all patients whose
    normalizer stage has completed.
    """

    successful_outputs: list[Path] = []

    failed_patients: list[
        tuple[str, Exception]
    ] = []

    ready_patients = (
        find_ready_patients()
    )

    for patient_name in ready_patients:
        try:
            output_path = process_patient(
                patient_name
            )

            successful_outputs.append(
                output_path
            )

        except KeyboardInterrupt:
            raise

        except Exception as error:
            failed_patients.append(
                (
                    patient_name,
                    error,
                )
            )

            error_path = save_error(
                patient_name=patient_name,
                task_name="complete_discharge",
                error=error,
            )

            print()
            print("=" * 72)
            print("DISCHARGE GENERATION FAILED")
            print("=" * 72)
            print(
                f"Patient: {patient_name}"
            )
            print(
                f"Error: "
                f"{type(error).__name__}: "
                f"{error}"
            )
            print(
                f"Error file: {error_path}"
            )
            print("=" * 72)

    return (
        successful_outputs,
        failed_patients,
    )


# ==========================================================
# OPTIONAL CONTINUOUS WATCH MODE
# ==========================================================

def watch_for_patients() -> None:
    """
    Continuously watch for patients whose normalizer stage finishes.

    This function is useful when the discharge generator is started
    together with OCR, LLM and the normalizer.

    It never reads partial normalized batches.
    """

    print()
    print(
        "Watching for completed normalized patient records...",
        flush=True,
    )

    while True:
        ready_patients = (
            find_ready_patients()
        )

        for patient_name in ready_patients:
            try:
                process_patient(
                    patient_name
                )

            except Exception as error:
                error_path = save_error(
                    patient_name=patient_name,
                    task_name="watch_mode_discharge",
                    error=error,
                )

                print(
                    f"Discharge failed for "
                    f"{patient_name}: {error}",
                    flush=True,
                )

                print(
                    f"Error saved: {error_path}",
                    flush=True,
                )

        time.sleep(
            POLL_SECONDS
        )


# ==========================================================
# MAIN
# ==========================================================

def main() -> int:
    """
    Run the discharge generator for all completed normalizer records.
    """

    started_at = perf_counter()

    print_header()

    try:
        ready_patients = (
            find_ready_patients()
        )

        if not ready_patients:
            print()
            print(
                "No patients with a completed normalized "
                "record are waiting for discharge generation."
            )

            print()
            print(
                "The generator requires:"
            )

            print(
                "  data/<patient>/normalizer/"
                "normalizer_complete.json"
            )

            return 0

        print()
        print(
            f"Ready patients: "
            f"{len(ready_patients)}"
        )

        successful_outputs, failed_patients = (
            process_ready_patients()
        )

        elapsed = (
            perf_counter()
            - started_at
        )

        print()
        print("=" * 72)
        print("DISCHARGE GENERATOR RUN SUMMARY")
        print("=" * 72)
        print(
            f"Ready patients : "
            f"{len(ready_patients)}"
        )
        print(
            f"Successful     : "
            f"{len(successful_outputs)}"
        )
        print(
            f"Failed         : "
            f"{len(failed_patients)}"
        )
        print(
            f"Total time     : "
            f"{elapsed:.2f} seconds"
        )

        if successful_outputs:
            print()
            print("Created summaries:")

            for output_path in (
                successful_outputs
            ):
                print(
                    f"  ✓ {output_path}"
                )

        if failed_patients:
            print()
            print("Failed patients:")

            for (
                patient_name,
                error,
            ) in failed_patients:
                print(
                    f"  ✗ {patient_name}: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

        print("=" * 72)

        if failed_patients:
            return 1

        return 0

    except KeyboardInterrupt:
        print()
        print(
            "Discharge generation stopped by user."
        )

        return 130

    except Exception as error:
        print()
        print("=" * 72)
        print("DISCHARGE GENERATOR FAILED")
        print("=" * 72)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print(
            traceback.format_exc()
        )
        print("=" * 72)

        return 1


if __name__ == "__main__":
    sys.exit(

        main()
    )