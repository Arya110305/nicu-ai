"""
NICU discharge-summary generator.

NEW PIPELINE ARCHITECTURE
=========================

This stage DOES NOT stream.

The Discharge Summary starts only after the Normalizer has fully
completed.

Required Normalizer inputs:

    normalizer/output/
        NORMALIZER_<patient>.txt

and:

    normalizer/temp/<patient>/
        normalizer_complete.json

The Normalizer completion marker is only created after its complete
combined TXT has been written and validated.

The Discharge Summary therefore NEVER reads:

    normalizer/temp/<patient>/parts/

and never consumes partial Normalizer batches.

Temporary discharge work is stored in:

    discharge_summary/temp/<patient>/
        sections/
        errors/

The final discharge summary is:

    discharge_summary/output/
        DISCHARGE_<patient>.txt

Temporary files are not deleted here.

api/pipeline.py will perform final cleanup and move useful outputs into:

    data/<patient>/

only after the complete pipeline succeeds.
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

from typing import (
    Callable,
    Final,
)

from .generator_model import (
    create_model,
)

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
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DISCHARGE_ROOT: Final[Path] = (
    Path(__file__)
    .resolve()
    .parent
)

DISCHARGE_TEMP_FOLDER: Final[Path] = (
    DISCHARGE_ROOT
    / "temp"
)

DISCHARGE_OUTPUT_FOLDER: Final[Path] = (
    DISCHARGE_ROOT
    / "output"
)

NORMALIZER_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "normalizer"
)

NORMALIZER_TEMP_FOLDER: Final[Path] = (
    NORMALIZER_ROOT
    / "temp"
)

NORMALIZER_OUTPUT_FOLDER: Final[Path] = (
    NORMALIZER_ROOT
    / "output"
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
    tuple[
        str,
        str,
        PromptBuilder,
    ],
    ...,
] = (
    (
        "patient_and_birth",
        "PATIENT AND BIRTH DETAILS",
        build_patient_and_birth_prompt,
    ),

    (
        "admission_and_diagnosis",
        (
            "ADMISSION, ANTENATAL HISTORY "
            "AND DIAGNOSIS"
        ),
        build_admission_and_diagnosis_prompt,
    ),

    (
        "hospital_course",
        (
            "MORBIDITIES AND COURSE "
            "DURING NICU STAY"
        ),
        build_hospital_course_prompt,
    ),

    (
        "investigations_and_treatment",
        (
            "INVESTIGATIONS, PROCEDURES "
            "AND TREATMENT"
        ),
        build_investigation_and_treatment_prompt,
    ),

    (
        "discharge_and_followup",
        (
            "CONDITION AT DISCHARGE, "
            "MEDICATIONS, ADVICE AND FOLLOW-UP"
        ),
        build_discharge_and_followup_prompt,
    ),
)


# ==========================================================
# PATIENT NAME
# ==========================================================

def safe_patient_name(
    value: str,
) -> str:
    """
    Convert a patient name into a safe canonical identifier.

    Temporary upload prefixes such as:

        pending_

    are removed.
    """

    value = (
        value
        .strip()
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


# ==========================================================
# NORMALIZER PATHS
# ==========================================================

def get_normalizer_workspace(
    patient_name: str,
) -> dict[
    str,
    Path,
]:
    """
    Return the Normalizer locations used by the Discharge stage.

    The Discharge generator reads ONLY:

        normalizer/output/NORMALIZER_<patient>.txt

    and:

        normalizer/temp/<patient>/normalizer_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    temp_root = (
        NORMALIZER_TEMP_FOLDER
        / patient_name
    )

    return {
        "temp_root": (
            temp_root
        ),

        "completion": (
            temp_root
            / "normalizer_complete.json"
        ),

        "final_output": (
            NORMALIZER_OUTPUT_FOLDER
            / (
                f"NORMALIZER_"
                f"{patient_name}"
                f".txt"
            )
        ),
    }


# ==========================================================
# DISCHARGE WORKSPACE
# ==========================================================

def get_discharge_workspace(
    patient_name: str,
) -> dict[
    str,
    Path,
]:
    """
    Create and return temporary discharge-stage locations.

    Structure:

        discharge_summary/temp/<patient>/
            sections/
            errors/
            discharge_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    root = (
        DISCHARGE_TEMP_FOLDER
        / patient_name
    )

    workspace = {
        "root": (
            root
        ),

        "sections": (
            root
            / "sections"
        ),

        "errors": (
            root
            / "errors"
        ),

        "completion": (
            root
            / "discharge_complete.json"
        ),
    }

    workspace[
        "root"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    workspace[
        "sections"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    workspace[
        "errors"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    return workspace


# ==========================================================
# FINAL DISCHARGE OUTPUT
# ==========================================================

def ensure_discharge_output_folder() -> Path:
    """
    Ensure the final discharge-output folder exists.
    """

    DISCHARGE_OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        DISCHARGE_OUTPUT_FOLDER
    )


def get_discharge_output_path(
    patient_name: str,
) -> Path:
    """
    Return the permanent stage output path.

    Example:

        discharge_summary/output/
            DISCHARGE_BO_VAGDEVI.txt
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    ensure_discharge_output_folder()

    return (
        DISCHARGE_OUTPUT_FOLDER
        / (
            f"DISCHARGE_"
            f"{patient_name}"
            f".txt"
        )
    )


def get_discharge_completion_marker_path(
    patient_name: str,
) -> Path:
    """
    Return the temporary discharge completion marker.
    """

    workspace = (
        get_discharge_workspace(
            patient_name
        )
    )

    return (
        workspace[
            "completion"
        ]
    )


# ==========================================================
# NORMALIZER COMPLETION MARKER
# ==========================================================

def get_normalizer_marker_path(
    patient_name: str,
) -> Path:
    """
    Return:

        normalizer/temp/<patient>/
            normalizer_complete.json
    """

    workspace = (
        get_normalizer_workspace(
            patient_name
        )
    )

    return (
        workspace[
            "completion"
        ]
    )


def load_normalizer_completion_data(
    patient_name: str,
) -> dict[
    str,
    object,
]:
    """
    Load and strictly validate the Normalizer completion marker.

    The Discharge Summary must not start unless this succeeds.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    marker_path = (
        get_normalizer_marker_path(
            patient_name
        )
    )

    if not marker_path.exists():

        raise FileNotFoundError(
            "Normalizer completion marker "
            "was not found:\n"
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
            "Could not read Normalizer "
            "completion marker:\n"
            f"{marker_path}"
        ) from error

    if not isinstance(
        payload,
        dict,
    ):

        raise RuntimeError(
            "Normalizer completion marker "
            "does not contain a JSON object."
        )

    if (
        payload.get(
            "stage"
        )
        != "normalizer"
    ):

        raise RuntimeError(
            "Normalizer completion marker "
            "has an invalid stage."
        )

    if (
        payload.get(
            "status"
        )
        != "completed"
    ):

        raise RuntimeError(
            "Normalizer completion marker "
            "does not show completed status."
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
            "Normalizer completion marker "
            "belongs to another patient. "
            f"Expected {patient_name}, "
            f"found {marker_patient}."
        )

    complete_file = str(
        payload.get(
            "complete_file",
            "",
        )
    ).strip()

    expected_filename = (
        f"NORMALIZER_"
        f"{patient_name}"
        f".txt"
    )

    if (
        complete_file
        != expected_filename
    ):

        raise RuntimeError(
            "Normalizer completion marker "
            "does not reference the expected "
            "final Normalizer TXT. "
            f"Expected {expected_filename}, "
            f"found {complete_file!r}."
        )

    return payload


# ==========================================================
# GET COMPLETE NORMALIZED FILE
# ==========================================================

def get_complete_normalized_file(
    patient_name: str,
) -> Path:
    """
    Return the ONLY input file the Discharge Summary may read.

    Requirements:

        1. normalizer_complete.json exists
        2. status == completed
        3. patient matches
        4. complete_file matches expected filename
        5. NORMALIZER_<patient>.txt exists
        6. TXT is non-empty
        7. character_count matches when supplied

    Partial Normalizer files are never considered.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    payload = (
        load_normalizer_completion_data(
            patient_name
        )
    )

    workspace = (
        get_normalizer_workspace(
            patient_name
        )
    )

    normalized_file = (
        workspace[
            "final_output"
        ]
    )

    if not normalized_file.exists():

        raise FileNotFoundError(
            "Complete Normalizer TXT "
            "was not found:\n"
            f"{normalized_file}"
        )

    if (
        normalized_file.suffix.lower()
        != ".txt"
    ):

        raise RuntimeError(
            "Discharge input must be "
            f"a TXT file: {normalized_file}"
        )

    normalized_text = (
        read_text_file(
            normalized_file
        )
    )

    if not normalized_text.strip():

        raise RuntimeError(
            "Complete Normalizer TXT "
            "is empty."
        )

    expected_character_count = (
        payload.get(
            "character_count"
        )
    )

    if (
        isinstance(
            expected_character_count,
            int,
        )
        and expected_character_count > 0
        and len(
            normalized_text
        )
        != expected_character_count
    ):

        raise RuntimeError(
            "Complete Normalizer TXT "
            "does not match the completion "
            "marker character count."
        )

    return normalized_file


# ==========================================================
# PATIENT READY CHECK
# ==========================================================

def patient_is_ready(
    patient_name: str,
) -> bool:
    """
    Return True only when the strict Normalizer -> Discharge
    gate is open.
    """

    try:

        get_complete_normalized_file(
            patient_name
        )

    except Exception:

        return False

    return True


# ==========================================================
# FIND READY PATIENTS
# ==========================================================

def find_ready_patients() -> list[str]:
    """
    Find patients whose Normalizer is fully completed and whose
    Discharge stage has not yet completed.

    Discovery source:

        normalizer/temp/

    NOT:

        data/
    """

    if not NORMALIZER_TEMP_FOLDER.exists():

        return []

    ready_patients: list[
        str
    ] = []

    for patient_folder in sorted(
        NORMALIZER_TEMP_FOLDER.iterdir(),
        key=lambda path: (
            path.name.lower()
        ),
    ):

        if not patient_folder.is_dir():

            continue

        patient_name = (
            safe_patient_name(
                patient_folder.name
            )
        )

        normalizer_marker = (
            patient_folder
            / "normalizer_complete.json"
        )

        if not normalizer_marker.exists():

            continue

        discharge_marker = (
            get_discharge_completion_marker_path(
                patient_name
            )
        )

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
# TEMPORARY SECTION PATH
# ==========================================================

def get_section_output_path(
    patient_name: str,
    task_key: str,
) -> Path:
    """
    Return the temporary TXT output for one discharge section.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    safe_task = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        task_key.strip(),
    ).strip(
        "_"
    )

    if not safe_task:

        safe_task = (
            "section"
        )

    workspace = (
        get_discharge_workspace(
            patient_name
        )
    )

    return (
        workspace[
            "sections"
        ]
        / (
            f"{safe_task}"
            f".txt"
        )
    )


# ==========================================================
# ERROR PATH
# ==========================================================

def get_error_path(
    patient_name: str,
    task_name: str,
) -> Path:
    """
    Return the temporary error-report path for a discharge task.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    safe_task_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        task_name.strip(),
    ).strip(
        "_"
    )

    if not safe_task_name:

        safe_task_name = (
            "discharge"
        )

    workspace = (
        get_discharge_workspace(
            patient_name
        )
    )

    timestamp = (
        time.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    return (
        workspace[
            "errors"
        ]
        / (
            f"{safe_task_name}"
            f"_{timestamp}"
            f"_ERROR.txt"
        )
    )


# ==========================================================
# SAVE ERROR
# ==========================================================

def save_error(
    *,
    patient_name: str,
    task_name: str,
    error: Exception,
) -> Path:
    """
    Save a readable temporary discharge-generation error report.
    """

    error_path = (
        get_error_path(
            patient_name,
            task_name,
        )
    )

    error_text = (
        f"Patient: "
        f"{safe_patient_name(patient_name)}\n"
        f"Task: "
        f"{task_name}\n"
        f"Error type: "
        f"{type(error).__name__}\n"
        f"Error: "
        f"{error}\n\n"
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
# GENERATE ONE DISCHARGE SECTION
# ==========================================================

def generate_task(
    *,
    task_key: str,
    heading: str,
    prompt_builder: PromptBuilder,
    complete_record: str,
    patient_name: str,
) -> tuple[
    str,
    str,
    str,
]:
    """
    Generate exactly one discharge-summary section.

    IMPORTANT:

    Every section receives the COMPLETE normalized patient record.

    This stage does not read partial Normalizer batches.
    """

    model = (
        create_model()
    )

    (
        system_prompt,
        user_prompt,
    ) = (
        prompt_builder(
            complete_record
        )
    )

    generated_text = (
        model.generate_text(
            system_prompt=(
                system_prompt
            ),
            user_prompt=(
                user_prompt
            ),
        )
    )

    cleaned_text = (
        clean_model_output(
            generated_text
        )
        .strip()
    )

    if (
        cleaned_text.upper()
        == "EMPTY"
    ):

        cleaned_text = ""

    # ------------------------------------------------------
    # SAVE TEMPORARY SECTION OUTPUT
    # ------------------------------------------------------

    section_path = (
        get_section_output_path(
            patient_name,
            task_key,
        )
    )

    write_text_file(
        section_path,
        cleaned_text,
    )

    return (
        task_key,
        heading,
        cleaned_text,
    )


# ==========================================================
# LOAD EXISTING SECTION
# ==========================================================

def load_existing_section(
    *,
    patient_name: str,
    task_key: str,
    heading: str,
) -> tuple[
    str,
    str,
] | None:
    """
    Restore a temporary section already created during the
    CURRENT pipeline run.

    api/pipeline.py will later clear discharge temp before a
    new run begins, preventing stale previous-run reuse.
    """

    section_path = (
        get_section_output_path(
            patient_name,
            task_key,
        )
    )

    if not section_path.exists():

        return None

    try:

        content = (
            read_text_file(
                section_path
            )
            .strip()
        )

    except Exception:

        return None

    if not content:

        return None

    return (
        heading,
        content,
    )


# ==========================================================
# RUN PARALLEL DISCHARGE TASKS
# ==========================================================

def run_parallel_tasks(
    *,
    complete_record: str,
    patient_name: str,
) -> dict[str, tuple[str, str]]:
    """
    Run five discharge-summary section tasks using four workers.

    A section is allowed to return EMPTY when there is genuinely
    no information for that section.

    But if a section task itself crashes or the model call fails,
    the complete Discharge stage must fail instead of silently
    generating a partial discharge summary.
    """

    completed_results: dict[
        str,
        tuple[str, str],
    ] = {}

    failed_tasks: list[
        tuple[
            str,
            str,
            Exception,
        ]
    ] = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
    ) as executor:

        futures: dict[
            Future[tuple[str, str, str]],
            tuple[str, str],
        ] = {}

        # ==================================================
        # SUBMIT ALL FIVE SECTION TASKS
        # ==================================================

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
                patient_name=patient_name,
            )

            futures[
                future
            ] = (
                task_key,
                heading,
            )

        # ==================================================
        # COLLECT ALL SECTION RESULTS
        # ==================================================

        for future in as_completed(
            futures
        ):

            (
                task_key,
                heading,
            ) = futures[
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

                failed_tasks.append(
                    (
                        task_key,
                        heading,
                        error,
                    )
                )

                print(
                    f"[DISCHARGE FAILED] "
                    f"{heading}: {error}",
                    flush=True,
                )

                print(
                    f"Error saved: "
                    f"{error_path}",
                    flush=True,
                )

    # ======================================================
    # DO NOT ALLOW PARTIAL SUCCESS AFTER A TASK FAILURE
    # ======================================================

    if failed_tasks:

        failed_names = ", ".join(
            (
                f"{task_key} "
                f"({heading})"
            )
            for (
                task_key,
                heading,
                _,
            ) in failed_tasks
        )

        raise RuntimeError(
            "One or more discharge-summary "
            "section tasks failed. "
            "Discharge generation has been "
            "stopped to prevent a partial "
            "clinical summary. "
            f"Failed task(s): {failed_names}"
        )

    # ======================================================
    # VERIFY EVERY TASK RETURNED A RESULT
    # ======================================================

    expected_task_keys = {
        task_key
        for (
            task_key,
            _,
            _,
        ) in TASKS
    }

    returned_task_keys = set(
        completed_results
    )

    missing_task_keys = (
        expected_task_keys
        - returned_task_keys
    )

    if missing_task_keys:

        raise RuntimeError(
            "Discharge section processing "
            "finished without results for: "
            + ", ".join(
                sorted(
                    missing_task_keys
                )
            )
        )

    return completed_results


# ==========================================================
# VALIDATE SECTION RESULTS
# ==========================================================

def validate_section_results(
    results: dict[
        str,
        tuple[
            str,
            str,
        ],
    ],
) -> None:
    """
    Confirm that at least one useful discharge section exists.

    Individual sections may legitimately be empty if the
    normalized record contains no evidence for that section.
    """

    if not results:

        raise RuntimeError(
            "No discharge section results "
            "were produced."
        )

    non_empty_sections = 0

    for (
        task_key,
        default_heading,
        _,
    ) in TASKS:

        (
            heading,
            content,
        ) = (
            results.get(
                task_key,
                (
                    default_heading,
                    "",
                ),
            )
        )

        del heading

        if (
            content.strip()
        ):

            non_empty_sections += 1

    if (
        non_empty_sections
        == 0
    ):

        raise RuntimeError(
            "All discharge-summary section "
            "tasks returned empty output."
        )


# ==========================================================
# BUILD DRAFT SUMMARY
# ==========================================================

def build_draft_summary(
    results: dict[
        str,
        tuple[
            str,
            str,
        ],
    ],
) -> str:
    """
    Combine generated discharge sections in the required order.

    Completion order from parallel workers does NOT affect final
    summary order.
    """

    validate_section_results(
        results
    )

    sections: list[
        str
    ] = []

    for (
        task_key,
        default_heading,
        _,
    ) in TASKS:

        (
            heading,
            content,
        ) = (
            results.get(
                task_key,
                (
                    default_heading,
                    "",
                ),
            )
        )

        content = (
            content.strip()
        )

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
            "The discharge draft contains "
            "no usable sections."
        )

    draft = (
        "NICU DISCHARGE SUMMARY\n"
        "======================\n\n"
        + "\n\n".join(
            sections
        )
    ).strip()

    if not draft:

        raise RuntimeError(
            "Combined discharge draft "
            "is empty."
        )

    return draft


# ==========================================================
# FINAL POLISH
# ==========================================================

def polish_final_summary(
    draft_summary: str,
) -> str:
    """
    Run one final model pass over the COMPLETE draft summary.

    This pass improves consistency and presentation after the
    parallel section tasks have finished.
    """

    if not draft_summary.strip():

        raise RuntimeError(
            "Cannot polish an empty "
            "discharge draft."
        )

    model = (
        create_model()
    )

    (
        system_prompt,
        user_prompt,
    ) = (
        build_final_polish_prompt(
            draft_summary
        )
    )

    generated_text = (
        model.generate_text(
            system_prompt=(
                system_prompt
            ),
            user_prompt=(
                user_prompt
            ),
        )
    )

    polished_summary = (
        clean_model_output(
            generated_text
        )
        .strip()
    )

    if not polished_summary:

        return (
            draft_summary
        )

    return (
        polished_summary
    )


# ==========================================================
# BASIC FINAL SUMMARY VALIDATION
# ==========================================================

def validate_final_summary(
    summary_text: str,
) -> None:
    """
    Reject an empty or implausibly tiny discharge summary.
    """

    cleaned = (
        summary_text.strip()
    )

    if not cleaned:

        raise RuntimeError(
            "Final discharge summary "
            "is empty."
        )

    if (
        len(
            cleaned
        )
        < 200
    ):

        raise RuntimeError(
            "Final discharge summary "
            "appears unexpectedly short. "
            f"Length: "
            f"{len(cleaned)} characters."
        )


# ==========================================================
# SAVE FINAL DISCHARGE OUTPUT
# ==========================================================

def save_final_discharge_output(
    *,
    patient_name: str,
    final_summary: str,
) -> Path:
    """
    Save:

        discharge_summary/output/
            DISCHARGE_<patient>.txt

    The file is written only after final validation.
    """

    validate_final_summary(
        final_summary
    )

    output_path = (
        get_discharge_output_path(
            patient_name
        )
    )

    write_text_file(
        output_path,
        final_summary,
    )

    # ------------------------------------------------------
    # READ-BACK VERIFICATION
    # ------------------------------------------------------

    saved_text = (
        read_text_file(
            output_path
        )
    )

    validate_final_summary(
        saved_text
    )

    return (
        output_path
    )


# ==========================================================
# SAVE DISCHARGE COMPLETION MARKER
# ==========================================================

def save_discharge_completion_marker(
    *,
    patient_name: str,
    normalized_file: Path,
    output_file: Path,
    elapsed_seconds: float,
) -> Path:
    """
    Save temporary workflow metadata after the final discharge
    TXT has been successfully written and validated.

    Clinical content remains in:

        DISCHARGE_<patient>.txt
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    if not output_file.exists():

        raise RuntimeError(
            "Cannot create discharge "
            "completion marker because "
            "the final summary TXT "
            "does not exist."
        )

    final_text = (
        read_text_file(
            output_file
        )
    )

    validate_final_summary(
        final_text
    )

    marker_path = (
        get_discharge_completion_marker_path(
            patient_name
        )
    )

    payload = {
        "patient_name": (
            patient_name
        ),

        "stage": (
            "discharge"
        ),

        "status": (
            "completed"
        ),

        "workers": (
            MAX_WORKERS
        ),

        "normalized_file": (
            normalized_file.name
        ),

        "summary_file": (
            output_file.name
        ),

        "summary_path": str(
            output_file.resolve()
        ),

        "character_count": len(
            final_text
        ),

        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),

        "completed_at": (
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        ),
    }

    write_text_file(
        marker_path,
        json.dumps(
            payload,
            indent=2,
        ),
    )

    return (
        marker_path
    )
# ==========================================================
# VALIDATE DISCHARGE COMPLETION
# ==========================================================

def validate_discharge_completion(
    patient_name: str,
) -> Path:
    """
    Validate that the Discharge stage has completely finished.

    BOTH must exist:

        discharge_summary/output/
            DISCHARGE_<patient>.txt

    and:

        discharge_summary/temp/<patient>/
            discharge_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    output_path = (
        get_discharge_output_path(
            patient_name
        )
    )

    marker_path = (
        get_discharge_completion_marker_path(
            patient_name
        )
    )

    if not output_path.exists():

        raise RuntimeError(
            "Final discharge TXT "
            "does not exist."
        )

    if not marker_path.exists():

        raise RuntimeError(
            "Discharge completion marker "
            "does not exist."
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
            "Could not read discharge "
            "completion marker."
        ) from error

    if not isinstance(
        payload,
        dict,
    ):

        raise RuntimeError(
            "Discharge completion marker "
            "must contain a JSON object."
        )

    if (
        payload.get(
            "stage"
        )
        != "discharge"
    ):

        raise RuntimeError(
            "Discharge completion marker "
            "has an invalid stage."
        )

    if (
        payload.get(
            "status"
        )
        != "completed"
    ):

        raise RuntimeError(
            "Discharge stage is not "
            "marked completed."
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
            "Discharge completion marker "
            "belongs to another patient."
        )

    expected_filename = (
        f"DISCHARGE_"
        f"{patient_name}"
        f".txt"
    )

    summary_file = str(
        payload.get(
            "summary_file",
            "",
        )
    ).strip()

    if (
        summary_file
        != expected_filename
    ):

        raise RuntimeError(
            "Discharge completion marker "
            "does not reference the "
            "expected summary TXT."
        )

    final_text = (
        read_text_file(
            output_path
        )
    )

    validate_final_summary(
        final_text
    )

    expected_character_count = (
        payload.get(
            "character_count"
        )
    )

    if (
        isinstance(
            expected_character_count,
            int,
        )
        and expected_character_count > 0
        and len(
            final_text
        )
        != expected_character_count
    ):

        raise RuntimeError(
            "Final discharge TXT changed "
            "after completion."
        )

    return output_path


# ==========================================================
# PROCESS ONE PATIENT
# ==========================================================

def process_patient(
    patient_name: str,
) -> Path:
    """
    Generate one discharge summary.

    STRICT INPUT RULE:

    The function first validates the complete Normalizer output.

    Therefore Discharge cannot start from:

        - partial normalized batches
        - a Normalizer TXT without its completion marker
        - a completion marker without its final TXT
        - another patient's Normalizer output

    Processing:

        complete Normalizer record
                  ↓
        five parallel extraction tasks
                  ↓
        ordered draft
                  ↓
        final polish
                  ↓
        DISCHARGE_<patient>.txt
                  ↓
        discharge_complete.json
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    start_time = (
        perf_counter()
    )

    # ======================================================
    # STRICT NORMALIZER -> DISCHARGE GATE
    # ======================================================

    normalized_file = (
        get_complete_normalized_file(
            patient_name
        )
    )

    print()
    print(
        "=" * 72
    )
    print(
        "NICU DISCHARGE SUMMARY GENERATOR"
    )
    print(
        "=" * 72
    )
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
    print(
        "=" * 72
    )

    # ======================================================
    # READ COMPLETE NORMALIZED RECORD
    # ======================================================

    patient_record = (
        read_text_file(
            normalized_file
        )
    )

    if not patient_record.strip():

        raise RuntimeError(
            "The complete normalized "
            "clinical record is empty."
        )

    # ======================================================
    # PREPARE COMPLETE RECORD
    # ======================================================

    cleaned_record = (
        clean_clinical_text(
            patient_record
        )
    )

    if not cleaned_record.strip():

        raise RuntimeError(
            "The normalized clinical "
            "record became empty after "
            "cleaning."
        )

    print(
        f"Normalized size : "
        f"{len(patient_record):,} "
        "characters"
    )

    print(
        f"Cleaned size    : "
        f"{len(cleaned_record):,} "
        "characters"
    )

    # ======================================================
    # RUN FIVE PARALLEL EXTRACTION TASKS
    # ======================================================

    results = (
        run_parallel_tasks(
            complete_record=(
                cleaned_record
            ),
            patient_name=(
                patient_name
            ),
        )
    )

    # ======================================================
    # BUILD ORDERED DRAFT
    # ======================================================

    print()
    print(
        "Combining generated "
        "discharge sections...",
        flush=True,
    )

    draft_summary = (
        build_draft_summary(
            results
        )
    )

    # ======================================================
    # FINAL POLISH
    # ======================================================

    print(
        "Running final clinical polish...",
        flush=True,
    )

    try:

        final_summary = (
            polish_final_summary(
                draft_summary
            )
        )

    except Exception as error:

        error_path = (
            save_error(
                patient_name=(
                    patient_name
                ),
                task_name=(
                    "final_polish"
                ),
                error=(
                    error
                ),
            )
        )

        print(
            "Final polish failed. "
            "Using the completed draft "
            "summary instead.",
            flush=True,
        )

        print(
            f"Error saved: "
            f"{error_path}",
            flush=True,
        )

        final_summary = (
            draft_summary
        )

    # ======================================================
    # VALIDATE FINAL SUMMARY
    # ======================================================

    validate_final_summary(
        final_summary
    )

    # ======================================================
    # WRITE FINAL DISCHARGE TXT
    # ======================================================

    output_file = (
        save_final_discharge_output(
            patient_name=(
                patient_name
            ),
            final_summary=(
                final_summary
            ),
        )
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    # ======================================================
    # ONLY NOW CREATE COMPLETION MARKER
    # ======================================================

    marker_path = (
        save_discharge_completion_marker(
            patient_name=(
                patient_name
            ),
            normalized_file=(
                normalized_file
            ),
            output_file=(
                output_file
            ),
            elapsed_seconds=(
                elapsed
            ),
        )
    )

    # ======================================================
    # FINAL READ-BACK / COMPLETION VALIDATION
    # ======================================================

    verified_output = (
        validate_discharge_completion(
            patient_name
        )
    )

    if (
        verified_output.resolve()
        != output_file.resolve()
    ):

        raise RuntimeError(
            "Discharge completion "
            "validation returned an "
            "unexpected output file."
        )

    print()
    print(
        "=" * 72
    )
    print(
        "DISCHARGE GENERATION COMPLETED"
    )
    print(
        "=" * 72
    )
    print(
        f"Patient      : "
        f"{patient_name}"
    )
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
    print(
        "=" * 72
    )

    # ======================================================
    # DO NOT CLEAN TEMP HERE
    # ======================================================
    #
    # api/pipeline.py will later perform cleanup only after
    # the ENTIRE patient pipeline has been confirmed
    # successful.
    #
    # If something fails after this point, temporary stage
    # information remains available for debugging.
    #
    # ======================================================

    return output_file


# ==========================================================
# WAIT FOR NORMALIZER
# ==========================================================

def wait_for_normalizer(
    patient_name: str,
) -> Path:
    """
    Wait until the strict Normalizer -> Discharge gate opens.

    The Discharge generator does not inspect partial Normalizer
    parts while waiting.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    print()
    print(
        "Waiting for complete normalized "
        "patient record...",
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
                "Normalizer completed: "
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
# PROCESS PATIENT AFTER WAITING
# ==========================================================

def wait_and_process_patient(
    patient_name: str,
) -> Path:
    """
    Wait for the complete Normalizer output and then generate
    the discharge summary.

    There is no Normalizer batch streaming into Discharge.
    """

    patient_name = (
        safe_patient_name(
            patient_name
        )
    )

    wait_for_normalizer(
        patient_name
    )

    return (
        process_patient(
            patient_name
        )
    )


# ==========================================================
# FIND PATIENTS WAITING FOR DISCHARGE
# ==========================================================

def find_waiting_patients() -> list[str]:
    """
    Find Normalizer patient workspaces that do not yet have a
    completed Discharge stage.

    Discovery uses:

        normalizer/temp/

    and never data/.
    """

    if not NORMALIZER_TEMP_FOLDER.exists():

        return []

    patients: list[
        str
    ] = []

    for patient_folder in sorted(
        NORMALIZER_TEMP_FOLDER.iterdir(),
        key=lambda path: (
            path.name.lower()
        ),
    ):

        if not patient_folder.is_dir():

            continue

        patient_name = (
            safe_patient_name(
                patient_folder.name
            )
        )

        normalizer_marker = (
            patient_folder
            / "normalizer_complete.json"
        )

        if not normalizer_marker.exists():

            continue

        discharge_marker = (
            get_discharge_completion_marker_path(
                patient_name
            )
        )

        if discharge_marker.exists():

            continue

        patients.append(
            patient_name
        )

    return patients


# ==========================================================
# PROCESS ALL READY PATIENTS
# ==========================================================

def process_ready_patients() -> tuple[
    list[Path],
    list[
        tuple[
            str,
            Exception,
        ]
    ],
]:
    """
    Generate discharge summaries for all patients whose strict
    Normalizer completion gate is open.
    """

    successful_outputs: list[
        Path
    ] = []

    failed_patients: list[
        tuple[
            str,
            Exception,
        ]
    ] = []

    ready_patients = (
        find_ready_patients()
    )

    for patient_name in (
        ready_patients
    ):

        try:

            output_path = (
                process_patient(
                    patient_name
                )
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

            error_path = (
                save_error(
                    patient_name=(
                        patient_name
                    ),
                    task_name=(
                        "complete_discharge"
                    ),
                    error=(
                        error
                    ),
                )
            )

            print()
            print(
                "=" * 72
            )
            print(
                "DISCHARGE GENERATION FAILED"
            )
            print(
                "=" * 72
            )
            print(
                f"Patient: "
                f"{patient_name}"
            )
            print(
                f"Error: "
                f"{type(error).__name__}: "
                f"{error}"
            )
            print(
                f"Error file: "
                f"{error_path}"
            )
            print(
                "=" * 72
            )

    return (
        successful_outputs,
        failed_patients,
    )


# ==========================================================
# OPTIONAL WATCH MODE
# ==========================================================

def watch_for_patients() -> None:
    """
    Continuously watch for completed Normalizer records.

    This is optional standalone behavior.

    It never reads partial Normalizer batches.
    """

    print()
    print(
        "Watching for completed normalized "
        "patient records...",
        flush=True,
    )

    while True:

        ready_patients = (
            find_ready_patients()
        )

        for patient_name in (
            ready_patients
        ):

            try:

                process_patient(
                    patient_name
                )

            except Exception as error:

                error_path = (
                    save_error(
                        patient_name=(
                            patient_name
                        ),
                        task_name=(
                            "watch_mode_discharge"
                        ),
                        error=(
                            error
                        ),
                    )
                )

                print(
                    "Discharge failed for "
                    f"{patient_name}: "
                    f"{error}",
                    flush=True,
                )

                print(
                    f"Error saved: "
                    f"{error_path}",
                    flush=True,
                )

        time.sleep(
            POLL_SECONDS
        )


# ==========================================================
# TERMINAL HEADER
# ==========================================================

def print_header() -> None:
    """
    Display the new Discharge architecture.
    """

    print(
        "=" * 72
    )
    print(
        "NICU COMPLETE-RECORD "
        "DISCHARGE GENERATOR"
    )
    print(
        "=" * 72
    )
    print(
        f"Normalizer temp   : "
        f"{NORMALIZER_TEMP_FOLDER}"
    )
    print(
        f"Normalizer output : "
        f"{NORMALIZER_OUTPUT_FOLDER}"
    )
    print(
        f"Discharge temp    : "
        f"{DISCHARGE_TEMP_FOLDER}"
    )
    print(
        f"Discharge output  : "
        f"{DISCHARGE_OUTPUT_FOLDER}"
    )
    print(
        f"Section workers   : "
        f"{MAX_WORKERS}"
    )
    print(
        f"Polling interval  : "
        f"{POLL_SECONDS}s"
    )
    print(
        "Input policy      : "
        "Complete Normalizer record only"
    )
    print(
        "=" * 72
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> int:
    """
    Run the discharge generator for all patients whose
    Normalizer stage has completely finished.
    """

    started_at = (
        perf_counter()
    )

    print_header()

    try:

        ready_patients = (
            find_ready_patients()
        )

        if not ready_patients:

            print()
            print(
                "No completed Normalizer "
                "records are waiting for "
                "discharge generation."
            )

            print()
            print(
                "The generator requires BOTH:"
            )

            print(
                "  normalizer/output/"
                "NORMALIZER_<patient>.txt"
            )

            print(
                "  normalizer/temp/<patient>/"
                "normalizer_complete.json"
            )

            return 0

        print()
        print(
            f"Ready patients: "
            f"{len(ready_patients)}"
        )

        (
            successful_outputs,
            failed_patients,
        ) = (
            process_ready_patients()
        )

        elapsed = (
            perf_counter()
            - started_at
        )

        print()
        print(
            "=" * 72
        )
        print(
            "DISCHARGE GENERATOR RUN SUMMARY"
        )
        print(
            "=" * 72
        )
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
            print(
                "Created summaries:"
            )

            for output_path in (
                successful_outputs
            ):

                print(
                    f"  ✓ "
                    f"{output_path}"
                )

        if failed_patients:

            print()
            print(
                "Failed patients:"
            )

            for (
                patient_name,
                error,
            ) in (
                failed_patients
            ):

                print(
                    f"  ✗ "
                    f"{patient_name}: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

        print(
            "=" * 72
        )

        if failed_patients:

            return 1

        return 0

    except KeyboardInterrupt:

        print()
        print(
            "Discharge generation "
            "stopped by user."
        )

        return 130

    except Exception as error:

        print()
        print(
            "=" * 72
        )
        print(
            "DISCHARGE GENERATOR FAILED"
        )
        print(
            "=" * 72
        )
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print(
            traceback.format_exc()
        )
        print(
            "=" * 72
        )

        return 1


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )