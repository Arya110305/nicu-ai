"""
NICU Medical Parser

For every patient:

1. Load the complete structured patient document.
2. Run each extraction category as a separate LLM prompt.
3. Give every prompt the complete patient document.
4. Append every result into one patient_record.txt.
5. Generate a final human-style clinical summary from the
   individually extracted sections.
"""

from __future__ import annotations

from pathlib import Path
import time

import ollama

from llm.models import MODEL_NAME

from parser.prompts import (
    COMMON_RULES,
    PARSER_TASKS,
    CLINICAL_SUMMARY_PROMPT,
    build_prompt,
)

from parser.utils import (
    get_patients,
    load_patient_document,
    create_output_folder,
    initialize_patient_record,
    append_section,
    read_patient_record,
)


# ============================================================
# PATH SETTINGS
# ============================================================

INPUT_ROOT = Path(
    "structurer/output"
)

OUTPUT_ROOT = Path(
    "parser/output"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

PARSER_MODEL = MODEL_NAME

PARSER_OPTIONS = {
    "temperature": 0,
    "num_ctx": 65536,
    "num_predict": 4096,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}


SUMMARY_OPTIONS = {
    "temperature": 0,
    "num_ctx": 65536,
    "num_predict": 8192,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}


# ============================================================
# EXTRACT RESPONSE CONTENT
# ============================================================

def get_response_content(
    response: dict,
) -> str:
    """
    Safely extracts the assistant message from an Ollama response.
    """

    message = response.get(
        "message",
        {},
    )

    content = message.get(
        "content",
        "",
    )

    content = content.strip()

    if not content:
        return "Not Found"

    return content


# ============================================================
# RUN ONE EXTRACTION TASK
# ============================================================

def run_parser_task(
    task: dict,
    document: str,
) -> str:
    """
    Runs one independent parser task.

    Every task receives the complete patient document.
    """

    prompt = build_prompt(
        task=task,
        document=document,
    )

    response = ollama.chat(
        model=PARSER_MODEL,
        messages=[
            {
                "role": "system",
                "content": COMMON_RULES,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options=PARSER_OPTIONS,
    )

    return get_response_content(
        response
    )


# ============================================================
# RUN FINAL CLINICAL SUMMARY
# ============================================================

def run_clinical_summary(
    extracted_record: str,
) -> str:
    """
    Produces one human-style NICU discharge summary using the
    individually extracted sections.
    """

    if not extracted_record.strip():
        return "Not Found"

    prompt = f"""
{CLINICAL_SUMMARY_PROMPT}

============================================================
EXTRACTED PATIENT SECTIONS
============================================================

{extracted_record}

============================================================
END OF EXTRACTED PATIENT SECTIONS
============================================================

Now prepare the integrated clinical summary.

Avoid citations.

Include morbidities and their details only when present.

Do not invent missing information.

Make the summary similar to one prepared by a human neonatal
resident.
""".strip()

    response = ollama.chat(
        model=PARSER_MODEL,
        messages=[
            {
                "role": "system",
                "content": CLINICAL_SUMMARY_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options=SUMMARY_OPTIONS,
    )

    return get_response_content(
        response
    )


# ============================================================
# PRINT DOCUMENT INFORMATION
# ============================================================

def print_document_information(
    document: str,
) -> None:
    """
    Prints approximate document size information.
    """

    character_count = len(
        document
    )

    word_count = len(
        document.split()
    )

    approximate_tokens = max(
        1,
        character_count // 4,
    )

    print(
        f"Characters        : {character_count:,}"
    )

    print(
        f"Words             : {word_count:,}"
    )

    print(
        f"Approximate tokens: {approximate_tokens:,}"
    )

    if approximate_tokens > PARSER_OPTIONS["num_ctx"]:

        print(
            "\nWARNING:"
        )

        print(
            "The complete document may be larger than the "
            "configured model context window."
        )

        print(
            "Ollama may truncate part of the document."
        )


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(
    patient_folder: Path,
    output_root: Path,
) -> None:
    """
    Processes one patient's complete structured record.
    """

    print(
        f"\nLoading patient: {patient_folder.name}"
    )

    document = load_patient_document(
        patient_folder
    )

    if not document.strip():

        print(
            "No structured patient document found."
        )

        return

    print_document_information(
        document
    )

    output_folder = create_output_folder(
        output_root=output_root,
        patient_name=patient_folder.name,
    )

    patient_record = initialize_patient_record(
        output_folder
    )

    successful = 0
    failed = 0

    total_tasks = len(
        PARSER_TASKS
    )

    for task_number, (
        task_key,
        task,
    ) in enumerate(
        PARSER_TASKS.items(),
        start=1,
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"[{task_number}/{total_tasks}] "
            f"Extracting {task['title']}"
        )

        print(
            "-" * 70
        )

        start_time = time.time()

        try:

            result = run_parser_task(
                task=task,
                document=document,
            )

            append_section(
                patient_record=patient_record,
                section_name=task["title"],
                content=result,
            )

            successful += 1

            elapsed = time.time() - start_time

            print(
                f"Completed in {elapsed:.1f} seconds"
            )

        except KeyboardInterrupt:

            print(
                "\nParser stopped by the user."
            )

            raise

        except Exception as error:

            failed += 1

            error_message = (
                f"Extraction failed: {error}"
            )

            print(
                error_message
            )

            append_section(
                patient_record=patient_record,
                section_name=task["title"],
                content=error_message,
            )

    # ========================================================
    # FINAL CLINICAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "-" * 70
    )

    print(
        "Generating Final Clinical Summary"
    )

    print(
        "-" * 70
    )

    try:

        extracted_record = read_patient_record(
            patient_record
        )

        clinical_summary = run_clinical_summary(
            extracted_record
        )

        append_section(
            patient_record=patient_record,
            section_name="Clinical Summary",
            content=clinical_summary,
        )

        successful += 1

        print(
            "Clinical summary completed."
        )

    except KeyboardInterrupt:

        print(
            "\nParser stopped by the user."
        )

        raise

    except Exception as error:

        failed += 1

        print(
            f"Clinical summary failed: {error}"
        )

        append_section(
            patient_record=patient_record,
            section_name="Clinical Summary",
            content=f"Summary generation failed: {error}",
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Patient completed: {patient_folder.name}"
    )

    print(
        f"Successful tasks : {successful}"
    )

    print(
        f"Failed tasks     : {failed}"
    )

    print(
        f"Output file      : {patient_record}"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Runs the parser for every patient folder.
    """

    print(
        "=" * 70
    )

    print(
        "NICU MEDICAL PARSER"
    )

    print(
        "=" * 70
    )

    print(
        f"Model       : {PARSER_MODEL}"
    )
    print(f"Context     : {PARSER_OPTIONS['num_ctx']}")

    print(
        f"Input root  : {INPUT_ROOT}"
    )

    print(
        f"Output root : {OUTPUT_ROOT}"
    )

    patients = get_patients(
        INPUT_ROOT
    )

    if not patients:

        print(
            "\nNo patient folders found."
        )

        return

    print(
        f"Patients found: {len(patients)}"
    )

    for patient_number, patient in enumerate(
        patients,
        start=1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"PATIENT {patient_number}/{len(patients)}"
        )

        print(
            patient.name
        )

        print(
            "=" * 70
        )

        try:

            process_patient(
                patient_folder=patient,
                output_root=OUTPUT_ROOT,
            )

        except KeyboardInterrupt:

            print(
                "\nParsing cancelled."
            )

            return

        except Exception as error:

            print(
                f"\nPatient processing failed: {error}"
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PARSING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()