from pathlib import Path
import ollama

from llm.models import MODEL_NAME
from parser.schema import SCHEMA
from parser.prompts import SYSTEM_PROMPT, TIMELINE_SYSTEM_PROMPT
from parser.utils import (
    build_prompt,
    build_timeline_prompt,
    load_document,
    save_sections,
    save_text,
)

from llm.models import MODEL_NAME

STRUCTURED_DIR = Path("structurer/output")
OUTPUT_DIR = Path("parser/output")


# --------------------------------------------------
# Patient Schema Extraction
# --------------------------------------------------

def extract_patient_schema(document: str):

    print("\n========== PATIENT SCHEMA ==========")

    results = {}

    for section, fields in SCHEMA.items():

        print(f"Extracting {section}...")

        prompt = build_prompt(section, fields, document)

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
    "temperature": 0,
    "num_predict": 4096,
}
        )

        results[section] = response["message"]["content"].strip()

    return results


# --------------------------------------------------
# Timeline Extraction
# --------------------------------------------------

def extract_timeline(document: str):

    print("\n========== TIMELINE ==========")

    prompt = build_timeline_prompt(document)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": TIMELINE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
    "temperature": 0,
    "num_predict": 4096,
}
    )

    return response["message"]["content"].strip()


# --------------------------------------------------
# Medical Facts Extraction
# --------------------------------------------------

def extract_medical_facts(document: str):

    print("\n========== MEDICAL FACTS ==========")

    prompt = f"""
You are an expert NICU Medical Information Extraction AI.

Extract EVERY important medical fact from the patient's record.

Include EVERYTHING.

Patient Information
Birth Details
Maternal History
Admission
Diagnoses
Clinical Findings
Vitals
Respiratory Support
Laboratory Results
Radiology
Procedures
Medications
Nutrition
Feeding
Progress Notes
Nursing Notes
Consultant Notes
Complications
Screenings
Vaccinations
Discharge

Rules

1. Never summarize.
2. Never explain.
3. Never recommend.
4. Copy exactly.
5. Preserve wording.
6. Preserve laboratory values.
7. Preserve medication names.
8. Preserve dates.
9. Preserve units.
10. Return plain text only.

Medical Record:

{document}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
    "temperature": 0,
    "num_predict": 4096,
}
    )

    return response["message"]["content"].strip()


# --------------------------------------------------
# Process One Patient
# --------------------------------------------------

def process_patient(patient_folder: Path):

    patient_name = patient_folder.name

    structured_file = (
        patient_folder /
        f"{patient_name}_structured_complete.md"
    )

    if not structured_file.exists():

        print(f"Missing structured file for {patient_name}")
        return

    print("\n" + "=" * 80)
    print(patient_name)
    print("=" * 80)

    document = load_document(structured_file)

    output_folder = OUTPUT_DIR / patient_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Patient Schema
    # ----------------------------

    patient_schema = extract_patient_schema(document)

    save_sections(
        output_folder /
        f"{patient_name}_schema.txt",
        patient_schema,
    )

    # ----------------------------
    # Timeline
    # ----------------------------

    timeline = extract_timeline(document)

    save_text(
        output_folder /
        f"{patient_name}_timeline.txt",
        timeline,
    )

    # ----------------------------
    # Medical Facts
    # ----------------------------

    facts = extract_medical_facts(document)

    save_text(
        output_folder /
        f"{patient_name}_medical_facts.txt",
        facts,
    )

    print(f"\nFinished {patient_name}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not STRUCTURED_DIR.exists():

        print("Structured output folder not found.")
        return

    patient_folders = sorted([
        folder
        for folder in STRUCTURED_DIR.iterdir()
        if folder.is_dir()
    ])

    if not patient_folders:

        print("No patient folders found.")
        return

    print(f"\nFound {len(patient_folders)} patient(s)\n")

    for patient in patient_folders:
        process_patient(patient)

    print("\nAll patients processed.")


if __name__ == "__main__":
    main()