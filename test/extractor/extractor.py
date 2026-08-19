from pathlib import Path
import ollama

from llm.models import MODEL_NAME

from extractor.prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    build_extraction_prompt,
)

from extractor.utils import (
    get_sections,
    load_section,
    save_text,
)


# --------------------------------------------------
# Extract One Section
# --------------------------------------------------

def extract_section(section_name, section_text):

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": EXTRACTOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_extraction_prompt(
                    section_name,
                    section_text,
                ),
            },
        ],
    )

    return response["message"]["content"]


# --------------------------------------------------
# Process One Patient
# --------------------------------------------------

def process_patient(patient_folder, output_folder):

    sections = get_sections(patient_folder)

    for section_folder in sections:

        section_name = section_folder.name

        print(f"\nExtracting {section_name}")

        section_text = load_section(section_folder)

        if section_text is None:
            continue

        extracted_text = extract_section(
            section_name,
            section_text,
        )

        save_text(
            output_folder,
            section_name,
            extracted_text,
        )

        print("Done")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    input_root = Path("classifier/output")

    output_root = Path("extractor/output")

    for patient in input_root.iterdir():

        if not patient.is_dir():
            continue

        print("=" * 70)
        print(patient.name)
        print("=" * 70)

        process_patient(
            patient,
            output_root / patient.name,
        )


if __name__ == "__main__":
    main()