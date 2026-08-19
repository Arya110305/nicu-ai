from pathlib import Path
import ollama

from llm.models import MODEL_NAME

from classifier.prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    build_classifier_prompt,
)

from classifier.utils import (
    load_patient_pages,
    load_page,
    parse_labels,
    group_pages,
    save_grouped_sections,
    save_page_labels,
)


# --------------------------------------------------
# Classify One Page
# --------------------------------------------------

def classify_page(page_text):

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": CLASSIFIER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_classifier_prompt(page_text),
            },
        ],
    )

    labels = parse_labels(
        response["message"]["content"]
    )

    return labels


# --------------------------------------------------
# Process One Patient
# --------------------------------------------------

def classify_patient(patient_folder: Path, output_folder: Path):

    pages = load_patient_pages(patient_folder)

    classifications = []

    for page in pages:

        print(f"Classifying {page.name}")

        text = load_page(page)

        labels = classify_page(text)
        print(f"Labels: {', '.join(labels)}")

        classifications.append(
            (
                page.name,
                text,
                labels,
            )
        )

    grouped = group_pages(classifications)

    save_grouped_sections(
        output_folder,
        grouped,
    )

    save_page_labels(
        output_folder,
        classifications,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    input_root = Path("structurer/output")

    output_root = Path("classifier/output")

    for patient in input_root.iterdir():

        if not patient.is_dir():
            continue

        print("=" * 60)
        print(patient.name)
        print("=" * 60)

        classify_patient(
            patient,
            output_root / patient.name,
        )

    print("\nDone.")