from pathlib import Path
import ollama

from llm.models import MODEL_NAME
from structurer.prompts import MEDICAL_STRUCTURE_PROMPT


INPUT_FOLDER = Path("llm/output")
OUTPUT_FOLDER = Path("structurer/output")

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def structure_page(text):

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",
                "content": MEDICAL_STRUCTURE_PROMPT
            },

            {
                "role": "user",
                "content": text
            }

        ]

    )

    return response["message"]["content"]


def process_patient(patient_folder):

    print()
    print("=" * 70)
    print(f"Structuring : {patient_folder.name}")
    print("=" * 70)

    page_files = sorted(
        patient_folder.glob("page_*_clean.txt")
    )

    if not page_files:

        print("No cleaned pages found.")
        return

    output_patient = OUTPUT_FOLDER / patient_folder.name

    output_patient.mkdir(
        parents=True,
        exist_ok=True
    )

    complete_document = ""

    total_pages = len(page_files)

    for index, page_file in enumerate(page_files, start=1):

        print(f"[{index}/{total_pages}] {page_file.name}")

        with open(
            page_file,
            "r",
            encoding="utf8"
        ) as f:

            cleaned_text = f.read()

        structured = structure_page(cleaned_text)

        output_page = output_patient / (
            page_file.stem.replace(
                "_clean",
                "_structured"
            ) + ".md"
        )

        with open(
            output_page,
            "w",
            encoding="utf8"
        ) as f:

            f.write(structured)

        complete_document += "\n"
        complete_document += "#" * 80
        complete_document += "\n"
        complete_document += page_file.stem.upper()
        complete_document += "\n"
        complete_document += "#" * 80
        complete_document += "\n\n"
        complete_document += structured
        complete_document += "\n"

    final_output = output_patient / (
        f"{patient_folder.name}_structured_complete.md"
    )

    with open(
        final_output,
        "w",
        encoding="utf8"
    ) as f:

        f.write(complete_document)

    print()
    print(f"Saved : {final_output.name}")


def main():

    patient_folders = [

        folder

        for folder in INPUT_FOLDER.iterdir()

        if folder.is_dir()

    ]

    if not patient_folders:

        print("No cleaned patient folders found.")
        return

    print("=" * 70)
    print("Medical Document Structurer")
    print("=" * 70)

    for patient in patient_folders:

        process_patient(patient)

    print()
    print("=" * 70)
    print("Structuring Completed Successfully")
    print("=" * 70)


if __name__ == "__main__":

    main()