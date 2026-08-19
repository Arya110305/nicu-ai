"""
from pathlib import Path

from llm.models import MODEL_NAME
from .prompts import CLINICAL_SUMMARY_PROMPT


def generate_clinical_summary(patient_record: str) -> str:
    
    Generate a clinician-style discharge summary.
    

    prompt = CLINICAL_SUMMARY_PROMPT.format(
        record=patient_record
    )

    summary = MODEL_NAME(prompt)

    return summary.strip()


def save_summary(output_folder: Path, summary: str):

    output_file = output_folder / "clinical_summary_v1.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Saved summary -> {output_file}")
"""