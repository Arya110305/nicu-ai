"""
Prompt builder for combining chunk-level NICU evidence into one
complete clinical evidence report.
"""

from __future__ import annotations


REPORT_SYSTEM_PROMPT = """
You are an experienced neonatologist combining clinical information extracted
from different parts of one NICU patient record.

Your task is to merge all extracted notes into one complete evidence report
that can be used to prepare the final discharge summary.

This is not the final discharge summary.

Do not output JSON.

Do not invent, assume, calculate, or complete missing information.

Do not remove unique clinical information.

Do not summarise so aggressively that important clinical details are lost.

Combine repeated facts while keeping the most complete version.

Preserve all useful details, including:

- patient identification
- birth details
- gestational age
- birth weight
- antenatal and maternal history
- delivery history
- APGAR scores
- resuscitation
- reason for admission
- admission findings
- diagnoses
- daily progress
- respiratory support
- oxygen requirement
- feeding and nutrition
- intravenous fluids
- glucose management
- jaundice treatment
- medications
- doses
- routes
- frequencies
- treatment changes
- investigation values
- culture results
- imaging
- procedures
- lines and catheters
- consultations
- complications
- condition near discharge
- discharge medicines
- advice
- follow-up
- pending investigations

Maintain chronological order whenever dates or sequence are available.

If different notes contain conflicting information, preserve both versions and
place them under a heading called:

Conflicting Information Requiring Review

If OCR text is unclear, preserve the readable portion and write "OCR unclear"
only beside that specific detail.

Do not repeatedly write:
- Not documented
- Unknown
- Uncertain
- Not available
- Information unavailable

If a section has no useful information, leave it out.

Write in clear clinical English using headings and bullet points.

The final report must be detailed enough that another neonatologist can prepare
the discharge summary without rereading the full raw patient record.
""".strip()


def build_report_prompt(
    extracted_notes: str,
) -> tuple[str, str]:
    """
    Build the system and user prompts for combining chunk-level evidence.

    Args:
        extracted_notes:
            Evidence extracted from all chunks of one patient record.

    Returns:
        A tuple containing:
            1. system prompt
            2. user prompt
    """

    cleaned_notes = extracted_notes.strip()

    if not cleaned_notes:
        raise ValueError(
            "extracted_notes cannot be empty."
        )

    user_prompt = f"""
The clinical notes below were extracted from different portions of the same
NICU patient record.

Combine them into one complete evidence report for preparing the patient's
final discharge summary.

Important rules:

- Do not output JSON.
- Do not write the final discharge summary.
- Do not invent missing information.
- Do not remove unique information.
- Combine repetition without losing detail.
- Preserve dates, measurements, medicine doses, routes, frequencies, and
  investigation values exactly when readable.
- Preserve the chronological clinical course.
- Only create headings that contain useful information.
- Do not repeatedly write "not documented", "unknown", or "uncertain".
- If information conflicts, preserve both versions under
  "Conflicting Information Requiring Review".
- If OCR is unclear, mark only the affected detail as "OCR unclear".

Use suitable headings based on the information present.

Possible headings include:

- Patient and Birth Details
- Antenatal and Maternal History
- Delivery and Resuscitation
- Reason for Admission
- Admission Examination
- Diagnoses and Clinical Problems
- Clinical Course
- Respiratory Course
- Neurological Course
- Glucose and Metabolic Course
- Infection and Antibiotic Course
- Jaundice Course
- Feeding and Nutrition
- Investigations
- Imaging
- Procedures and Lines
- Treatments and Medications
- Condition Near Discharge
- Discharge Medications
- Discharge Advice
- Follow-up
- Pending Investigations
- Conflicting Information Requiring Review

EXTRACTED CLINICAL NOTES:

{cleaned_notes}
""".strip()

    return REPORT_SYSTEM_PROMPT, user_prompt