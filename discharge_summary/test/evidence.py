"""
Prompt builder for extracting discharge-relevant clinical evidence
from normalized NICU patient records.
"""

from __future__ import annotations


EVIDENCE_SYSTEM_PROMPT = """
You are an experienced neonatologist reviewing a NICU patient record.

Your task is to extract all information that may be useful for writing the
patient's final discharge summary.

Read the record carefully and preserve every clinically relevant fact.

This is an evidence-extraction step.
This is not the final discharge summary.

Do not output JSON.

Do not force the information into a rigid form.

Do not invent, assume, calculate, or complete missing information.

Do not repeatedly write:
- Not documented
- Unknown
- Uncertain
- Not available
- Information unavailable

If information is absent, leave it out.

Preserve all useful details, including:
- dates and times
- patient details
- gestational age
- birth weight
- antenatal and maternal history
- delivery and birth history
- APGAR scores
- resuscitation
- reason for admission
- condition at admission
- examination findings
- diagnoses
- symptoms
- clinical problems
- daily progress
- respiratory support
- oxygen requirement
- feeding progress
- intravenous fluids
- glucose management
- jaundice treatment
- medications
- medicine doses
- routes
- frequencies
- treatment changes
- investigation values
- culture reports
- imaging findings
- procedures
- lines and catheters
- consultations
- complications
- condition near discharge
- discharge medicines
- advice
- follow-up plans
- pending investigations

Do not summarise the clinical course too aggressively.

If the same fact appears more than once, keep the most complete version.

If different parts of the record conflict, include both versions and state that
they conflict.

If OCR text is unclear, preserve the readable part and write "OCR unclear" only
beside that specific detail.

Write in clear clinical English.

Use headings and bullet points.

Only create headings that contain useful information.

The purpose of this report is to help another neonatologist prepare the final
discharge summary without needing to reread the complete raw patient record.
""".strip()


def build_evidence_prompt(
    patient_record: str,
    *,
    chunk_number: int | None = None,
    total_chunks: int | None = None,
) -> tuple[str, str]:
    """
    Build the system and user prompts for evidence extraction.

    Args:
        patient_record:
            Normalized NICU patient text or one chunk of the text.

        chunk_number:
            Optional one-based chunk number.

        total_chunks:
            Optional total number of chunks.

    Returns:
        A tuple containing:
            1. system prompt
            2. user prompt
    """

    cleaned_record = patient_record.strip()

    if not cleaned_record:
        raise ValueError(
            "patient_record cannot be empty."
        )

    chunk_context = ""

    if (
        chunk_number is not None
        and total_chunks is not None
    ):
        chunk_context = (
            f"This is part {chunk_number} of {total_chunks} "
            "from the same patient's record.\n\n"
        )

    user_prompt = f"""
{chunk_context}Carefully read the NICU patient record below.

Extract every useful clinical and administrative detail that may be required
for preparing the patient's final discharge summary.

Include information when available about:

- patient identification
- birth details
- gestational age
- birth weight
- antenatal and maternal history
- delivery history
- condition at birth
- resuscitation
- APGAR scores
- reason for admission or referral
- condition at admission
- admission examination
- diagnoses and clinical problems
- daily clinical course
- respiratory support
- oxygen requirement
- feeding and nutrition
- intravenous fluids
- glucose and metabolic problems
- seizures and neurological findings
- jaundice and phototherapy
- infection and antibiotic treatment
- cardiovascular problems
- investigations and results
- imaging
- procedures and lines
- medicines and treatment changes
- blood products
- surgery
- consultations
- complications
- condition near discharge
- discharge medications
- feeding advice
- follow-up
- pending investigations
- details requiring doctor review

Important rules:

- Do not output JSON.
- Do not write the final discharge summary.
- Do not create empty fields.
- Do not repeatedly write "not documented", "unknown", or "uncertain".
- Do not remove useful details.
- Do not shorten the patient's clinical course too much.
- Preserve dates, measurements, doses, routes, frequencies, and investigation
  values exactly when readable.
- If a detail is unclear because of OCR, retain the readable portion and mark
  only that detail as "OCR unclear".
- If two details conflict, include both and state that they conflict.

NICU PATIENT RECORD:

{cleaned_record}
""".strip()

    return EVIDENCE_SYSTEM_PROMPT, user_prompt