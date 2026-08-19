"""
Prompt builder for generating the final NICU discharge summary.
"""

from __future__ import annotations


DISCHARGE_SYSTEM_PROMPT = """
You are an experienced consultant neonatologist.

Your task is to prepare a professional hospital discharge summary using:

1. The supplied hospital discharge template.
2. The clinical evidence report.

This is the final discharge summary.

Rules:

- Follow the hospital template exactly.
- Preserve the section order.
- Use only information supported by the evidence report.
- Never invent facts.
- Never guess missing values.
- Do not calculate values that are not explicitly documented.
- Leave unsupported fields blank rather than writing "Unknown" or "Not documented".
- Preserve dates, medicine names, doses, routes, frequencies, laboratory values,
  and investigation findings exactly when documented.
- Resolve duplicate information by using the most complete documented version.
- If conflicting information exists, use the version that is best supported by
  the evidence. If the conflict cannot be resolved, leave the field blank.
- Write in clear professional medical English suitable for inclusion in the
  patient's hospital record.
- Do not output JSON.
- Do not add explanations.
- Do not add notes outside the discharge summary.
- Return only the completed discharge summary.
""".strip()


def build_discharge_prompt(
    evidence_report: str,
    template_text: str,
) -> tuple[str, str]:
    """
    Build the prompts for generating the final discharge summary.

    Args:
        evidence_report:
            Merged clinical evidence.

        template_text:
            Hospital discharge template.

    Returns:
        (system_prompt, user_prompt)
    """

    evidence = evidence_report.strip()
    template = template_text.strip()

    if not evidence:
        raise ValueError(
            "evidence_report cannot be empty."
        )

    if not template:
        raise ValueError(
            "template_text cannot be empty."
        )

    user_prompt = f"""
Using the evidence report below, complete the hospital discharge summary.

Instructions:

- Follow the hospital template exactly.
- Fill only fields supported by the evidence.
- Leave unsupported fields blank.
- Do not invent information.
- Preserve medical terminology.
- Preserve chronology where appropriate.
- Preserve medication names, doses, routes, frequencies and durations exactly.
- Preserve investigation values exactly.
- Preserve dates exactly when documented.
- Produce a discharge summary ready for direct clinical use.

=========================
HOSPITAL TEMPLATE
=========================

{template}

=========================
CLINICAL EVIDENCE REPORT
=========================

{evidence}
""".strip()

    return DISCHARGE_SYSTEM_PROMPT, user_prompt