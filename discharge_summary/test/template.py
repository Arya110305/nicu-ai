"""
Prompt builder for selecting the most suitable NICU discharge-summary template.
"""

from __future__ import annotations


ALLOWED_TEMPLATES = (
    "PRETERM_JAUNDICE",
    "PRETERM_RESPIRATORY",
    "TERM_JAUNDICE",
    "TERM_RESPIRATORY",
)


TEMPLATE_SYSTEM_PROMPT = """
You are an experienced neonatologist selecting the most suitable hospital
discharge-summary template for a NICU patient.

Choose exactly one template from the allowed list.

Base the choice only on the supplied clinical evidence.

Do not explain your reasoning.

Do not add punctuation, headings, markdown, or extra text.

Return only the exact template name.
""".strip()


def build_template_prompt(
    evidence_report: str,
) -> tuple[str, str]:
    """
    Build the system and user prompts for discharge-template selection.

    Args:
        evidence_report:
            The merged clinical evidence report for one NICU patient.

    Returns:
        A tuple containing:
            1. system prompt
            2. user prompt
    """

    cleaned_report = evidence_report.strip()

    if not cleaned_report:
        raise ValueError(
            "evidence_report cannot be empty."
        )

    allowed_template_text = "\n".join(
        f"- {template_name}"
        for template_name in ALLOWED_TEMPLATES
    )

    user_prompt = f"""
Select the most suitable discharge-summary template for this NICU patient.

Allowed templates:

{allowed_template_text}

Selection guidance:

- Use PRETERM when the baby is documented as preterm or the gestational age
  is below 37 completed weeks.
- Use TERM when the baby is documented as term or the gestational age is
  37 completed weeks or more.
- Use JAUNDICE when jaundice, hyperbilirubinaemia, phototherapy, or bilirubin
  management is the main reason for admission or the main clinical problem.
- Use RESPIRATORY when respiratory distress, oxygen support, CPAP,
  ventilation, apnoea, or another respiratory problem is the main reason for
  admission or the main clinical problem.
- When both jaundice and respiratory problems are present, select the template
  matching the main reason for admission and the dominant hospital course.
- Do not assume missing gestational age or diagnosis.
- If gestational-age information conflicts, use the most explicit documented
  diagnosis and clinical description.
- Return only one exact template name.
- Do not provide an explanation.

CLINICAL EVIDENCE REPORT:

{cleaned_report}
""".strip()

    return TEMPLATE_SYSTEM_PROMPT, user_prompt


def validate_template_name(template_name: str) -> str:
    """
    Validate and clean a template name returned by the model.

    Args:
        template_name:
            Raw model response.

    Returns:
        A valid template name.

    Raises:
        ValueError:
            If the response does not contain one allowed template name.
    """

    cleaned_name = template_name.strip().upper()

    if cleaned_name in ALLOWED_TEMPLATES:
        return cleaned_name

    for allowed_name in ALLOWED_TEMPLATES:
        if allowed_name in cleaned_name:
            return allowed_name

    allowed_names = ", ".join(ALLOWED_TEMPLATES)

    raise ValueError(
        f"Invalid template name: {template_name!r}. "
        f"Expected one of: {allowed_names}."
    )