"""
Hospital template loading and template-selection prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

try:
    from docx import Document
except ImportError as exc:
    raise ImportError(
        "python-docx is required. Install it using: "
        "python3 -m pip install python-docx"
    ) from exc


PRETERM_NNJ: Final[str] = "PRETERM_NNJ"
PRETERM_RD_TTNB: Final[str] = "PRETERM_RD_TTNB"
TERM_NNJ: Final[str] = "TERM_NNJ"
TERM_RD_TTNB: Final[str] = "TERM_RD_TTNB"

VALID_TEMPLATE_NAMES: Final[tuple[str, ...]] = (
    PRETERM_NNJ,
    PRETERM_RD_TTNB,
    TERM_NNJ,
    TERM_RD_TTNB,
)


PROMPTS_DIRECTORY: Final[Path] = Path(__file__).resolve().parent
DISCHARGE_SUMMARY_DIRECTORY: Final[Path] = PROMPTS_DIRECTORY.parent
TEMPLATE_DIRECTORY: Final[Path] = (
    DISCHARGE_SUMMARY_DIRECTORY / "templates"
)

TEMPLATE_FILES: Final[dict[str, Path]] = {
    PRETERM_NNJ: (
        TEMPLATE_DIRECTORY / "PRETERM DISCHARGE NNJ TEMPLATE.docx"
    ),
    PRETERM_RD_TTNB: (
        TEMPLATE_DIRECTORY / "PRETERM DISCHARGE RD TTNB TEMPLATE.docx"
    ),
    TERM_NNJ: (
        TEMPLATE_DIRECTORY / "TERM DISCHARGE NNJ TEMPLATE.docx"
    ),
    TERM_RD_TTNB: (
        TEMPLATE_DIRECTORY / "TERM DISCHARGE RD TTNB TEMPLATE.docx"
    ),
}


class TemplateError(RuntimeError):
    """Raised when a hospital DOCX template cannot be loaded."""


def get_template_path(template_name: str) -> Path:
    """Return the path of a selected hospital template."""

    cleaned_name = template_name.strip().upper()

    if cleaned_name not in TEMPLATE_FILES:
        valid_names = ", ".join(VALID_TEMPLATE_NAMES)

        raise ValueError(
            f"Invalid template name '{template_name}'. "
            f"Expected one of: {valid_names}"
        )

    template_path = TEMPLATE_FILES[cleaned_name]

    if not template_path.exists():
        raise TemplateError(
            f"Hospital template was not found: {template_path}"
        )

    if not template_path.is_file():
        raise TemplateError(
            f"Hospital template path is not a file: {template_path}"
        )

    return template_path


def _extract_paragraph_text(element) -> str:
    """Extract text from a DOCX paragraph XML element."""

    text_parts: list[str] = []

    for node in element.iter():
        tag_name = node.tag.rsplit("}", maxsplit=1)[-1]

        if tag_name == "t" and node.text:
            text_parts.append(node.text)

    return "".join(text_parts).strip()


def _extract_table_rows(table_element) -> list[str]:
    """Extract rows from a DOCX table XML element."""

    extracted_rows: list[str] = []

    for row in table_element.iter():
        row_tag = row.tag.rsplit("}", maxsplit=1)[-1]

        if row_tag != "tr":
            continue

        cells: list[str] = []

        for cell in row.iterchildren():
            cell_tag = cell.tag.rsplit("}", maxsplit=1)[-1]

            if cell_tag != "tc":
                continue

            text_parts: list[str] = []

            for node in cell.iter():
                node_tag = node.tag.rsplit("}", maxsplit=1)[-1]

                if node_tag == "t" and node.text:
                    text_parts.append(node.text.strip())

            cell_text = " ".join(
                part for part in text_parts if part
            ).strip()

            cells.append(cell_text)

        if cells:
            extracted_rows.append(" | ".join(cells))

    return extracted_rows


def load_template_text(template_name: str) -> str:
    """
    Load paragraphs and tables from a hospital DOCX template.
    """

    template_path = get_template_path(template_name)

    try:
        document = Document(str(template_path))
    except Exception as exc:
        raise TemplateError(
            f"Could not open hospital template: {template_path}"
        ) from exc

    extracted_blocks: list[str] = []

    for child in document.element.body.iterchildren():
        tag_name = child.tag.rsplit("}", maxsplit=1)[-1]

        if tag_name == "p":
            paragraph_text = _extract_paragraph_text(child)

            if paragraph_text:
                extracted_blocks.append(paragraph_text)

        elif tag_name == "tbl":
            table_rows = _extract_table_rows(child)
            extracted_blocks.extend(table_rows)

    template_text = "\n".join(extracted_blocks).strip()

    if not template_text:
        raise TemplateError(
            f"No readable content was found in: {template_path}"
        )

    return template_text


def build_template_selection_prompt(
    clinical_evidence: str,
) -> str:
    """Build the prompt used to select one of four templates."""

    evidence = clinical_evidence.strip()

    if not evidence:
        raise ValueError(
            "Clinical evidence cannot be empty for template selection."
        )

    return f"""
Select the closest hospital discharge-summary template.

VALID OUTPUTS:

{PRETERM_NNJ}
{PRETERM_RD_TTNB}
{TERM_NNJ}
{TERM_RD_TTNB}
TEMPLATE_UNCERTAIN

RULES:

1. PRETERM means gestational age below 37 completed weeks.

2. TERM means gestational age of 37 completed weeks or more.

3. NNJ means neonatal jaundice is the principal condition represented by the
   hospital course.

4. RD_TTNB means respiratory distress, delayed transition or transient
   tachypnoea of the newborn is the principal condition represented by the
   hospital course.

5. Do not infer gestational age from birth weight.

6. Do not infer a diagnosis only from treatment.

7. If gestational category cannot be established, return:
   TEMPLATE_UNCERTAIN

8. If the principal condition cannot be distinguished, return:
   TEMPLATE_UNCERTAIN

9. Other diagnoses may still be added to the selected base template.

10. Return only the template name.

PATIENT CLINICAL EVIDENCE:

--- BEGIN CLINICAL EVIDENCE ---

{evidence}

--- END CLINICAL EVIDENCE ---
""".strip()


def validate_template_selection(model_output: str) -> str:
    """Validate the template name returned by the model."""

    cleaned_output = (
        model_output
        .strip()
        .upper()
        .replace("`", "")
        .replace("*", "")
    )

    first_line = cleaned_output.splitlines()[0].strip()

    valid_outputs = set(VALID_TEMPLATE_NAMES)
    valid_outputs.add("TEMPLATE_UNCERTAIN")

    if first_line not in valid_outputs:
        raise ValueError(
            "Template-selection model returned an invalid value: "
            f"{model_output!r}"
        )

    return first_line


def validate_all_templates() -> dict[str, Path]:
    """Verify that all four hospital templates exist and are readable."""

    validated_templates: dict[str, Path] = {}

    for template_name in VALID_TEMPLATE_NAMES:
        template_path = get_template_path(template_name)
        load_template_text(template_name)
        validated_templates[template_name] = template_path

    return validated_templates