"""
Prompt Builder for Section Extractor
"""

from extractor.schema import SECTION_SCHEMAS


EXTRACTOR_SYSTEM_PROMPT = """
You are an expert Neonatal (NICU) Medical Information Extraction System.

Your job is to extract ONLY the requested information from the supplied section.

Rules

1. Do NOT return JSON.
2. Do NOT return Markdown.
3. Do NOT explain.
4. Do NOT summarize unless asked.
5. Do NOT invent information.
6. If a field is missing, write:

Not Found

Return the output exactly like this:

Field Name : Value

One field per line.
"""


def build_extraction_prompt(section_name: str, section_text: str):

    fields = SECTION_SCHEMAS.get(section_name)

    if fields is None:
        raise ValueError(f"Unknown section {section_name}")

    field_list = "\n".join(
        f"- {field}"
        for field in fields
    )

    return f"""
Extract ONLY the following fields.

Fields

{field_list}

Return format

Field Name : Value

If information is missing

Field Name : Not Found

Medical Section

------------------------------------------------

{section_text}
"""