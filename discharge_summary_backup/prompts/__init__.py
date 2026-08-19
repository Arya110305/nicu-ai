"""
NICU discharge-summary prompt package.

This file exposes all important prompts and prompt-building functions so that
generator.py can import them from one place.
"""

from .system_prompts import (
    CLINICAL_SYSTEM_PROMPT,
    EVIDENCE_EXTRACTION_SYSTEM_PROMPT,
    EVIDENCE_REPORT_SYSTEM_PROMPT,
    TEMPLATE_SELECTION_SYSTEM_PROMPT,
)

from .extraction_prompts import (
    build_chunk_extraction_prompt,
    build_evidence_consolidation_prompt,
)

from .template_prompt import (
    PRETERM_NNJ,
    PRETERM_RD_TTNB,
    TERM_NNJ,
    TERM_RD_TTNB,
    VALID_TEMPLATE_NAMES,
    TemplateError,
    build_template_selection_prompt,
    get_template_path,
    load_template_text,
    validate_all_templates,
    validate_template_selection,
)

from .evidence_report_prompt import build_evidence_report_prompt

from .discharge_prompt import build_final_discharge_prompt


__all__ = [
    "CLINICAL_SYSTEM_PROMPT",
    "EVIDENCE_EXTRACTION_SYSTEM_PROMPT",
    "EVIDENCE_REPORT_SYSTEM_PROMPT",
    "TEMPLATE_SELECTION_SYSTEM_PROMPT",
    "PRETERM_NNJ",
    "PRETERM_RD_TTNB",
    "TERM_NNJ",
    "TERM_RD_TTNB",
    "VALID_TEMPLATE_NAMES",
    "TemplateError",
    "build_chunk_extraction_prompt",
    "build_evidence_consolidation_prompt",
    "build_template_selection_prompt",
    "validate_template_selection",
    "get_template_path",
    "load_template_text",
    "validate_all_templates",
    "build_evidence_report_prompt",
    "build_final_discharge_prompt",
]