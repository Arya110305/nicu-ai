"""
Prompt package exports.
"""

from .evidence import (
    EVIDENCE_SYSTEM_PROMPT,
    build_evidence_prompt,
)

from .report import (
    REPORT_SYSTEM_PROMPT,
    build_report_prompt,
)

from .template import (
    TEMPLATE_SYSTEM_PROMPT,
    build_template_prompt,
    validate_template_name,
)

from .discharge import (
    DISCHARGE_SYSTEM_PROMPT,
    build_discharge_prompt,
)

__all__ = [
    "EVIDENCE_SYSTEM_PROMPT",
    "REPORT_SYSTEM_PROMPT",
    "TEMPLATE_SYSTEM_PROMPT",
    "DISCHARGE_SYSTEM_PROMPT",
    "build_evidence_prompt",
    "build_report_prompt",
    "build_template_prompt",
    "build_discharge_prompt",
    "validate_template_name",
]