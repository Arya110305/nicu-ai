"""
Controlled clinical terminology normalization for NICU records.

This module performs deterministic corrections for high-confidence OCR
variants after the LLM normalization step.

Only terms with a clear and clinically safe interpretation should be added
here. Ambiguous text must remain unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalTermRule:
    """
    One controlled terminology-normalization rule.
    """

    canonical_term: str
    patterns: tuple[str, ...]


CLINICAL_TERM_RULES: tuple[ClinicalTermRule, ...] = (
    ClinicalTermRule(
        canonical_term="Levetiracetam",
        patterns=(
            r"\blevetiracetam\b",
            r"\blevetiracetem\b",
            r"\blevetiracitam\b",
            r"\blevetiracitam\b",
            r"\blevipil\b",
            r"\blevipl\b",
            r"\blevipill\b",
            r"\blevupil\b",
            r"\blevipal\b",
            r"\blenipil\b",
            r"\bleuipil\b",
            r"\bleuipill\b",
            r"\bleuipil\b",
            r"\ble\s+vi\s+pi\s+l\b",
            r"\ble\s*vi\s*pi\s*l\b",
        ),
    ),
    ClinicalTermRule(
        canonical_term="Phenobarbitone",
        patterns=(
            r"\bphenobarbitone\b",
            r"\bphenobarbital\b",
            r"\bphenobarbit\b",
            r"\bphenobarb\b",
            r"\bphenobarbitonc\b",
            r"\bphenobasbitone\b",
            r"\bphenobasbitonc\b",
            r"\bphenobarbiton\b",
            r"\bphenobabitone\b",
            r"\bphenobarbitione\b",
            r"\bphenoba1sitou22\b",
            r"\bpheno\s*barbitone\b",
            r"\bpheno\s*barbital\b",
            r"\bgardenal\b",
            r"\bgrardenal\b",
        ),
    ),
    ClinicalTermRule(
        canonical_term="MRI brain",
        patterns=(
            r"\bmri\s+brain\b",
            r"\bmri\s+bran\b",
            r"\bmri\s+boan\b",
            r"\bmri\s+blawn\b",
            r"\bmri\s+bram\b",
            r"\bmri\s+braim\b",
            r"\bmri\s+brin\b",
            r"\bmri\s+of\s+brain\b",
        ),
    ),
    ClinicalTermRule(
        canonical_term="EEG",
        patterns=(
            r"\beeg\b",
            r"\be6g\b",
            r"\beeg\s*nlsq\b",
            r"\beegnlsq\b",
        ),
    ),
    ClinicalTermRule(
        canonical_term="seizure",
        patterns=(
            r"\bseizure\b",
            r"\bserzure\b",
            r"\bserzenre\b",
            r"\bserzenres\b",
            r"\bseizvua\b",
            r"\bseizua\b",
            r"\bseizurea\b",
        ),
    ),
    ClinicalTermRule(
        canonical_term="seizures",
        patterns=(
            r"\bseizures\b",
            r"\bseizvures\b",
            r"\bseizureses\b",
            r"\bserzures\b",
            r"\bseizueres\b",
        ),
    ),
    ClinicalTermRule(
        canonical_term="hypoglycemia",
        patterns=(
            r"\bhypoglycemia\b",
            r"\bhypoglycaemia\b",
            r"\bhypogylcemia\b",
            r"\bhypoglycemla\b",
            r"\bhypoglyceamia\b",
        ),
    ),
    ClinicalTermRule(
        canonical_term="Cefotaxime",
        patterns=(
            r"\bcefotaxime\b",
            r"\bcefotaxim\b",
            r"\bcefotaxime calcium\b",
            r"\btaxim\b",
            r"\btaxiym\b",
        ),
    ),
    ClinicalTermRule(
        canonical_term="pyridoxine",
        patterns=(
            r"\bpyridoxine\b",
            r"\bpyedowne\b",
            r"\bpyeedowne\b",
            r"\bpyecidoxine\b",
            r"\bpyridoxin\b",
        ),
    ),
)


def normalize_spacing(text: str) -> str:
    """
    Correct safe spacing issues without changing document structure.
    """

    text = re.sub(
        r"(?i)\bMRI\s*[-:]\s*brain\b",
        "MRI brain",
        text,
    )

    text = re.sub(
        r"(?i)\bE\s*E\s*G\b",
        "EEG",
        text,
    )

    text = re.sub(
        r"(?i)\bL\s*E\s*V\s*I\s*P\s*I\s*L\b",
        "Levetiracetam",
        text,
    )

    text = re.sub(
        r"(?i)\bP\s*H\s*E\s*N\s*O\s*B\s*A\s*R\s*B\s*I\s*T\s*O\s*N\s*E\b",
        "Phenobarbitone",
        text,
    )

    return text


def apply_rule(
    text: str,
    rule: ClinicalTermRule,
) -> tuple[str, int]:
    """
    Apply all OCR patterns belonging to one canonical clinical term.

    Returns:
        Updated text and the number of replacements.
    """

    replacement_count = 0
    updated_text = text

    for pattern in rule.patterns:
        updated_text, count = re.subn(
            pattern=pattern,
            repl=rule.canonical_term,
            string=updated_text,
            flags=re.IGNORECASE,
        )

        replacement_count += count

    return updated_text, replacement_count


def normalize_clinical_terms(
    text: str,
) -> tuple[str, dict[str, int]]:
    """
    Normalize high-confidence clinical terminology.

    The function returns:

    1. The corrected text.
    2. A dictionary showing how many replacements were made for each
       canonical term.

    No clinical information is removed.
    """

    if not isinstance(text, str):
        raise TypeError(
            "Clinical terminology input must be a string."
        )

    if not text.strip():
        return text, {}

    normalized_text = normalize_spacing(text)
    replacement_summary: dict[str, int] = {}

    for rule in CLINICAL_TERM_RULES:
        normalized_text, count = apply_rule(
            text=normalized_text,
            rule=rule,
        )

        if count > 0:
            replacement_summary[rule.canonical_term] = (
                replacement_summary.get(rule.canonical_term, 0)
                + count
            )

    return normalized_text, replacement_summary


def print_replacement_summary(
    replacement_summary: dict[str, int],
    prefix: str = "    ",
) -> None:
    """
    Print deterministic terminology corrections for progress reporting.
    """

    if not replacement_summary:
        print(f"{prefix}Terminology mapping: no replacements")
        return

    total_replacements = sum(replacement_summary.values())

    print(
        f"{prefix}Terminology mapping: "
        f"{total_replacements} replacement(s)"
    )

    for term, count in sorted(
        replacement_summary.items(),
        key=lambda item: item[0].lower(),
    ):
        print(f"{prefix}  - {term}: {count}")