"""
Utility functions for cleaning normalized clinical text and extracting
relevant blocks for discharge-summary generation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final


EMPTY_VALUES: Final[set[str]] = {
    "",
    "0",
    "0.0",
    "-",
    "--",
    "---",
    "_",
    "__",
    "___",
    "empty",
    "none",
    "nil",
    "null",
    "unknown",
    "not known",
    "not available",
    "not applicable",
    "not documented",
    "none documented",
    "no data",
    "no information",
    "missing",
    "blank",
    "[blank]",
    "[missing]",
    "[uncertain]",
    "[uncertain: missing]",
    "[uncertain: blank]",
    "n/a",
    "na",
}

NOISE_PATTERNS: Final[tuple[str, ...]] = (
    r"^\s*page[\s_:.-]*\d+\s*$",
    r"^\s*page[_\s-]*\d+\s*$",
    r"^\s*generated\s+on\b.*$",
    r"^\s*printed\s+on\b.*$",
    r"^\s*print\s+time\b.*$",
    r"^\s*hospital\s+copy\s*$",
    r"^\s*patient\s+copy\s*$",
    r"^\s*confidential\s*$",
    r"^\s*barcode\s*$",
    r"^\s*qr\s*code\s*$",
    r"^\s*signature\s*:\s*$",
    r"^\s*employee\s+id\s*:\s*$",
    r"^\s*remarks\s*:\s*$",
    r"^\s*name\s*:\s*$",
    r"^\s*date\s*:\s*$",
    r"^\s*time\s*:\s*$",
    r"^\s*ref\.?\s*interval\s*:\s*$",
    r"^\s*method\s*:\s*$",
    r"^\s*interpretation\s*:\s*$",
    r"^\s*[=_*#-]{4,}\s*$",
)

BILLING_KEYWORDS: Final[tuple[str, ...]] = (
    "billing record",
    "pcc/billing",
    "billing staff",
    "charge sheet",
    "tariff",
    "rate card",
)

CHECKLIST_KEYWORDS: Final[tuple[str, ...]] = (
    "requisition form",
    "laboratory investigations form",
    "procedure checklist",
    "available procedures",
    "morning shift:",
    "evening shift:",
    "night shift:",
)

PATIENT_KEYWORDS: Final[tuple[str, ...]] = (
    "patient name",
    "baby name",
    "b/o",
    "uhid",
    "mr no",
    "mr number",
    "ipd",
    "ip no",
    "ip number",
    "date of birth",
    "dob",
    "gender",
    "sex",
    "gestational age",
    "birth weight",
    "admission weight",
    "discharge weight",
    "current weight",
    "length",
    "head circumference",
    "blood group",
    "mother name",
    "father name",
    "contact number",
    "date of admission",
    "date of discharge",
    "inborn",
    "outborn",
)

ADMISSION_KEYWORDS: Final[tuple[str, ...]] = (
    "antenatal history",
    "maternal history",
    "perinatal history",
    "birth history",
    "delivery",
    "ls cs",
    "lscs",
    "normal vaginal delivery",
    "cry immediately",
    "apgar",
    "resuscitation",
    "reason for admission",
    "reason for referral",
    "referred",
    "admission examination",
    "examination at admission",
    "diagnosis",
    "seizure",
    "hypoglycemia",
    "hypoglycaemia",
    "respiratory distress",
    "jaundice",
    "infection",
    "sepsis",
)

COURSE_KEYWORDS: Final[tuple[str, ...]] = (
    "clinical condition",
    "clinical course",
    "hospital course",
    "course in hospital",
    "daily progress",
    "progress note",
    "nicu course",
    "seizure",
    "no further seizures",
    "last seizure",
    "hypoglycemia",
    "hypoglycaemia",
    "grbs",
    "gir",
    "dextrose",
    "oxygen",
    "room air",
    "cpap",
    "hfnc",
    "lfnc",
    "ventilation",
    "intubation",
    "extubation",
    "feeding",
    "feeds",
    "npo",
    "ogt",
    "spoon feeds",
    "breast milk",
    "formula",
    "hemodynamically stable",
    "haemodynamically stable",
    "activity improved",
    "sensorium",
    "neurology",
    "consultation",
)

INVESTIGATION_KEYWORDS: Final[tuple[str, ...]] = (
    "laboratory report",
    "biochemistry",
    "hematology",
    "haematology",
    "complete blood picture",
    "cbc",
    "cbp",
    "hemoglobin",
    "haemoglobin",
    "platelet",
    "wbc",
    "crp",
    "creatinine",
    "urea",
    "bun",
    "sodium",
    "potassium",
    "calcium",
    "bilirubin",
    "liver function",
    "renal function",
    "blood gas",
    "culture",
    "blood culture",
    "urine culture",
    "cue",
    "mri",
    "eeg",
    "neurosonogram",
    "nsg",
    "x-ray",
    "xray",
    "echo",
    "echocardiography",
    "ammonia",
    "lactate",
)

TREATMENT_KEYWORDS: Final[tuple[str, ...]] = (
    "medication",
    "medicine",
    "treatment",
    "inj.",
    "inj ",
    "injection",
    "syrup",
    "tablet",
    "tab.",
    "drops",
    "phenobarbitone",
    "phenobarbital",
    "gardenal",
    "levetiracetam",
    "levipil",
    "cefotaxime",
    "taxim",
    "midazolam",
    "calcium gluconate",
    "calcimax",
    "pyridoxine",
    "vitamin k",
    "optineuron",
    "dextrose",
    "iv fluids",
    "antibiotic",
    "anticonvulsant",
    "phototherapy",
    "transfusion",
    "uvc",
    "uac",
    "picc",
    "foley",
    "cannula",
)

DISCHARGE_KEYWORDS: Final[tuple[str, ...]] = (
    "condition at discharge",
    "discharge condition",
    "discharge medication",
    "medications at discharge",
    "advice at discharge",
    "discharge advice",
    "follow-up",
    "follow up",
    "review after",
    "review on",
    "hearing screening",
    "bera",
    "rop",
    "thyroid profile",
    "immunization",
    "immunisation",
    "warm care",
    "danger signs",
    "feeding advice",
    "discharged",
    "comfortable at room air",
    "tolerating feeds",
)


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not text:
        raise ValueError("Input file is empty.")

    return text


def write_text_file(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text.strip() + "\n",
        encoding="utf-8",
    )


def patient_name(file: Path) -> str:
    name = file.stem

    suffixes = (
        "_normalized_complete",
        "_normalized",
        "_structured_complete",
        "_structured",
    )

    lowered = name.lower()

    for suffix in suffixes:
        if lowered.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name.strip()


def clean_model_output(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        cleaned = "\n".join(lines)

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    if cleaned.upper() == "EMPTY":
        return ""

    return cleaned


def normalize_line(line: str) -> str:
    line = line.replace("\t", " ")
    line = line.replace("\u00a0", " ")
    line = line.replace("–", "-")
    line = line.replace("—", "-")

    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    line = re.sub(
        r"\s+([,:;.)])",
        r"\1",
        line,
    )

    line = re.sub(
        r"([(])\s+",
        r"\1",
        line,
    )

    return line.strip()


def is_noise_line(line: str) -> bool:
    normalized = normalize_line(line)
    lowered = normalized.lower().strip()

    if lowered in EMPTY_VALUES:
        return True

    if not lowered:
        return True

    if re.fullmatch(
        r"[\W_]+",
        lowered,
    ):
        return True

    if re.fullmatch(
        r"(?:0|0\.0|00|000)",
        lowered,
    ):
        return True

    if re.fullmatch(
        r".*:\s*(?:0|0\.0|empty|none|nil|unknown|n/?a|not documented|not available|\[blank\]|\[missing\]|\[uncertain.*\])\s*$",
        lowered,
    ):
        return True

    if "[uncertain:" in lowered:
        return True

    if "[blank]" in lowered:
        return True

    if "[missing]" in lowered:
        return True

    for pattern in NOISE_PATTERNS:
        if re.match(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def is_billing_or_checklist_line(line: str) -> bool:
    lowered = line.lower()

    if any(
        keyword in lowered
        for keyword in BILLING_KEYWORDS
    ):
        return True

    if any(
        keyword in lowered
        for keyword in CHECKLIST_KEYWORDS
    ):
        return True

    if re.match(
        r"^\s*o\s+[a-z]",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(
        r"^\s*\d+\s+(arterial|ascitic|bone marrow|central line|cpr|cvp|dressing|intubation|foleys|hemodialysis|icd|iv cannulation|lumbar|tracheostomy|nebulization|picc|pigtail|pleural|ryles|sedation|surfactant)",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def remove_duplicate_lines(
    lines: list[str],
) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for line in lines:
        key = re.sub(
            r"\s+",
            " ",
            line.strip().lower(),
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        output.append(line)

    return output


def clean_clinical_text(text: str) -> str:
    raw_lines = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    ).splitlines()

    cleaned_lines: list[str] = []

    skip_billing_block = False

    for raw_line in raw_lines:
        line = normalize_line(raw_line)

        if not line:
            continue

        lowered = line.lower()

        if any(
            keyword in lowered
            for keyword in BILLING_KEYWORDS
        ):
            skip_billing_block = True
            continue

        if skip_billing_block:
            if is_probable_section_heading(line):
                skip_billing_block = False
            else:
                continue

        if is_noise_line(line):
            continue

        if is_billing_or_checklist_line(line):
            continue

        cleaned_lines.append(line)

    cleaned_lines = remove_duplicate_lines(
        cleaned_lines
    )

    return "\n".join(
        cleaned_lines
    ).strip()


def is_probable_section_heading(line: str) -> bool:
    cleaned = line.strip()

    if not cleaned:
        return False

    if len(cleaned) > 120:
        return False

    if cleaned.endswith(":"):
        return True

    if cleaned.startswith("**") and cleaned.endswith("**"):
        return True

    letters = [
        character
        for character in cleaned
        if character.isalpha()
    ]

    if letters and all(
        character.isupper()
        for character in letters
    ):
        return True

    return False


def get_context_lines(
    lines: list[str],
    index: int,
    before: int = 2,
    after: int = 5,
) -> list[str]:
    start = max(
        0,
        index - before,
    )

    end = min(
        len(lines),
        index + after + 1,
    )

    return lines[start:end]


def extract_keyword_block(
    text: str,
    keywords: tuple[str, ...],
    *,
    before: int = 2,
    after: int = 6,
    maximum_characters: int = 18_000,
) -> str:
    lines = text.splitlines()

    selected_lines: list[str] = []

    for index, line in enumerate(lines):
        lowered = line.lower()

        if any(
            keyword in lowered
            for keyword in keywords
        ):
            selected_lines.extend(
                get_context_lines(
                    lines,
                    index,
                    before=before,
                    after=after,
                )
            )

    selected_lines = remove_duplicate_lines(
        [
            normalize_line(line)
            for line in selected_lines
            if not is_noise_line(line)
            and not is_billing_or_checklist_line(line)
        ]
    )

    result = "\n".join(
        selected_lines
    ).strip()

    if len(result) > maximum_characters:
        result = result[:maximum_characters]

    return result


def extract_patient_block(text: str) -> str:
    return extract_keyword_block(
        text,
        PATIENT_KEYWORDS,
        before=1,
        after=4,
        maximum_characters=12_000,
    )


def extract_admission_block(text: str) -> str:
    return extract_keyword_block(
        text,
        ADMISSION_KEYWORDS,
        before=2,
        after=7,
        maximum_characters=16_000,
    )


def extract_course_block(text: str) -> str:
    return extract_keyword_block(
        text,
        COURSE_KEYWORDS,
        before=3,
        after=10,
        maximum_characters=22_000,
    )


def extract_investigation_treatment_block(
    text: str,
) -> str:
    investigation_text = extract_keyword_block(
        text,
        INVESTIGATION_KEYWORDS,
        before=2,
        after=8,
        maximum_characters=18_000,
    )

    treatment_text = extract_keyword_block(
        text,
        TREATMENT_KEYWORDS,
        before=2,
        after=8,
        maximum_characters=18_000,
    )

    combined = "\n".join(
        part
        for part in (
            investigation_text,
            treatment_text,
        )
        if part
    )

    return "\n".join(
        remove_duplicate_lines(
            combined.splitlines()
        )
    ).strip()


def extract_discharge_block(text: str) -> str:
    return extract_keyword_block(
        text,
        DISCHARGE_KEYWORDS,
        before=3,
        after=10,
        maximum_characters=16_000,
    )


def extract_generator_blocks(
    text: str,
) -> dict[str, str]:
    cleaned_text = clean_clinical_text(
        text
    )

    blocks = {
        "patient_and_birth": extract_patient_block(
            cleaned_text
        ),
        "admission_and_diagnosis": extract_admission_block(
            cleaned_text
        ),
        "hospital_course": extract_course_block(
            cleaned_text
        ),
        "investigations_and_treatment": (
            extract_investigation_treatment_block(
                cleaned_text
            )
        ),
        "discharge_and_followup": extract_discharge_block(
            cleaned_text
        ),
    }

    for key, value in blocks.items():
        if not value:
            blocks[key] = cleaned_text[:12_000]

    return blocks