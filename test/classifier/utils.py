from pathlib import Path
from collections import defaultdict
import json

ALLOWED_LABELS = {
    "PATIENT_INFORMATION",
    "BIRTH_DETAILS",
    "MATERNAL_HISTORY",
    "ADMISSION",
    "NICU_COURSE",
    "VITALS",
    "LABORATORY",
    "MEDICATIONS",
    "PROCEDURES",
    "DAILY_PROGRESS",
    "SCREENING",
    "DISCHARGE",
    "OTHER",
}

# --------------------------------------------------
# Read One Structured Page
# --------------------------------------------------

def load_page(page_path: Path) -> str:
    with open(page_path, "r", encoding="utf8") as f:
        return f.read()


# --------------------------------------------------
# Read All Structured Pages
# --------------------------------------------------

def load_patient_pages(patient_folder: Path):
    return sorted(patient_folder.glob("page_*_structured.md"))


# --------------------------------------------------
# Parse LLM Labels
# --------------------------------------------------

def parse_labels(response: str):

    labels = []

    seen = set()

    for line in response.splitlines():

        label = line.strip().upper()

        if not label:
            continue

        if label not in ALLOWED_LABELS:
            print(f"[WARNING] Unknown label ignored: {label}")
            continue

        if label not in seen:
            labels.append(label)
            seen.add(label)

    if not labels:
        labels.append("OTHER")

    return labels


# --------------------------------------------------
# Group Pages By Label
# --------------------------------------------------

def group_pages(classifications):
    grouped = defaultdict(list)

    for page_name, page_text, labels in classifications:

        for label in labels:

            grouped[label].append(
                {
                    "page": page_name,
                    "text": page_text,
                }
            )

    return grouped


# --------------------------------------------------
# Save Sections
# --------------------------------------------------

def save_grouped_sections(output_folder: Path, grouped):

    output_folder.mkdir(parents=True, exist_ok=True)

    for label, pages in grouped.items():

        section_folder = output_folder / label
        section_folder.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------
        # Save each page individually
        # --------------------------------------------

        for page in pages:

            page_file = section_folder / page["page"]

            with open(page_file, "w", encoding="utf8") as f:
                f.write(page["text"])

        # --------------------------------------------
        # Save merged section
        # --------------------------------------------

        merged_file = section_folder / f"{label.lower()}.md"

        with open(merged_file, "w", encoding="utf8") as f:

            f.write(f"# {label}\n\n")

            for page in pages:

                f.write("=" * 80 + "\n")
                f.write(f"PAGE : {page['page']}\n")
                f.write("=" * 80 + "\n\n")

                f.write(page["text"])
                f.write("\n\n")

        # --------------------------------------------
        # Save metadata
        # --------------------------------------------

        metadata = {
            "section": label,
            "page_count": len(pages),
            "pages": [page["page"] for page in pages],
        }

        metadata_file = section_folder / "metadata.json"

        with open(metadata_file, "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=4)


# --------------------------------------------------
# Save Page Labels
# --------------------------------------------------

def save_page_labels(output_folder: Path, classifications):

    output_folder.mkdir(parents=True, exist_ok=True)

    file = output_folder / "page_labels.txt"

    with open(file, "w", encoding="utf8") as f:

        for page_name, _, labels in classifications:

            f.write(page_name + "\n")

            for label in labels:
                f.write(f"  - {label}\n")

            f.write("\n")