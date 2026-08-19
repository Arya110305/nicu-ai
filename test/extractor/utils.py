from pathlib import Path


# --------------------------------------------------
# Read Merged Section File
# --------------------------------------------------

def load_section(section_folder: Path):

    section_name = section_folder.name

    merged_file = section_folder / f"{section_name.lower()}.md"

    if not merged_file.exists():
        return None

    return merged_file.read_text(encoding="utf8")


# --------------------------------------------------
# Get All Sections
# --------------------------------------------------

def get_sections(patient_folder: Path):

    sections = []

    for folder in sorted(patient_folder.iterdir()):

        if folder.is_dir():
            sections.append(folder)

    return sections


# --------------------------------------------------
# Save Text Output
# --------------------------------------------------

def save_text(output_folder: Path,
              section_name: str,
              text: str):

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    outfile = output_folder / f"{section_name.lower()}.txt"

    with open(outfile, "w", encoding="utf8") as f:
        f.write(text)