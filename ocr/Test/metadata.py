"""
import json
from pathlib import Path


def save_page_metadata(
    metadata_folder: Path,
    pdf_name: str,
    page_number: int,
    page_result: dict,
):
    
    Saves the raw OCR output of one page.

    Output example:

    metadata/
        BO_VAGDEVI/
            page_001.json
            page_002.json
            ...
    

    pdf_folder = metadata_folder / pdf_name

    pdf_folder.mkdir(parents=True, exist_ok=True)

    output_file = pdf_folder / f"page_{page_number:03}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            page_result,
            f,
            indent=4,
            ensure_ascii=False,
        )
"""