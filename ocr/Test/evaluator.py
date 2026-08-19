"""
from pathlib import Path
import fitz
import csv

INPUT_PDF = Path("ocr/input")
OCR_TEXT = Path("ocr/output")

REPORT = Path("reports")
REPORT.mkdir(exist_ok=True)

pdfs = list(INPUT_PDF.glob("*.pdf"))

if not pdfs:
    print("No PDF found.")
    exit()

for pdf in pdfs:

    txt_file = OCR_TEXT / f"{pdf.stem}.txt"

    if not txt_file.exists():
        print(f"OCR output missing for {pdf.name}")
        continue

    print(f"\nEvaluating {pdf.name}")

    # Read OCR text
    with open(txt_file, "r", encoding="utf8") as f:
        text = f.read()

    # Split using page markers
    pages = text.split("========== PAGE ")

    doc = fitz.open(pdf)

    total_pdf_pages = len(doc)

    report_rows = []

    total_words = 0

    poor_pages = 0

    empty_pages = 0

    print("-" * 50)

    for i in range(total_pdf_pages):

        if i + 1 < len(pages):
            page_text = pages[i + 1]
        else:
            page_text = ""

        words = len(page_text.split())

        total_words += words

        if words == 0:
            status = "EMPTY"
            empty_pages += 1

        elif words < 50:
            status = "POOR"
            poor_pages += 1

        else:
            status = "GOOD"

        print(f"Page {i+1:03} | Words: {words:5} | {status}")

        report_rows.append(
            {
                "Page": i + 1,
                "Words": words,
                "Status": status
            }
        )

    csv_file = REPORT / f"{pdf.stem}_ocr_report.csv"

    with open(csv_file, "w", newline="") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=["Page", "Words", "Status"]
        )

        writer.writeheader()

        writer.writerows(report_rows)

    avg_words = total_words / total_pdf_pages

    good_pages = total_pdf_pages - poor_pages - empty_pages

    coverage = (good_pages / total_pdf_pages) * 100

    print("\n========== OCR REPORT ==========")

    print(f"PDF Pages        : {total_pdf_pages}")
    print(f"Average Words    : {avg_words:.1f}")
    print(f"Good Pages       : {good_pages}")
    print(f"Poor Pages       : {poor_pages}")
    print(f"Empty Pages      : {empty_pages}")
    print(f"Coverage         : {coverage:.2f}%")

    print(f"\nCSV Saved -> {csv_file.name}")
    """