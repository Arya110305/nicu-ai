from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


CSV_HEADERS = [
    "Page",
    "Words",
    "Characters",
    "Lines",
    "Blocks",
    "Numbers",
    "Medical Fields",
    "Confidence",
    "Health",
    "Time",
    "Status",
]


def print_page_report(
    page_number: int,
    total_pages: int,
    page_stats: dict[str, Any],
    processing_time: float,
) -> None:
    confidence = page_stats["average_confidence"]
    health = page_stats["ocr_health_score"]

    bar_length = 30
    filled = max(0, min(bar_length, int((health / 100) * bar_length)))
    bar = "█" * filled + "░" * (bar_length - filled)

    print(
        f"[DONE] Page {page_number}/{total_pages} | "
        f"Time {processing_time:.2f}s | "
        f"Words {page_stats['words']} | "
        f"Confidence {confidence:.2f}% | "
        f"Health [{bar}] {health:.2f}% | "
        f"{page_stats['status']}"
    )


def save_csv_report(report_rows: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(report_rows)


def print_final_report(
    pdf_name: str,
    total_pages: int,
    report_rows: list[dict[str, Any]],
    total_time: float,
) -> None:
    successful_rows = [
        row for row in report_rows
        if not str(row["Status"]).startswith("FAILED")
    ]
    failed_rows = [
        row for row in report_rows
        if str(row["Status"]).startswith("FAILED")
    ]

    completed = len(successful_rows)
    total_words = sum(row["Words"] for row in successful_rows)
    total_characters = sum(row["Characters"] for row in successful_rows)

    avg_conf = (
        sum(row["Confidence"] for row in successful_rows) / completed
        if completed else 0
    )
    avg_health = (
        sum(row["Health"] for row in successful_rows) / completed
        if completed else 0
    )

    print("\n" + "=" * 70)
    print("OCR FINAL REPORT")
    print("=" * 70)
    print(f"PDF Name           : {pdf_name}")
    print(f"Total Pages        : {total_pages}")
    print(f"Successful Pages   : {completed}")
    print(f"Failed Pages       : {len(failed_rows)}")
    print(f"Total Words        : {total_words:,}")
    print(f"Total Characters   : {total_characters:,}")
    print(f"Average Confidence : {avg_conf:.2f}%")
    print(f"OCR Health Score   : {avg_health:.2f}%")
    print(f"Wall-clock Time    : {total_time:.2f} sec")
    if total_pages:
        print(f"Average/Page       : {total_time / total_pages:.2f} sec")
    print("=" * 70)