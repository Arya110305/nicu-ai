from __future__ import annotations

import re
from typing import Any

from ocr.hospital_dictionary import ALL_MEDICAL_FIELDS


def calculate_basic_statistics(texts: list[str]) -> dict[str, Any]:
    full_text = "\n".join(str(text) for text in texts)

    return {
        "text": full_text,
        "words": len(full_text.split()),
        "characters": len(full_text),
        "lines": len(texts),
        "blocks": len(texts),
        "numbers": len(re.findall(r"\d+\.?\d*", full_text)),
    }


def calculate_confidence(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"average": 0.0, "minimum": 0.0, "maximum": 0.0}

    clean_scores = [float(score) for score in scores]

    return {
        "average": sum(clean_scores) / len(clean_scores),
        "minimum": min(clean_scores),
        "maximum": max(clean_scores),
    }


def detect_medical_fields(text: str) -> dict[str, Any]:
    lowered = text.lower()
    found = {
        keyword
        for keyword in ALL_MEDICAL_FIELDS
        if keyword and keyword.lower() in lowered
    }

    return {"count": len(found), "fields": sorted(found)}


def calculate_health_score(
    statistics: dict[str, Any],
    confidence: dict[str, float],
    medical: dict[str, Any],
) -> dict[str, Any]:
    score = 0.0
    score += confidence["average"] * 35
    score += min(statistics["words"] / 600, 1) * 20
    score += min(statistics["lines"] / 60, 1) * 10
    score += min(medical["count"] / 15, 1) * 25
    score += min(statistics["numbers"] / 40, 1) * 10
    score = round(min(max(score, 0), 100), 2)

    if score >= 95:
        status = "★★★★★ Excellent"
    elif score >= 85:
        status = "★★★★☆ Very Good"
    elif score >= 70:
        status = "★★★☆☆ Good"
    elif score >= 50:
        status = "★★☆☆☆ Review"
    else:
        status = "★☆☆☆☆ Failed"

    return {"score": score, "status": status}


def analyze_page_result(
    pdf_name: str,
    page_number: int,
    texts: list[str],
    scores: list[float],
    processing_time: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    statistics = calculate_basic_statistics(texts)
    confidence = calculate_confidence(scores)
    medical = detect_medical_fields(statistics["text"])
    health = calculate_health_score(statistics, confidence, medical)

    page_stats = {
        "words": statistics["words"],
        "characters": statistics["characters"],
        "lines": statistics["lines"],
        "blocks": statistics["blocks"],
        "numbers_found": statistics["numbers"],
        "medical_keywords_found": medical["count"],
        "medical_fields": medical["fields"],
        "average_confidence": round(confidence["average"] * 100, 2),
        "minimum_confidence": round(confidence["minimum"] * 100, 2),
        "maximum_confidence": round(confidence["maximum"] * 100, 2),
        "ocr_health_score": health["score"],
        "status": health["status"],
    }

    metadata = {
        "pdf_name": pdf_name,
        "page": page_number,
        "processing_time": round(processing_time, 2),
        "statistics": {
            "words": statistics["words"],
            "characters": statistics["characters"],
            "lines": statistics["lines"],
            "blocks": statistics["blocks"],
            "numbers": statistics["numbers"],
        },
        "confidence": {
            "average": page_stats["average_confidence"],
            "minimum": page_stats["minimum_confidence"],
            "maximum": page_stats["maximum_confidence"],
        },
        "medical": medical,
        "health": health,
        "ocr_text": texts,
    }

    return page_stats, metadata