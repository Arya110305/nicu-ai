"""
Configuration for the streaming NICU clinical normalizer.
"""

from __future__ import annotations

from pathlib import Path


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = PROJECT_ROOT / "data"

ERROR_FOLDER = (
    PROJECT_ROOT
    / "normalizer"
    / "errors"
)

ERROR_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "gemma3:12b"
# MODEL_NAME = "qwen2.5:7b"
# MODEL_NAME = "gemma3:27b"

OLLAMA_HOST = "http://127.0.0.1:11434"

OLLAMA_GENERATE_ENDPOINT = (
    f"{OLLAMA_HOST.rstrip('/')}/api/generate"
)

OLLAMA_TAGS_ENDPOINT = (
    f"{OLLAMA_HOST.rstrip('/')}/api/tags"
)


# ==========================================================
# NORMALIZER WORKERS
# ==========================================================

# Number of LLM batches normalized simultaneously.
NORMALIZER_MAX_WORKERS = 4

# How often the normalizer checks for new LLM-ready batches.
NORMALIZER_POLL_SECONDS = 1.0

# Maximum time allowed for one batch request.
TIMEOUT_SECONDS = 1800

# Additional attempts after the first failed request.
RETRIES = 2

# Delay between retry attempts.
RETRY_DELAY_SECONDS = 5


# ==========================================================
# OLLAMA OPTIONS
# ==========================================================

OPTIONS = {
    "temperature": 0,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_ctx": 16384,
    "num_predict": 8192,
}

KEEP_ALIVE = "30m"


# ==========================================================
# FILE SETTINGS
# ==========================================================

FILE_ENCODING = "utf-8"

LLM_COMPLETE_SUFFIX = "_llm_complete.txt"

NORMALIZED_COMPLETE_SUFFIX = (
    "_normalized_complete.txt"
)

LLM_BATCH_READY_SUFFIX = ".ready"


# ==========================================================
# STORAGE POLICY
# ==========================================================

"""
Permanent files retained after successful processing:

data/<patient>/
    original/
        patient.pdf

    ocr/
        <patient>_ocr_complete.txt

    llm/
        <patient>_llm_complete.txt

    normalizer/
        <patient>_normalized_complete.txt

    discharge/
        <patient>_discharge_summary.txt

Temporary files while processing:

data/<patient>/ocr/batches/
data/<patient>/llm/batches/
data/<patient>/normalizer/batches/

After normalization succeeds:

- temporary LLM batches are removed
- temporary normalized batches are removed after merging
- the complete LLM file remains
- the complete normalized file remains
"""