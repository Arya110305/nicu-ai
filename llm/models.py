# ==========================================================
# OLLAMA MODEL CONFIGURATION
# ==========================================================

MODEL_NAME = "gemma3:12b"
# MODEL_NAME = "qwen2.5:7b"
# MODEL_NAME = "gemma3:27b"


# ==========================================================
# GENERAL SETTINGS
# ==========================================================

OLLAMA_KEEP_ALIVE = "30m"

OLLAMA_HOST = "http://127.0.0.1:11434"


# ==========================================================
# STREAMING LLM SETTINGS
# ==========================================================

# Number of OCR batches submitted concurrently.
EXTRACTOR_MAX_WORKERS = 1

# Maximum time for one LLM batch.
EXTRACTOR_TIMEOUT_SECONDS = 1800

# Additional attempts after the first failure.
EXTRACTOR_RETRIES = 2

# Delay between retries.
EXTRACTOR_RETRY_DELAY_SECONDS = 5

# How frequently the LLM checks for new OCR-ready batches.
EXTRACTOR_POLL_SECONDS = 1.0


# ==========================================================
# STRUCTURER OPTIONS
# ==========================================================

STRUCTURER_OPTIONS = {
    "temperature": 0,
    "num_ctx": 16384,
    "num_predict": 8192,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}


# ==========================================================
# NORMALIZER SETTINGS
# ==========================================================

NORMALIZER_MAX_WORKERS = 1

NORMALIZER_TIMEOUT_SECONDS = 1800

NORMALIZER_RETRIES = 2

NORMALIZER_RETRY_DELAY_SECONDS = 5

NORMALIZER_POLL_SECONDS = 1.0

NORMALIZER_OPTIONS = {
    "temperature": 0,
    "num_ctx": 16384,
    "num_predict": 8192,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}


# ==========================================================
# DISCHARGE GENERATOR OPTIONS
# ==========================================================

DISCHARGE_OPTIONS = {
    "temperature": 0,
    "num_ctx": 16384,
    "num_predict": 8192,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}