from __future__ import annotations

import os
from pathlib import Path


# ==========================================================
# PROJECT FOLDERS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = PROJECT_ROOT / "ocr"

INPUT_FOLDER = OCR_ROOT / "input"
TEMP_FOLDER = OCR_ROOT / "temp"
DEBUG_FOLDER = OCR_ROOT / "debug"

REPORT_FOLDER = PROJECT_ROOT / "reports"

# Permanent storage for every patient.
DATA_FOLDER = PROJECT_ROOT / "data"

for folder in (
    INPUT_FOLDER,
    TEMP_FOLDER,
    DEBUG_FOLDER,
    REPORT_FOLDER,
    DATA_FOLDER,
):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==========================================================
# OCR SETTINGS
# ==========================================================

# Rendering quality.
OCR_DPI = 220

USE_DOC_ORIENTATION = False
USE_DOC_UNWARPING = False
USE_TEXTLINE_ORIENTATION = False


# ==========================================================
# OCR PARALLELISM
# ==========================================================

# Number of PDF pages OCR processes simultaneously.
MAX_OCR_WORKERS = 4

# CPU threads used inside each OCR worker.
#
# Keep this at 1 so four OCR processes do not each create
# several BLAS / OpenMP threads and overload the machine.
WORKER_CPU_THREADS = 1


# ==========================================================
# CHARACTER-BASED STREAMING
# ==========================================================

# We NO LONGER create batches by a fixed number of pages.
#
# OCR text is accumulated in page order.
#
# Once the accumulated text reaches MIN_BATCH_CHARACTERS,
# the batch becomes eligible to be sent to the LLM.
#
# Additional complete pages may be added as long as the
# total does not exceed MAX_BATCH_CHARACTERS.
#
# Example:
#
# Page 1-4  -> 11,500 chars
# Page 5    -> +4,200 chars
#
# Total     -> 15,700 chars
#
# Batch can now be released.
#
# If Page 6 would increase the batch above 20,000 chars,
# Batch 1 closes and Page 6 starts Batch 2.

MIN_BATCH_CHARACTERS = 15_000

MAX_BATCH_CHARACTERS = 20_000


# ==========================================================
# PAGE PROCESSING
# ==========================================================

# Retry one failed OCR page this many additional times.
PAGE_RETRIES = 1

# Page TXT and metadata are not stored permanently,
# so persistent resume is disabled.
RESUME_EXISTING_PAGES = False


# ==========================================================
# OUTPUT SETTINGS
# ==========================================================

# Do NOT permanently store individual page text files.
SAVE_PAGE_TEXT = False

# Do NOT permanently store individual page metadata JSON.
SAVE_METADATA = False

# Do NOT keep page images unless this is enabled manually.
SAVE_DEBUG_IMAGES = False

# Remove rendered PNG after successful OCR.
DELETE_SUCCESSFUL_TEMP_IMAGES = True

# Save one complete OCR text file after the entire PDF finishes.
SAVE_COMPLETE_TEXT = True

# Save CSV OCR quality/performance report.
SAVE_CSV_REPORT = True

# Create a small .ready marker when a character batch
# is available for the downstream LLM watcher.
CREATE_BATCH_READY_MARKERS = True


# ==========================================================
# DEBUG THRESHOLDS
# ==========================================================

DEBUG_HEALTH_THRESHOLD = 70

DEBUG_CONFIDENCE_THRESHOLD = 85


# ==========================================================
# DATA STORAGE POLICY
# ==========================================================

"""
Permanent storage:

data/
    Patient_Name/
        original/
            patient.pdf

        ocr/
            Patient_Name_ocr_complete.txt

        llm/
            Patient_Name_llm_complete.txt

        normalizer/
            Patient_Name_normalized_complete.txt

        discharge/
            Patient_Name_discharge_summary.txt


Temporary streaming folders:

data/
    Patient_Name/
        ocr/
            batches/

        llm/
            batches/

        normalizer/
            batches/


Temporary rendered OCR images:

ocr/
    temp/


Temporary batch files are used only for communication between
pipeline stages.

The complete stage outputs remain the permanent patient record.

Individual OCR page images and page TXT files are not retained.
"""


# ==========================================================
# ENVIRONMENT LIMITS
# ==========================================================

os.environ.setdefault(
    "OMP_NUM_THREADS",
    str(WORKER_CPU_THREADS),
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    str(WORKER_CPU_THREADS),
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    str(WORKER_CPU_THREADS),
)

os.environ.setdefault(
    "VECLIB_MAXIMUM_THREADS",
    str(WORKER_CPU_THREADS),
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    str(WORKER_CPU_THREADS),
)