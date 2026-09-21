# NICU AI Setup

This file contains the software, Python packages, and Ollama models required
to move this project to another computer.

## 1. System requirements

- Python 3.11 or newer
- Git
- Ollama
- Enough disk space for the Python environment and the Ollama models

Ollama installation instructions are available at:

    https://ollama.com/download

On macOS or Linux, Ollama can also be installed with:

    curl -fsSL https://ollama.com/install.sh | sh

On Windows, install Ollama using the official installer.

## 2. Download the project

    git clone https://github.com/Arya110305/nicu-ai.git
    cd nicu-ai

## 3. Create the Python environment

macOS/Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Windows PowerShell:

    py -3 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## 4. Install the Ollama models

Start Ollama first. On systems where Ollama does not run automatically, use:

    ollama serve

In another terminal, download both models used by the current pipeline:

    ollama pull qwen2.5:7b
    ollama pull gemma3:12b

Verify that they are available:

    ollama list

The model roles are currently:

- `qwen2.5:7b`: OCR cleanup and LLM extraction
- `gemma3:12b`: normalization and discharge summary generation

The Ollama server is expected at `http://127.0.0.1:11434`. To use another
host, update the Ollama host settings in `llm/models.py` and the related
normalizer/discharge configuration files.

## 5. Start the API

From the repository root, with the virtual environment activated:

    uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

The API is available at:

    http://127.0.0.1:8000

API documentation is available at:

    http://127.0.0.1:8000/docs

## 6. Start the frontend

In a second terminal:

    cd frontend
    python3 -m http.server 5500

Open:

    http://127.0.0.1:5500

## 7. Verify the installation

Run these checks from the repository root:

    python -c "import fastapi, ollama, paddleocr, fitz, requests; print('Python dependencies OK')"
    curl http://127.0.0.1:11434/api/tags
    curl http://127.0.0.1:8000/docs

The Ollama response should list both downloaded models, and the API should be
running before processing a patient PDF.

## Troubleshooting

- `Could not connect to Ollama`: start Ollama with `ollama serve` and verify port `11434`.
- Model not found: run both `ollama pull` commands again and check `ollama list`.
- `ModuleNotFoundError`: activate `.venv` and run `python -m pip install -r requirements.txt`.
- Upload or form-data errors: confirm `python-multipart` is installed in the active environment.