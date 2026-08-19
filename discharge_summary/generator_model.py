"""
Simple Ollama client for discharge summary generation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gemma3:12b"


class OllamaConnectionError(RuntimeError):
    """
    Raised when Ollama cannot be reached.
    """


class OllamaGenerationError(RuntimeError):
    """
    Raised when Ollama fails to generate output.
    """


class OllamaModel:
    """
    Small client for the local Ollama server.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        url: str = OLLAMA_URL,
    ) -> None:
        self.model_name = model_name
        self.url = url

    def health_check(self) -> bool:
        """
        Check whether Ollama is running.
        """

        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags",
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=10,
            ) as response:
                return response.status == 200

        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            raise OllamaConnectionError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ) from exc

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Generate text using the local Ollama model.
        """

        full_prompt = f"""
SYSTEM INSTRUCTIONS
-------------------
{system_prompt}

USER REQUEST
------------
{user_prompt}
""".strip()

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 16384,
                "num_predict": 8192,
            },
        }

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=900,
            ) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise OllamaGenerationError(
                f"Ollama HTTP error: {error_text}"
            ) from exc

        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            raise OllamaConnectionError(
                "Connection to Ollama failed during generation."
            ) from exc

        except json.JSONDecodeError as exc:
            raise OllamaGenerationError(
                "Ollama returned invalid JSON."
            ) from exc

        generated_text = response_data.get(
            "response",
            "",
        ).strip()

        if not generated_text:
            raise OllamaGenerationError(
                "Ollama returned an empty response."
            )

        return generated_text


def create_model() -> OllamaModel:
    """
    Create the default Ollama model.
    """

    return OllamaModel()