"""
Local Ollama client for NICU discharge-summary generation.

This module handles only communication with the local Ollama server.

It provides the exact interface expected by discharge_summary/generator.py:

    OllamaModel
    OllamaConnectionError
    OllamaGenerationError
    create_model()

The OllamaModel object supports:

    health_check()
    warmup()
    generate_text(
        system_prompt=...,
        user_prompt=...,
    )
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Final, Mapping


LOGGER = logging.getLogger(__name__)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_MODEL_NAME: Final[str] = "gemma3:12b"

DEFAULT_OLLAMA_BASE_URL: Final[str] = (
    "http://127.0.0.1:11434"
)

DEFAULT_TIMEOUT_SECONDS: Final[int] = 900

DEFAULT_MAX_RETRIES: Final[int] = 3

DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 3.0

DEFAULT_TEMPERATURE: Final[float] = 0.1

DEFAULT_NUM_CTX: Final[int] = 16_384

DEFAULT_NUM_PREDICT: Final[int] = 8_192


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class OllamaModelError(RuntimeError):
    """
    Base exception for Ollama model failures.
    """


class OllamaConnectionError(OllamaModelError):
    """
    Raised when the local Ollama server cannot be reached.
    """


class OllamaGenerationError(OllamaModelError):
    """
    Raised when text generation fails or returns invalid output.
    """


class OllamaResponseError(OllamaGenerationError):
    """
    Raised when Ollama returns an invalid HTTP or JSON response.

    This class is retained as a more specific generation-error type.
    """


class OllamaModelNotFoundError(OllamaGenerationError):
    """
    Raised when the requested model is not installed locally.
    """


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class ModelConfig:
    """
    Configuration for one Ollama model client.
    """

    model_name: str = DEFAULT_MODEL_NAME

    base_url: str = DEFAULT_OLLAMA_BASE_URL

    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    max_retries: int = DEFAULT_MAX_RETRIES

    retry_delay_seconds: float = (
        DEFAULT_RETRY_DELAY_SECONDS
    )

    temperature: float = DEFAULT_TEMPERATURE

    num_ctx: int = DEFAULT_NUM_CTX

    num_predict: int = DEFAULT_NUM_PREDICT

    @classmethod
    def from_environment(
        cls,
    ) -> "ModelConfig":
        """
        Build model configuration using environment variables.
        """

        return cls(
            model_name=os.getenv(
                "DISCHARGE_MODEL_NAME",
                DEFAULT_MODEL_NAME,
            ).strip(),
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                DEFAULT_OLLAMA_BASE_URL,
            ).strip().rstrip("/"),
            timeout_seconds=_read_int_environment(
                "DISCHARGE_MODEL_TIMEOUT",
                DEFAULT_TIMEOUT_SECONDS,
                minimum=30,
            ),
            max_retries=_read_int_environment(
                "DISCHARGE_MODEL_RETRIES",
                DEFAULT_MAX_RETRIES,
                minimum=1,
            ),
            retry_delay_seconds=(
                _read_float_environment(
                    "DISCHARGE_MODEL_RETRY_DELAY",
                    DEFAULT_RETRY_DELAY_SECONDS,
                    minimum=0.0,
                )
            ),
            temperature=_read_float_environment(
                "DISCHARGE_MODEL_TEMPERATURE",
                DEFAULT_TEMPERATURE,
                minimum=0.0,
                maximum=2.0,
            ),
            num_ctx=_read_int_environment(
                "DISCHARGE_MODEL_NUM_CTX",
                DEFAULT_NUM_CTX,
                minimum=2_048,
            ),
            num_predict=_read_int_environment(
                "DISCHARGE_MODEL_NUM_PREDICT",
                DEFAULT_NUM_PREDICT,
                minimum=1,
            ),
        )

    def __post_init__(self) -> None:
        """
        Validate model configuration.
        """

        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if not self.base_url.strip():
            raise ValueError(
                "base_url cannot be empty."
            )

        if self.timeout_seconds < 30:
            raise ValueError(
                "timeout_seconds must be at least 30."
            )

        if self.max_retries < 1:
            raise ValueError(
                "max_retries must be at least 1."
            )

        if self.retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds cannot be negative."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        if self.num_ctx < 2_048:
            raise ValueError(
                "num_ctx must be at least 2048."
            )

        if self.num_predict < 1:
            raise ValueError(
                "num_predict must be greater than zero."
            )


# =============================================================================
# GENERATION RESULT
# =============================================================================

@dataclass(frozen=True, slots=True)
class GenerationResult:
    """
    Detailed result returned by an Ollama generation request.
    """

    text: str

    model_name: str

    total_duration_seconds: float | None

    load_duration_seconds: float | None

    prompt_tokens: int | None

    generated_tokens: int | None

    raw_response: Mapping[str, Any]


# =============================================================================
# OLLAMA MODEL CLIENT
# =============================================================================

class OllamaModel:
    """
    Local client for the Ollama HTTP API.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
    ) -> None:
        """
        Initialise the Ollama model client.
        """

        self.config = (
            config
            if config is not None
            else ModelConfig.from_environment()
        )

    # =========================================================================
    # SERVER AND MODEL VALIDATION
    # =========================================================================

    def health_check(self) -> bool:
        """
        Confirm that Ollama is running and the configured model exists.

        Returns:
            True when the server is reachable and the model is installed.

        Raises:
            OllamaConnectionError:
                If the Ollama server cannot be reached.

            OllamaModelNotFoundError:
                If the configured model is not installed.
        """

        self.check_server()
        self.ensure_model_available()

        return True

    def check_server(self) -> bool:
        """
        Confirm that the local Ollama server is reachable.
        """

        endpoint = (
            f"{self.config.base_url}/api/tags"
        )

        try:
            self._request_json(
                endpoint=endpoint,
                method="GET",
                payload=None,
                timeout_seconds=min(
                    self.config.timeout_seconds,
                    30,
                ),
            )

            return True

        except OllamaModelError:
            raise

        except Exception as exc:
            raise OllamaConnectionError(
                "Could not connect to Ollama at "
                f"{self.config.base_url}. "
                "Make sure Ollama is running."
            ) from exc

    def list_models(self) -> list[str]:
        """
        Return model names installed in the local Ollama server.
        """

        endpoint = (
            f"{self.config.base_url}/api/tags"
        )

        response = self._request_json(
            endpoint=endpoint,
            method="GET",
            payload=None,
            timeout_seconds=min(
                self.config.timeout_seconds,
                30,
            ),
        )

        models = response.get("models")

        if not isinstance(models, list):
            raise OllamaResponseError(
                "Ollama returned an invalid model-list response."
            )

        model_names: list[str] = []

        for model in models:
            if not isinstance(model, dict):
                continue

            name = (
                model.get("name")
                or model.get("model")
            )

            if isinstance(name, str) and name.strip():
                model_names.append(
                    name.strip()
                )

        return model_names

    def ensure_model_available(self) -> None:
        """
        Confirm that the configured Ollama model is installed.
        """

        available_models = self.list_models()

        configured_name = (
            self.config.model_name
        )

        configured_base_name = configured_name.split(
            ":",
            maxsplit=1,
        )[0]

        exact_match = (
            configured_name in available_models
        )

        compatible_base_match = any(
            installed_name.split(
                ":",
                maxsplit=1,
            )[0]
            == configured_base_name
            for installed_name in available_models
        )

        if exact_match or compatible_base_match:
            return

        installed_models = (
            ", ".join(available_models)
            if available_models
            else "none"
        )

        raise OllamaModelNotFoundError(
            f"Model '{configured_name}' is not installed. "
            f"Installed models: {installed_models}. "
            f"Run: ollama pull {configured_name}"
        )

    # =========================================================================
    # MODEL WARMUP
    # =========================================================================

    def warmup(self) -> None:
        """
        Load the model into memory using a minimal request.

        This is the method expected by generator.py.
        """

        LOGGER.info(
            "Warming up Ollama model '%s'.",
            self.config.model_name,
        )

        response = self.generate_text(
            system_prompt=(
                "You are a local model health-check process. "
                "Follow the instruction exactly."
            ),
            user_prompt=(
                "Respond with exactly the word READY."
            ),
            temperature=0.0,
            num_predict=8,
        )

        if "READY" not in response.upper():
            LOGGER.warning(
                "Model warmup completed but returned an "
                "unexpected response: %s",
                response[:200],
            )

    def warm_up(self) -> None:
        """
        Backwards-compatible alias for warmup().
        """

        self.warmup()

    # =========================================================================
    # TEXT GENERATION
    # =========================================================================

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        **generation_kwargs: Any,
    ) -> str:
        """
        Generate text and return only the generated response.

        Args:
            system_prompt:
                Model-level instructions.

            user_prompt:
                Task-specific prompt.

            generation_kwargs:
                Optional values passed to generate(), such as temperature,
                num_ctx, num_predict, seed or stop.

        Returns:
            Generated response text.
        """

        result = self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            **generation_kwargs,
        )

        return result.text

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        seed: int | None = 42,
        stop: list[str] | None = None,
        keep_alive: str = "10m",
        extra_options: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """
        Generate text using the configured local Ollama model.
        """

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError(
                "The generation prompt cannot be empty."
            )

        resolved_temperature = (
            self.config.temperature
            if temperature is None
            else temperature
        )

        resolved_num_ctx = (
            self.config.num_ctx
            if num_ctx is None
            else num_ctx
        )

        resolved_num_predict = (
            self.config.num_predict
            if num_predict is None
            else num_predict
        )

        if not 0.0 <= resolved_temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        if resolved_num_ctx < 2_048:
            raise ValueError(
                "num_ctx must be at least 2048."
            )

        if resolved_num_predict < 1:
            raise ValueError(
                "num_predict must be greater than zero."
            )

        options: dict[str, Any] = {
            "temperature": resolved_temperature,
            "num_ctx": resolved_num_ctx,
            "num_predict": resolved_num_predict,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05,
        }

        if seed is not None:
            options["seed"] = seed

        if stop:
            options["stop"] = stop

        if extra_options:
            options.update(
                dict(extra_options)
            )

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "prompt": cleaned_prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        }

        if (
            system_prompt is not None
            and system_prompt.strip()
        ):
            payload["system"] = (
                system_prompt.strip()
            )

        endpoint = (
            f"{self.config.base_url}/api/generate"
        )

        started_at = time.monotonic()

        LOGGER.info(
            "Starting Ollama generation: "
            "model=%s, prompt_chars=%d, "
            "num_ctx=%d, num_predict=%d",
            self.config.model_name,
            len(cleaned_prompt),
            resolved_num_ctx,
            resolved_num_predict,
        )

        response = self._request_with_retries(
            endpoint=endpoint,
            payload=payload,
        )

        elapsed_seconds = (
            time.monotonic() - started_at
        )

        generated_text = response.get(
            "response"
        )

        if not isinstance(generated_text, str):
            raise OllamaResponseError(
                "Ollama response did not contain "
                "generated text."
            )

        generated_text = generated_text.strip()

        if not generated_text:
            raise OllamaResponseError(
                "Ollama returned an empty generation response."
            )

        if response.get("done") is False:
            raise OllamaResponseError(
                "Ollama generation ended before completion."
            )

        result = GenerationResult(
            text=generated_text,
            model_name=str(
                response.get("model")
                or self.config.model_name
            ),
            total_duration_seconds=(
                _nanoseconds_to_seconds(
                    response.get("total_duration")
                )
            ),
            load_duration_seconds=(
                _nanoseconds_to_seconds(
                    response.get("load_duration")
                )
            ),
            prompt_tokens=_safe_integer(
                response.get(
                    "prompt_eval_count"
                )
            ),
            generated_tokens=_safe_integer(
                response.get("eval_count")
            ),
            raw_response=response,
        )

        LOGGER.info(
            "Ollama generation completed: "
            "elapsed=%.2fs, prompt_tokens=%s, "
            "generated_tokens=%s, output_chars=%d",
            elapsed_seconds,
            result.prompt_tokens,
            result.generated_tokens,
            len(result.text),
        )

        return result

    # =========================================================================
    # HTTP REQUEST HANDLING
    # =========================================================================

    def _request_with_retries(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Send an Ollama generation request with retry handling.
        """

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.config.max_retries + 1,
        ):
            try:
                return self._request_json(
                    endpoint=endpoint,
                    method="POST",
                    payload=payload,
                    timeout_seconds=(
                        self.config.timeout_seconds
                    ),
                )

            except OllamaModelNotFoundError:
                raise

            except (
                OllamaConnectionError,
                OllamaResponseError,
            ) as exc:
                last_error = exc

                if (
                    attempt
                    >= self.config.max_retries
                ):
                    break

                delay = (
                    self.config.retry_delay_seconds
                    * attempt
                )

                LOGGER.warning(
                    "Ollama request failed on attempt "
                    "%d/%d: %s. Retrying in %.1f seconds.",
                    attempt,
                    self.config.max_retries,
                    exc,
                    delay,
                )

                time.sleep(delay)

        raise OllamaGenerationError(
            "Ollama generation failed after "
            f"{self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    def _request_json(
        self,
        *,
        endpoint: str,
        method: str,
        payload: Mapping[str, Any] | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """
        Send an HTTP request and decode the JSON response.
        """

        request_data: bytes | None = None

        headers = {
            "Accept": "application/json",
        }

        if payload is not None:
            try:
                request_data = json.dumps(
                    payload,
                    ensure_ascii=False,
                ).encode("utf-8")

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    "Ollama request payload is not "
                    "JSON serialisable."
                ) from exc

            headers["Content-Type"] = (
                "application/json"
            )

        request = urllib.request.Request(
            endpoint,
            data=request_data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                raw_body = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            message = _extract_ollama_error(
                raw_error
            )

            if (
                exc.code == 404
                and (
                    "model" in message.lower()
                    or "not found" in message.lower()
                )
            ):
                raise OllamaModelNotFoundError(
                    "Ollama model "
                    f"'{self.config.model_name}' "
                    "was not found. Run: "
                    f"ollama pull {self.config.model_name}"
                ) from exc

            raise OllamaResponseError(
                f"Ollama returned HTTP {exc.code}: "
                f"{message}"
            ) from exc

        except (
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise OllamaConnectionError(
                "Could not connect to Ollama at "
                f"{self.config.base_url}. "
                "Start Ollama and confirm that its "
                "local API is available."
            ) from exc

        except OSError as exc:
            raise OllamaConnectionError(
                "Operating-system error while "
                f"contacting Ollama: {exc}"
            ) from exc

        if not raw_body.strip():
            raise OllamaResponseError(
                "Ollama returned an empty HTTP response."
            )

        try:
            decoded_response = json.loads(
                raw_body
            )

        except json.JSONDecodeError as exc:
            preview = raw_body[
                :500
            ].replace(
                "\n",
                " ",
            )

            raise OllamaResponseError(
                "Ollama returned invalid JSON: "
                f"{preview}"
            ) from exc

        if not isinstance(
            decoded_response,
            dict,
        ):
            raise OllamaResponseError(
                "Ollama returned JSON in an "
                "unexpected format."
            )

        error_message = decoded_response.get(
            "error"
        )

        if (
            isinstance(error_message, str)
            and error_message.strip()
        ):
            cleaned_error = error_message.strip()

            if "not found" in cleaned_error.lower():
                raise OllamaModelNotFoundError(
                    "Ollama model "
                    f"'{self.config.model_name}' "
                    f"was not found: {cleaned_error}"
                )

            raise OllamaResponseError(
                "Ollama reported an error: "
                f"{cleaned_error}"
            )

        return decoded_response


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = (
        DEFAULT_RETRY_DELAY_SECONDS
    ),
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    base_url: str | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    verify_server: bool = False,
    verify_model: bool = False,
) -> OllamaModel:
    """
    Create an Ollama model using the interface expected by generator.py.

    Startup validation is normally performed later through health_check(),
    so verify_server and verify_model default to False.
    """

    resolved_base_url = (
        base_url
        or os.getenv(
            "OLLAMA_BASE_URL",
            DEFAULT_OLLAMA_BASE_URL,
        )
    ).strip().rstrip("/")

    config = ModelConfig(
        model_name=model_name.strip(),
        base_url=resolved_base_url,
        timeout_seconds=timeout,
        max_retries=max_retries,
        retry_delay_seconds=(
            retry_delay_seconds
        ),
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )

    model = OllamaModel(
        config=config
    )

    if verify_server:
        model.check_server()

    if verify_model:
        model.ensure_model_available()

    return model


def create_discharge_model(
    config: ModelConfig | None = None,
    *,
    verify_server: bool = True,
    verify_model: bool = True,
) -> OllamaModel:
    """
    Backwards-compatible model factory retained for older code.
    """

    model = OllamaModel(
        config=config
    )

    if verify_server:
        model.check_server()

    if verify_model:
        model.ensure_model_available()

    return model


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _read_int_environment(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """
    Read and validate an integer environment variable.
    """

    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(
            raw_value.strip()
        )

    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} "
            "must be an integer."
        ) from exc

    if (
        minimum is not None
        and value < minimum
    ):
        raise ValueError(
            f"Environment variable {name} "
            f"must be at least {minimum}."
        )

    if (
        maximum is not None
        and value > maximum
    ):
        raise ValueError(
            f"Environment variable {name} "
            f"must not exceed {maximum}."
        )

    return value


def _read_float_environment(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """
    Read and validate a floating-point environment variable.
    """

    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(
            raw_value.strip()
        )

    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} "
            "must be a number."
        ) from exc

    if (
        minimum is not None
        and value < minimum
    ):
        raise ValueError(
            f"Environment variable {name} "
            f"must be at least {minimum}."
        )

    if (
        maximum is not None
        and value > maximum
    ):
        raise ValueError(
            f"Environment variable {name} "
            f"must not exceed {maximum}."
        )

    return value


def _nanoseconds_to_seconds(
    value: Any,
) -> float | None:
    """
    Convert Ollama timing values from nanoseconds to seconds.
    """

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return (
            float(value)
            / 1_000_000_000
        )

    return None


def _safe_integer(
    value: Any,
) -> int | None:
    """
    Convert a value into an integer without raising an exception.
    """

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        try:
            return int(
                value.strip()
            )
        except ValueError:
            return None

    return None


def _extract_ollama_error(
    raw_body: str,
) -> str:
    """
    Extract a readable error message from an Ollama response.
    """

    if not raw_body.strip():
        return (
            "No error details were returned."
        )

    try:
        decoded = json.loads(
            raw_body
        )

    except json.JSONDecodeError:
        return raw_body[
            :1_000
        ].strip()

    if isinstance(decoded, dict):
        error = decoded.get(
            "error"
        )

        if (
            isinstance(error, str)
            and error.strip()
        ):
            return error.strip()

    return raw_body[
        :1_000
    ].strip()


# =============================================================================
# DIRECT MODULE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    try:
        discharge_model = create_model(
            verify_server=True,
            verify_model=True,
        )

        print(
            "Model ready: "
            f"{discharge_model.config.model_name}"
        )

        print(
            "Ollama server: "
            f"{discharge_model.config.base_url}"
        )

        print(
            "Health check: "
            f"{discharge_model.health_check()}"
        )

    except OllamaModelError as error:
        LOGGER.error(
            "%s",
            error,
        )

        raise SystemExit(
            1
        ) from error