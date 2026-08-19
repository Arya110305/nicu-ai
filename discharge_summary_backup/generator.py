"""
Main NICU discharge-summary generation pipeline.

Pipeline:

1. Read normalized patient text.
2. Split the text into manageable chunks.
3. Extract discharge-relevant evidence from every chunk.
4. Consolidate all extracted evidence.
5. Generate a doctor-friendly evidence report.
6. Save the evidence report as plain text.
7. Select the appropriate hospital discharge template.
8. Generate the final hospital discharge summary.
9. Save the final summary as TXT, DOCX and optionally PDF.

This file coordinates the pipeline.

Model communication is handled by generator_model.py.
Prompt construction is handled by the prompts package.
File handling is handled by utils.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

from .generator_model import (
    OllamaModel,
    OllamaConnectionError,
    OllamaGenerationError,
    create_model,
)

from .prompts import (
    CLINICAL_SYSTEM_PROMPT,
    EVIDENCE_EXTRACTION_SYSTEM_PROMPT,
    EVIDENCE_REPORT_SYSTEM_PROMPT,
    TEMPLATE_SELECTION_SYSTEM_PROMPT,
    TemplateError,
    build_chunk_extraction_prompt,
    build_evidence_consolidation_prompt,
    build_evidence_report_prompt,
    build_final_discharge_prompt,
    build_template_selection_prompt,
    validate_all_templates,
    validate_template_selection,
)

from .utils import (
    ERROR_DIRECTORY,
    NORMALIZER_OUTPUT_DIRECTORY,
    OUTPUT_DIRECTORY,
    clean_model_output,
    ensure_directories,
    extract_patient_name_from_filename,
    find_normalized_files,
    read_text_file,
    safe_filename,
    save_consolidated_evidence,
    save_discharge_summary_documents,
    save_error_report,
    save_evidence_report,
    save_raw_extracted_evidence,
    split_text_into_chunks,
    text_statistics,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL_NAME: Final[str] = os.getenv(
    "DISCHARGE_MODEL_NAME",
    "gemma3:12b",
)

DEFAULT_FALLBACK_MODEL_NAME: Final[str] = os.getenv(
    "DISCHARGE_FALLBACK_MODEL_NAME",
    "qwen2.5:7b",
)

DEFAULT_CHUNK_SIZE: Final[int] = int(
    os.getenv(
        "DISCHARGE_CHUNK_SIZE",
        "14000",
    )
)

DEFAULT_CHUNK_OVERLAP: Final[int] = int(
    os.getenv(
        "DISCHARGE_CHUNK_OVERLAP",
        "800",
    )
)

DEFAULT_TEMPERATURE: Final[float] = float(
    os.getenv(
        "DISCHARGE_TEMPERATURE",
        "0.1",
    )
)

DEFAULT_MAX_RETRIES: Final[int] = int(
    os.getenv(
        "DISCHARGE_MAX_RETRIES",
        "3",
    )
)

DEFAULT_RETRY_DELAY_SECONDS: Final[float] = float(
    os.getenv(
        "DISCHARGE_RETRY_DELAY_SECONDS",
        "3",
    )
)

DEFAULT_REQUEST_TIMEOUT: Final[int] = int(
    os.getenv(
        "DISCHARGE_REQUEST_TIMEOUT",
        "900",
    )
)

DEFAULT_SAVE_INTERMEDIATE_FILES: Final[bool] = (
    os.getenv(
        "DISCHARGE_SAVE_INTERMEDIATE",
        "true",
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

DEFAULT_CREATE_PDF: Final[bool] = (
    os.getenv(
        "DISCHARGE_CREATE_PDF",
        "true",
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


# =============================================================================
# LOGGING
# =============================================================================

LOGGER_NAME: Final[str] = "nicu_discharge_summary"

logger = logging.getLogger(LOGGER_NAME)


def configure_logging(
    *,
    verbose: bool = False,
) -> None:
    """
    Configure console and file logging.

    Args:
        verbose:
            When True, DEBUG messages are also displayed.
    """

    log_level = (
        logging.DEBUG
        if verbose
        else logging.INFO
    )

    ERROR_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        ERROR_DIRECTORY
        / "discharge_summary_pipeline.log"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    logger.setLevel(log_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler(
        sys.stdout
    )
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(slots=True)
class GeneratorConfig:
    """
    Runtime configuration for the discharge-summary pipeline.
    """

    model_name: str = DEFAULT_MODEL_NAME

    fallback_model_name: str | None = (
        DEFAULT_FALLBACK_MODEL_NAME
    )

    chunk_size: int = DEFAULT_CHUNK_SIZE

    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    temperature: float = DEFAULT_TEMPERATURE

    max_retries: int = DEFAULT_MAX_RETRIES

    retry_delay_seconds: float = (
        DEFAULT_RETRY_DELAY_SECONDS
    )

    request_timeout: int = DEFAULT_REQUEST_TIMEOUT

    save_intermediate_files: bool = (
        DEFAULT_SAVE_INTERMEDIATE_FILES
    )

    create_pdf: bool = DEFAULT_CREATE_PDF

    input_directory: Path = (
        NORMALIZER_OUTPUT_DIRECTORY
    )

    output_directory: Path = OUTPUT_DIRECTORY

    def validate(self) -> None:
        """
        Validate configuration values before processing.
        """

        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.chunk_size < 1_000:
            raise ValueError(
                "chunk_size must be at least 1000."
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than "
                "chunk_size."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        if self.max_retries < 1:
            raise ValueError(
                "max_retries must be at least 1."
            )

        if self.retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds cannot be negative."
            )

        if self.request_timeout < 30:
            raise ValueError(
                "request_timeout must be at least "
                "30 seconds."
            )


@dataclass(slots=True)
class PatientGenerationResult:
    """
    Stores the result of processing one patient.
    """

    patient_name: str
    input_file: Path

    success: bool = False

    selected_template: str | None = None

    evidence_report_path: Path | None = None

    discharge_text_path: Path | None = None

    discharge_docx_path: Path | None = None

    discharge_pdf_path: Path | None = None

    raw_extraction_path: Path | None = None

    consolidated_evidence_path: Path | None = None

    error_report_path: Path | None = None

    chunk_count: int = 0

    elapsed_seconds: float = 0.0

    error_message: str | None = None

    statistics: dict[str, dict[str, int]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class PipelineRunSummary:
    """
    Stores statistics for one complete generator run.
    """

    started_at: datetime = field(
        default_factory=datetime.now
    )

    completed_at: datetime | None = None

    results: list[PatientGenerationResult] = field(
        default_factory=list
    )

    @property
    def total_files(self) -> int:
        """Return the total number of processed files."""

        return len(self.results)

    @property
    def successful_files(self) -> int:
        """Return the number of successful files."""

        return sum(
            1
            for result in self.results
            if result.success
        )

    @property
    def failed_files(self) -> int:
        """Return the number of failed files."""

        return sum(
            1
            for result in self.results
            if not result.success
        )

    @property
    def elapsed_seconds(self) -> float:
        """Return the total pipeline runtime."""

        end_time = (
            self.completed_at
            or datetime.now()
        )

        return (
            end_time - self.started_at
        ).total_seconds()


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class DischargePipelineError(RuntimeError):
    """
    Base exception for discharge-summary pipeline failures.
    """


class StartupValidationError(
    DischargePipelineError
):
    """
    Raised when the generator cannot pass startup checks.
    """


class EvidenceExtractionError(
    DischargePipelineError
):
    """
    Raised when clinical evidence extraction fails.
    """


class EvidenceConsolidationError(
    DischargePipelineError
):
    """
    Raised when extracted evidence cannot be consolidated.
    """


class EvidenceReportGenerationError(
    DischargePipelineError
):
    """
    Raised when the doctor-friendly report cannot be generated.
    """


class TemplateSelectionError(
    DischargePipelineError
):
    """
    Raised when a hospital template cannot be selected.
    """


class DischargeGenerationError(
    DischargePipelineError
):
    """
    Raised when the final discharge summary cannot be generated.
    """


# =============================================================================
# DISCHARGE-SUMMARY GENERATOR
# =============================================================================

class DischargeSummaryGenerator:
    """
    Coordinates the complete NICU discharge-summary pipeline.
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        *,
        model: OllamaModel | None = None,
        fallback_model: OllamaModel | None = None,
    ) -> None:
        """
        Initialise the discharge-summary generator.

        Args:
            config:
                Optional runtime configuration.

            model:
                Optional pre-created primary Ollama model.

            fallback_model:
                Optional pre-created fallback model.
        """

        self.config = (
            config
            if config is not None
            else GeneratorConfig()
        )

        self.config.validate()

        ensure_directories()

        self.model = (
            model
            if model is not None
            else self._create_model(
                self.config.model_name
            )
        )

        self.fallback_model = fallback_model

        if (
            self.fallback_model is None
            and self.config.fallback_model_name
            and self.config.fallback_model_name
            != self.config.model_name
        ):
            self.fallback_model = self._create_model(
                self.config.fallback_model_name
            )

        self._startup_validated = False

        logger.debug(
            "DischargeSummaryGenerator initialised. "
            "Primary model: %s | Fallback model: %s",
            self.config.model_name,
            (
                self.config.fallback_model_name
                or "None"
            ),
        )

    def _create_model(
        self,
        model_name: str,
    ) -> OllamaModel:
        """
        Create an Ollama model using the project model factory.
        """

        return create_model(
            model_name=model_name,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            retry_delay_seconds=(
                self.config.retry_delay_seconds
            ),
            timeout=self.config.request_timeout,
        )

    def validate_startup(self) -> None:
        """
        Validate templates, folders and Ollama connectivity.

        This should be called once before processing patient files.
        """

        logger.info(
            "Running discharge-generator startup checks."
        )

        try:
            self.config.validate()
            ensure_directories()

            self._validate_input_directory()
            self._validate_templates()
            self._validate_primary_model()
            self._validate_fallback_model()

        except Exception as exc:
            raise StartupValidationError(
                f"Startup validation failed: {exc}"
            ) from exc

        self._startup_validated = True

        logger.info(
            "All startup checks passed."
        )

    def _validate_input_directory(self) -> None:
        """
        Validate the configured normalized-text input directory.
        """

        input_directory = (
            self.config.input_directory
        )

        input_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not input_directory.is_dir():
            raise NotADirectoryError(
                "Configured input path is not a "
                f"directory: {input_directory}"
            )

        logger.info(
            "Input directory ready: %s",
            input_directory,
        )

    def _validate_templates(self) -> None:
        """
        Confirm that all hospital DOCX templates are readable.
        """

        validated_templates = (
            validate_all_templates()
        )

        logger.info(
            "Validated %d hospital templates.",
            len(validated_templates),
        )

        for template_name, path in (
            validated_templates.items()
        ):
            logger.debug(
                "Template ready: %s -> %s",
                template_name,
                path,
            )

    def _validate_primary_model(self) -> None:
        """
        Confirm that the primary Ollama model is available.
        """

        logger.info(
            "Checking primary Ollama model: %s",
            self.config.model_name,
        )

        try:
            is_healthy = self.model.health_check()
        except Exception as exc:
            raise OllamaConnectionError(
                "Could not connect to the primary "
                f"Ollama model: {exc}"
            ) from exc

        if not is_healthy:
            raise OllamaConnectionError(
                "Primary Ollama model health check "
                f"failed: {self.config.model_name}"
            )

        logger.info(
            "Primary model is available."
        )

    def _validate_fallback_model(self) -> None:
        """
        Check the fallback model without failing the whole startup.
        """

        if self.fallback_model is None:
            logger.info(
                "No fallback model configured."
            )
            return

        logger.info(
            "Checking fallback Ollama model: %s",
            self.config.fallback_model_name,
        )

        try:
            fallback_healthy = (
                self.fallback_model.health_check()
            )
        except Exception as exc:
            logger.warning(
                "Fallback model health check failed: %s",
                exc,
            )
            self.fallback_model = None
            return

        if not fallback_healthy:
            logger.warning(
                "Fallback model is unavailable. "
                "Processing will continue using only "
                "the primary model."
            )
            self.fallback_model = None
            return

        logger.info(
            "Fallback model is available."
        )

    def warmup_models(self) -> None:
        """
        Warm up the available Ollama models.

        Warmup reduces delay during the first real generation call.
        """

        if not self._startup_validated:
            self.validate_startup()

        logger.info(
            "Warming up primary model: %s",
            self.config.model_name,
        )

        try:
            self.model.warmup()
        except Exception as exc:
            raise StartupValidationError(
                "Primary model warmup failed: "
                f"{exc}"
            ) from exc

        if self.fallback_model is not None:
            logger.info(
                "Warming up fallback model: %s",
                self.config.fallback_model_name,
            )

            try:
                self.fallback_model.warmup()
            except Exception as exc:
                logger.warning(
                    "Fallback model warmup failed: %s",
                    exc,
                )
                self.fallback_model = None

        logger.info(
            "Model warmup complete."
        )

    def discover_input_files(
        self,
    ) -> list[Path]:
        """
        Find all normalized patient records ready for processing.
        """

        input_files = find_normalized_files(
            self.config.input_directory
        )

        logger.info(
            "Found %d normalized patient file(s).",
            len(input_files),
        )

        return input_files

    def prepare_patient_input(
        self,
        input_file: Path,
    ) -> tuple[str, str, list[str]]:
        """
        Read and split one normalized patient record.

        Returns:
            Tuple containing:

            1. Patient name
            2. Complete normalized text
            3. Text chunks
        """

        patient_name = (
            extract_patient_name_from_filename(
                input_file
            )
        )

        logger.info(
            "Reading normalized record for: %s",
            patient_name,
        )

        normalized_text = read_text_file(
            input_file
        )

        chunks = split_text_into_chunks(
            normalized_text,
            maximum_characters=(
                self.config.chunk_size
            ),
            overlap_characters=(
                self.config.chunk_overlap
            ),
        )

        logger.info(
            "Prepared %d chunk(s) for %s.",
            len(chunks),
            patient_name,
        )

        logger.debug(
            "Normalized record statistics for %s: %s",
            patient_name,
            text_statistics(normalized_text),
        )

        return (
            patient_name,
            normalized_text,
            chunks,
        )
        # =========================================================================
    # MODEL GENERATION HELPERS
    # =========================================================================

    def _generate_with_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage_name: str,
    ) -> str:
        """
        Generate text using the primary model.

        If the primary model fails and a fallback model is available, retry
        using the fallback model.

        Args:
            system_prompt:
                System-level instructions.

            user_prompt:
                Task-specific prompt.

            stage_name:
                Human-readable pipeline stage used in logs and errors.

        Returns:
            Cleaned model output.

        Raises:
            OllamaGenerationError:
                If both primary and fallback generation fail.
        """

        logger.debug(
            "Starting model generation for stage: %s",
            stage_name,
        )

        primary_error: Exception | None = None

        try:
            response = self.model.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            cleaned_response = clean_model_output(
                response
            )

            if not cleaned_response:
                raise OllamaGenerationError(
                    f"Primary model returned empty output "
                    f"during stage: {stage_name}"
                )

            return cleaned_response

        except Exception as exc:
            primary_error = exc

            logger.warning(
                "Primary model failed during %s: %s",
                stage_name,
                exc,
            )

        if self.fallback_model is not None:
            logger.info(
                "Retrying stage '%s' using fallback model: %s",
                stage_name,
                self.config.fallback_model_name,
            )

            try:
                response = (
                    self.fallback_model.generate_text(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                )

                cleaned_response = clean_model_output(
                    response
                )

                if not cleaned_response:
                    raise OllamaGenerationError(
                        "Fallback model returned empty "
                        f"output during stage: {stage_name}"
                    )

                return cleaned_response

            except Exception as fallback_error:
                logger.error(
                    "Fallback model also failed during "
                    "%s: %s",
                    stage_name,
                    fallback_error,
                )

                raise OllamaGenerationError(
                    f"Both models failed during "
                    f"'{stage_name}'. "
                    f"Primary error: {primary_error}. "
                    f"Fallback error: {fallback_error}"
                ) from fallback_error

        raise OllamaGenerationError(
            f"Primary model failed during "
            f"'{stage_name}' and no fallback model "
            f"was available. Error: {primary_error}"
        ) from primary_error

    # =========================================================================
    # CHUNK EVIDENCE EXTRACTION
    # =========================================================================

    def extract_evidence_from_chunk(
        self,
        *,
        clinical_chunk: str,
        chunk_number: int,
        total_chunks: int,
        patient_name: str,
    ) -> str:
        """
        Extract discharge-relevant evidence from one clinical-text chunk.

        Args:
            clinical_chunk:
                One section of the normalized patient record.

            chunk_number:
                One-based chunk number.

            total_chunks:
                Total number of chunks.

            patient_name:
                Patient name used in progress logs.

        Returns:
            Structured clinical evidence extracted from the chunk.
        """

        logger.info(
            "[%s] Extracting evidence from chunk %d/%d.",
            patient_name,
            chunk_number,
            total_chunks,
        )

        prompt = build_chunk_extraction_prompt(
            clinical_chunk,
            chunk_number=chunk_number,
            total_chunks=total_chunks,
        )

        try:
            extracted_evidence = (
                self._generate_with_fallback(
                    system_prompt=(
                        EVIDENCE_EXTRACTION_SYSTEM_PROMPT
                    ),
                    user_prompt=prompt,
                    stage_name=(
                        f"evidence extraction chunk "
                        f"{chunk_number}/{total_chunks}"
                    ),
                )
            )
        except Exception as exc:
            raise EvidenceExtractionError(
                f"Could not extract evidence from "
                f"chunk {chunk_number} of {total_chunks}: "
                f"{exc}"
            ) from exc

        logger.debug(
            "[%s] Chunk %d extraction statistics: %s",
            patient_name,
            chunk_number,
            text_statistics(extracted_evidence),
        )

        return extracted_evidence

    def extract_all_chunk_evidence(
        self,
        *,
        chunks: list[str],
        patient_name: str,
    ) -> list[str]:
        """
        Extract evidence from every patient-text chunk.

        Returns:
            List containing one extracted-evidence block per chunk.
        """

        if not chunks:
            raise EvidenceExtractionError(
                "No clinical chunks were provided for "
                "evidence extraction."
            )

        extracted_chunks: list[str] = []
        total_chunks = len(chunks)

        logger.info(
            "[%s] Starting evidence extraction from "
            "%d chunk(s).",
            patient_name,
            total_chunks,
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            extracted_evidence = (
                self.extract_evidence_from_chunk(
                    clinical_chunk=chunk,
                    chunk_number=index,
                    total_chunks=total_chunks,
                    patient_name=patient_name,
                )
            )

            extracted_chunks.append(
                extracted_evidence
            )

        if not extracted_chunks:
            raise EvidenceExtractionError(
                "Evidence extraction completed without "
                "producing any output."
            )

        logger.info(
            "[%s] Evidence extraction completed.",
            patient_name,
        )

        return extracted_chunks

    def combine_extracted_chunks(
        self,
        extracted_chunks: list[str],
    ) -> str:
        """
        Combine chunk-level extracted evidence into one text block.

        Chunk labels are retained for traceability before consolidation.
        """

        if not extracted_chunks:
            raise EvidenceExtractionError(
                "Cannot combine an empty evidence list."
            )

        combined_sections: list[str] = []

        total_chunks = len(extracted_chunks)

        for index, evidence in enumerate(
            extracted_chunks,
            start=1,
        ):
            cleaned_evidence = clean_model_output(
                evidence
            )

            if not cleaned_evidence:
                continue

            section = (
                f"===== EXTRACTED EVIDENCE "
                f"CHUNK {index} OF {total_chunks} =====\n\n"
                f"{cleaned_evidence}"
            )

            combined_sections.append(section)

        combined_text = "\n\n".join(
            combined_sections
        ).strip()

        if not combined_text:
            raise EvidenceExtractionError(
                "All extracted chunk evidence was empty."
            )

        return combined_text

    # =========================================================================
    # EVIDENCE CONSOLIDATION
    # =========================================================================

    def consolidate_evidence(
        self,
        *,
        extracted_evidence: str,
        patient_name: str,
    ) -> str:
        """
        Consolidate all chunk-level evidence into one clinical evidence record.
        """

        if not extracted_evidence.strip():
            raise EvidenceConsolidationError(
                "Extracted evidence cannot be empty."
            )

        logger.info(
            "[%s] Consolidating extracted clinical evidence.",
            patient_name,
        )

        prompt = build_evidence_consolidation_prompt(
            extracted_evidence
        )

        try:
            consolidated_evidence = (
                self._generate_with_fallback(
                    system_prompt=(
                        EVIDENCE_EXTRACTION_SYSTEM_PROMPT
                    ),
                    user_prompt=prompt,
                    stage_name="evidence consolidation",
                )
            )
        except Exception as exc:
            raise EvidenceConsolidationError(
                "Could not consolidate extracted "
                f"evidence: {exc}"
            ) from exc

        if not consolidated_evidence.strip():
            raise EvidenceConsolidationError(
                "Evidence consolidation returned empty "
                "output."
            )

        logger.info(
            "[%s] Evidence consolidation completed.",
            patient_name,
        )

        logger.debug(
            "[%s] Consolidated evidence statistics: %s",
            patient_name,
            text_statistics(consolidated_evidence),
        )

        return consolidated_evidence

    # =========================================================================
    # DOCTOR-FRIENDLY EVIDENCE REPORT
    # =========================================================================

    def generate_evidence_report(
        self,
        *,
        consolidated_evidence: str,
        patient_name: str,
    ) -> str:
        """
        Generate the doctor-friendly evidence report.

        This is the first clinical output.
        """

        if not consolidated_evidence.strip():
            raise EvidenceReportGenerationError(
                "Consolidated evidence cannot be empty."
            )

        logger.info(
            "[%s] Generating doctor-friendly "
            "evidence report.",
            patient_name,
        )

        prompt = build_evidence_report_prompt(
            consolidated_evidence
        )

        try:
            evidence_report = (
                self._generate_with_fallback(
                    system_prompt=(
                        EVIDENCE_REPORT_SYSTEM_PROMPT
                    ),
                    user_prompt=prompt,
                    stage_name=(
                        "doctor-friendly evidence report"
                    ),
                )
            )
        except Exception as exc:
            raise EvidenceReportGenerationError(
                "Could not generate the doctor-friendly "
                f"evidence report: {exc}"
            ) from exc

        if not evidence_report.strip():
            raise EvidenceReportGenerationError(
                "Evidence-report generation returned "
                "empty output."
            )

        logger.info(
            "[%s] Doctor-friendly evidence report "
            "generated.",
            patient_name,
        )

        logger.debug(
            "[%s] Evidence report statistics: %s",
            patient_name,
            text_statistics(evidence_report),
        )

        return evidence_report

    # =========================================================================
    # TEMPLATE SELECTION
    # =========================================================================

    def select_template(
        self,
        *,
        consolidated_evidence: str,
        patient_name: str,
    ) -> str:
        """
        Select the closest available hospital discharge template.

        Template selection uses the consolidated evidence rather than the
        doctor-friendly report because the consolidated evidence preserves more
        extraction detail.
        """

        if not consolidated_evidence.strip():
            raise TemplateSelectionError(
                "Consolidated evidence cannot be empty "
                "for template selection."
            )

        logger.info(
            "[%s] Selecting hospital discharge template.",
            patient_name,
        )

        prompt = build_template_selection_prompt(
            consolidated_evidence
        )

        try:
            raw_selection = (
                self._generate_with_fallback(
                    system_prompt=(
                        TEMPLATE_SELECTION_SYSTEM_PROMPT
                    ),
                    user_prompt=prompt,
                    stage_name="template selection",
                )
            )

            selected_template = (
                validate_template_selection(
                    raw_selection
                )
            )

        except Exception as exc:
            raise TemplateSelectionError(
                f"Could not select a discharge "
                f"template: {exc}"
            ) from exc

        if selected_template == "TEMPLATE_UNCERTAIN":
            raise TemplateSelectionError(
                "The model could not determine the "
                "appropriate hospital template. "
                "Gestational category or principal "
                "condition requires clinical review."
            )

        logger.info(
            "[%s] Selected template: %s",
            patient_name,
            selected_template,
        )

        return selected_template

    # =========================================================================
    # FINAL DISCHARGE SUMMARY GENERATION
    # =========================================================================

    def generate_discharge_summary(
        self,
        *,
        template_name: str,
        evidence_report: str,
        patient_name: str,
    ) -> str:
        """
        Generate the final hospital discharge-summary text.

        This is the second clinical output.
        """

        if not evidence_report.strip():
            raise DischargeGenerationError(
                "Evidence report cannot be empty."
            )

        logger.info(
            "[%s] Generating final hospital "
            "discharge summary.",
            patient_name,
        )

        try:
            prompt = build_final_discharge_prompt(
                template_name=template_name,
                evidence_report=evidence_report,
            )
        except Exception as exc:
            raise DischargeGenerationError(
                "Could not build the final discharge "
                f"prompt: {exc}"
            ) from exc

        try:
            discharge_summary = (
                self._generate_with_fallback(
                    system_prompt=(
                        CLINICAL_SYSTEM_PROMPT
                    ),
                    user_prompt=prompt,
                    stage_name=(
                        "final hospital discharge summary"
                    ),
                )
            )
        except Exception as exc:
            raise DischargeGenerationError(
                "Could not generate the final hospital "
                f"discharge summary: {exc}"
            ) from exc

        if not discharge_summary.strip():
            raise DischargeGenerationError(
                "Discharge-summary generation returned "
                "empty output."
            )

        logger.info(
            "[%s] Final discharge summary generated.",
            patient_name,
        )

        logger.debug(
            "[%s] Discharge summary statistics: %s",
            patient_name,
            text_statistics(discharge_summary),
        )

        return discharge_summary

    # =========================================================================
    # SINGLE PATIENT PROCESSING
    # =========================================================================

    def process_patient_file(
        self,
        input_file: Path,
    ) -> PatientGenerationResult:
        """
        Process one normalized patient file through the complete generation
        workflow.

        This method generates:

        1. Doctor-friendly evidence report.
        2. Hospital discharge summary.
        """

        started_at = time.perf_counter()

        patient_name = (
            extract_patient_name_from_filename(
                input_file
            )
        )

        result = PatientGenerationResult(
            patient_name=patient_name,
            input_file=input_file,
        )

        logger.info(
            "=" * 70
        )
        logger.info(
            "Starting discharge generation for: %s",
            patient_name,
        )
        logger.info(
            "Input file: %s",
            input_file,
        )

        try:
            (
                patient_name,
                normalized_text,
                chunks,
            ) = self.prepare_patient_input(
                input_file
            )

            result.patient_name = patient_name
            result.chunk_count = len(chunks)

            result.statistics["normalized_input"] = (
                text_statistics(normalized_text)
            )

            extracted_chunks = (
                self.extract_all_chunk_evidence(
                    chunks=chunks,
                    patient_name=patient_name,
                )
            )

            combined_extracted_evidence = (
                self.combine_extracted_chunks(
                    extracted_chunks
                )
            )

            result.statistics["raw_extraction"] = (
                text_statistics(
                    combined_extracted_evidence
                )
            )

            if self.config.save_intermediate_files:
                result.raw_extraction_path = (
                    save_raw_extracted_evidence(
                        patient_name,
                        combined_extracted_evidence,
                    )
                )

            consolidated_evidence = (
                self.consolidate_evidence(
                    extracted_evidence=(
                        combined_extracted_evidence
                    ),
                    patient_name=patient_name,
                )
            )

            result.statistics[
                "consolidated_evidence"
            ] = text_statistics(
                consolidated_evidence
            )

            if self.config.save_intermediate_files:
                result.consolidated_evidence_path = (
                    save_consolidated_evidence(
                        patient_name,
                        consolidated_evidence,
                    )
                )

            evidence_report = (
                self.generate_evidence_report(
                    consolidated_evidence=(
                        consolidated_evidence
                    ),
                    patient_name=patient_name,
                )
            )

            result.statistics["evidence_report"] = (
                text_statistics(evidence_report)
            )

            result.evidence_report_path = (
                save_evidence_report(
                    patient_name,
                    evidence_report,
                )
            )

            selected_template = (
                self.select_template(
                    consolidated_evidence=(
                        consolidated_evidence
                    ),
                    patient_name=patient_name,
                )
            )

            result.selected_template = (
                selected_template
            )

            discharge_summary = (
                self.generate_discharge_summary(
                    template_name=selected_template,
                    evidence_report=evidence_report,
                    patient_name=patient_name,
                )
            )

            result.statistics[
                "discharge_summary"
            ] = text_statistics(
                discharge_summary
            )

            output_paths = (
                save_discharge_summary_documents(
                    patient_name,
                    discharge_summary,
                    create_pdf=(
                        self.config.create_pdf
                    ),
                )
            )

            result.discharge_text_path = (
                output_paths["text"]
            )

            result.discharge_docx_path = (
                output_paths["docx"]
            )

            result.discharge_pdf_path = (
                output_paths["pdf"]
            )

            result.success = True

            logger.info(
                "[%s] Processing completed successfully.",
                patient_name,
            )

        except Exception as exc:
            result.success = False
            result.error_message = str(exc)

            logger.exception(
                "[%s] Processing failed: %s",
                patient_name,
                exc,
            )

            try:
                result.error_report_path = (
                    save_error_report(
                        patient_name=patient_name,
                        stage="patient_processing",
                        error=exc,
                        additional_details=(
                            f"Input file: {input_file}"
                        ),
                    )
                )
            except Exception as error_save_exc:
                logger.error(
                    "Could not save error report for "
                    "%s: %s",
                    patient_name,
                    error_save_exc,
                )

        finally:
            result.elapsed_seconds = (
                time.perf_counter() - started_at
            )

            logger.info(
                "[%s] Runtime: %.1f seconds.",
                patient_name,
                result.elapsed_seconds,
            )

        return result

        # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    def process_all_files(
        self,
        input_files: list[Path] | None = None,
    ) -> PipelineRunSummary:
        """
        Process all discovered normalized patient files.

        Args:
            input_files:
                Optional list of files to process. When omitted, files are
                discovered from the configured normalizer output directory.

        Returns:
            PipelineRunSummary containing one result per patient.
        """

        if not self._startup_validated:
            self.validate_startup()

        files_to_process = (
            input_files
            if input_files is not None
            else self.discover_input_files()
        )

        summary = PipelineRunSummary()

        if not files_to_process:
            logger.warning(
                "No normalized patient files were found."
            )

            summary.completed_at = datetime.now()
            return summary

        total_files = len(files_to_process)

        logger.info(
            "Starting batch processing for %d patient file(s).",
            total_files,
        )

        for index, input_file in enumerate(
            files_to_process,
            start=1,
        ):
            logger.info(
                "Processing patient file %d of %d.",
                index,
                total_files,
            )

            result = self.process_patient_file(
                input_file
            )

            summary.results.append(result)

        summary.completed_at = datetime.now()

        logger.info(
            "Batch processing complete. "
            "Successful: %d | Failed: %d | Total: %d",
            summary.successful_files,
            summary.failed_files,
            summary.total_files,
        )

        return summary

    def process_single_file(
        self,
        input_file: Path,
    ) -> PatientGenerationResult:
        """
        Validate startup and process one specific normalized patient file.
        """

        if not self._startup_validated:
            self.validate_startup()

        if not input_file.exists():
            raise FileNotFoundError(
                f"Input file was not found: {input_file}"
            )

        if not input_file.is_file():
            raise ValueError(
                f"Input path is not a file: {input_file}"
            )

        return self.process_patient_file(
            input_file
        )


# =============================================================================
# TERMINAL OUTPUT HELPERS
# =============================================================================

def format_duration(seconds: float) -> str:
    """
    Format seconds as a readable duration.
    """

    total_seconds = max(
        0,
        int(round(seconds)),
    )

    hours, remainder = divmod(
        total_seconds,
        3600,
    )

    minutes, remaining_seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours}h {minutes}m "
            f"{remaining_seconds}s"
        )

    if minutes:
        return (
            f"{minutes}m "
            f"{remaining_seconds}s"
        )

    return f"{remaining_seconds}s"


def print_patient_result(
    result: PatientGenerationResult,
) -> None:
    """
    Display the result of processing one patient.
    """

    print()
    print("=" * 70)
    print(f"Patient: {result.patient_name}")
    print(f"Input: {result.input_file}")
    print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Chunks processed: {result.chunk_count}")
    print(
        "Runtime: "
        f"{format_duration(result.elapsed_seconds)}"
    )

    if result.selected_template:
        print(
            "Selected template: "
            f"{result.selected_template}"
        )

    if result.success:
        print()
        print("Generated outputs:")

        if result.evidence_report_path:
            print(
                "  Evidence report: "
                f"{result.evidence_report_path}"
            )

        if result.discharge_text_path:
            print(
                "  Discharge summary TXT: "
                f"{result.discharge_text_path}"
            )

        if result.discharge_docx_path:
            print(
                "  Discharge summary DOCX: "
                f"{result.discharge_docx_path}"
            )

        if result.discharge_pdf_path:
            print(
                "  Discharge summary PDF: "
                f"{result.discharge_pdf_path}"
            )
        else:
            print(
                "  Discharge summary PDF: "
                "Not generated"
            )

        if result.raw_extraction_path:
            print(
                "  Raw extracted evidence: "
                f"{result.raw_extraction_path}"
            )

        if result.consolidated_evidence_path:
            print(
                "  Consolidated evidence: "
                f"{result.consolidated_evidence_path}"
            )

    else:
        print()
        print(
            "Error: "
            f"{result.error_message or 'Unknown error'}"
        )

        if result.error_report_path:
            print(
                "Error report: "
                f"{result.error_report_path}"
            )

    if result.statistics:
        print()
        print("Text statistics:")

        for stage_name, statistics in (
            result.statistics.items()
        ):
            readable_name = (
                stage_name
                .replace("_", " ")
                .title()
            )

            print(
                f"  {readable_name}: "
                f"{statistics.get('characters', 0):,} chars, "
                f"{statistics.get('words', 0):,} words, "
                f"{statistics.get('lines', 0):,} lines"
            )


def print_pipeline_summary(
    summary: PipelineRunSummary,
) -> None:
    """
    Display the result of a complete batch run.
    """

    print()
    print("=" * 70)
    print("NICU DISCHARGE SUMMARY PIPELINE RESULT")
    print("=" * 70)
    print(
        f"Total patient files: {summary.total_files}"
    )
    print(
        f"Successful: {summary.successful_files}"
    )
    print(
        f"Failed: {summary.failed_files}"
    )
    print(
        "Total runtime: "
        f"{format_duration(summary.elapsed_seconds)}"
    )

    for result in summary.results:
        print_patient_result(result)


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate a doctor-friendly NICU evidence report "
            "and hospital discharge summary from normalized "
            "patient records."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "Process one normalized patient TXT file. "
            "When omitted, all normalized files are processed."
        ),
    )

    parser.add_argument(
        "--input-directory",
        type=Path,
        default=NORMALIZER_OUTPUT_DIRECTORY,
        help=(
            "Directory containing normalized patient files. "
            f"Default: {NORMALIZER_OUTPUT_DIRECTORY}"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "Primary Ollama model name. "
            f"Default: {DEFAULT_MODEL_NAME}"
        ),
    )

    parser.add_argument(
        "--fallback-model",
        type=str,
        default=DEFAULT_FALLBACK_MODEL_NAME,
        help=(
            "Fallback Ollama model name. "
            "Use 'none' to disable fallback."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Maximum characters per normalized-text chunk. "
            f"Default: {DEFAULT_CHUNK_SIZE}"
        ),
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=(
            "Character overlap between chunks. "
            f"Default: {DEFAULT_CHUNK_OVERLAP}"
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Ollama generation temperature. "
            f"Default: {DEFAULT_TEMPERATURE}"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Model request timeout in seconds. "
            f"Default: {DEFAULT_REQUEST_TIMEOUT}"
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum Ollama request attempts. "
            f"Default: {DEFAULT_MAX_RETRIES}"
        ),
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help=(
            "Delay between request retries in seconds. "
            f"Default: {DEFAULT_RETRY_DELAY_SECONDS}"
        ),
    )

    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help=(
            "Skip DOCX-to-PDF conversion."
        ),
    )

    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help=(
            "Do not save raw extracted evidence or "
            "consolidated evidence files."
        ),
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help=(
            "Skip model warmup before generation."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Run template, directory and Ollama checks "
            "without processing patient files."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Enable detailed DEBUG logging."
        ),
    )

    return parser


def normalize_optional_model_name(
    model_name: str | None,
) -> str | None:
    """
    Convert values such as 'none' or an empty string into None.
    """

    if model_name is None:
        return None

    cleaned_name = model_name.strip()

    if cleaned_name.lower() in {
        "",
        "none",
        "null",
        "disabled",
        "off",
    }:
        return None

    return cleaned_name


def create_config_from_arguments(
    arguments: argparse.Namespace,
) -> GeneratorConfig:
    """
    Build GeneratorConfig from command-line arguments.
    """

    fallback_model_name = (
        normalize_optional_model_name(
            arguments.fallback_model
        )
    )

    config = GeneratorConfig(
        model_name=arguments.model.strip(),
        fallback_model_name=fallback_model_name,
        chunk_size=arguments.chunk_size,
        chunk_overlap=arguments.chunk_overlap,
        temperature=arguments.temperature,
        max_retries=arguments.max_retries,
        retry_delay_seconds=arguments.retry_delay,
        request_timeout=arguments.timeout,
        save_intermediate_files=(
            not arguments.no_intermediate
        ),
        create_pdf=not arguments.no_pdf,
        input_directory=(
            arguments.input_directory.resolve()
        ),
        output_directory=OUTPUT_DIRECTORY,
    )

    config.validate()

    return config


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """
    Run the NICU discharge-summary generator from the terminal.

    Returns:
        Shell exit code:

        0 = all processing succeeded
        1 = one or more patient files failed
        2 = startup or configuration failure
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    configure_logging(
        verbose=arguments.verbose
    )

    logger.info(
        "NICU discharge-summary pipeline started."
    )

    try:
        config = create_config_from_arguments(
            arguments
        )

        generator = DischargeSummaryGenerator(
            config=config
        )

        generator.validate_startup()

        if arguments.validate_only:
            print()
            print(
                "Startup validation passed successfully."
            )
            print(
                f"Primary model: {config.model_name}"
            )
            print(
                "Fallback model: "
                f"{config.fallback_model_name or 'Disabled'}"
            )
            print(
                f"Input directory: {config.input_directory}"
            )
            print(
                f"Output directory: {config.output_directory}"
            )
            print(
                "PDF generation: "
                f"{'Enabled' if config.create_pdf else 'Disabled'}"
            )
            print(
                "Intermediate files: "
                f"{'Enabled' if config.save_intermediate_files else 'Disabled'}"
            )

            return 0

        if not arguments.skip_warmup:
            generator.warmup_models()

        if arguments.input_file is not None:
            input_file = arguments.input_file.resolve()

            result = generator.process_single_file(
                input_file
            )

            print_patient_result(result)

            return 0 if result.success else 1

        summary = generator.process_all_files()

        print_pipeline_summary(summary)

        if summary.failed_files > 0:
            return 1

        return 0

    except (
        StartupValidationError,
        TemplateError,
        OllamaConnectionError,
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
    ) as exc:
        logger.exception(
            "Pipeline startup failed: %s",
            exc,
        )

        print()
        print("=" * 70)
        print("PIPELINE COULD NOT START")
        print("=" * 70)
        print(str(exc))

        return 2

    except KeyboardInterrupt:
        logger.warning(
            "Pipeline stopped by the user."
        )

        print()
        print(
            "Processing stopped by the user."
        )

        return 130

    except Exception as exc:
        logger.exception(
            "Unexpected pipeline failure: %s",
            exc,
        )

        print()
        print("=" * 70)
        print("UNEXPECTED PIPELINE ERROR")
        print("=" * 70)
        print(str(exc))

        return 2


if __name__ == "__main__":
    raise SystemExit(main())