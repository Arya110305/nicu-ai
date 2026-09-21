/* ==========================================================
   NICU AI — FRONTEND APPLICATION
   ========================================================== */

const API_BASE_URL = "http://127.0.0.1:8000/api";


/* ==========================================================
   ELEMENTS
   ========================================================== */

const sidebar =
    document.getElementById("sidebar");

const mobileMenuButton =
    document.getElementById("mobileMenuButton");

const uploadSection =
    document.getElementById("uploadSection");

const processingSection =
    document.getElementById("processingSection");

const resultSection =
    document.getElementById("resultSection");

const summarySection =
    document.getElementById("summarySection");

const errorSection =
    document.getElementById("errorSection");

const uploadZone =
    document.getElementById("uploadZone");

const uploadPrompt =
    document.getElementById("uploadPrompt");

const uploadSelectedName =
    document.getElementById("uploadSelectedName");

const pdfInput =
    document.getElementById("pdfInput");

const selectedFileCard =
    document.getElementById("selectedFileCard");

const selectedFileName =
    document.getElementById("selectedFileName");

const selectedFileMeta =
    document.getElementById("selectedFileMeta");

const removeFileButton =
    document.getElementById("removeFileButton");

const generateButton =
    document.getElementById("generateButton");

const processingMessage =
    document.getElementById("processingMessage");

const elapsedTime =
    document.getElementById("elapsedTime");

const finalProcessingTime =
    document.getElementById("finalProcessingTime");

const viewSummaryButton =
    document.getElementById("viewSummaryButton");

const downloadSummaryButton =
    document.getElementById("downloadSummaryButton");

const newPatientButton =
    document.getElementById("newPatientButton");

const closeSummaryButton =
    document.getElementById("closeSummaryButton");

const summaryContent =
    document.getElementById("summaryContent");

const errorMessage =
    document.getElementById("errorMessage");

const retryButton =
    document.getElementById("retryButton");


/* ==========================================================
   PIPELINE UI STAGES
   ========================================================== */

const stageElements = {

    upload:
        document.querySelector(
            '[data-stage="upload"]'
        ),

    ocr:
        document.querySelector(
            '[data-stage="ocr"]'
        ),

    llm:
        document.querySelector(
            '[data-stage="llm"]'
        ),

    normalizer:
        document.querySelector(
            '[data-stage="normalizer"]'
        ),

    discharge:
        document.querySelector(
            '[data-stage="discharge"]'
        ),
};


/* ==========================================================
   STATE
   ========================================================== */

let selectedFile = null;

let progressTimer = null;

let elapsedTimer = null;

let processStartedAt = null;

let processFinishedAt = null;

let summaryText = "";

let summaryFilename =
    "nicu_discharge_summary.txt";


/* ==========================================================
   MOBILE MENU
   ========================================================== */

if (
    mobileMenuButton
    && sidebar
) {

    mobileMenuButton.addEventListener(
        "click",
        () => {

            sidebar.classList.toggle(
                "mobile-open"
            );
        }
    );
}


/* ==========================================================
   FILE SELECTION
   ========================================================== */

if (pdfInput) {

    pdfInput.addEventListener(
        "change",
        () => {

            if (!pdfInput.files.length) {

                clearSelectedFile();

                return;
            }


            setSelectedFile(
                pdfInput.files[0]
            );
        }
    );
}


function setSelectedFile(file) {

    if (!file) {
        return;
    }


    const isPdf =
        file.type === "application/pdf"
        ||
        file.name
            .toLowerCase()
            .endsWith(".pdf");


    if (!isPdf) {

        clearSelectedFile();

        window.alert(
            "Please select a PDF file."
        );

        return;
    }


    selectedFile = file;


    if (uploadSelectedName) {

        uploadSelectedName.textContent =
            file.name;

        uploadSelectedName.classList.remove(
            "hidden"
        );
    }


    if (uploadPrompt) {

        uploadPrompt.textContent =
            "Selected patient record";
    }


    if (selectedFileName) {

        selectedFileName.textContent =
            file.name;
    }


    if (selectedFileMeta) {

        selectedFileMeta.textContent =
            `${formatFileSize(file.size)} • PDF document`;
    }


    if (selectedFileCard) {

        selectedFileCard.classList.remove(
            "hidden"
        );
    }


    if (generateButton) {

        generateButton.disabled =
            false;
    }
}


function clearSelectedFile() {

    selectedFile = null;


    if (pdfInput) {

        pdfInput.value = "";
    }


    if (uploadSelectedName) {

        uploadSelectedName.textContent =
            "";

        uploadSelectedName.classList.add(
            "hidden"
        );
    }


    if (uploadPrompt) {

        uploadPrompt.textContent =
            "Click to choose a PDF file or drag and drop it here.";
    }


    if (selectedFileName) {

        selectedFileName.textContent =
            "";
    }


    if (selectedFileMeta) {

        selectedFileMeta.textContent =
            "";
    }


    if (selectedFileCard) {

        selectedFileCard.classList.add(
            "hidden"
        );
    }


    if (generateButton) {

        generateButton.disabled =
            true;
    }
}


function formatFileSize(bytes) {

    if (!Number.isFinite(bytes)) {

        return "";
    }


    if (bytes < 1024) {

        return `${bytes} bytes`;
    }


    if (
        bytes
        < 1024 * 1024
    ) {

        return `${(
            bytes / 1024
        ).toFixed(1)} KB`;
    }


    return `${(
        bytes
        / (1024 * 1024)
    ).toFixed(1)} MB`;
}


/* ==========================================================
   REMOVE FILE
   ========================================================== */

if (removeFileButton) {

    removeFileButton.addEventListener(
        "click",
        clearSelectedFile
    );
}


/* ==========================================================
   DRAG + DROP
   ========================================================== */

if (uploadZone) {

    [
        "dragenter",
        "dragover",
    ].forEach(
        eventName => {

            uploadZone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();

                    event.stopPropagation();

                    uploadZone.classList.add(
                        "drag-active"
                    );
                }
            );
        }
    );


    [
        "dragleave",
        "drop",
    ].forEach(
        eventName => {

            uploadZone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();

                    event.stopPropagation();

                    uploadZone.classList.remove(
                        "drag-active"
                    );
                }
            );
        }
    );


    uploadZone.addEventListener(
        "drop",
        event => {

            const files =
                event.dataTransfer.files;


            if (!files.length) {

                return;
            }


            setSelectedFile(
                files[0]
            );
        }
    );
}


/* ==========================================================
   START PROCESSING
   ========================================================== */

if (generateButton) {

    generateButton.addEventListener(
        "click",
        uploadPatient
    );
}


async function uploadPatient() {

    if (!selectedFile) {

        return;
    }


    prepareProcessingState();


    const formData =
        new FormData();


    formData.append(
        "file",
        selectedFile
    );


    try {

        const response =
            await fetch(
                `${API_BASE_URL}/upload`,
                {
                    method: "POST",
                    body: formData,
                }
            );


        const data =
            await readJsonResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                data.detail
                || data.error
                || "The patient PDF could not be uploaded."
            );
        }


        markStageComplete(
            "upload"
        );


        markStageActive(
            "ocr"
        );


        if (processingMessage) {

            processingMessage.textContent =
                "OCR is reading the complete patient record.";
        }


        startProgressPolling();

    } catch (error) {

        showError(
            error.message
        );
    }
}


/* ==========================================================
   PREPARE PROCESSING SCREEN
   ========================================================== */

function prepareProcessingState() {

    stopProgressPolling();

    stopElapsedTimer();


    summaryText = "";

    summaryFilename =
        "nicu_discharge_summary.txt";


    processStartedAt =
        Date.now();

    processFinishedAt =
        null;


    if (uploadSection) {

        uploadSection.classList.add(
            "hidden"
        );
    }


    if (resultSection) {

        resultSection.classList.add(
            "hidden"
        );
    }


    if (summarySection) {

        summarySection.classList.add(
            "hidden"
        );
    }


    if (errorSection) {

        errorSection.classList.add(
            "hidden"
        );
    }


    if (processingSection) {

        processingSection.classList.remove(
            "hidden"
        );
    }


    resetPipelineStages();


    markStageActive(
        "upload"
    );


    if (processingMessage) {

        processingMessage.textContent =
            "Uploading patient PDF...";
    }


    if (elapsedTime) {

        elapsedTime.textContent =
            "00:00";
    }


    startElapsedTimer();
}


/* ==========================================================
   TIMER
   ========================================================== */

function startElapsedTimer() {

    stopElapsedTimer();


    updateElapsedDisplay();


    elapsedTimer =
        window.setInterval(
            updateElapsedDisplay,
            1000
        );
}


function stopElapsedTimer() {

    if (
        elapsedTimer
        !== null
    ) {

        window.clearInterval(
            elapsedTimer
        );


        elapsedTimer =
            null;
    }
}


function updateElapsedDisplay() {

    if (
        !processStartedAt
        || !elapsedTime
    ) {

        return;
    }


    const endTime =
        processFinishedAt
        || Date.now();


    const elapsedMilliseconds =
        Math.max(
            0,
            endTime
            - processStartedAt
        );


    elapsedTime.textContent =
        formatDuration(
            elapsedMilliseconds
        );
}


function formatDuration(
    milliseconds
) {

    const totalSeconds =
        Math.floor(
            milliseconds / 1000
        );


    const hours =
        Math.floor(
            totalSeconds / 3600
        );


    const minutes =
        Math.floor(
            (
                totalSeconds
                % 3600
            )
            / 60
        );


    const seconds =
        totalSeconds % 60;


    if (hours > 0) {

        return (
            String(hours)
            + ":"
            + String(minutes)
                .padStart(2, "0")
            + ":"
            + String(seconds)
                .padStart(2, "0")
        );
    }


    return (
        String(minutes)
            .padStart(2, "0")
        + ":"
        + String(seconds)
            .padStart(2, "0")
    );
}


/* ==========================================================
   POLLING
   ========================================================== */

function startProgressPolling() {

    stopProgressPolling();


    fetchProgress();


    progressTimer =
        window.setInterval(
            fetchProgress,
            1500
        );
}


function stopProgressPolling() {

    if (
        progressTimer
        !== null
    ) {

        window.clearInterval(
            progressTimer
        );


        progressTimer =
            null;
    }
}


async function fetchProgress() {

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/progress`,
                {
                    cache: "no-store",
                }
            );


        const progress =
            await readJsonResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                progress.detail
                || progress.error
                || "Could not read processing progress."
            );
        }


        updateProgressInterface(
            progress
        );


        const stage =
            String(
                progress.stage
                || ""
            )
            .trim()
            .toLowerCase();


        if (
            stage === "completed"
        ) {

            stopProgressPolling();


            processFinishedAt =
                Date.now();


            updateElapsedDisplay();

            stopElapsedTimer();


            await loadSummary();


            return;
        }


        if (
            stage === "failed"
        ) {

            stopProgressPolling();

            stopElapsedTimer();


            showError(
                progress.error
                || progress.message
                || "The processing pipeline failed."
            );
        }

    } catch (error) {

        stopProgressPolling();

        stopElapsedTimer();


        showError(
            error.message
        );
    }
}


/* ==========================================================
   SEQUENTIAL VISUAL PROGRESS

   IMPORTANT:
   Backend still streams OCR → LLM → Normalizer in parallel.

   The interface displays the stages sequentially based on
   COMPLETE stages, not merely ACTIVE worker threads.
   ========================================================== */

function updateProgressInterface(
    progress
) {

    const completed =
        new Set(
            Array.isArray(
                progress.completed_stages
            )
                ? progress.completed_stages
                : []
        );


    const backendStage =
        String(
            progress.stage
            || ""
        )
        .trim()
        .toLowerCase();


    /*
    ----------------------------------------------------------
    UPLOAD
    ----------------------------------------------------------
    */

    if (
        !completed.has(
            "upload"
        )
    ) {

        setVisualStage(
            "upload"
        );


        setProcessingMessage(
            "Uploading patient PDF..."
        );


        return;
    }


    /*
    ----------------------------------------------------------
    OCR

    OCR remains visually active until the ENTIRE OCR stage
    has completed, even though LLM and Normalizer workers
    may already be running internally.
    ----------------------------------------------------------
    */

    if (
        !completed.has(
            "ocr"
        )
    ) {

        markStageComplete(
            "upload"
        );


        setVisualStage(
            "ocr"
        );


        setProcessingMessage(
            "OCR is reading the complete patient record."
        );


        return;
    }


    /*
    ----------------------------------------------------------
    CLINICAL EXTRACTION / LLM

    Only show this once OCR is fully completed.
    ----------------------------------------------------------
    */

    if (
        !completed.has(
            "llm"
        )
    ) {

        markStageComplete(
            "upload"
        );


        markStageComplete(
            "ocr"
        );


        setVisualStage(
            "llm"
        );


        setProcessingMessage(
            "OCR completed. Extracting clinical information."
        );


        return;
    }


    /*
    ----------------------------------------------------------
    NORMALIZER

    Only show after full LLM completion.
    ----------------------------------------------------------
    */

    if (
        !completed.has(
            "normalizer"
        )
    ) {

        markStageComplete(
            "upload"
        );


        markStageComplete(
            "ocr"
        );


        markStageComplete(
            "llm"
        );


        setVisualStage(
            "normalizer"
        );


        setProcessingMessage(
            "Clinical extraction completed. Normalizing patient information."
        );


        return;
    }


    /*
    ----------------------------------------------------------
    DISCHARGE SUMMARY

    Normalizer must be completely finished first.
    ----------------------------------------------------------
    */

    if (
        !completed.has(
            "discharge"
        )
    ) {

        markStageComplete(
            "upload"
        );


        markStageComplete(
            "ocr"
        );


        markStageComplete(
            "llm"
        );


        markStageComplete(
            "normalizer"
        );


        setVisualStage(
            "discharge"
        );


        setProcessingMessage(
            "Patient record normalized. Generating the discharge summary."
        );


        return;
    }


    /*
    ----------------------------------------------------------
    FINALIZATION

    Backend has an internal finalizing stage. Keep the visible
    discharge step active/completed while files are being saved.
    ----------------------------------------------------------
    */

    markStageComplete(
        "upload"
    );


    markStageComplete(
        "ocr"
    );


    markStageComplete(
        "llm"
    );


    markStageComplete(
        "normalizer"
    );


    markStageComplete(
        "discharge"
    );


    if (
        backendStage
        === "finalizing"
    ) {

        setProcessingMessage(
            "Discharge summary generated. Saving final patient record."
        );

        return;
    }


    if (
        backendStage
        === "completed"
    ) {

        setProcessingMessage(
            "Patient record completed successfully."
        );
    }
}


/* ==========================================================
   VISUAL STAGE
   ========================================================== */

function setVisualStage(
    stageName
) {

    Object.entries(
        stageElements
    ).forEach(
        ([
            name,
            element,
        ]) => {

            if (!element) {

                return;
            }


            if (
                name !== stageName
                &&
                !element.classList.contains(
                    "complete"
                )
            ) {

                element.classList.remove(
                    "active"
                );
            }
        }
    );


    markStageActive(
        stageName
    );
}


function setProcessingMessage(
    message
) {

    if (
        processingMessage
    ) {

        processingMessage.textContent =
            message;
    }
}


/* ==========================================================
   RESET STAGES
   ========================================================== */

function resetPipelineStages() {

    Object.values(
        stageElements
    ).forEach(
        element => {

            if (!element) {

                return;
            }


            element.classList.remove(
                "active",
                "complete"
            );


            const indicator =
                element.querySelector(
                    ".pipeline-indicator"
                );


            if (!indicator) {

                return;
            }


            if (
                !indicator.dataset
                    .originalNumber
            ) {

                indicator.dataset
                    .originalNumber =
                    indicator.textContent.trim();
            }


            indicator.textContent =
                indicator.dataset
                    .originalNumber;
        }
    );
}


/* ==========================================================
   ACTIVE STAGE
   ========================================================== */

function markStageActive(
    stageName
) {

    const element =
        stageElements[
            stageName
        ];


    if (!element) {

        return;
    }


    if (
        element.classList.contains(
            "complete"
        )
    ) {

        return;
    }


    element.classList.add(
        "active"
    );
}


/* ==========================================================
   COMPLETE STAGE
   ========================================================== */

function markStageComplete(
    stageName
) {

    const element =
        stageElements[
            stageName
        ];


    if (!element) {

        return;
    }


    element.classList.remove(
        "active"
    );


    element.classList.add(
        "complete"
    );


    const indicator =
        element.querySelector(
            ".pipeline-indicator"
        );


    if (indicator) {

        if (
            !indicator.dataset
                .originalNumber
        ) {

            indicator.dataset
                .originalNumber =
                indicator.textContent.trim();
        }


        indicator.textContent =
            "✓";
    }
}


/* ==========================================================
   LOAD SUMMARY
   ========================================================== */

async function loadSummary() {

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/summary/text`,
                {
                    cache: "no-store",
                }
            );


        const data =
            await readJsonResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                data.detail
                || data.error
                || "The discharge summary could not be loaded."
            );
        }


        summaryFilename =
            data.filename
            || "nicu_discharge_summary.txt";


        summaryText =
            data.summary
            || data.text
            || "";


        if (summaryContent) {

            summaryContent.textContent =
                summaryText;
        }


        showSuccessfulResult();

    } catch (error) {

        showError(
            error.message
        );
    }
}


/* ==========================================================
   SUCCESS
   ========================================================== */

function showSuccessfulResult() {

    stopProgressPolling();

    stopElapsedTimer();


    if (!processFinishedAt) {

        processFinishedAt =
            Date.now();
    }


    if (processingSection) {

        processingSection.classList.add(
            "hidden"
        );
    }


    if (errorSection) {

        errorSection.classList.add(
            "hidden"
        );
    }


    if (summarySection) {

        summarySection.classList.add(
            "hidden"
        );
    }


    if (resultSection) {

        resultSection.classList.remove(
            "hidden"
        );
    }


    if (
        finalProcessingTime
        && processStartedAt
    ) {

        finalProcessingTime.textContent =
            formatFriendlyDuration(
                processFinishedAt
                - processStartedAt
            );
    }
}


function formatFriendlyDuration(
    milliseconds
) {

    const totalSeconds =
        Math.max(
            0,
            Math.floor(
                milliseconds / 1000
            )
        );


    const hours =
        Math.floor(
            totalSeconds / 3600
        );


    const minutes =
        Math.floor(
            (
                totalSeconds
                % 3600
            )
            / 60
        );


    const seconds =
        totalSeconds % 60;


    if (hours > 0) {

        return (
            `${hours} hr `
            + `${minutes} min `
            + `${seconds} sec`
        );
    }


    if (minutes > 0) {

        return (
            `${minutes} min `
            + `${seconds} sec`
        );
    }


    return `${seconds} sec`;
}


/* ==========================================================
   VIEW SUMMARY
   ========================================================== */

if (viewSummaryButton) {

    viewSummaryButton.addEventListener(
        "click",
        () => {

            if (!summarySection) {

                return;
            }


            summarySection.classList.remove(
                "hidden"
            );


            summarySection.scrollIntoView({
                behavior: "smooth",
                block: "start",
            });
        }
    );
}


if (closeSummaryButton) {

    closeSummaryButton.addEventListener(
        "click",
        () => {

            if (!summarySection) {

                return;
            }


            summarySection.classList.add(
                "hidden"
            );
        }
    );
}


/* ==========================================================
   DOWNLOAD
   ========================================================== */

if (downloadSummaryButton) {

    downloadSummaryButton.addEventListener(
        "click",
        () => {

            if (!summaryText) {

                return;
            }


            const blob =
                new Blob(
                    [summaryText],
                    {
                        type:
                            "text/plain;charset=utf-8",
                    }
                );


            const downloadUrl =
                URL.createObjectURL(
                    blob
                );


            const link =
                document.createElement(
                    "a"
                );


            link.href =
                downloadUrl;


            link.download =
                summaryFilename;


            document.body.appendChild(
                link
            );


            link.click();


            link.remove();


            URL.revokeObjectURL(
                downloadUrl
            );
        }
    );
}


/* ==========================================================
   ERROR
   ========================================================== */

function showError(
    message
) {

    stopProgressPolling();

    stopElapsedTimer();


    if (processingSection) {

        processingSection.classList.add(
            "hidden"
        );
    }


    if (resultSection) {

        resultSection.classList.add(
            "hidden"
        );
    }


    if (summarySection) {

        summarySection.classList.add(
            "hidden"
        );
    }


    if (errorSection) {

        errorSection.classList.remove(
            "hidden"
        );
    }


    if (errorMessage) {

        errorMessage.textContent =
            message
            || "An unknown processing error occurred.";
    }
}


/* ==========================================================
   RESET
   ========================================================== */

if (newPatientButton) {

    newPatientButton.addEventListener(
        "click",
        resetApplication
    );
}


if (retryButton) {

    retryButton.addEventListener(
        "click",
        resetApplication
    );
}


function resetApplication() {

    stopProgressPolling();

    stopElapsedTimer();


    processStartedAt =
        null;

    processFinishedAt =
        null;


    summaryText =
        "";

    summaryFilename =
        "nicu_discharge_summary.txt";


    clearSelectedFile();

    resetPipelineStages();


    if (processingSection) {

        processingSection.classList.add(
            "hidden"
        );
    }


    if (resultSection) {

        resultSection.classList.add(
            "hidden"
        );
    }


    if (summarySection) {

        summarySection.classList.add(
            "hidden"
        );
    }


    if (errorSection) {

        errorSection.classList.add(
            "hidden"
        );
    }


    if (uploadSection) {

        uploadSection.classList.remove(
            "hidden"
        );
    }


    if (elapsedTime) {

        elapsedTime.textContent =
            "00:00";
    }


    if (finalProcessingTime) {

        finalProcessingTime.textContent =
            "—";
    }


    if (processingMessage) {

        processingMessage.textContent =
            "Preparing patient record...";
    }
}


/* ==========================================================
   RESPONSE HELPER
   ========================================================== */

async function readJsonResponse(
    response
) {

    const text =
        await response.text();


    if (!text) {

        return {};
    }


    try {

        return JSON.parse(
            text
        );

    } catch {

        return {
            detail: text,
        };
    }
}