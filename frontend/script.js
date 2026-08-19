const API_BASE_URL = "http://127.0.0.1:8000/api";

const pdfInput = document.getElementById("pdfInput");
const selectedFileName = document.getElementById("selectedFileName");
const generateButton = document.getElementById("generateButton");

const progressSection = document.getElementById("progressSection");
const resultSection = document.getElementById("resultSection");
const errorSection = document.getElementById("errorSection");

const progressBar = document.getElementById("progressBar");
const progressPercent = document.getElementById("progressPercent");
const statusMessage = document.getElementById("statusMessage");

const resultMessage = document.getElementById("resultMessage");
const summaryPanel = document.getElementById("summaryPanel");
const summaryContent = document.getElementById("summaryContent");
const errorMessage = document.getElementById("errorMessage");

const viewSummaryButton = document.getElementById("viewSummaryButton");
const downloadTextButton = document.getElementById("downloadTextButton");
const closeSummaryButton = document.getElementById("closeSummaryButton");
const newPatientButton = document.getElementById("newPatientButton");
const retryButton = document.getElementById("retryButton");

const stageElements = {
    upload: document.querySelector('[data-stage="upload"]'),
    ocr: document.querySelector('[data-stage="ocr"]'),
    structurer: document.querySelector('[data-stage="structurer"]'),
    llm: document.querySelector('[data-stage="llm"]'),
    normalizer: document.querySelector('[data-stage="normalizer"]'),
    discharge: document.querySelector('[data-stage="discharge"]'),
};

let selectedFile = null;
let progressTimer = null;
let summaryFilename = "nicu_discharge_summary.txt";
let summaryText = "";


pdfInput.addEventListener("change", () => {
    if (!pdfInput.files.length) {
        selectedFile = null;
        selectedFileName.textContent = "No file selected";
        generateButton.disabled = true;
        return;
    }

    selectedFile = pdfInput.files[0];

    if (
        selectedFile.type !== "application/pdf"
        && !selectedFile.name.toLowerCase().endsWith(".pdf")
    ) {
        selectedFile = null;
        pdfInput.value = "";
        selectedFileName.textContent = "Please select a PDF file";
        generateButton.disabled = true;
        return;
    }

    selectedFileName.textContent = selectedFile.name;
    generateButton.disabled = false;
});


generateButton.addEventListener("click", uploadPatient);


async function uploadPatient() {
    if (!selectedFile) {
        return;
    }

    resetPageForProcessing();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch(
            `${API_BASE_URL}/upload`,
            {
                method: "POST",
                body: formData,
            }
        );

        const data = await readJsonResponse(response);

        if (!response.ok) {
            throw new Error(
                data.detail || "The patient PDF could not be uploaded."
            );
        }

        statusMessage.textContent =
            data.message || "Processing has started.";

        startProgressPolling();

    } catch (error) {
        showError(error.message);
    }
}


function startProgressPolling() {
    stopProgressPolling();

    fetchProgress();

    progressTimer = window.setInterval(
        fetchProgress,
        1500
    );
}


function stopProgressPolling() {
    if (progressTimer !== null) {
        window.clearInterval(progressTimer);
        progressTimer = null;
    }
}


async function fetchProgress() {
    try {
        const response = await fetch(
            `${API_BASE_URL}/progress`,
            {
                cache: "no-store",
            }
        );

        const data = await readJsonResponse(response);

        if (!response.ok) {
            throw new Error(
                data.detail || "Could not read pipeline progress."
            );
        }

        updateProgressInterface(data);

        if (data.stage === "completed") {
            stopProgressPolling();
            await loadSummary();
        }

        if (data.stage === "failed") {
            stopProgressPolling();
            showError(
                data.error
                || data.message
                || "The processing pipeline failed."
            );
        }

    } catch (error) {
        stopProgressPolling();
        showError(error.message);
    }
}


function updateProgressInterface(progress) {
    const percentage = Number(progress.percentage || 0);

    progressBar.style.width = `${percentage}%`;
    progressPercent.textContent = `${percentage}%`;
    statusMessage.textContent =
        progress.message || progress.stage_label || "Processing...";

    const completedStages = new Set(
        Array.isArray(progress.completed_stages)
            ? progress.completed_stages
            : []
    );

    Object.entries(stageElements).forEach(
        ([stageName, element]) => {
            if (!element) {
                return;
            }

            element.classList.remove(
                "active",
                "complete"
            );

            if (completedStages.has(stageName)) {
                element.classList.add("complete");
            }

            if (
                progress.stage === stageName
                && progress.running
            ) {
                element.classList.add("active");
            }

            if (progress.stage === "completed") {
                element.classList.remove("active");
                element.classList.add("complete");
            }
        }
    );
}


async function loadSummary() {
    try {
        const response = await fetch(
            `${API_BASE_URL}/summary/text`,
            {
                cache: "no-store",
            }
        );

        const data = await readJsonResponse(response);

        if (!response.ok) {
            throw new Error(
                data.detail || "The discharge summary could not be loaded."
            );
        }

        summaryFilename =
            data.filename || "nicu_discharge_summary.txt";

        summaryText =
            data.summary || "";

        summaryContent.textContent = summaryText;

        progressSection.classList.add("hidden");
        errorSection.classList.add("hidden");
        resultSection.classList.remove("hidden");

        resultMessage.textContent =
            "The patient record was processed successfully.";

    } catch (error) {
        showError(error.message);
    }
}


function resetPageForProcessing() {
    stopProgressPolling();

    generateButton.disabled = true;
    pdfInput.disabled = true;

    resultSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    summaryPanel.classList.add("hidden");
    progressSection.classList.remove("hidden");

    progressBar.style.width = "0%";
    progressPercent.textContent = "0%";
    statusMessage.textContent = "Uploading patient PDF...";

    Object.values(stageElements).forEach(element => {
        if (!element) {
            return;
        }

        element.classList.remove(
            "active",
            "complete"
        );
    });

    if (stageElements.upload) {
        stageElements.upload.classList.add("active");
    }
}


function showError(message) {
    stopProgressPolling();

    progressSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    errorSection.classList.remove("hidden");

    errorMessage.textContent =
        message || "An unknown error occurred.";

    generateButton.disabled = false;
    pdfInput.disabled = false;
}


async function readJsonResponse(response) {
    const text = await response.text();

    if (!text) {
        return {};
    }

    try {
        return JSON.parse(text);
    } catch {
        return {
            detail: text,
        };
    }
}


viewSummaryButton.addEventListener("click", () => {
    summaryPanel.classList.remove("hidden");
});


closeSummaryButton.addEventListener("click", () => {
    summaryPanel.classList.add("hidden");
});


downloadTextButton.addEventListener("click", () => {
    if (!summaryText) {
        return;
    }

    const blob = new Blob(
        [summaryText],
        {
            type: "text/plain;charset=utf-8",
        }
    );

    const downloadUrl = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = summaryFilename;

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(downloadUrl);
});


newPatientButton.addEventListener("click", resetApplication);
retryButton.addEventListener("click", resetApplication);


function resetApplication() {
    stopProgressPolling();

    selectedFile = null;
    summaryText = "";
    summaryFilename = "nicu_discharge_summary.txt";

    pdfInput.value = "";
    pdfInput.disabled = false;

    selectedFileName.textContent = "No file selected";
    generateButton.disabled = true;

    progressSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    summaryPanel.classList.add("hidden");

    progressBar.style.width = "0%";
    progressPercent.textContent = "0%";
    statusMessage.textContent = "Waiting to begin.";

    Object.values(stageElements).forEach(element => {
        if (!element) {
            return;
        }

        element.classList.remove(
            "active",
            "complete"
        );
    });
}