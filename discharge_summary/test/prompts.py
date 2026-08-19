"""
Prompt construction for the NICU discharge-summary generator.

This module:

1. Defines the clinical safety instructions.
2. Creates prompts for extracting evidence from normalized records.
3. Creates the prompt used to select one of four hospital templates.
4. Reads the selected hospital DOCX template.
5. Inserts the real template content into the final generation prompt.

The actual DOCX templates remain the source of truth. If the hospital changes
a template, replace the corresponding DOCX file without changing this code.

Required package:
    python-docx

Install using:
    pip install python-docx
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from docx import Document


# =============================================================================
# TEMPLATE NAMES
# =============================================================================

PRETERM_NNJ: Final[str] = "PRETERM_NNJ"
PRETERM_RD_TTNB: Final[str] = "PRETERM_RD_TTNB"
TERM_NNJ: Final[str] = "TERM_NNJ"
TERM_RD_TTNB: Final[str] = "TERM_RD_TTNB"

VALID_TEMPLATE_NAMES: Final[tuple[str, ...]] = (
    PRETERM_NNJ,
    PRETERM_RD_TTNB,
    TERM_NNJ,
    TERM_RD_TTNB,
)


# =============================================================================
# TEMPLATE FILE PATHS
# =============================================================================

MODULE_DIRECTORY: Final[Path] = Path(__file__).resolve().parent
TEMPLATE_DIRECTORY: Final[Path] = MODULE_DIRECTORY / "templates"

TEMPLATE_FILES: Final[dict[str, Path]] = {
    PRETERM_NNJ: (
        TEMPLATE_DIRECTORY / "PRETERM DISCHARGE NNJ TEMPLATE.docx"
    ),
    PRETERM_RD_TTNB: (
        TEMPLATE_DIRECTORY / "PRETERM DISCHARGE RD TTNB TEMPLATE.docx"
    ),
    TERM_NNJ: (
        TEMPLATE_DIRECTORY / "TERM DISCHARGE NNJ TEMPLATE.docx"
    ),
    TERM_RD_TTNB: (
        TEMPLATE_DIRECTORY / "TERM DISCHARGE RD TTNB TEMPLATE.docx"
    ),
}


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

CLINICAL_SYSTEM_PROMPT: Final[str] = """
You are assisting a consultant neonatologist in preparing a NICU discharge
summary from patient records.

This is a clinical documentation task.

Follow these rules strictly:

1. Use only information explicitly supported by the supplied patient record.
2. Never invent patient information.
3. Never assume that a standard template statement is true for this patient.
4. Never invent or estimate:
   - names
   - dates
   - times
   - gestational age
   - weights
   - diagnoses
   - examination findings
   - laboratory values
   - imaging findings
   - medications
   - doses
   - routes
   - frequencies
   - durations
   - follow-up dates
5. Preserve medication names and doses exactly as documented.
6. Preserve dates and chronology exactly as documented.
7. Do not convert uncertain information into a definite clinical statement.
8. Keep conflicting information visible and mark it for review.
9. Do not silently resolve conflicts.
10. Do not provide medical advice beyond the hospital template.
11. Do not output JSON.
12. Do not explain your reasoning.
13. Return only the requested clinical documentation.
14. The completed discharge summary must be reviewed and approved by the
    responsible clinician before clinical use.
""".strip()


EVIDENCE_EXTRACTION_SYSTEM_PROMPT: Final[str] = """
You are extracting evidence from NICU clinical records.

Your task is not to write the discharge summary.

Extract all discharge-relevant clinical facts while preserving:

- chronology
- dates
- times
- exact values
- units
- medication doses
- respiratory settings
- investigation results
- treatment changes
- uncertainty
- conflicting entries

Do not invent, correct, merge or clinically reinterpret unsupported details.

Return structured plain text only.
Do not return JSON.
Do not explain your reasoning.
""".strip()


TEMPLATE_SELECTION_SYSTEM_PROMPT: Final[str] = """
You are selecting the closest hospital discharge-summary template for a NICU
patient.

The selection is based only on:

1. Whether the baby is term or preterm.
2. Whether the principal discharge condition is:
   - neonatal jaundice, or
   - respiratory distress / TTNB.

Do not invent gestational age or diagnosis.

Return only one valid template name or TEMPLATE_UNCERTAIN.
Do not provide an explanation.
""".strip()


# =============================================================================
# TEMPLATE LOADING
# =============================================================================

class TemplateError(RuntimeError):
    """Raised when a hospital discharge template cannot be loaded."""


def get_template_path(template_name: str) -> Path:
    """
    Return the DOCX path corresponding to a template name.

    Args:
        template_name:
            One of:
            PRETERM_NNJ
            PRETERM_RD_TTNB
            TERM_NNJ
            TERM_RD_TTNB

    Returns:
        Path to the corresponding hospital DOCX template.

    Raises:
        ValueError:
            If the template name is invalid.

        TemplateError:
            If the template file does not exist.
    """

    cleaned_name = template_name.strip().upper()

    if cleaned_name not in TEMPLATE_FILES:
        valid_names = ", ".join(VALID_TEMPLATE_NAMES)

        raise ValueError(
            f"Invalid template name '{template_name}'. "
            f"Expected one of: {valid_names}"
        )

    template_path = TEMPLATE_FILES[cleaned_name]

    if not template_path.exists():
        raise TemplateError(
            f"Hospital template was not found: {template_path}"
        )

    if not template_path.is_file():
        raise TemplateError(
            f"Hospital template path is not a file: {template_path}"
        )

    return template_path


def load_template_text(template_name: str) -> str:
    """
    Extract readable text from the selected DOCX hospital template.

    Both normal paragraphs and table contents are extracted. Table rows are
    represented using pipe-separated cells so that medication and screening
    tables remain understandable to the model.

    Args:
        template_name:
            Selected hospital template name.

    Returns:
        Plain-text representation of the DOCX template.

    Raises:
        TemplateError:
            If the DOCX cannot be opened or contains no readable content.
    """

    template_path = get_template_path(template_name)

    try:
        document = Document(template_path)
    except Exception as exc:
        raise TemplateError(
            f"Could not open hospital template: {template_path}"
        ) from exc

    extracted_blocks: list[str] = []

    for block in _iter_document_blocks(document):
        cleaned_block = _clean_template_block(block)

        if cleaned_block:
            extracted_blocks.append(cleaned_block)

    template_text = "\n".join(extracted_blocks).strip()

    if not template_text:
        raise TemplateError(
            f"No readable content was found in template: {template_path}"
        )

    return template_text


def _iter_document_blocks(document: Document) -> list[str]:
    """
    Extract paragraphs and tables from a DOCX in their document order.

    python-docx normally exposes paragraphs and tables separately. This helper
    walks through the underlying document body so the original sequence is
    retained.
    """

    blocks: list[str] = []

    for child in document.element.body.iterchildren():
        tag_name = child.tag.rsplit("}", maxsplit=1)[-1]

        if tag_name == "p":
            paragraph_text = "".join(
                node.text or ""
                for node in child.iter()
                if node.tag.rsplit("}", maxsplit=1)[-1] == "t"
            ).strip()

            if paragraph_text:
                blocks.append(paragraph_text)

        elif tag_name == "tbl":
            table_rows: list[str] = []

            for row in child.iter():
                row_tag = row.tag.rsplit("}", maxsplit=1)[-1]

                if row_tag != "tr":
                    continue

                cells: list[str] = []

                for cell in row.iterchildren():
                    cell_tag = cell.tag.rsplit("}", maxsplit=1)[-1]

                    if cell_tag != "tc":
                        continue

                    cell_text_parts = [
                        node.text or ""
                        for node in cell.iter()
                        if node.tag.rsplit("}", maxsplit=1)[-1] == "t"
                    ]

                    cell_text = " ".join(
                        part.strip()
                        for part in cell_text_parts
                        if part.strip()
                    ).strip()

                    cells.append(cell_text)

                if cells:
                    table_rows.append(" | ".join(cells))

            if table_rows:
                blocks.extend(table_rows)

    return blocks


def _clean_template_block(text: str) -> str:
    """Clean spacing without changing the clinical wording."""

    cleaned_lines: list[str] = []

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned_line = " ".join(line.split()).strip()

        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


# =============================================================================
# TEMPLATE SELECTION PROMPT
# =============================================================================

def build_template_selection_prompt(clinical_evidence: str) -> str:
    """
    Create the prompt used to select the closest hospital template.

    The model must return one of the four template identifiers or
    TEMPLATE_UNCERTAIN.
    """

    evidence = clinical_evidence.strip()

    if not evidence:
        raise ValueError(
            "Clinical evidence cannot be empty for template selection."
        )

    return f"""
Select the closest hospital discharge-summary template for this patient.

VALID OUTPUTS:

{PRETERM_NNJ}
{PRETERM_RD_TTNB}
{TERM_NNJ}
{TERM_RD_TTNB}
TEMPLATE_UNCERTAIN

SELECTION RULES:

1. PRETERM means documented gestational age below 37 completed weeks.

2. TERM means documented gestational age of 37 completed weeks or more.

3. NNJ means neonatal jaundice is the principal condition represented by the
   discharge course.

4. RD_TTNB means respiratory distress, delayed transition or transient
   tachypnoea of the newborn is the principal condition represented by the
   discharge course.

5. Do not infer gestational age from birth weight.

6. Do not infer a diagnosis only from treatment.

7. If gestational category cannot be established, return TEMPLATE_UNCERTAIN.

8. If the principal condition cannot be distinguished between NNJ and
   RD/TTNB, return TEMPLATE_UNCERTAIN.

9. Return only the template name.

PATIENT CLINICAL EVIDENCE:

--- BEGIN CLINICAL EVIDENCE ---

{evidence}

--- END CLINICAL EVIDENCE ---
""".strip()


def validate_template_selection(model_output: str) -> str:
    """
    Validate and clean the template-selection model output.

    Returns:
        A valid template name or TEMPLATE_UNCERTAIN.

    Raises:
        ValueError:
            If the model returned an unexpected response.
    """

    cleaned_output = (
        model_output
        .strip()
        .upper()
        .replace("`", "")
        .replace("*", "")
    )

    first_line = cleaned_output.splitlines()[0].strip()

    valid_outputs = set(VALID_TEMPLATE_NAMES)
    valid_outputs.add("TEMPLATE_UNCERTAIN")

    if first_line not in valid_outputs:
        raise ValueError(
            "Template-selection model returned an invalid value: "
            f"{model_output!r}"
        )

    return first_line


# =============================================================================
# CHUNK EVIDENCE EXTRACTION PROMPT
# =============================================================================

def build_chunk_extraction_prompt(
    clinical_chunk: str,
    *,
    chunk_number: int,
    total_chunks: int,
) -> str:
    """
    Create the prompt used to extract clinical facts from one normalized chunk.

    Args:
        clinical_chunk:
            Portion of the normalized patient record.

        chunk_number:
            One-based index of the current chunk.

        total_chunks:
            Total number of chunks.

    Returns:
        Prompt for discharge-relevant evidence extraction.
    """

    chunk = clinical_chunk.strip()

    if not chunk:
        raise ValueError("Clinical chunk cannot be empty.")

    if chunk_number < 1:
        raise ValueError("chunk_number must be at least 1.")

    if total_chunks < 1:
        raise ValueError("total_chunks must be at least 1.")

    if chunk_number > total_chunks:
        raise ValueError(
            "chunk_number cannot be greater than total_chunks."
        )

    return f"""
Extract all information from this NICU record section that may be required to
complete the hospital discharge summary.

This is chunk {chunk_number} of {total_chunks}.

IMPORTANT:

- Do not write the final discharge summary.
- Do not select a template.
- Do not summarise away specific values.
- Do not remove a fact merely because it appears more than once.
- Preserve the source chronology.
- Preserve all dates, times, units and doses.
- Preserve uncertain wording such as "possible", "suspected", "query" or "?":
  do not convert it into a confirmed diagnosis.
- Record contradictions under CONFLICTS OR ITEMS FOR CLINICAL REVIEW.
- Do not state "normal" unless the source explicitly documents it.
- Do not fill missing information with NOT AVAILABLE at this stage.
- Write "No relevant information in this chunk" under a heading only when
  necessary.
- Return plain text only.

Use these headings:

PATIENT IDENTIFIERS
- MR number
- IP number
- baby name or B/O name
- mother name
- father name
- contact number
- referral doctor
- inborn or outborn status

ADMISSION AND DISCHARGE
- admission date and time
- discharge date and time
- referring hospital
- referral reason
- age/day/hour of life at admission

BABY DETAILS
- date and time of birth
- sex
- gestational age
- term or preterm status when explicitly documented
- birth weight
- admission weight
- discharge weight
- growth classification
- length
- head circumference
- percentiles

MATERNAL AND ANTENATAL HISTORY
- maternal age
- gravida/parity
- antenatal care
- antenatal scans
- maternal illnesses
- pregnancy complications
- medications
- relevant blood groups
- infection risk factors

PERINATAL AND BIRTH HISTORY
- singleton or multiple pregnancy
- mode of delivery
- indication for delivery
- place of birth
- cry at birth
- resuscitation
- APGAR scores
- meconium
- cord events
- immediate post-birth problems

DIAGNOSES
- confirmed diagnoses
- suspected diagnoses
- gestational and growth classifications
- diagnosis status and supporting source wording

ADMISSION EXAMINATION
- temperature
- heart rate
- respiratory rate
- SpO2
- perfusion and pulses
- head-to-toe examination
- respiratory examination
- cardiovascular examination
- abdominal examination
- neurological examination
- skin and jaundice findings
- SAS or respiratory distress score

NEONATAL JAUNDICE COURSE
- onset day/hour
- Kramer zone or extent of icterus
- mother and baby blood groups
- TCB and TSB values with date/time
- phototherapy type
- phototherapy start and stop
- phototherapy duration
- repeat bilirubin results
- exchange transfusion
- BIND findings
- cholestasis findings
- rebound bilirubin or rebound risk

RESPIRATORY COURSE
- distress findings
- chest X-ray findings
- oxygen support
- T-piece
- HFNC
- CPAP
- NIMV
- mechanical ventilation
- PEEP
- FiO2
- flow
- surfactant
- support start and stop times
- support duration
- weaning
- duration comfortable in room air

CARDIOVASCULAR COURSE
- haemodynamic status
- murmurs
- echocardiography
- inotropes
- blood pressure
- perfusion

NEUROLOGICAL COURSE
- activity
- tone
- cry
- suck
- seizures
- anticonvulsants
- neurosonogram
- neurological diagnoses

INFECTION AND SEPSIS COURSE
- sepsis risk factors
- cultures
- CRP
- blood counts
- antibiotics
- start and stop dates
- confirmed or suspected infection

NUTRITION AND FEEDING
- NPO status
- OGT feeds
- spoon feeds
- expressed breast milk
- direct breastfeeding
- formula
- feed volumes
- feed frequency
- feed progression
- feed intolerance
- total parenteral nutrition
- feeding status at discharge

LINES AND PROCEDURES
- UVC
- UAC
- PICC
- peripheral lines
- line insertion/removal
- line duration
- procedures
- transfusions

INVESTIGATIONS
- laboratory tests
- imaging
- dates
- values
- units
- interpretations explicitly documented

MEDICATIONS DURING ADMISSION
- exact drug name
- concentration
- dose
- route
- frequency
- indication
- start date
- stop date
- duration
- changes

SCREENING TESTS
- pulse oximetry
- DDH
- hearing screening
- thyroid profile
- ROP
- neurosonogram
- immunisation
- blood groups
- screening result and date

DISCHARGE CONDITION
- respiratory status
- haemodynamic status
- feeding status
- urine output
- stools
- temperature status
- glucose status
- activity
- saturation
- final weight
- other documented condition

DISCHARGE MEDICATIONS
- exact drug name
- concentration
- dose
- route
- frequency
- duration
- special instruction

DISCHARGE ADVICE AND FOLLOW-UP
- feeding advice
- warm care
- KMC
- bathing
- immunisation
- ROP appointment
- hearing appointment
- thyroid testing
- neurosonogram
- referral-doctor review
- neonatal review
- neurodevelopmental follow-up
- jaundice advice
- danger signs
- other documented advice

CONFLICTS OR ITEMS FOR CLINICAL REVIEW
- conflicting dates
- conflicting values
- uncertain handwriting/OCR
- inconsistent diagnosis
- inconsistent drug details
- unsupported template statements

CLINICAL RECORD CHUNK:

--- BEGIN CHUNK {chunk_number} ---

{chunk}

--- END CHUNK {chunk_number} ---
""".strip()


# =============================================================================
# EVIDENCE CONSOLIDATION PROMPT
# =============================================================================

def build_evidence_consolidation_prompt(
    extracted_evidence: str,
) -> str:
    """
    Build the prompt that consolidates facts extracted from multiple chunks.

    This stage removes exact duplication but must retain unique facts,
    chronology, conflicts and dose changes.
    """

    evidence = extracted_evidence.strip()

    if not evidence:
        raise ValueError("Extracted clinical evidence cannot be empty.")

    return f"""
Consolidate the extracted NICU evidence into one complete clinical evidence
record for discharge-summary generation.

This is not the final discharge summary.

RULES:

1. Preserve every unique clinical fact.
2. Remove only exact duplicate statements.
3. Preserve chronology.
4. Preserve all dates, times, values and units.
5. Preserve medication changes.
6. Preserve respiratory-setting changes.
7. Do not combine conflicting values into one value.
8. Place contradictions under CONFLICTS OR ITEMS FOR CLINICAL REVIEW.
9. Do not invent missing details.
10. Do not add routine normal findings.
11. Do not output JSON.
12. Return structured plain text only.

Use the same clinical headings present in the extracted evidence.

EXTRACTED CHUNK EVIDENCE:

--- BEGIN EXTRACTED EVIDENCE ---

{evidence}

--- END EXTRACTED EVIDENCE ---
""".strip()


# =============================================================================
# FINAL DISCHARGE-SUMMARY PROMPT
# =============================================================================

def build_final_discharge_prompt(
    *,
    template_name: str,
    evidence_report: str,
) -> str:
    """
    Build the final discharge-generation prompt using the real DOCX template.

    The hospital template is loaded dynamically from the templates directory.

    Args:
        template_name:
            Selected template identifier.

        clinical_evidence:
            Consolidated evidence extracted from the normalized patient record.

    Returns:
        Final prompt containing both the template and patient evidence.
    """

    evidence = evidence_report.strip()

    if not evidence:
        raise ValueError(
            "Evidence report cannot be empty for discharge generation."
        )

    template_text = load_template_text(template_name)

    return f"""
Complete the supplied hospital NICU discharge-summary template using the
patient evidence.

SELECTED TEMPLATE:
{template_name}

CRITICAL INSTRUCTIONS:

1. The hospital template is the required structure.

2. Preserve:
   - heading order
   - clinical section order
   - hospital wording
   - standard advice
   - danger signs
   - disclaimers
   - consultant details
   - signature sections

3. Replace template blanks only with information supported by the patient
   evidence.

4. When an individual patient field is genuinely unavailable, retain a clear
   blank placeholder as:

   __________

5. Do not write NOT AVAILABLE inside the final hospital form unless the
   hospital explicitly requests that wording.

6. Do not copy a prewritten normal finding from the template unless:
   - it is supported by the patient record, or
   - it is clearly fixed hospital administrative text rather than a
     patient-specific finding.

7. Do not assume:
   - normal antenatal history
   - no comorbidities
   - normal examination
   - normal cardiovascular status
   - normal neurological status
   - successful screening
   - full feeds
   - room-air stability
   unless supported by the patient evidence.

8. Preserve the exact:
   - patient names
   - dates
   - times
   - gestational age
   - weights
   - diagnoses
   - investigation values
   - medication names
   - concentrations
   - doses
   - routes
   - frequencies
   - durations

9. Do not calculate or estimate missing values.

10. Do not infer gestational age from weight.

11. Do not convert a suspected diagnosis into a confirmed diagnosis.

12. The diagnosis line must include all important supported discharge
    diagnoses, not only the diagnosis named in the selected template.

13. The COURSE IN HOSPITAL section must reflect the actual patient course.

14. Include additional clinically relevant subsections under COURSE IN
    HOSPITAL when supported, including:
    - sepsis
    - hypoglycaemia
    - seizures
    - HIE
    - NEC
    - apnoea
    - congenital anomalies
    - cardiovascular conditions
    - neurological conditions
    - other documented diagnoses

15. For neonatal jaundice, include only documented:
    - onset
    - bilirubin values
    - blood groups
    - phototherapy details
    - treatment duration
    - repeat bilirubin
    - BIND/cholestasis findings
    - rebound information

16. For RD/TTNB, include only documented:
    - clinical distress signs
    - SAS score
    - chest X-ray result
    - respiratory support
    - PEEP
    - FiO2
    - support duration
    - weaning
    - room-air observation duration

17. Do not automatically prescribe medicines listed in the blank hospital
    template. Include a discharge medicine only when:
    - it is supported by the patient evidence, or
    - it is explicitly confirmed as the hospital's standard prescribed
      discharge medicine for this patient.

18. Do not generate JSON.

19. Do not generate Markdown formatting.

20. Do not add commentary before or after the completed summary.

21. Return only the completed discharge-summary text.

22. Any conflicts listed in the evidence must not be silently resolved. Keep
    the affected field blank or mark it visibly as:

    [VERIFY WITH CONSULTANT]

HOSPITAL TEMPLATE:

--- BEGIN HOSPITAL TEMPLATE ---

{template_text}

--- END HOSPITAL TEMPLATE ---

CONSOLIDATED PATIENT EVIDENCE:

--- BEGIN PATIENT EVIDENCE ---

{evidence}

--- END PATIENT EVIDENCE ---
""".strip()


# =============================================================================
# REVIEW PROMPT
# =============================================================================

def build_discharge_review_prompt(
    *,
    generated_summary: str,
    clinical_evidence: str,
) -> str:
    """
    Build a final quality-control prompt.

    The review stage checks whether the generated summary contains unsupported
    information, omitted facts or altered medication details.

    It does not approve the summary for clinical use.
    """

    summary = generated_summary.strip()
    evidence = clinical_evidence.strip()

    if not summary:
        raise ValueError("Generated discharge summary cannot be empty.")

    if not evidence:
        raise ValueError("Clinical evidence cannot be empty.")

    return f"""
Review the generated NICU discharge summary against the supplied patient
evidence.

Identify:

1. Unsupported patient-specific statements.
2. Invented normal findings.
3. Missing important diagnoses.
4. Missing important hospital-course events.
5. Incorrect dates or times.
6. Incorrect laboratory values.
7. Incorrect medication names.
8. Incorrect concentrations.
9. Incorrect doses.
10. Incorrect routes.
11. Incorrect frequencies.
12. Incorrect durations.
13. Unresolved contradictions.
14. Template blanks that could have been filled from the evidence.
15. Statements that require consultant verification.

Do not rewrite the entire discharge summary.

Return a concise plain-text review using:

STATUS:
PASS

or

STATUS:
REQUIRES_REVIEW

Then list findings under:

UNSUPPORTED INFORMATION
OMITTED INFORMATION
MEDICATION DISCREPANCIES
DATE OR VALUE DISCREPANCIES
ITEMS FOR CONSULTANT VERIFICATION

If no issue exists under a heading, write:

None identified

GENERATED DISCHARGE SUMMARY:

--- BEGIN GENERATED SUMMARY ---

{summary}

--- END GENERATED SUMMARY ---

PATIENT EVIDENCE:

--- BEGIN PATIENT EVIDENCE ---

{evidence}

--- END PATIENT EVIDENCE ---
""".strip()


# =============================================================================
# TEMPLATE VALIDATION
# =============================================================================

def validate_all_templates() -> dict[str, Path]:
    """
    Confirm that all four hospital template files are present and readable.

    This can be called by generator.py during startup.

    Returns:
        Dictionary of template names and validated file paths.

    Raises:
        TemplateError:
            If any template is missing or unreadable.
    """

    validated_templates: dict[str, Path] = {}

    for template_name in VALID_TEMPLATE_NAMES:
        template_path = get_template_path(template_name)
        load_template_text(template_name)
        validated_templates[template_name] = template_path

    return validated_templates


if __name__ == "__main__":
    try:
        templates = validate_all_templates()

        print("All hospital templates are available:")

        for name, path in templates.items():
            template_text = load_template_text(name)

            print(
                f"- {name}: {path.name} "
                f"({len(template_text):,} characters)"
            )

    except (TemplateError, ValueError) as error:
        print(f"Template validation failed: {error}")
        raise SystemExit(1) from error
    
    # =============================================================================
# DOCTOR-FRIENDLY EVIDENCE REPORT
# =============================================================================

EVIDENCE_REPORT_SYSTEM_PROMPT: Final[str] = """
You are preparing a clinician-readable evidence report from NICU patient
records.

The report is intended to help the treating doctor quickly review all clinical
information extracted from the patient's records before approving the final
discharge summary.

This is not the hospital-formatted discharge summary.

Follow these rules strictly:

1. Use only information contained in the supplied clinical evidence.
2. Never invent or assume patient information.
3. Never add routine normal findings unless explicitly documented.
4. Preserve all clinically important facts.
5. Preserve chronology.
6. Preserve exact dates, times, values, units, diagnoses and medication details.
7. Preserve uncertainty such as:
   - suspected
   - possible
   - query
   - provisional
8. Do not convert suspected diagnoses into confirmed diagnoses.
9. Do not silently resolve conflicting information.
10. Clearly mention conflicts or unclear information under:
    ITEMS REQUIRING DOCTOR REVIEW
11. Write in clear, professional and easy-to-understand medical English.
12. Explain the clinical course in complete sentences.
13. Use bullet points for structured details such as investigations,
    treatments and discharge advice.
14. Do not produce JSON.
15. Do not produce a hospital form.
16. Do not include unsupported standard advice.
17. Do not explain your reasoning.
18. Return only the completed evidence report.
""".strip()


def build_evidence_report_prompt(
    clinical_evidence: str,
) -> str:
    """
    Build a doctor-friendly clinical evidence report.

    The report presents all discharge-relevant information extracted from the
    patient's clinical records in readable medical language.

    It becomes the reviewed source used to generate the hospital discharge
    summary.

    Args:
        clinical_evidence:
            Consolidated clinical evidence created after processing all
            normalized-text chunks.

    Returns:
        Prompt for generating the clinician-readable evidence report.
    """

    evidence = clinical_evidence.strip()

    if not evidence:
        raise ValueError(
            "Clinical evidence cannot be empty for evidence-report generation."
        )

    return f"""
Prepare a complete clinician-readable NICU evidence report using the supplied
clinical evidence.

The report should allow the treating doctor to understand:

- who the patient is
- why the patient was admitted
- what happened during the NICU stay
- what investigations were performed
- what treatments were given
- what the baby's condition was at discharge
- what medications and follow-up were advised
- which details remain missing, unclear or conflicting

Use the exact structure given below.

============================================================
REQUIRED OUTPUT STRUCTURE
============================================================

DISCHARGE SUMMARY

Demographics

Present the available information using bullet points:

- Patient Name
- MR Number
- IP Number
- Age at Admission
- Date of Birth
- Time of Birth
- Gender
- Birth Weight
- Admission Weight
- Discharge Weight
- Gestational Age
- Term or Preterm Classification
- Growth Classification such as AGA, SGA or LGA
- Blood Group
- Inborn or Outborn Status
- Referring Hospital
- Referral Doctor
- Mother Name
- Father Name
- Contact Number
- Date of Admission
- Date of Discharge

Do not display fields that are completely unavailable unless they are
clinically important or require doctor verification.

Example style:

**Demographics**
* **Patient Name:** Baby of Vagdevi
* **Age at Admission:** 4 days
* **Gender:** Male
* **Birth Weight:** 2.11 kg
* **Gestational Age:** 36 weeks, late preterm
* **Blood Group:** B Positive


Antenatal and Perinatal History

Write a concise clinical paragraph covering all documented information about:

- maternal age
- gravida and parity
- antenatal care
- pregnancy complications
- maternal illnesses
- antenatal medications
- antenatal scans
- mode of delivery
- indication for delivery
- place of birth
- cry at birth
- resuscitation
- APGAR scores
- feeding before admission
- relevant perinatal events

Do not write that the antenatal period was normal unless this is explicitly
documented.


History of Present Illness / Reason for Admission

Write a chronological paragraph explaining:

- day or hour of life when symptoms began
- symptoms observed
- evaluation at another hospital
- initial investigation results
- treatments given before referral
- referral reason
- condition on arrival

Clearly distinguish treatment received outside the hospital from treatment
given at Ankura Hospital.


Admission Examination

Begin with a short description of the baby's general condition.

Then present documented examination findings using bullet points where useful.

Include:

- general condition
- hydration
- activity
- sensorium
- temperature
- heart rate
- respiratory rate
- oxygen saturation
- blood pressure
- respiratory support at admission
- respiratory examination
- cardiovascular examination
- abdominal examination
- neurological examination
- tone
- suck
- cry
- skin or jaundice findings
- other abnormal findings

Do not copy normal template findings unless supported by the evidence.


Morbidities and Course during NICU Stay

Create a separate bullet subsection for every major morbidity or clinical
problem.

Examples include:

- Symptomatic Hypoglycaemia
- Neonatal Seizures
- Respiratory Distress / TTNB
- Neonatal Jaundice
- Sepsis / Suspected Infection
- Prematurity
- Small for Gestational Age
- Feeding Difficulty
- Apnoea
- HIE
- NEC
- Cardiovascular Problems
- Neurological Problems
- Electrolyte Abnormalities
- Other documented diagnoses

Use this style:

* **Condition or Morbidity:**
Write a chronological clinical paragraph describing:

- initial presentation
- important examination findings
- relevant investigations
- treatments
- treatment changes
- response to treatment
- last abnormal event
- resolution or condition at discharge

Preserve medication changes, respiratory-setting changes, glucose infusion
rates, feeding progression and important dates.

Do not merge unrelated morbidities into one paragraph.


Investigations and Interpretation

Present investigations as bullet points grouped logically.

Suggested groups:

- Complete Blood Picture
- Infection Markers
- Blood Culture
- Urine Examination and Culture
- Renal Function
- Electrolytes
- Liver Function
- Bilirubin
- Blood Glucose
- Blood Gas
- Metabolic Investigations
- Neurosonogram
- EEG
- MRI Brain
- Chest X-ray
- Echocardiography
- Screening Tests
- Other Investigations

Use this style:

* **Complete Blood Picture:** Hb: ..., WBC: ..., Platelets: ...
* **Infection Markers:** CRP: ...
* **MRI Brain:** ...

Only provide interpretations such as normal, elevated, reduced, no growth or
abnormal when:

- explicitly documented in the record, or
- directly evident from an interpretation already stated in the clinical
  evidence.

Do not independently diagnose based on laboratory values.

Include the date or time when clinically relevant.


Procedures and Interventions

Present all documented procedures and supportive interventions using bullet
points.

Include:

- UVC
- UAC
- PICC
- peripheral IV cannulation
- lumbar puncture
- transfusion
- phototherapy
- exchange transfusion
- respiratory support
- oxygen therapy
- mechanical ventilation
- CPAP
- HFNC
- LFNC
- NIMV
- surfactant
- imaging procedures
- other procedures

Include durations and settings when available.


Treatments Given

Group treatments using clinically meaningful headings.

Possible groups:

- IV Fluids
- Glucose Management
- Anticonvulsants
- Antibiotics
- Respiratory Treatment
- Phototherapy
- Cardiovascular Medicines
- Vitamins and Supplements
- Electrolyte Correction
- Feeding and Nutrition
- Other Medicines

For every medicine, preserve:

- exact drug name
- brand name if documented
- concentration
- dose
- route
- frequency
- duration
- start and stop information
- reason for discontinuation when documented

Clearly distinguish between:

- treatment given during admission
- treatment stopped before discharge
- treatment continued at discharge

Do not infer a dose or duration.


Condition at Discharge

Write a concise paragraph describing the documented condition of the baby at
discharge.

Include when available:

- activity and sensorium
- seizure-free duration
- respiratory status
- room-air status
- haemodynamic status
- feeding status
- urine output
- stools
- temperature stability
- blood-glucose stability
- weight
- jaundice status
- neurological condition
- remaining clinical concerns


Discharge Medications

List only medications specifically documented to continue after discharge.

Use this style:

* **Medicine Name:** concentration, dose, route, frequency and duration.

Example:

* **Syrup Gardenal:** 2 ml orally twice daily until reviewed by the paediatric
  neurologist.

Do not move an inpatient medication into this section unless continuation at
discharge is explicitly supported.

If the medication is documented but its dose is unclear, write:

[DOSE REQUIRES VERIFICATION]


Discharge Advice

Present all documented advice using bullet points.

Include only supported advice regarding:

- feeding
- feed volume and frequency
- breastfeeding
- expressed breast milk
- formula
- warm care
- kangaroo mother care
- hygiene
- bathing
- immunisation
- glucose monitoring
- seizure precautions
- jaundice monitoring
- danger signs
- other instructions


Follow-up Plan

List every documented follow-up separately.

Include:

- review date
- doctor name
- department
- paediatric neurologist review
- neonatologist review
- ROP screening
- hearing screening
- BERA
- thyroid profile
- neurosonogram
- developmental assessment
- laboratory follow-up
- imaging follow-up
- referral-doctor review

Do not invent a follow-up date.


Items Requiring Doctor Review

Always include this section.

List:

- conflicting dates
- conflicting weights
- inconsistent medication details
- unreadable OCR text
- uncertain diagnosis
- missing discharge-medication dose
- missing follow-up date
- investigation mentioned without result
- template field that cannot be established
- any information requiring consultant confirmation

Use bullet points.

If no conflicts or missing critical items are identified, write:

* No major conflicts or missing critical information identified from the
  supplied record.

============================================================
FORMATTING RULES
============================================================

1. Use Markdown-style headings and bullet points.

2. Use this exact heading order:

**DISCHARGE SUMMARY**

**Demographics**

**Antenatal and Perinatal History**

**History of Present Illness / Reason for Admission**

**Admission Examination**

**Morbidities and Course during NICU Stay**

**Investigations and Interpretation**

**Procedures and Interventions**

**Treatments Given**

**Condition at Discharge**

**Discharge Medications**

**Discharge Advice**

**Follow-up Plan**

**Items Requiring Doctor Review**

3. Omit a clinical subsection only when no relevant information is available.

4. Never omit the Items Requiring Doctor Review section.

5. Use complete sentences for clinical history and hospital course.

6. Use bullet points for values, medicines, investigations and advice.

7. Avoid repetitive information.

8. Preserve every unique clinical fact.

9. Do not include raw extraction labels or chunk numbers.

10. Do not mention the AI, model, prompt, OCR pipeline or evidence-extraction
    process.

11. Do not add a disclaimer unless one exists in the supplied evidence.

12. Return only the evidence report.

============================================================
CONSOLIDATED CLINICAL EVIDENCE
============================================================

--- BEGIN CLINICAL EVIDENCE ---

{evidence}

--- END CLINICAL EVIDENCE ---
""".strip()