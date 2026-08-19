"""
Prompt for generating the doctor-friendly evidence report.
"""

from __future__ import annotations


def build_evidence_report_prompt(
    clinical_evidence: str,
) -> str:
    """
    Build a doctor-friendly report from consolidated clinical evidence.
    """

    evidence = clinical_evidence.strip()

    if not evidence:
        raise ValueError(
            "Clinical evidence cannot be empty for evidence-report generation."
        )

    return f"""
Prepare a complete clinician-readable NICU evidence report using the supplied
clinical evidence.

This report is for the treating doctor to review before the final hospital
discharge summary is approved.

Use the exact heading order shown below.

**DISCHARGE SUMMARY**

**Demographics**

Present available demographic details as bullet points.

Possible fields include:

* **Patient Name:**
* **MR Number:**
* **IP Number:**
* **Age at Admission:**
* **Date of Birth:**
* **Time of Birth:**
* **Gender:**
* **Birth Weight:**
* **Admission Weight:**
* **Discharge Weight:**
* **Gestational Age:**
* **Term/Preterm Classification:**
* **Growth Classification:**
* **Blood Group:**
* **Inborn/Outborn Status:**
* **Referring Hospital:**
* **Referral Doctor:**
* **Mother Name:**
* **Father Name:**
* **Contact Number:**
* **Date of Admission:**
* **Date of Discharge:**

Do not display fields that are completely unavailable unless their absence
requires doctor review.

**Antenatal and Perinatal History**

Write a clear clinical paragraph covering documented:

- maternal age
- gravida and parity
- antenatal history
- maternal illnesses
- pregnancy complications
- antenatal scans
- maternal medicines
- mode of delivery
- indication for delivery
- place of birth
- cry at birth
- resuscitation
- APGAR scores
- feeding before admission
- relevant perinatal events

Do not describe the history as normal unless explicitly documented.

**History of Present Illness / Reason for Admission**

Write a chronological paragraph explaining:

- day or hour of life when symptoms started
- symptoms observed
- outside-hospital evaluation
- investigation findings before referral
- treatment given before referral
- reason for referral
- condition when brought to Ankura Hospital

Clearly separate treatment given outside from treatment given at Ankura.

**Admission Examination**

Begin with a short paragraph describing the baby's general condition.

Use bullet points for documented examination findings.

Possible categories:

* **Vitals:**
* **Respiratory Examination:**
* **Cardiovascular Examination:**
* **Abdominal Examination:**
* **Neurological Examination:**
* **Skin/Jaundice:**
* **Other Findings:**

Do not add unsupported normal findings.

**Morbidities and Course during NICU Stay**

Create a separate bullet subsection for every major morbidity.

Use this format:

* **Condition or Morbidity:**
  Write a chronological paragraph describing presentation, investigations,
  treatment, treatment changes, response and condition at discharge.

Possible conditions include:

- hypoglycaemia
- neonatal seizures
- respiratory distress
- TTNB
- neonatal jaundice
- suspected infection
- confirmed sepsis
- prematurity
- SGA
- feeding difficulty
- apnoea
- HIE
- NEC
- cardiovascular problems
- neurological problems
- electrolyte abnormalities
- other documented conditions

Preserve:

- dates and times
- last seizure or abnormal event
- glucose values
- glucose infusion rate
- respiratory settings
- feeding progression
- medication changes
- clinical response

Do not merge unrelated problems into one subsection.

**Investigations and Interpretation**

Group investigations logically using bullet points.

Example format:

* **Complete Blood Picture:** Hb: ..., WBC: ..., Platelets: ...
* **Infection Markers:** CRP: ...
* **Renal Function and Electrolytes:** ...
* **Liver Function Test:** ...
* **Blood Gas:** ...
* **EEG:** ...
* **MRI Brain:** ...
* **Chest X-ray:** ...

Include exact values and units.

Only include interpretations such as normal, elevated, reduced or no growth
when explicitly supported by the clinical evidence.

Do not independently diagnose a condition from a laboratory value.

**Procedures and Interventions**

List all documented procedures and support using bullet points.

Examples:

- UVC
- UAC
- PICC
- peripheral IV cannulation
- lumbar puncture
- transfusion
- phototherapy
- oxygen therapy
- LFNC
- HFNC
- CPAP
- NIMV
- mechanical ventilation
- surfactant
- imaging procedures
- other interventions

Include settings and duration when available.

**Treatments Given**

Group treatments under meaningful headings.

Possible groups:

* **IV Fluids:**
* **Glucose Management:**
* **Anticonvulsants:**
* **Antibiotics:**
* **Respiratory Treatment:**
* **Phototherapy:**
* **Vitamins and Supplements:**
* **Electrolyte Correction:**
* **Feeding and Nutrition:**
* **Other Medicines:**

For each medicine preserve:

- exact medicine name
- brand name
- concentration
- dose
- route
- frequency
- duration
- start and stop information
- reason for discontinuation when documented

Clearly identify whether treatment was:

- given during admission
- stopped before discharge
- continued after discharge

**Condition at Discharge**

Write a concise paragraph describing documented:

- activity
- sensorium
- seizure-free duration
- respiratory status
- room-air status
- haemodynamic status
- feeding status
- glucose stability
- urine output
- stools
- temperature stability
- jaundice status
- neurological status
- discharge weight
- remaining concerns

**Discharge Medications**

List only medicines explicitly documented to continue after discharge.

Use this style:

* **Medicine Name:** concentration, dose, route, frequency and duration.

Do not move an inpatient medicine into this section unless continuation after
discharge is explicitly documented.

When a discharge medicine is documented but the dose is unclear, write:

[DOSE REQUIRES VERIFICATION]

**Discharge Advice**

List only documented advice.

Possible advice includes:

- breastfeeding
- expressed breast milk
- formula
- feed volume and frequency
- spoon feeds
- warm care
- KMC
- hand hygiene
- bathing
- immunisation
- seizure precautions
- glucose monitoring
- jaundice monitoring
- danger signs
- other instructions

**Follow-up Plan**

List every documented follow-up separately.

Include when available:

- review date
- doctor
- department
- paediatric neurologist
- neonatologist
- ROP screening
- hearing screening
- BERA
- thyroid testing
- neurosonogram
- developmental assessment
- laboratory follow-up
- imaging follow-up

Do not invent a review date.

**Items Requiring Doctor Review**

Always include this section.

List:

- conflicting dates
- conflicting weights
- inconsistent medicine details
- unreadable OCR information
- uncertain diagnosis
- missing medicine dose
- missing follow-up date
- investigation mentioned without a result
- important missing clinical information
- information requiring consultant confirmation

If no major issue is present, write:

* No major conflicts or missing critical information identified from the
  supplied record.

FORMATTING RULES:

1. Use the exact headings shown above.
2. Use Markdown-style bold headings and bullet points.
3. Use paragraphs for history and clinical course.
4. Use bullet points for values, medicines, investigations and advice.
5. Preserve every unique clinical fact.
6. Avoid unnecessary repetition.
7. Do not include raw extraction headings.
8. Do not include chunk numbers.
9. Do not mention OCR, AI, prompts or models.
10. Do not output JSON.
11. Return only the completed evidence report.

CONSOLIDATED CLINICAL EVIDENCE:

--- BEGIN CLINICAL EVIDENCE ---

{evidence}

--- END CLINICAL EVIDENCE ---
""".strip()