"""
Prompts for extracting and consolidating clinical evidence.
"""

from __future__ import annotations


def build_chunk_extraction_prompt(
    clinical_chunk: str,
    *,
    chunk_number: int,
    total_chunks: int,
) -> str:
    """
    Build a prompt for extracting discharge-relevant information from one
    section of normalized clinical text.
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
Extract all discharge-relevant information from this NICU clinical-record
section.

This is chunk {chunk_number} of {total_chunks}.

IMPORTANT RULES:

- Do not write the final discharge summary.
- Do not select a template.
- Do not summarise away specific clinical values.
- Preserve dates, times, units and doses.
- Preserve chronology.
- Preserve medication changes.
- Preserve respiratory-support changes.
- Preserve feeding changes.
- Preserve uncertain wording.
- Do not convert suspected conditions into confirmed diagnoses.
- Do not invent missing information.
- Do not state that a finding is normal unless explicitly documented.
- Record contradictions under CONFLICTS OR ITEMS FOR CLINICAL REVIEW.
- Return structured plain text only.
- Do not output JSON.

Use these headings:

PATIENT IDENTIFIERS
- MR number
- IP number
- baby name
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
- age or day of life at admission

BABY DETAILS
- date and time of birth
- sex
- gestational age
- term or preterm classification
- birth weight
- admission weight
- discharge weight
- AGA, SGA or LGA classification
- blood group
- length
- head circumference

MATERNAL AND ANTENATAL HISTORY
- maternal age
- gravida and parity
- antenatal care
- maternal illnesses
- pregnancy complications
- antenatal scans
- maternal medications
- blood group
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
- immediate post-birth problems
- feeding before admission

HISTORY OF PRESENT ILLNESS
- symptom onset
- day or hour of life
- initial symptoms
- outside-hospital evaluation
- outside investigations
- treatment before referral
- referral reason
- condition on arrival

DIAGNOSES
- confirmed diagnoses
- suspected diagnoses
- provisional diagnoses
- gestational classification
- growth classification

ADMISSION EXAMINATION
- general condition
- hydration
- activity
- sensorium
- temperature
- heart rate
- respiratory rate
- oxygen saturation
- blood pressure
- respiratory support
- respiratory examination
- cardiovascular examination
- abdominal examination
- neurological examination
- tone
- suck
- cry
- skin findings
- jaundice
- other abnormal findings

DAILY CLINICAL COURSE
- chronological daily events
- clinical improvement or deterioration
- treatment changes
- last abnormal event
- response to treatment

GLUCOSE AND HYPOGLYCAEMIA
- glucose values
- symptoms
- dextrose bolus
- glucose infusion rate
- IV fluid changes
- glucose monitoring
- glucose stability

NEUROLOGICAL AND SEIZURE COURSE
- seizure type
- seizure timing
- last seizure
- anticonvulsants
- neurological examination
- EEG
- MRI
- neurosonogram
- neurology consultation
- neurological condition at discharge

NEONATAL JAUNDICE COURSE
- onset day or hour
- bilirubin values
- blood groups
- phototherapy
- treatment duration
- repeat bilirubin
- BIND findings
- cholestasis findings
- rebound bilirubin

RESPIRATORY COURSE
- distress findings
- chest X-ray
- oxygen support
- LFNC
- HFNC
- CPAP
- NIMV
- mechanical ventilation
- PEEP
- FiO2
- flow
- surfactant
- support duration
- weaning
- room-air stability

INFECTION AND SEPSIS COURSE
- infection risk factors
- cultures
- CRP
- blood counts
- urine examination
- antibiotics
- antibiotic duration
- confirmed or suspected infection

CARDIOVASCULAR COURSE
- haemodynamic status
- blood pressure
- perfusion
- murmurs
- echocardiography
- inotropes

NUTRITION AND FEEDING
- NPO
- IV nutrition
- OGT feeds
- spoon feeds
- expressed breast milk
- direct breastfeeding
- formula
- feed volume
- feed frequency
- feed progression
- feed intolerance
- feeding status at discharge

LINES AND PROCEDURES
- UVC
- UAC
- PICC
- peripheral IV line
- lumbar puncture
- transfusion
- imaging procedures
- other procedures
- insertion and removal dates
- duration

INVESTIGATIONS
- complete blood picture
- CRP
- cultures
- urine examination
- renal function
- electrolytes
- liver function
- bilirubin
- blood glucose
- blood gas
- metabolic tests
- MRI
- EEG
- neurosonogram
- chest X-ray
- echocardiography
- screening tests
- exact values, dates and units

MEDICATIONS DURING ADMISSION
For every medicine preserve:
- exact medicine name
- brand name
- concentration
- dose
- route
- frequency
- start date
- stop date
- duration
- indication
- reason for stopping when documented

SCREENING TESTS
- hearing screening
- BERA
- ROP
- pulse oximetry
- thyroid profile
- neurosonogram
- immunisation
- other neonatal screening

DISCHARGE CONDITION
- activity
- sensorium
- respiratory status
- haemodynamic status
- feeding status
- glucose status
- seizure-free duration
- urine output
- stools
- temperature stability
- weight
- neurological status
- remaining concerns

DISCHARGE MEDICATIONS
- exact medicine name
- concentration
- dose
- route
- frequency
- duration
- special instructions

DISCHARGE ADVICE
- feeding
- warm care
- hygiene
- KMC
- bathing
- immunisation
- seizure advice
- jaundice advice
- danger signs
- other instructions

FOLLOW-UP
- review date
- doctor
- department
- neurology
- neonatology
- ROP
- hearing
- BERA
- thyroid testing
- neurosonogram
- developmental follow-up
- laboratory follow-up

CONFLICTS OR ITEMS FOR CLINICAL REVIEW
- conflicting dates
- conflicting weights
- conflicting values
- uncertain OCR
- uncertain diagnosis
- inconsistent medication details
- missing important information

CLINICAL RECORD CHUNK:

--- BEGIN CHUNK {chunk_number} ---

{chunk}

--- END CHUNK {chunk_number} ---
""".strip()


def build_evidence_consolidation_prompt(
    extracted_evidence: str,
) -> str:
    """
    Build the prompt for merging evidence extracted from all chunks.
    """

    evidence = extracted_evidence.strip()

    if not evidence:
        raise ValueError("Extracted clinical evidence cannot be empty.")

    return f"""
Consolidate the extracted NICU evidence into one complete clinical evidence
record.

This is not the final discharge summary and not the doctor-friendly evidence
report.

RULES:

1. Preserve every unique clinical fact.
2. Remove only exact duplicates.
3. Preserve chronology.
4. Preserve dates, times, values and units.
5. Preserve medication names, doses and changes.
6. Preserve respiratory settings and changes.
7. Preserve feeding progression.
8. Preserve uncertain diagnoses.
9. Do not merge conflicting values.
10. Place contradictions under:
    CONFLICTS OR ITEMS FOR CLINICAL REVIEW
11. Do not invent missing details.
12. Do not add routine normal findings.
13. Do not output JSON.
14. Return structured plain text only.

Use clear clinical headings.

EXTRACTED CHUNK EVIDENCE:

--- BEGIN EXTRACTED EVIDENCE ---

{evidence}

--- END EXTRACTED EVIDENCE ---
""".strip()