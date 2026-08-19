"""
Prompt for generating the final hospital discharge summary.
"""

from __future__ import annotations

from .template_prompt import load_template_text


def build_final_discharge_prompt(
    *,
    template_name: str,
    evidence_report: str,
) -> str:
    """
    Build the final hospital discharge-summary prompt using the selected
    hospital DOCX template and doctor-friendly evidence report.
    """

    evidence = evidence_report.strip()

    if not evidence:
        raise ValueError(
            "Evidence report cannot be empty for discharge generation."
        )

    template_text = load_template_text(template_name)

    return f"""
Complete the supplied hospital NICU discharge-summary template using the
doctor-friendly clinical evidence report.

SELECTED TEMPLATE:
{template_name}

The evidence report is the only source of patient-specific clinical
information.

Do not introduce patient information that is absent from the evidence report.

CRITICAL INSTRUCTIONS:

1. Use the hospital template as the required document structure.

2. Preserve:
   - heading order
   - section order
   - hospital wording
   - standard administrative text
   - hospital advice
   - danger signs
   - disclaimers
   - consultant details
   - signature sections

3. Fill patient-specific fields only when supported by the evidence report.

4. When a patient-specific field is unavailable, retain:

   __________

5. Do not write NOT AVAILABLE inside the final hospital form unless the
   hospital specifically requires that wording.

6. Do not copy patient-specific normal findings from the blank template unless
   they are supported by the evidence report.

7. Do not assume:
   - normal antenatal history
   - no maternal comorbidities
   - normal examination
   - normal cardiovascular status
   - normal neurological status
   - successful screening
   - room-air stability
   - full-feed tolerance
   - normal urine output
   - normal stool pattern

8. Preserve exactly:
   - patient names
   - MR and IP numbers
   - dates
   - times
   - gestational age
   - weights
   - diagnoses
   - investigation values
   - units
   - medicine names
   - concentrations
   - doses
   - routes
   - frequencies
   - durations
   - follow-up dates

9. Do not calculate, estimate or infer missing values.

10. Do not infer gestational age from birth weight.

11. Do not convert suspected, possible or provisional diagnoses into confirmed
    diagnoses.

12. Include all important supported discharge diagnoses, not only the
    condition represented by the selected base template.

13. The COURSE IN HOSPITAL section must describe the patient's actual course.

14. Add clinically relevant subsections under COURSE IN HOSPITAL when
    supported, including:
    - symptomatic hypoglycaemia
    - neonatal seizures
    - sepsis or suspected infection
    - neonatal jaundice
    - respiratory distress
    - TTNB
    - prematurity
    - SGA
    - feeding problems
    - apnoea
    - HIE
    - NEC
    - cardiovascular problems
    - neurological problems
    - electrolyte abnormalities
    - other documented conditions

15. For neonatal jaundice, include only documented:
    - onset day or hour
    - bilirubin values
    - mother and baby blood groups
    - phototherapy type
    - phototherapy duration
    - repeat bilirubin values
    - BIND findings
    - cholestasis findings
    - rebound information

16. For respiratory distress or TTNB, include only documented:
    - distress signs
    - SAS score
    - chest X-ray findings
    - respiratory-support type
    - flow
    - PEEP
    - FiO2
    - support duration
    - weaning details
    - room-air observation period

17. Do not automatically prescribe medicines present in the blank template.

18. Include a discharge medicine only when the evidence report explicitly
    states that it should continue after discharge.

19. Do not silently resolve conflicting information.

20. For unresolved conflicts, write:

    [VERIFY WITH CONSULTANT]

21. Do not generate JSON.

22. Do not use Markdown formatting in the hospital discharge summary.

23. Do not include explanations before or after the summary.

24. Return only the completed hospital discharge-summary text.

HOSPITAL TEMPLATE:

--- BEGIN HOSPITAL TEMPLATE ---

{template_text}

--- END HOSPITAL TEMPLATE ---

DOCTOR-FRIENDLY CLINICAL EVIDENCE REPORT:

--- BEGIN EVIDENCE REPORT ---

{evidence}

--- END EVIDENCE REPORT ---
""".strip()