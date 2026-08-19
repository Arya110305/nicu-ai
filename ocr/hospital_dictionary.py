"""
Hospital Dictionary
===================

Central repository of hospital and NICU terminology.

Used by:
- OCR Quality Analyzer
- OCR Health Score
- Future Page Classifier
- OCR Correction
- LLM Extraction
"""

# ==========================================================
# PATIENT INFORMATION
# ==========================================================

PATIENT_FIELDS = [

    "Patient Name",
    "Mother Name",
    "Father Name",
    "UHID",
    "MR No",
    "MR Number",
    "IPD",
    "IP Number",
    "IP No",
    "OP Number",
    "Age",
    "Gender",
    "DOB",
    "Date of Birth",
    "Address",
    "Phone",
    "Consultant",

]

# ==========================================================
# ADMISSION
# ==========================================================

ADMISSION_FIELDS = [

    "Admission",
    "Admission Date",
    "Admission Time",
    "Reason for Admission",
    "Chief Complaint",
    "Referral",
    "Diagnosis",
    "Provisional Diagnosis",
    "Final Diagnosis",

]

# ==========================================================
# MATERNAL HISTORY
# ==========================================================

MATERNAL_FIELDS = [

    "Mother Blood Group",
    "Gravida",
    "Parity",
    "LSCS",
    "Normal Delivery",
    "Caesarean",
    "PROM",
    "PIH",
    "GDM",
    "Antenatal",
    "Antenatal History",
    "Liquor",
    "Meconium",
    "Birth Asphyxia",

]

# ==========================================================
# NICU
# ==========================================================

NICU_FIELDS = [

    "Gestational Age",
    "GA",
    "Birth Weight",
    "Current Weight",
    "Discharge Weight",
    "Length",
    "Head Circumference",
    "HC",
    "APGAR",
    "Preterm",
    "Term",
    "Low Birth Weight",
    "VLBW",
    "ELBW",
    "NICU",
    "Phototherapy",
    "Sepsis",
    "CPAP",
    "HFNC",
    "Ventilator",
    "Surfactant",
    "Incubator",

]

# ==========================================================
# VITALS
# ==========================================================

VITAL_FIELDS = [

    "Temperature",
    "Heart Rate",
    "HR",
    "Respiratory Rate",
    "RR",
    "Pulse",
    "Pulse Rate",
    "SpO2",
    "SPO2",
    "Blood Pressure",
    "BP",
    "MAP",
    "CRT",
    "GRBS",
    "Weight",

]

# ==========================================================
# LAB TESTS
# ==========================================================

LAB_FIELDS = [

    "CBC",
    "Hemoglobin",
    "HB",
    "PCV",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
    "RBC",
    "WBC",
    "Platelets",
    "Neutrophils",
    "Lymphocytes",
    "Monocytes",
    "Eosinophils",
    "Basophils",

    "CRP",
    "Procalcitonin",

    "Blood Urea",
    "BUN",
    "Creatinine",
    "Uric Acid",

    "Sodium",
    "Potassium",
    "Chloride",
    "Calcium",
    "Ionic Calcium",

    "Bilirubin",
    "Total Bilirubin",
    "Direct Bilirubin",
    "Indirect Bilirubin",
    "Albumin",
    "Globulin",
    "ALP",
    "AST",
    "ALT",

    "Blood Gas",
    "pH",
    "PCO2",
    "PO2",
    "HCO3",
    "Base Excess",
    "Lactate",

    "Blood Culture",
    "Urine Culture",
    "Culture",
    "Sensitivity",

]

# ==========================================================
# URINE
# ==========================================================

URINE_FIELDS = [

    "Protein",
    "Sugar",
    "Ketone",
    "Blood",
    "Leukocytes",
    "Nitrite",
    "Specific Gravity",
    "pH",
    "Pus Cells",
    "Epithelial Cells",

]

# ==========================================================
# MEDICATIONS
# ==========================================================

MEDICATION_FIELDS = [

    "Medication",
    "Medicine",
    "Dose",
    "Dosage",
    "Route",
    "Frequency",

    "Ampicillin",
    "Amikacin",
    "Gentamicin",
    "Meropenem",
    "Vancomycin",
    "Cefotaxime",
    "Ceftriaxone",

    "Paracetamol",
    "Vitamin K",
    "Calcium Gluconate",
    "Dextrose",

]

# ==========================================================
# PROCEDURES
# ==========================================================

PROCEDURE_FIELDS = [

    "Procedure",
    "Intubation",
    "Extubation",
    "Umbilical Catheter",
    "Central Line",
    "Peripheral Line",
    "Lumbar Puncture",
    "Exchange Transfusion",

]

# ==========================================================
# SCREENING
# ==========================================================

SCREENING_FIELDS = [

    "ROP",
    "Hearing Screening",
    "Thyroid",
    "Pulse Ox",
    "Vaccination",
    "BCG",
    "OPV",
    "Hepatitis B",

]

# ==========================================================
# DISCHARGE
# ==========================================================

DISCHARGE_FIELDS = [

    "Discharge",
    "Follow Up",
    "Advice",
    "Condition",
    "Feeding",
    "Breast Feeding",
    "Formula Feed",

]

# ==========================================================
# DOCUMENT TYPES
# ==========================================================

DOCUMENT_TYPES = [

    "Admission Record",
    "Initial Assessment",
    "Progress Notes",
    "Nursing Notes",
    "Discharge Summary",
    "Laboratory Report",
    "Biochemistry",
    "Hematology",
    "Microbiology",
    "Radiology",
    "Patient Movement Record",
    "Medication Chart",
    "Prescription",

]

# ==========================================================
# COMMON ABBREVIATIONS
# ==========================================================

ABBREVIATIONS = [

    "NICU",
    "PICU",
    "ICU",
    "GA",
    "BW",
    "HC",
    "RR",
    "HR",
    "BP",
    "SpO2",
    "CBC",
    "CRP",
    "LFT",
    "RFT",
    "ABG",
    "CPAP",
    "HFNC",
    "IV",
    "IM",
    "PO",

]

# ==========================================================
# COMMON OCR CORRECTIONS
# ==========================================================

OCR_CORRECTIONS = {

    "ankuro": "Ankura",
    "ankera": "Ankura",
    "ankurs": "Ankura",
    "phototheropy": "Phototherapy",
    "bilirubine": "Bilirubin",
    "haemoglobin": "Hemoglobin",
    "sp02": "SpO2",
    "spo 2": "SpO2",

}

# ==========================================================
# MERGED LIST
# ==========================================================

ALL_MEDICAL_FIELDS = (

    PATIENT_FIELDS +
    ADMISSION_FIELDS +
    MATERNAL_FIELDS +
    NICU_FIELDS +
    VITAL_FIELDS +
    LAB_FIELDS +
    URINE_FIELDS +
    MEDICATION_FIELDS +
    PROCEDURE_FIELDS +
    SCREENING_FIELDS +
    DISCHARGE_FIELDS +
    DOCUMENT_TYPES +
    ABBREVIATIONS

)
# ==========================================================
# MEDICATIONS
# ==========================================================

MEDICATION_FIELDS = [

    # General
    "Medication",
    "Medicine",
    "Drug",
    "Dose",
    "Dosage",
    "Route",
    "Frequency",
    "Infusion",
    "Bolus",

    # Antibiotics
    "Ampicillin",
    "Amikacin",
    "Gentamicin",
    "Meropenem",
    "Vancomycin",
    "Cefotaxime",
    "Ceftriaxone",
    "Piperacillin",
    "Tazobactam",
    "Piperacillin-Tazobactam",
    "Linezolid",
    "Colistin",
    "Flucloxacillin",
    "Cloxacillin",
    "Azithromycin",

    # Antifungals
    "Fluconazole",
    "Amphotericin B",
    "Micafungin",
    "Voriconazole",

    # Antivirals
    "Acyclovir",

    # Respiratory
    "Surfactant",
    "Beractant",
    "Poractant Alfa",
    "Caffeine Citrate",
    "Salbutamol",
    "Budesonide",

    # Cardiovascular
    "Dopamine",
    "Dobutamine",
    "Adrenaline",
    "Epinephrine",
    "Noradrenaline",
    "Norepinephrine",
    "Milrinone",
    "Hydrocortisone",

    # Fluids
    "Normal Saline",
    "NS",
    "Ringer Lactate",
    "RL",
    "Dextrose",
    "D5",
    "D10",
    "D12.5",
    "D25",
    "D50",

    # Electrolytes
    "Calcium Gluconate",
    "Potassium Chloride",
    "Magnesium Sulphate",
    "Sodium Bicarbonate",

    # Nutrition
    "TPN",
    "Parenteral Nutrition",
    "Lipids",
    "Intralipid",
    "SMOFlipid",
    "Amino Acid",

    # Vitamins
    "Vitamin K",
    "Vitamin D",
    "Multivitamin",
    "Iron",
    "Folic Acid",

    # Pain / Sedation
    "Paracetamol",
    "Morphine",
    "Fentanyl",
    "Midazolam",
    "Ketamine",

    # Seizures
    "Phenobarbitone",
    "Phenobarbital",
    "Levetiracetam",
    "Phenytoin",
    "Lorazepam",

    # GI
    "Pantoprazole",
    "Omeprazole",
    "Domperidone",
    "Lactulose",

    # Blood Products
    "Fresh Frozen Plasma",
    "FFP",
    "Packed Red Cells",
    "PRBC",
    "Platelet Concentrate",
    "Cryoprecipitate",

    # Vaccines
    "BCG",
    "OPV",
    "Hepatitis B Vaccine",

]
MEDICATION_ROUTES = [

    "IV",
    "IM",
    "PO",
    "NG",
    "OG",
    "ET",
    "Neb",
    "SC",

]
MEDICATION_FREQUENCIES = [

    "OD",
    "BD",
    "TDS",
    "QID",
    "SOS",
    "STAT",
    "PRN",
    "q6h",
    "q8h",
    "q12h",
    "Continuous",

]
LAB_PARAMETERS = [

    "Hemoglobin",
    "WBC",
    "RBC",
    "Platelets",
    "CRP",
    "Procalcitonin",
    "Bilirubin",
    "Creatinine",
    "Blood Urea",
    "BUN",
    "Sodium",
    "Potassium",
    "Chloride",
    "Calcium",
    "Magnesium",
    "Phosphorus",
    "Blood Sugar",
    "Glucose",
    "Lactate",
    "pH",
    "PCO2",
    "PO2",
    "HCO3",

]