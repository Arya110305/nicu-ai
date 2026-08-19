"""
Section Extraction Schema

Each section corresponds to one classified folder.

The extractor uses this schema to know what information
to retrieve from each section.
"""

SECTION_SCHEMAS = {

    "PATIENT_INFORMATION": [
        "Patient Name",
        "UHID",
        "Hospital Number",
        "Date of Birth",
        "Sex",
        "Gestational Age",
        "Birth Weight",
        "Current Weight",
        "Blood Group",
    ],

    "BIRTH_DETAILS": [
        "Date and Time of Birth",
        "Mode of Delivery",
        "Apgar Score",
        "Birth Weight",
        "Head Circumference",
        "Length",
    ],

    "MATERNAL_HISTORY": [
        "Mother Age",
        "Gravida",
        "Parity",
        "Medical History",
        "Pregnancy Complications",
        "Medications",
        "Antenatal Steroids",
        "Blood Group",
    ],

    "ADMISSION": [
        "Admission Date",
        "Admission Diagnosis",
        "Reason for Admission",
        "Referring Hospital",
    ],

    "NICU_COURSE": [
        "Hospital Course",
        "Respiratory Support",
        "Nutrition",
        "Major Events",
    ],

    "VITALS": [
        "Temperature",
        "Heart Rate",
        "Respiratory Rate",
        "Blood Pressure",
        "SpO2",
    ],

    "LABORATORY": [
        "CBC",
        "CRP",
        "Blood Culture",
        "Electrolytes",
        "Bilirubin",
        "Blood Gas",
        "Blood Sugar",
    ],

    "MEDICATIONS": [
        "Medication Name",
        "Dose",
        "Route",
        "Frequency",
        "Duration",
    ],

    "PROCEDURES": [
        "Procedure Name",
        "Procedure Date",
        "Outcome",
    ],

    "DAILY_PROGRESS": [
        "Clinical Progress",
        "Doctor Notes",
        "Nursing Notes",
    ],

    "SCREENING": [
        "ROP",
        "Hearing",
        "Metabolic Screening",
        "Vaccination",
    ],

    "DISCHARGE": [
        "Discharge Date",
        "Final Diagnosis",
        "Condition at Discharge",
        "Follow-up",
    ],
    "OTHER": [
    "Miscellaneous Information",
    ],
}
