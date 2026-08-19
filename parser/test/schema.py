# parser/schema.py

SCHEMA = {
    "patient_information": [
        "Patient Name",
        "UHID",
        "IPD Number",
        "MR Number",
        "Gender",
        "Age",
        "Date of Birth",
        "Time of Birth",
        "Father Name",
        "Mother Name",
        "Consultant",
        "Hospital"
    ],

    "birth_details": [
        "Birth Weight",
        "Birth Length",
        "Head Circumference",
        "Gestational Age",
        "Mode of Delivery",
        "Presentation",
        "APGAR 1 Minute",
        "APGAR 5 Minutes",
        "APGAR 10 Minutes",
        "Resuscitation Required"
    ],

    "maternal_history": [
        "Mother Age",
        "Gravida",
        "Parity",
        "Blood Group",
        "Diabetes",
        "Hypertension",
        "Thyroid Disease",
        "PROM",
        "Meconium",
        "Antenatal Steroids",
        "Maternal Fever",
        "Maternal Infection",
        "Risk Factors"
    ],

    "admission": [
        "Admission Date",
        "Admission Time",
        "Admission Diagnosis",
        "Reason for Admission",
        "Initial Condition",
        "NICU Bed",
        "Consultant"
    ],

    "nicu_course": [
        "Primary Diagnosis",
        "Secondary Diagnosis",
        "Respiratory Support",
        "Cardiac Support",
        "Nutrition",
        "Complications",
        "Clinical Progress",
        "Final Outcome"
    ],

    "vitals": [
        "Temperature",
        "Heart Rate",
        "Respiratory Rate",
        "Blood Pressure",
        "SpO2",
        "Weight",
        "Length",
        "Head Circumference"
    ],

    "laboratory": [
        "Complete Blood Count",
        "CRP",
        "Blood Culture",
        "Urine Culture",
        "Liver Function Test",
        "Renal Function Test",
        "Electrolytes",
        "Blood Gas",
        "Blood Glucose",
        "Bilirubin",
        "Ammonia",
        "Calcium",
        "Other Laboratory Results"
    ],

    "medications": [
        "Antibiotics",
        "IV Fluids",
        "Nutrition",
        "Vitamin Supplements",
        "Respiratory Medications",
        "Other Medications"
    ],

    "procedures": [
        "Peripheral Cannulation",
        "Central Line",
        "Umbilical Venous Catheter",
        "Umbilical Arterial Catheter",
        "Intubation",
        "Ventilation",
        "CPAP",
        "Phototherapy",
        "Exchange Transfusion",
        "Blood Transfusion",
        "Lumbar Puncture",
        "Other Procedures"
    ],

    "daily_progress": [
        "Daily Progress Notes",
        "Consultant Notes",
        "Nursing Notes",
        "Feeding Progress",
        "Respiratory Progress",
        "Clinical Changes"
    ],

    "screening": [
        "Hearing Screening",
        "ROP Screening",
        "Metabolic Screening",
        "Cardiac Screening",
        "Neurological Screening",
        "Vaccinations"
    ],

    "discharge": [
        "Discharge Date",
        "Discharge Diagnosis",
        "Condition At Discharge",
        "Discharge Weight",
        "Discharge Medications",
        "Follow Up",
        "Advice"
    ]
}