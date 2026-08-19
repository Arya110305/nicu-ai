"""
SQLite database for the NICU AI pipeline.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIRECTORY = PROJECT_ROOT / "database"
DATABASE_DIRECTORY.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_DIRECTORY / "nicu_ai.db"


# =============================================================================
# DATABASE
# =============================================================================

class Database:

    def __init__(self) -> None:
        self.connection = sqlite3.connect(
            DATABASE_PATH,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self.create_tables()

    def create_tables(self) -> None:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                patient_name TEXT,

                uhid TEXT,

                status TEXT,

                upload_time TEXT,

                pdf_path TEXT,

                ocr_path TEXT,

                structured_path TEXT,

                normalized_path TEXT,

                discharge_path TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.connection.commit()

    def add_patient(
        self,
        *,
        patient_name: str,
        pdf_path: str,
    ) -> int:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO patients
            (
                patient_name,
                pdf_path,
                status
            )
            VALUES
            (
                ?,
                ?,
                ?
            )
            """,
            (
                patient_name,
                pdf_path,
                "Uploaded",
            ),
        )

        self.connection.commit()

        return cursor.lastrowid

    def update_status(
        self,
        patient_id: int,
        status: str,
    ) -> None:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            UPDATE patients

            SET status=?

            WHERE id=?
            """,
            (
                status,
                patient_id,
            ),
        )

        self.connection.commit()

    def update_stage_path(
        self,
        patient_id: int,
        column: str,
        value: str,
    ) -> None:

        allowed = {
            "ocr_path",
            "structured_path",
            "normalized_path",
            "discharge_path",
        }

        if column not in allowed:
            raise ValueError(
                f"Invalid column: {column}"
            )

        cursor = self.connection.cursor()

        cursor.execute(
            f"""
            UPDATE patients

            SET {column}=?

            WHERE id=?
            """,
            (
                value,
                patient_id,
            ),
        )

        self.connection.commit()

    def get_patient(
        self,
        patient_id: int,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT *

            FROM patients

            WHERE id=?
            """,
            (
                patient_id,
            ),
        )

        return cursor.fetchone()

    def list_patients(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT *

            FROM patients

            ORDER BY id DESC
            """
        )

        return cursor.fetchall()

    def close(self):

        self.connection.close()


database = Database()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    print("Database:", DATABASE_PATH)

    patients = database.list_patients()

    print(f"Patients: {len(patients)}")