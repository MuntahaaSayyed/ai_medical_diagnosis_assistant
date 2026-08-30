"""
db_setup.py
------------
Creates and initializes the SQLite database used by the AI Medical
Diagnosis Assistant. Run this file directly once before starting the
Flask app for the first time:

    python database/db_setup.py

It creates three tables:
  1. users      -> staff/doctor login accounts
  2. patients   -> basic patient records
  3. diagnoses  -> history of AI-assisted diagnosis results
"""

import sqlite3
import os
import sys
from werkzeug.security import generate_password_hash

# Allow running this file directly (adds project root to path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config


def get_connection():
    """Return a SQLite connection to the project's database file."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # ---- Users table (doctors / staff) ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Doctor',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---- Patients table ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            contact TEXT,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (created_by) REFERENCES users (id)
        )
    """)

    # ---- Diagnoses table ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            symptoms_json TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            confidence REAL NOT NULL,
            xray_filename TEXT,
            xray_result TEXT,
            xray_confidence REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    """)

    conn.commit()

    # ---- Create a default demo doctor account if none exist ----
    cursor.execute("SELECT COUNT(*) as count FROM users")
    if cursor.fetchone()["count"] == 0:
        cursor.execute(
            "INSERT INTO users (full_name, username, password_hash, role) VALUES (?, ?, ?, ?)",
            (
                "Dr. Sarah Mitchell",
                "admin",
                generate_password_hash("admin123"),
                "Administrator",
            ),
        )
        conn.commit()
        print("Default account created -> username: admin | password: admin123")

    conn.close()
    print(f"Database initialized successfully at: {Config.DATABASE_PATH}")


if __name__ == "__main__":
    init_db()
