"""
config.py
----------
Central configuration for the AI Medical Diagnosis Assistant.
Keeping configuration in one place makes the app easier to maintain
and deploy (e.g. switching database paths or secret keys later).
"""

import os

# Base directory of the project (folder where this file lives)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Flask secret key -> used to sign session cookies.
    # In a real production app, load this from an environment variable.
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me-in-production")

    # SQLite database file location
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "medical_assistant.db")

    # Folder where uploaded medical images (e.g. X-rays) are stored
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

    # Allowed image extensions for uploads
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    # Maximum upload size: 5 MB
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024

    # Path to the trained symptom-risk Keras model
    SYMPTOM_MODEL_PATH = os.path.join(BASE_DIR, "models", "symptom_risk_model.h5")

    # Path to the demo X-ray CNN model
    XRAY_MODEL_PATH = os.path.join(BASE_DIR, "models", "xray_demo_model.h5")

    # Path to the synthetic training dataset
    SYMPTOM_DATASET_PATH = os.path.join(BASE_DIR, "data", "symptoms_dataset.csv")
