"""
train_model.py
----------------
Trains a small TensorFlow/Keras neural network that estimates a patient's
RISK LEVEL (Low / Moderate / High) from reported symptoms and vitals.

This is a DEMO / EDUCATIONAL model trained on synthetically generated
data (see data/symptoms_dataset.csv). It illustrates the full pipeline
(data -> preprocessing -> training -> saved model -> inference) but is
NOT validated for real clinical use.

Run once before starting the Flask app:
    python models/train_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

FEATURE_COLUMNS = [
    "age", "fever", "cough", "fatigue", "shortness_of_breath",
    "chest_pain", "headache", "body_ache", "sore_throat",
    "heart_rate", "temperature", "spo2"
]


def load_data():
    df = pd.read_csv(Config.SYMPTOM_DATASET_PATH)
    X = df[FEATURE_COLUMNS].values.astype("float32")
    y_raw = df["risk_level"].astype(str).values
    return X, y_raw


def build_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading dataset...")
    X, y_raw = load_data()

    # Encode text labels (Low/Moderate/High) into integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Scale numeric features so the network trains smoothly
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building model...")
    model = build_model(input_dim=X.shape[1], num_classes=len(label_encoder.classes_))

    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=16,
        verbose=2,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.3f}  |  Test loss: {loss:.3f}")

    # Save model
    model.save(Config.SYMPTOM_MODEL_PATH)
    print(f"Model saved to: {Config.SYMPTOM_MODEL_PATH}")

    # Save scaler parameters + label classes so model_utils.py can
    # reproduce the exact same preprocessing at inference time.
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": label_encoder.classes_.tolist(),
    }
    meta_path = os.path.join(os.path.dirname(Config.SYMPTOM_MODEL_PATH), "symptom_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
