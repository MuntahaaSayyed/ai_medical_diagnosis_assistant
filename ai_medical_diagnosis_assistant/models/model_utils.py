"""
model_utils.py
----------------
Central place where the Flask app talks to the TensorFlow/Keras models.

Two models are managed here:

1. SYMPTOM RISK MODEL
   A small dense neural network that takes structured symptom/vitals
   input and predicts a risk level: Low / Moderate / High.
   Trained on data/symptoms_dataset.csv (see train_model.py).

2. X-RAY DEMO MODEL
   A small CNN that takes a preprocessed chest X-ray image and predicts
   Normal vs Abnormal. Because no real, freely-licensed labeled X-ray
   dataset is bundled with this project, the CNN is trained on
   SYNTHETIC placeholder images generated locally with OpenCV/NumPy.
   This is purely to demonstrate the full image-pipeline (upload ->
   preprocess -> CNN -> result) end-to-end. It is NOT a clinically
   validated radiology model.

Both models auto-train on first run if a saved .h5 file is not found,
so the project works immediately after `pip install -r requirements.txt`
without any manual training step.
"""

import os
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

# TensorFlow is imported lazily (inside functions) in some places to
# keep Flask's startup fast when models are already cached in memory.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODELS_DIR = os.path.dirname(Config.SYMPTOM_MODEL_PATH)
XRAY_META_PATH = os.path.join(MODELS_DIR, "xray_model_meta.json")
SYMPTOM_META_PATH = os.path.join(MODELS_DIR, "symptom_model_meta.json")
IMG_SIZE = 128

# In-memory caches so we don't reload from disk on every request
_symptom_model_cache = {"model": None, "meta": None}
_xray_model_cache = {"model": None, "meta": None}


# ----------------------------------------------------------------------
# SYMPTOM RISK MODEL
# ----------------------------------------------------------------------

def _train_symptom_model():
    """Train the symptom-risk model on the synthetic CSV dataset."""
    from train_model import main as train_main
    train_main()


def get_symptom_model():
    """Load the symptom-risk model (training it first if necessary)."""
    if _symptom_model_cache["model"] is not None:
        return _symptom_model_cache["model"], _symptom_model_cache["meta"]

    if not os.path.exists(Config.SYMPTOM_MODEL_PATH) or not os.path.exists(SYMPTOM_META_PATH):
        print("[model_utils] Symptom model not found - training now (first run)...")
        _train_symptom_model()

    model = keras.models.load_model(Config.SYMPTOM_MODEL_PATH)
    with open(SYMPTOM_META_PATH) as f:
        meta = json.load(f)

    _symptom_model_cache["model"] = model
    _symptom_model_cache["meta"] = meta
    return model, meta


def predict_symptom_risk(input_dict: dict) -> dict:
    """
    input_dict keys must match FEATURE_COLUMNS in train_model.py:
    age, fever, cough, fatigue, shortness_of_breath, chest_pain,
    headache, body_ache, sore_throat, heart_rate, temperature, spo2

    Returns:
    {
        "risk_level": "Low" | "Moderate" | "High",
        "confidence": float (0-1),
        "probabilities": {"Low": .., "Moderate": .., "High": ..}
    }
    """
    model, meta = get_symptom_model()
    feature_columns = meta["feature_columns"]
    mean = np.array(meta["scaler_mean"])
    scale = np.array(meta["scaler_scale"])
    classes = meta["classes"]

    # Build feature vector in the exact same order used during training
    x = np.array([[float(input_dict[col]) for col in feature_columns]], dtype="float32")
    x_scaled = (x - mean) / scale

    probs = model.predict(x_scaled, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))

    return {
        "risk_level": classes[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(classes, probs)},
    }


# ----------------------------------------------------------------------
# X-RAY DEMO CNN MODEL
# ----------------------------------------------------------------------

def _generate_synthetic_xray_dataset(n_per_class=250):
    """
    Generates simple synthetic grayscale images that stand in for chest
    X-rays, purely so the CNN pipeline has something to train on locally
    without requiring a downloaded medical dataset.

    "Normal"   -> smooth radial gradient with light noise
    "Abnormal" -> same gradient PLUS random bright/dark blob patches
                  simulating an opacity/irregularity
    """
    import cv2
    rng = np.random.default_rng(42)
    images = []
    labels = []

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cy, cx = IMG_SIZE / 2, IMG_SIZE / 2
    base_gradient = 255 - (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (IMG_SIZE / 2) * 180)
    base_gradient = np.clip(base_gradient, 0, 255)

    for _ in range(n_per_class):
        # ---- Normal sample ----
        noise = rng.normal(0, 8, (IMG_SIZE, IMG_SIZE))
        img_normal = np.clip(base_gradient + noise, 0, 255).astype("uint8")
        images.append(img_normal)
        labels.append(0)  # Normal

        # ---- Abnormal sample: add 1-3 random opacity blobs ----
        img_abn = img_normal.copy()
        num_blobs = rng.integers(1, 4)
        for _ in range(num_blobs):
            bx = int(rng.integers(20, IMG_SIZE - 20))
            by = int(rng.integers(20, IMG_SIZE - 20))
            radius = int(rng.integers(6, 16))
            intensity = int(rng.integers(180, 255))
            cv2.circle(img_abn, (bx, by), radius, intensity, -1)
        images.append(img_abn)
        labels.append(1)  # Abnormal

    X = np.array(images, dtype="float32") / 255.0
    X = np.expand_dims(X, axis=-1)  # (N, H, W, 1)
    y = np.array(labels, dtype="int32")

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _build_xray_cnn():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _train_xray_model():
    print("[model_utils] Training demo X-ray CNN on synthetic images (first run)...")
    X, y = _generate_synthetic_xray_dataset(n_per_class=250)
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = _build_xray_cnn()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, batch_size=16, verbose=2)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(Config.XRAY_MODEL_PATH)

    meta = {"classes": ["Normal", "Abnormal (Needs Review)"], "img_size": IMG_SIZE}
    with open(XRAY_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[model_utils] X-ray demo model saved to: {Config.XRAY_MODEL_PATH}")


def get_xray_model():
    if _xray_model_cache["model"] is not None:
        return _xray_model_cache["model"], _xray_model_cache["meta"]

    if not os.path.exists(Config.XRAY_MODEL_PATH) or not os.path.exists(XRAY_META_PATH):
        _train_xray_model()

    model = keras.models.load_model(Config.XRAY_MODEL_PATH)
    with open(XRAY_META_PATH) as f:
        meta = json.load(f)

    _xray_model_cache["model"] = model
    _xray_model_cache["meta"] = meta
    return model, meta


def predict_xray(preprocessed_image: np.ndarray) -> dict:
    """
    preprocessed_image: numpy array of shape (1, IMG_SIZE, IMG_SIZE, 1)
    as produced by utils/image_processing.preprocess_xray()

    Returns:
    {
        "result": "Normal" | "Abnormal (Needs Review)",
        "confidence": float (0-1)
    }
    """
    model, meta = get_xray_model()
    probs = model.predict(preprocessed_image, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    return {
        "result": meta["classes"][predicted_idx],
        "confidence": float(probs[predicted_idx]),
    }
