"""
image_processing.py
---------------------
OpenCV-based helper functions for preparing uploaded medical images
(e.g. chest X-rays) before they are fed into the demo CNN model.

Pipeline:
    1. Read image from disk
    2. Convert to grayscale (X-rays are single-channel)
    3. Resize to a fixed shape the CNN expects
    4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
       to enhance visibility of lung fields / structures
    5. Normalize pixel values to [0, 1]
"""

import cv2
import numpy as np

IMG_SIZE = 128  # width & height the CNN model expects


def preprocess_xray(image_path: str) -> np.ndarray:
    """
    Load an X-ray image from disk and preprocess it for model inference.

    Returns a numpy array of shape (1, IMG_SIZE, IMG_SIZE, 1) ready to be
    passed to model.predict().
    """
    # 1. Read image in grayscale mode directly
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # 2. Resize to the CNN's expected input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 3. Enhance contrast so subtle features are more visible
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 4. Normalize to [0, 1] range
    img = img.astype("float32") / 255.0

    # 5. Reshape to (1, H, W, 1) -> batch of 1, single channel
    img = np.expand_dims(img, axis=-1)   # (H, W, 1)
    img = np.expand_dims(img, axis=0)    # (1, H, W, 1)

    return img


def generate_preview_image(image_path: str, save_path: str) -> None:
    """
    Save a human-viewable, contrast-enhanced preview of the uploaded
    X-ray (grayscale + CLAHE, but NOT normalized) so it can be displayed
    back to the doctor in the UI next to the AI result.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    cv2.imwrite(save_path, img)
