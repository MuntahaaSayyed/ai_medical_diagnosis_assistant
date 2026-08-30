# 🏥 AI Medical Diagnosis Assistant

A full-stack, hospital-styled dashboard that uses AI (TensorFlow/Keras) to
estimate symptom-based risk levels and screen chest X-rays, built with
Flask, SQLite, OpenCV, Pandas, and a modern glassmorphism UI.

> ⚠️ **Disclaimer:** This is an educational/portfolio project. The models
> are demo models (the symptom model is trained on synthetic data; the
> X-ray CNN is trained on synthetic placeholder images since no real
> licensed X-ray dataset is bundled). **This is NOT a certified medical
> device and must never be used for real clinical decisions.** Always
> consult a licensed healthcare professional.

---

## ✨ Features

- 🔐 Secure login/registration (hashed passwords, SQLite)
- 🧑‍⚕️ Patient registration & searchable patient list
- 🧠 AI symptom-risk classifier (Keras dense neural network)
- 🩻 Chest X-ray upload + OpenCV preprocessing + demo CNN screening
- 📊 Dashboard with live stats and Chart.js risk-distribution chart
- 🖨️ Printable / PDF-exportable diagnosis reports
- 📱 Fully responsive hospital-themed UI (sidebar + topbar, glassmorphism)

## 🛠️ Tech Stack

| Layer          | Technology                          |
|----------------|--------------------------------------|
| Backend        | Python, Flask                        |
| ML             | TensorFlow / Keras, scikit-learn     |
| Image handling | OpenCV                               |
| Data           | Pandas, NumPy                        |
| Database       | SQLite                               |
| Frontend       | HTML5, CSS3, JavaScript              |
| Icons/Charts   | Bootstrap Icons (CDN), Chart.js (CDN)|

---

## 📁 Project Structure

```
ai_medical_diagnosis_assistant/
│
├── app.py                     # Main Flask app (routes, auth, logic)
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
│
├── database/
│   ├── db_setup.py            # Creates SQLite tables + default admin user
│   └── medical_assistant.db   # (created automatically on first run)
│
├── models/
│   ├── train_model.py         # Trains the symptom-risk Keras model
│   ├── model_utils.py         # Loads/auto-trains models, runs predictions
│   ├── symptom_risk_model.h5  # (created automatically)
│   └── xray_demo_model.h5     # (created automatically)
│
├── utils/
│   └── image_processing.py    # OpenCV preprocessing for X-ray images
│
├── data/
│   └── symptoms_dataset.csv   # Synthetic training dataset
│
├── static/
│   ├── css/style.css          # Hospital dashboard theme
│   ├── js/main.js             # Sidebar toggle, flash auto-dismiss
│   ├── images/                # (logo/illustrations if you add any)
│   └── uploads/                # Uploaded X-ray previews (created automatically)
│
└── templates/
    ├── base.html               # Shared layout (sidebar + topbar)
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── patients.html
    ├── new_patient.html
    ├── patient_detail.html
    ├── diagnose.html
    ├── report.html
    └── history.html
```

---

## 🚀 Getting Started

### 1. Extract the project and open a terminal in its folder

```bash
cd ai_medical_diagnosis_assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize the database

```bash
python database/db_setup.py
```

This creates `database/medical_assistant.db` and a default login:

```
Username: admin
Password: admin123
```

### 5. (Optional) Pre-train the AI models

You can pre-train the models now, or just skip this — the app will
auto-train both models the **first time it starts** if the `.h5` files
don't exist yet (takes ~10–30 seconds, one-time only).

```bash
python models/train_model.py
```

### 6. Run the app

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 🧠 How the AI Works

### Symptom Risk Model
A dense neural network (`Dense(32) → Dropout → Dense(16) → Dense(3, softmax)`)
trained on a synthetically generated dataset of symptoms + vitals
(`data/symptoms_dataset.csv`), predicting **Low / Moderate / High** risk.

### X-Ray Screening Model
A small CNN (`Conv2D → MaxPool ×3 → Dense`) that classifies preprocessed
chest X-ray images as **Normal** or **Abnormal (Needs Review)**. Since no
real, freely-licensed labeled X-ray dataset ships with this project, the
CNN trains on locally generated synthetic placeholder images so you can
see the full upload → OpenCV preprocessing → CNN inference pipeline work
end-to-end. Swap in a real licensed dataset (e.g. NIH ChestX-ray14) and
retrain `model_utils.py`'s `_train_xray_model()` if you want real
predictive value — with appropriate clinical validation and regulatory
review before any real-world use.

---

## 🔧 Customization Ideas

- Add more symptom fields and retrain `train_model.py`
- Swap the demo X-ray CNN for a transfer-learning model (e.g. MobileNetV2)
- Add PDF export via a library like `reportlab`
- Add role-based permissions (Doctor vs Nurse vs Admin)
- Deploy with Gunicorn + Nginx for a production-style setup

---

## 📜 License

Free to use and modify for learning, portfolio, and demo purposes.
