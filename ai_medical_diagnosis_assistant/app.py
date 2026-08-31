"""
app.py
-------
Main Flask application for the AI Medical Diagnosis Assistant.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
Default login -> username: admin | password: admin123
"""

import os
import json
import uuid
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from database.db_setup import get_connection, init_db
from models.model_utils import predict_symptom_risk, predict_xray
from utils.image_processing import preprocess_xray, generate_preview_image

app = Flask(__name__)
app.config.from_object(Config)

# Ensure database + upload folder exist before the app starts serving requests
os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
if not os.path.exists(Config.DATABASE_PATH):
    init_db()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def login_required(view_func):
    """Simple decorator to protect routes that require an authenticated user."""
    from functools import wraps

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# ----------------------------------------------------------------------
# Auth routes
# ----------------------------------------------------------------------

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]
            flash(f"Welcome back, {user['full_name']}.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "Doctor")

        if not full_name or not username or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        conn = get_connection()
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            flash("Username already taken. Please choose another.", "danger")
            conn.close()
            return render_template("register.html")

        conn.execute(
            "INSERT INTO users (full_name, username, password_hash, role) VALUES (?, ?, ?, ?)",
            (full_name, username, generate_password_hash(password), role),
        )
        conn.commit()
        conn.close()

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_connection()

    total_patients = conn.execute("SELECT COUNT(*) c FROM patients").fetchone()["c"]
    total_diagnoses = conn.execute("SELECT COUNT(*) c FROM diagnoses").fetchone()["c"]
    high_risk_count = conn.execute(
        "SELECT COUNT(*) c FROM diagnoses WHERE risk_level = 'High'"
    ).fetchone()["c"]

    risk_breakdown = conn.execute("""
        SELECT risk_level, COUNT(*) as count
        FROM diagnoses
        GROUP BY risk_level
    """).fetchall()
    risk_data = {"Low": 0, "Moderate": 0, "High": 0}
    for row in risk_breakdown:
        risk_data[row["risk_level"]] = row["count"]

    recent_diagnoses = conn.execute("""
        SELECT d.id, d.risk_level, d.confidence, d.created_at, p.full_name as patient_name
        FROM diagnoses d
        JOIN patients p ON d.patient_id = p.id
        ORDER BY d.created_at DESC
        LIMIT 6
    """).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_diagnoses=total_diagnoses,
        high_risk_count=high_risk_count,
        risk_data=risk_data,
        recent_diagnoses=recent_diagnoses,
    )


# ----------------------------------------------------------------------
# Patients
# ----------------------------------------------------------------------

@app.route("/patients")
@login_required
def patients():
    conn = get_connection()
    search = request.args.get("q", "").strip()
    if search:
        rows = conn.execute(
            "SELECT * FROM patients WHERE full_name LIKE ? ORDER BY created_at DESC",
            (f"%{search}%",),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template("patients.html", patients=rows, search=search)


@app.route("/patients/new", methods=["GET", "POST"])
@login_required
def new_patient():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        age = request.form.get("age", "")
        gender = request.form.get("gender", "")
        contact = request.form.get("contact", "").strip()

        if not full_name or not age or not gender:
            flash("Name, age and gender are required.", "danger")
            return render_template("new_patient.html")

        conn = get_connection()
        cursor = conn.execute(
            "INSERT INTO patients (full_name, age, gender, contact, created_by) VALUES (?, ?, ?, ?, ?)",
            (full_name, int(age), gender, contact, session["user_id"]),
        )
        conn.commit()
        patient_id = cursor.lastrowid
        conn.close()

        flash(f"Patient '{full_name}' added successfully.", "success")
        return redirect(url_for("diagnose", patient_id=patient_id))

    return render_template("new_patient.html")


@app.route("/patients/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    conn = get_connection()
    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient:
        conn.close()
        flash("Patient not found.", "danger")
        return redirect(url_for("patients"))

    diagnoses = conn.execute(
        "SELECT * FROM diagnoses WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,),
    ).fetchall()
    conn.close()

    return render_template("patient_detail.html", patient=patient, diagnoses=diagnoses)


# ----------------------------------------------------------------------
# Diagnosis (symptom form + optional X-ray upload)
# ----------------------------------------------------------------------

@app.route("/diagnose/<int:patient_id>", methods=["GET", "POST"])
@login_required
def diagnose(patient_id):
    conn = get_connection()
    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient:
        conn.close()
        flash("Patient not found.", "danger")
        return redirect(url_for("patients"))

    if request.method == "POST":
        # ---- 1. Collect symptom form data ----
        symptom_input = {
            "age": patient["age"],
            "fever": 1 if request.form.get("fever") else 0,
            "cough": 1 if request.form.get("cough") else 0,
            "fatigue": 1 if request.form.get("fatigue") else 0,
            "shortness_of_breath": 1 if request.form.get("shortness_of_breath") else 0,
            "chest_pain": 1 if request.form.get("chest_pain") else 0,
            "headache": 1 if request.form.get("headache") else 0,
            "body_ache": 1 if request.form.get("body_ache") else 0,
            "sore_throat": 1 if request.form.get("sore_throat") else 0,
            "heart_rate": float(request.form.get("heart_rate") or 80),
            "temperature": float(request.form.get("temperature") or 37.0),
            "spo2": float(request.form.get("spo2") or 98),
        }
        notes = request.form.get("notes", "").strip()

        # ---- 2. Run symptom-risk model ----
        risk_result = predict_symptom_risk(symptom_input)

        # ---- 3. Optional X-ray upload + CNN prediction ----
        xray_filename = None
        xray_result = None
        xray_confidence = None

        file = request.files.get("xray_image")
        if file and file.filename and allowed_file(file.filename):
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            save_path = os.path.join(Config.UPLOAD_FOLDER, unique_name)
            file.save(save_path)

            # Generate a nice contrast-enhanced preview for the UI
            preview_name = f"preview_{unique_name}"
            preview_path = os.path.join(Config.UPLOAD_FOLDER, preview_name)
            generate_preview_image(save_path, preview_path)

            # Preprocess + run CNN
            processed = preprocess_xray(save_path)
            xray_prediction = predict_xray(processed)

            xray_filename = preview_name
            xray_result = xray_prediction["result"]
            xray_confidence = xray_prediction["confidence"]

        # ---- 4. Save diagnosis record ----
        conn.execute("""
            INSERT INTO diagnoses (
                patient_id, symptoms_json, risk_level, confidence,
                xray_filename, xray_result, xray_confidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            json.dumps(symptom_input),
            risk_result["risk_level"],
            risk_result["confidence"],
            xray_filename,
            xray_result,
            xray_confidence,
            notes,
        ))
        conn.commit()
        diagnosis_id = conn.execute("SELECT last_insert_rowid() as id").fetchone()["id"]
        conn.close()

        return redirect(url_for("report", diagnosis_id=diagnosis_id))

    conn.close()
    return render_template("diagnose.html", patient=patient)


@app.route("/report/<int:diagnosis_id>")
@login_required
def report(diagnosis_id):
    conn = get_connection()
    diagnosis = conn.execute("SELECT * FROM diagnoses WHERE id = ?", (diagnosis_id,)).fetchone()
    if not diagnosis:
        conn.close()
        flash("Diagnosis report not found.", "danger")
        return redirect(url_for("dashboard"))

    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (diagnosis["patient_id"],)).fetchone()
    conn.close()

    symptoms = json.loads(diagnosis["symptoms_json"])

    return render_template(
        "report.html",
        diagnosis=diagnosis,
        patient=patient,
        symptoms=symptoms,
        generated_at=datetime.now().strftime("%B %d, %Y %H:%M"),
    )


# ----------------------------------------------------------------------
# History (all diagnoses across all patients)
# ----------------------------------------------------------------------

@app.route("/history")
@login_required
def history():
    conn = get_connection()
    rows = conn.execute("""
        SELECT d.id, d.risk_level, d.confidence, d.created_at,
               p.full_name as patient_name, p.id as patient_id
        FROM diagnoses d
        JOIN patients p ON d.patient_id = p.id
        ORDER BY d.created_at DESC
    """).fetchall()
    conn.close()
    return render_template("history.html", diagnoses=rows)


# ----------------------------------------------------------------------
# Static uploads route (serves stored X-ray previews)
# ----------------------------------------------------------------------

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
