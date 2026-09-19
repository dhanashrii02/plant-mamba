"""
Plant Mamba - SQLite Database Layer
Handles:
1. User Authentication (Registration, Password Hashing, Login, User Profiles)
2. Consultation & Diagnosis History (Saving scans, thumbnails, CAM images, reloading past diagnoses)
3. Statistical Aggregations for Farm Analytics
"""

import os
import sqlite3
import hashlib
import json
from datetime import datetime
from PIL import Image

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant_mamba.db")
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)


def _get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.strip().encode("utf-8")).hexdigest()


def init_db():
    """Initializes SQLite database tables and default admin/demo accounts."""
    conn = _get_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'Agronomist',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Consultations / Diagnostic History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            report_id TEXT UNIQUE NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            crop_name TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            clean_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity_level TEXT NOT NULL,
            infected_ratio REAL NOT NULL,
            city TEXT NOT NULL,
            weather_temp REAL,
            weather_humidity REAL,
            risk_level TEXT,
            risk_explanation TEXT,
            orig_img_path TEXT,
            cam_img_path TEXT,
            treatment_json TEXT,
            similar_cases_json TEXT,
            review_status TEXT DEFAULT 'AI Preliminary',
            expert_notes TEXT DEFAULT '',
            certified_by TEXT DEFAULT '',
            confirmed_disease TEXT DEFAULT '',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Check and add new columns if table existed without them
    cursor.execute("PRAGMA table_info(consultations)")
    existing_cols = [col[1] for col in cursor.fetchall()]
    for new_col, col_type, default_val in [
        ("review_status", "TEXT", "'AI Preliminary'"),
        ("expert_notes", "TEXT", "''"),
        ("certified_by", "TEXT", "''"),
        ("confirmed_disease", "TEXT", "''")
    ]:
        if new_col not in existing_cols:
            cursor.execute(f"ALTER TABLE consultations ADD COLUMN {new_col} {col_type} DEFAULT {default_val}")

    # 3. Regional Outbreak Alerts Table (Broadcasted by Agronomists for Farmers)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS outbreak_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agronomist_name TEXT NOT NULL,
            region TEXT NOT NULL,
            crop TEXT NOT NULL,
            disease TEXT NOT NULL,
            severity TEXT NOT NULL,
            alert_message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Seed demo users if not present
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, ?)",
            ("admin", hash_password("admin123"), "Dr. Dhanashri (Chief Agronomist)", "dhanashri@plantmamba.ai", "Chief Agronomist")
        )
    cursor.execute("SELECT id FROM users WHERE username = 'farmer'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, ?)",
            ("farmer", hash_password("farmer123"), "Ramesh Patil (Progressive Farmer)", "ramesh@kisan.in", "Farmer")
        )
    cursor.execute("SELECT id FROM users WHERE username = 'student'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, ?)",
            ("student", hash_password("student123"), "Aarav Deshmukh (Agri Research Scholar)", "aarav@agriuni.edu", "Student / Researcher")
        )

    # Seed sample outbreak alert if table empty
    cursor.execute("SELECT COUNT(*) FROM outbreak_alerts")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO outbreak_alerts (agronomist_name, region, crop, disease, severity, alert_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "Dr. Dhanashri (Chief Agronomist)",
            "Nagpur & Vidarbha",
            "Tomato / Chilli",
            "Late Blight & Powdery Mildew",
            "High Risk",
            "Continuous overcast weather and 85%+ relative humidity detected. Farmers are advised to apply preventive Mancozeb 75 WP (2g/L) before foliar sporulation spreads."
        ))
    conn.commit()
    conn.close()


def register_user(username, password, full_name, email, role="Farmer") -> tuple[bool, str]:
    """Registers a new user account."""
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, ?)",
            (username, hash_password(password), full_name.strip(), email.strip(), role)
        )
        conn.commit()
        return True, "Registration successful! You can now log in."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' is already registered. Please choose another."
    finally:
        conn.close()


def authenticate_user(username, password) -> dict | None:
    """Authenticates user credentials and returns user dictionary."""
    username = username.strip().lower()
    pw_hash = hash_password(password)

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, full_name, email, role, created_at FROM users WHERE username = ? AND password_hash = ?",
        (username, pw_hash)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def save_consultation(user_id, diagnosis_result, weather_info, risk_info, similar_cases) -> str:
    """
    Saves a completed diagnosis consultation into the database and persists image files locally.
    Returns the generated report_id.
    """
    now = datetime.now()
    report_id = f"PMR-{now.strftime('%y%m%d%H%M%S')}-{int(now.microsecond / 1000):03d}"

    # Save images to history_storage
    orig_img_path = os.path.join(STORAGE_DIR, f"{report_id}_orig.jpg")
    cam_img_path = os.path.join(STORAGE_DIR, f"{report_id}_cam.jpg")

    try:
        diagnosis_result["original_image"].save(orig_img_path, "JPEG")
        diagnosis_result["cam_image"].save(cam_img_path, "JPEG")
    except Exception as e:
        print(f"[DB] Error saving images for report {report_id}: {e}")

    clean_name = diagnosis_result.get("clean_name", diagnosis_result["predicted_class"])
    crop_name = clean_name.split()[0] if " " in clean_name else clean_name

    treatment_json = json.dumps(diagnosis_result.get("treatment", {}))
    similar_cases_json = json.dumps(similar_cases if similar_cases else [])

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO consultations (
            user_id, report_id, crop_name, disease_name, clean_name, confidence,
            severity_level, infected_ratio, city, weather_temp, weather_humidity,
            risk_level, risk_explanation, orig_img_path, cam_img_path, treatment_json, similar_cases_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        report_id,
        crop_name,
        diagnosis_result.get("predicted_class", ""),
        clean_name,
        float(diagnosis_result.get("confidence", 0.0)),
        diagnosis_result.get("severity_level", "Unknown"),
        float(diagnosis_result.get("infected_ratio", 0.0)),
        weather_info.get("city", "Nagpur"),
        float(weather_info.get("temperature", 25.0)),
        float(weather_info.get("humidity", 80.0)),
        risk_info.get("risk_level", "Moderate Risk"),
        risk_info.get("explanation", ""),
        orig_img_path,
        cam_img_path,
        treatment_json,
        similar_cases_json
    ))
    conn.commit()
    conn.close()
    return report_id


def get_user_consultations(user_id=None, limit=50) -> list[dict]:
    """Retrieves past consultations for a user (or all if user_id is None)."""
    conn = _get_connection()
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute(
            "SELECT * FROM consultations WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        )
    else:
        cursor.execute(
            "SELECT * FROM consultations ORDER BY id DESC LIMIT ?",
            (limit,)
        )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d["treatment"] = json.loads(d["treatment_json"]) if d.get("treatment_json") else {}
        d["similar_cases"] = json.loads(d["similar_cases_json"]) if d.get("similar_cases_json") else []
        results.append(d)
    return results


def get_consultation_by_id(consultation_id) -> dict | None:
    """Retrieves a single consultation record by database ID."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM consultations WHERE id = ?", (consultation_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["treatment"] = json.loads(d["treatment_json"]) if d.get("treatment_json") else {}
        d["similar_cases"] = json.loads(d["similar_cases_json"]) if d.get("similar_cases_json") else []
        return d
    return None


def delete_consultation(consultation_id, user_id=None) -> bool:
    """Deletes a consultation and cleans up stored image files."""
    conn = _get_connection()
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute("SELECT orig_img_path, cam_img_path FROM consultations WHERE id = ? AND user_id = ?", (consultation_id, user_id))
    else:
        cursor.execute("SELECT orig_img_path, cam_img_path FROM consultations WHERE id = ?", (consultation_id,))
    row = cursor.fetchone()
    if row:
        for p in [row["orig_img_path"], row["cam_img_path"]]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        if user_id is not None:
            cursor.execute("DELETE FROM consultations WHERE id = ? AND user_id = ?", (consultation_id, user_id))
        else:
            cursor.execute("DELETE FROM consultations WHERE id = ?", (consultation_id,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


def get_consultation_stats(user_id=None) -> dict:
    """Returns analytics statistics for user consultations."""
    conn = _get_connection()
    cursor = conn.cursor()

    query_filter = "WHERE user_id = ?" if user_id is not None else ""
    params = (user_id,) if user_id is not None else ()

    cursor.execute(f"SELECT COUNT(*) as total_scans FROM consultations {query_filter}", params)
    total_scans = cursor.fetchone()["total_scans"]

    cursor.execute(f"""
        SELECT crop_name, COUNT(*) as count 
        FROM consultations {query_filter} 
        GROUP BY crop_name 
        ORDER BY count DESC LIMIT 5
    """, params)
    top_crops = [dict(r) for r in cursor.fetchall()]

    cursor.execute(f"""
        SELECT severity_level, COUNT(*) as count 
        FROM consultations {query_filter} 
        GROUP BY severity_level
    """, params)
    severity_breakdown = {r["severity_level"]: r["count"] for r in cursor.fetchall()}

    cursor.execute(f"""
        SELECT AVG(confidence) as avg_conf 
        FROM consultations {query_filter}
    """, params)
    avg_conf = cursor.fetchone()["avg_conf"] or 0.0

    conn.close()
    return {
        "total_scans": total_scans,
        "top_crops": top_crops,
        "severity_breakdown": severity_breakdown,
        "avg_confidence": round(float(avg_conf), 1)
    }


def certify_consultation(consultation_id, agronomist_name, confirmed_disease=None, severity_level=None, expert_notes="") -> bool:
    """Allows an Agronomist to review, confirm/override AI diagnosis, and sign off."""
    conn = _get_connection()
    cursor = conn.cursor()
    update_fields = ["review_status = 'Agronomist Certified'", "certified_by = ?", "expert_notes = ?"]
    params = [agronomist_name, expert_notes.strip()]

    if confirmed_disease:
        update_fields.append("confirmed_disease = ?")
        params.append(confirmed_disease)
        # also update clean_name and crop_name if disease overridden
        crop = confirmed_disease.split()[0] if " " in confirmed_disease else confirmed_disease
        update_fields.append("clean_name = ?")
        params.append(confirmed_disease)
        update_fields.append("crop_name = ?")
        params.append(crop)

    if severity_level:
        update_fields.append("severity_level = ?")
        params.append(severity_level)

    params.append(consultation_id)
    sql = f"UPDATE consultations SET {', '.join(update_fields)} WHERE id = ?"
    cursor.execute(sql, tuple(params))
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def get_all_consultations(limit=100, filter_status=None) -> list[dict]:
    """Retrieves consultations across all users for Agronomist review and auditing."""
    conn = _get_connection()
    cursor = conn.cursor()
    if filter_status:
        cursor.execute(
            "SELECT c.*, u.full_name as submitter_name FROM consultations c LEFT JOIN users u ON c.user_id = u.id WHERE c.review_status = ? ORDER BY c.id DESC LIMIT ?",
            (filter_status, limit)
        )
    else:
        cursor.execute(
            "SELECT c.*, u.full_name as submitter_name FROM consultations c LEFT JOIN users u ON c.user_id = u.id ORDER BY c.id DESC LIMIT ?",
            (limit,)
        )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d["treatment"] = json.loads(d["treatment_json"]) if d.get("treatment_json") else {}
        d["similar_cases"] = json.loads(d["similar_cases_json"]) if d.get("similar_cases_json") else []
        results.append(d)
    return results


def add_outbreak_alert(agronomist_name, region, crop, disease, severity, alert_message) -> int:
    """Allows an Agronomist to broadcast a regional disease epidemic alert to farmers."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO outbreak_alerts (agronomist_name, region, crop, disease, severity, alert_message)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (agronomist_name, region, crop, disease, severity, alert_message))
    conn.commit()
    alert_id = cursor.lastrowid
    conn.close()
    return alert_id


def get_outbreak_alerts(limit=20) -> list[dict]:
    """Fetches active regional epidemic alerts broadcasted by agronomists."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM outbreak_alerts ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_outbreak_alert(alert_id) -> bool:
    """Deletes an outbreak alert."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM outbreak_alerts WHERE id = ?", (alert_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# Auto-initialize database on module import
init_db()

