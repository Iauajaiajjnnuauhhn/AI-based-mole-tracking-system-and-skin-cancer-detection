"""
db.py — shared SQLite layer for DermaScan AI
Tables: patients, reports, report_images
"""

import sqlite3
import json
import datetime
import random
import string
import os
import cv2
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dermascan.db")


# ── Connection ────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    with get_conn() as conn:
        conn.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS patients (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id    TEXT    NOT NULL UNIQUE,   -- e.g. DS-4F2A
            name          TEXT    NOT NULL,
            email         TEXT,
            dob           TEXT,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reports (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id    TEXT    NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
            timestamp     TEXT    NOT NULL,
            risk_level    TEXT    NOT NULL,
            risk_score    REAL    NOT NULL,
            tds_baseline  REAL    NOT NULL,
            tds_current   REAL    NOT NULL,
            delta_tds     REAL    NOT NULL,
            asymmetry_b   REAL, border_b   REAL, color_b   REAL, diameter_b   REAL,
            asymmetry_c   REAL, border_c   REAL, color_c   REAL, diameter_c   REAL,
            color_flags_b TEXT,
            color_flags_c TEXT,
            similarity    REAL,
            confidence    REAL,
            change_summary TEXT,
            recommendation TEXT,
            flags_json    TEXT,
            ok_flags_json TEXT
        );

        CREATE TABLE IF NOT EXISTS report_images (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id       INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
            img_baseline    BLOB,
            img_current     BLOB,
            img_seg_base    BLOB,
            img_seg_curr    BLOB
        );
        """)


# ── Patient helpers ───────────────────────────────────────────────────────────

def _gen_patient_id() -> str:
    """Generate a unique 4-char alphanumeric ID like DS-4F2A."""
    chars = string.ascii_uppercase + string.digits
    while True:
        code = "DS-" + "".join(random.choices(chars, k=4))
        with get_conn() as conn:
            row = conn.execute("SELECT 1 FROM patients WHERE patient_id=?", (code,)).fetchone()
        if not row:
            return code


def _hash(password: str) -> str:
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()


def register_patient(name: str, password: str, email: str = "", dob: str = "") -> str:
    """Create a new patient. Returns the assigned patient_id."""
    pid = _gen_patient_id()
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO patients (patient_id,name,email,dob,password_hash,created_at) VALUES (?,?,?,?,?,?)",
            (pid, name.strip(), email.strip(), dob, _hash(password), ts)
        )
    return pid


def login_patient(patient_id: str, password: str):
    """Returns patient Row if credentials match, else None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id=? AND password_hash=?",
            (patient_id.strip().upper(), _hash(password))
        ).fetchone()
    return row


def get_patient(patient_id: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM patients WHERE patient_id=?", (patient_id,)
        ).fetchone()


# ── Image helpers ─────────────────────────────────────────────────────────────

def bgr_to_jpeg(bgr: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def rgb_to_jpeg(rgb: np.ndarray, quality: int = 82) -> bytes:
    return bgr_to_jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), quality)


def jpeg_to_rgb(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Report helpers ────────────────────────────────────────────────────────────

def save_report(patient_id: str, data: dict, bgr1: np.ndarray, bgr2: np.ndarray,
                overlay_fn) -> int:
    """Persist a full analysis + images. Returns new report id."""
    ab  = data["abcd_baseline"]
    ac  = data["abcd_current"]
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seg_base = overlay_fn(bgr1, ab["mask"])
    seg_curr = overlay_fn(bgr2, ac["mask"])

    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO reports (
                patient_id, timestamp, risk_level, risk_score,
                tds_baseline, tds_current, delta_tds,
                asymmetry_b, border_b, color_b, diameter_b,
                asymmetry_c, border_c, color_c, diameter_c,
                color_flags_b, color_flags_c,
                similarity, confidence,
                change_summary, recommendation,
                flags_json, ok_flags_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            patient_id, ts, data["risk_level"], data["risk_score"],
            ab["tds"], ac["tds"], data["delta_tds"],
            ab["asymmetry"], ab["border"], ab["color"], ab["diameter"],
            ac["asymmetry"], ac["border"], ac["color"], ac["diameter"],
            json.dumps(ab["color_flags"]), json.dumps(ac["color_flags"]),
            data["similarity_index"], data["confidence"],
            data["change_summary"], data["recommendation"],
            json.dumps(data["flags"]), json.dumps(data["ok_flags"]),
        ))
        rid = cur.lastrowid
        conn.execute("""
            INSERT INTO report_images (report_id,img_baseline,img_current,img_seg_base,img_seg_curr)
            VALUES (?,?,?,?,?)
        """, (rid, bgr_to_jpeg(bgr1), bgr_to_jpeg(bgr2),
              rgb_to_jpeg(seg_base), rgb_to_jpeg(seg_curr)))
    return rid


def load_patient_reports(patient_id: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM reports WHERE patient_id=? ORDER BY id ASC", (patient_id,)
        ).fetchall()


def load_report_images(report_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM report_images WHERE report_id=?", (report_id,)
        ).fetchone()


def delete_report(report_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM reports WHERE id=?", (report_id,))


def delete_all_patient_reports(patient_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM reports WHERE patient_id=?", (patient_id,))


# Run on import
init_db()
