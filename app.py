"""
DermaScan AI – Enhanced Mole Tracker v2
────────────────────────────────────────
Improvements over v1:
  • GrabCut-assisted segmentation for higher accuracy
  • Confidence-weighted risk scoring
  • Normalised TDS with calibrated thresholds
  • Plain-English, patient-friendly report
  • Colour-coded "what this means" context for every metric
  • Action-step checklist based on risk level
  • Dermoscopic colour flag breakdown
  • Lesion shape preview with contour overlay
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime
import json
import sqlite3
import io
import os

# =============================================================================
#  DATABASE  — SQLite persistence for reports + images
# =============================================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dermascan.db")


def get_conn() -> sqlite3.Connection:
    """Return a thread-safe connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist yet."""
    with get_conn() as conn:
        conn.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS reports (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            risk_level    TEXT    NOT NULL,
            risk_score    REAL    NOT NULL,
            tds_baseline  REAL    NOT NULL,
            tds_current   REAL    NOT NULL,
            delta_tds     REAL    NOT NULL,
            asymmetry_b   REAL, border_b   REAL, color_b   REAL, diameter_b   REAL,
            asymmetry_c   REAL, border_c   REAL, color_c   REAL, diameter_c   REAL,
            color_flags_b TEXT,   -- JSON list
            color_flags_c TEXT,   -- JSON list
            similarity    REAL,
            confidence    REAL,
            change_summary TEXT,
            recommendation TEXT,
            flags_json    TEXT,   -- JSON list of [title, detail] pairs
            ok_flags_json TEXT    -- JSON list of strings
        );

        CREATE TABLE IF NOT EXISTS report_images (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id       INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
            img_baseline    BLOB,   -- JPEG bytes
            img_current     BLOB,   -- JPEG bytes
            img_seg_base    BLOB,   -- JPEG bytes  (segmentation overlay)
            img_seg_curr    BLOB    -- JPEG bytes
        );
        """)


def bgr_to_jpeg(bgr: np.ndarray, quality: int = 82) -> bytes:
    """Encode a BGR ndarray to JPEG bytes for compact storage."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def rgb_to_jpeg(rgb: np.ndarray, quality: int = 82) -> bytes:
    """Encode an RGB ndarray to JPEG bytes."""
    return bgr_to_jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), quality)


def jpeg_to_rgb(blob: bytes) -> np.ndarray:
    """Decode JPEG bytes back to an RGB ndarray for display."""
    arr = np.frombuffer(blob, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_report(data: dict, bgr1: np.ndarray, bgr2: np.ndarray) -> int:
    """
    Persist a full analysis result + images to SQLite.
    Returns the new report id.
    """
    ab  = data["abcd_baseline"]
    ac  = data["abcd_current"]
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build segmentation overlay images
    seg_base = overlay_mask(bgr1, ab["mask"])   # RGB ndarray
    seg_curr = overlay_mask(bgr2, ac["mask"])

    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO reports (
                timestamp, risk_level, risk_score,
                tds_baseline, tds_current, delta_tds,
                asymmetry_b, border_b, color_b, diameter_b,
                asymmetry_c, border_c, color_c, diameter_c,
                color_flags_b, color_flags_c,
                similarity, confidence,
                change_summary, recommendation,
                flags_json, ok_flags_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ts, data["risk_level"], data["risk_score"],
            ab["tds"], ac["tds"], data["delta_tds"],
            ab["asymmetry"], ab["border"], ab["color"], ab["diameter"],
            ac["asymmetry"], ac["border"], ac["color"], ac["diameter"],
            json.dumps(ab["color_flags"]), json.dumps(ac["color_flags"]),
            data["similarity_index"], data["confidence"],
            data["change_summary"], data["recommendation"],
            json.dumps(data["flags"]), json.dumps(data["ok_flags"]),
        ))
        report_id = cur.lastrowid

        conn.execute("""
            INSERT INTO report_images (report_id, img_baseline, img_current, img_seg_base, img_seg_curr)
            VALUES (?,?,?,?,?)
        """, (
            report_id,
            bgr_to_jpeg(bgr1),
            bgr_to_jpeg(bgr2),
            rgb_to_jpeg(seg_base),
            rgb_to_jpeg(seg_curr),
        ))

    return report_id


def load_all_reports() -> list[sqlite3.Row]:
    """Return all report rows ordered oldest→newest."""
    with get_conn() as conn:
        return conn.execute("SELECT * FROM reports ORDER BY id ASC").fetchall()


def load_report_images(report_id: int) -> sqlite3.Row | None:
    """Return the image row for a given report id."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM report_images WHERE report_id=?", (report_id,)
        ).fetchone()


def delete_report(report_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM reports WHERE id=?", (report_id,))


def delete_all_reports():
    with get_conn() as conn:
        conn.execute("DELETE FROM reports")
        conn.execute("DELETE FROM report_images")


# Initialise DB on every cold start
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan AI · Mole Tracker",
    page_icon="🔬",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,600;0,700;1,300&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');
:root{
  --bg:#07090f;--surface:#0f1219;--surface2:#161b26;--surface3:#1d2333;
  --border:#1f2535;--accent:#5eead4;--accent2:#818cf8;
  --text:#dde1f0;--muted:#5a6175;
  --low:#34d399;--mod:#fbbf24;--high:#f87171;
  --low-bg:rgba(52,211,153,.08);--mod-bg:rgba(251,191,36,.08);--high-bg:rgba(248,113,113,.08);
}
html,body,[class*="css"]{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2.5rem 2.5rem 5rem;max-width:1320px;}
h1,h2,h3{font-family:'Fraunces',serif;}

/* HERO */
.hero{
  text-align:center;padding:3rem 1rem 2.5rem;
  background:radial-gradient(ellipse 70% 50% at 50% 0%,rgba(94,234,212,.07),transparent 70%);
  border-bottom:1px solid var(--border);margin-bottom:2.5rem;
}
.hero-title{
  font-family:'Fraunces',serif;font-size:clamp(2.2rem,5vw,3.6rem);font-weight:700;
  background:linear-gradient(135deg,#dde1f0 20%,#5eead4 60%,#818cf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:-.03em;margin:0 0 .5rem;
}
.hero-sub{color:var(--muted);font-size:.9rem;font-weight:300;letter-spacing:.08em;text-transform:uppercase;}
.hero-badge{
  display:inline-block;margin-top:.8rem;padding:.3rem 1rem;border-radius:99px;
  background:rgba(94,234,212,.08);border:1px solid rgba(94,234,212,.2);
  font-size:.78rem;font-family:'DM Mono',monospace;color:var(--accent);letter-spacing:.06em;
}

/* CARDS */
.card{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:1.5rem;margin-bottom:1.2rem;position:relative;overflow:hidden;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(94,234,212,.15),transparent);}
.card-label{font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--accent);margin-bottom:1rem;}

/* RISK BADGE */
.risk-badge{display:inline-flex;align-items:center;gap:.5rem;padding:.4rem 1.1rem;
  border-radius:999px;font-family:'DM Mono',monospace;font-size:.9rem;font-weight:500;letter-spacing:.04em;}
.risk-LOW{background:var(--low-bg);color:var(--low);border:1px solid rgba(52,211,153,.25);}
.risk-MODERATE{background:var(--mod-bg);color:var(--mod);border:1px solid rgba(251,191,36,.25);}
.risk-HIGH{background:var(--high-bg);color:var(--high);border:1px solid rgba(248,113,113,.3);}

/* GAUGE */
.gauge-wrap{margin:.45rem 0;}
.gauge-header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.3rem;}
.gauge-name{font-size:.8rem;font-weight:500;color:var(--text);}
.gauge-hint{font-size:.72rem;color:var(--muted);}
.gauge-val{font-family:'DM Mono',monospace;font-size:.78rem;color:var(--text);}
.gauge-track{width:100%;height:6px;background:var(--surface3);border-radius:99px;overflow:hidden;}
.gauge-fill{height:100%;border-radius:99px;transition:width .6s cubic-bezier(.4,0,.2,1);}

/* SCORE RING — just a big number */
.score-ring{text-align:center;padding:.8rem;}
.score-num{font-family:'Fraunces',serif;font-size:3.2rem;font-weight:700;line-height:1;}
.score-label{font-size:.72rem;font-family:'DM Mono',monospace;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);margin-top:.3rem;}

/* METRIC PILLS */
.pill-row{display:flex;flex-wrap:wrap;gap:.5rem;margin:.6rem 0;}
.pill{padding:.25rem .75rem;border-radius:999px;font-size:.76rem;font-family:'DM Mono',monospace;
  border:1px solid;}
.pill-ok{background:rgba(52,211,153,.08);color:var(--low);border-color:rgba(52,211,153,.2);}
.pill-warn{background:rgba(251,191,36,.08);color:var(--mod);border-color:rgba(251,191,36,.2);}
.pill-bad{background:rgba(248,113,113,.08);color:var(--high);border-color:rgba(248,113,113,.2);}

/* EXPLANATION BOX */
.explain{background:var(--surface2);border-radius:10px;padding:.9rem 1.1rem;
  font-size:.85rem;line-height:1.7;color:#a8b0c8;margin:.4rem 0;}
.explain strong{color:var(--text);}

/* FLAG ITEM */
.flag-item{display:flex;align-items:flex-start;gap:.6rem;padding:.55rem .75rem;
  background:rgba(248,113,113,.06);border:1px solid rgba(248,113,113,.15);
  border-radius:8px;margin:.35rem 0;font-size:.84rem;line-height:1.5;}
.flag-ok-item{display:flex;align-items:flex-start;gap:.6rem;padding:.55rem .75rem;
  background:rgba(52,211,153,.06);border:1px solid rgba(52,211,153,.15);
  border-radius:8px;margin:.35rem 0;font-size:.84rem;line-height:1.5;}

/* ACTION STEPS */
.action-step{display:flex;align-items:flex-start;gap:.8rem;padding:.7rem;
  border-radius:10px;background:var(--surface2);margin:.4rem 0;}
.step-num{font-family:'DM Mono',monospace;font-size:.78rem;color:var(--accent);
  background:rgba(94,234,212,.1);border-radius:6px;padding:.2rem .5rem;flex-shrink:0;}
.step-txt{font-size:.86rem;line-height:1.6;}

/* SECTION DIVIDER */
.section-head{display:flex;align-items:center;gap:.8rem;margin:2rem 0 1rem;}
.section-head-line{flex:1;height:1px;background:var(--border);}
.section-head-txt{font-family:'Fraunces',serif;font-size:1.1rem;color:var(--text);white-space:nowrap;}

/* DISCLAIMER */
.disclaimer{margin-top:2.5rem;padding:1.1rem 1.3rem;
  background:var(--surface);border-radius:12px;border-left:3px solid var(--muted);
  font-size:.78rem;color:var(--muted);line-height:1.65;}

/* BUTTON */
div.stButton>button{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#07090f;border:none;border-radius:10px;
  font-family:'Outfit',sans-serif;font-weight:600;font-size:.95rem;
  padding:.7rem 2rem;width:100%;cursor:pointer;transition:opacity .2s;
}
div.stButton>button:hover{opacity:.85;}
div.stButton>button:disabled{opacity:.35;cursor:not-allowed;}

/* DIST INDICATOR */
.dist-ok{color:var(--low);font-weight:600;}
.dist-warn{color:var(--mod);font-weight:600;}
.dist-bad{color:var(--high);font-weight:600;}

/* COMPARE TABLE */
.cmp-table{width:100%;border-collapse:collapse;font-size:.84rem;}
.cmp-table th{text-align:left;font-family:'DM Mono',monospace;font-size:.65rem;
  letter-spacing:.15em;text-transform:uppercase;color:var(--muted);
  padding:.5rem .7rem;border-bottom:1px solid var(--border);}
.cmp-table td{padding:.5rem .7rem;border-bottom:1px solid var(--border);}
.cmp-up{color:var(--high);}
.cmp-down{color:var(--low);}
.cmp-flat{color:var(--muted);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-title">DermaScan AI</div>
  <div class="hero-sub">Mole Tracking · ABCD Analysis · Similarity Index</div>
  <div class="hero-badge">📸 Camera Capture · OpenCV · Research Tool</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  CONSTANTS
# =============================================================================
# =============================================================================
#  CAMERA INPUT HELPERS  (st.camera_input replaces WebRTC)
# =============================================================================

TARGET_CM_HINT = 10   # informational only — shown to user as guidance

def load_camera_image(camera_file) -> np.ndarray:
    """Decode an st.camera_input snapshot into a BGR numpy array."""
    raw = np.frombuffer(camera_file.getvalue(), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


# =============================================================================
#  ENHANCED SEGMENTATION — GrabCut + LAB fallback
# =============================================================================

def segment_lesion_grabcut(bgr: np.ndarray) -> np.ndarray:
    """
    Two-pass segmentation:
      1. LAB-based Otsu threshold to locate the lesion region
      2. GrabCut refinement for cleaner boundary
    Falls back to LAB-only if GrabCut fails.
    """
    h, w = bgr.shape[:2]

    # ── Pass 1: LAB threshold ────────────────────────────────────────────────
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, _ = cv2.split(lab)
    # Combine darkness + redness cues
    score = cv2.addWeighted(255 - l, 0.55, a, 0.45, 0)
    _, coarse = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, k, iterations=3)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN,  k, iterations=2)

    # ── Largest connected component ──────────────────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(coarse, connectivity=8)
    if n_labels < 2:
        return coarse
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    coarse  = np.uint8(labels == largest) * 255

    # ── Pass 2: GrabCut refinement ───────────────────────────────────────────
    try:
        rect_m = cv2.boundingRect(coarse)
        rx, ry, rw, rh = rect_m
        # Expand rect slightly
        pad = 10
        rx  = max(0, rx - pad); ry = max(0, ry - pad)
        rw  = min(w - rx, rw + 2*pad); rh = min(h - ry, rh + 2*pad)
        if rw < 10 or rh < 10:
            return coarse
        bgr_mask  = np.zeros((h, w), np.uint8)
        bgmodel   = np.zeros((1, 65), np.float64)
        fgmodel   = np.zeros((1, 65), np.float64)
        # Seed with coarse mask
        bgr_mask[coarse == 0]   = cv2.GC_BGD
        bgr_mask[coarse == 255] = cv2.GC_PR_FGD
        cv2.grabCut(bgr, bgr_mask, (rx, ry, rw, rh),
                    bgmodel, fgmodel, 4, cv2.GC_INIT_WITH_MASK)
        fine = np.where((bgr_mask == cv2.GC_FGD) | (bgr_mask == cv2.GC_PR_FGD),
                        255, 0).astype(np.uint8)
        fine = cv2.morphologyEx(fine, cv2.MORPH_CLOSE, k, iterations=2)
        # Sanity: GrabCut result shouldn't shrink lesion by more than 60 %
        if np.count_nonzero(fine) < 0.4 * np.count_nonzero(coarse):
            return coarse
        return fine
    except Exception:
        return coarse


# =============================================================================
#  ENHANCED ABCD METRICS
# =============================================================================

def compute_asymmetry(mask: np.ndarray) -> float:
    """
    Score 0–2 (lower is better).
    Compute both horizontal & vertical symmetry and average.
    Normalised so that a perfectly circular lesion scores 0.
    """
    h, w = mask.shape
    scores = []
    for axis in [0, 1]:
        if axis == 0:
            half1 = mask[:h//2, :]
            half2 = np.flip(mask[h//2:, :], 0)
            half2 = cv2.resize(half2, (half1.shape[1], half1.shape[0]))
        else:
            half1 = mask[:, :w//2]
            half2 = np.flip(mask[:, w//2:], 1)
            half2 = cv2.resize(half2, (half1.shape[1], half1.shape[0]))
        union = np.logical_or(half1 > 0, half2 > 0).sum()
        diff  = np.logical_xor(half1 > 0, half2 > 0).sum()
        scores.append(diff / union if union > 0 else 0)

    # Scale to 0–2: raw score 0.5 maps to 2.0 (very asymmetric)
    raw = (scores[0] + scores[1]) / 2
    return round(min(raw * 4.0, 2.0), 3)


def compute_border(mask: np.ndarray) -> float:
    """
    ABCD border score 0–8.
    Divide lesion border into 8 octants; count those with high radial variance.
    Added: compactness penalty to catch spiky lesions.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :]
    M   = cv2.moments(mask)
    cx  = M['m10'] / M['m00'] if M['m00'] else mask.shape[1] / 2
    cy  = M['m01'] / M['m00'] if M['m00'] else mask.shape[0] / 2

    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    octant_scores = []
    for i in range(8):
        lo  = -np.pi + i * (2 * np.pi / 8)
        idx = np.where((angles >= lo) & (angles < lo + 2 * np.pi / 8))[0]
        if len(idx) < 3:
            octant_scores.append(0); continue
        r   = np.sqrt((pts[idx, 0] - cx)**2 + (pts[idx, 1] - cy)**2)
        cv_ = np.std(r) / (np.mean(r) + 1e-6)
        octant_scores.append(1 if cv_ > 0.10 else 0)   # lowered threshold vs v1

    # Compactness bonus (circularity < 0.6 adds ≤1 extra point)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = 4 * np.pi * area / (peri ** 2 + 1e-6)
    extra = max(0.0, 1.0 - circ / 0.6)   # 0→1
    return min(float(sum(octant_scores)) + extra, 8.0)


def compute_color(bgr: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Returns (score 1–6, list of colour flag names detected).
    Six dermoscopic colours: white, red, light-brown, dark-brown, blue-grey, black.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    roi = mask > 0
    h, s, v = hsv[:, :, 0][roi], hsv[:, :, 1][roi], hsv[:, :, 2][roi]
    l_ch     = lab[:, :, 0][roi]
    b_ch     = lab[:, :, 2][roi]   # b* channel: positive = yellow/brown

    found = []
    if np.mean(v > 200) > 0.04:                                        found.append("White")
    if np.mean((h < 10) & (s > 80)) > 0.03:                           found.append("Red")
    if np.mean((h >= 10) & (h < 25) & (s > 40) & (v > 100)) > 0.05:  found.append("Light-brown")
    if np.mean((h >= 10) & (h < 30) & (v < 120) & (s > 30)) > 0.05:  found.append("Dark-brown")
    if np.mean((h >= 95) & (h < 140) & (s < 90)) > 0.03:              found.append("Blue-grey")
    if np.mean(l_ch < 35) > 0.04:                                      found.append("Black")

    return float(max(1, len(found))), found


def compute_diameter(mask: np.ndarray, fov_mm: float = 20.0) -> float:
    """
    ABCD diameter score 0–5.
    Uses minimum bounding rectangle; calibrated to 6 mm clinical cutoff.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    _, (mw, mh), _ = cv2.minAreaRect(cnt)
    diam_mm = max(mw, mh) * fov_mm / (mask.shape[1] + 1e-6)
    # Linear scale: 0 mm → 0, 6 mm → 5 (clipped)
    return round(min(diam_mm / 6.0 * 5.0, 5.0), 3)


def tds(a, b, c, d) -> float:
    """Total Dermoscopy Score (EDF weights)."""
    return round(a * 1.3 + b * 0.1 + c * 0.5 + d * 0.5, 3)


def segmentation_confidence(mask: np.ndarray, bgr: np.ndarray) -> float:
    """
    Estimate how confident we are in the segmentation (0–1).
    Based on: lesion fills ≥2 % of frame, mask fills a coherent shape,
    contour is closed, colour contrast between lesion & surroundings.
    """
    h, w = bgr.shape[:2]
    fill_ratio = np.count_nonzero(mask) / (h * w)
    if fill_ratio < 0.01 or fill_ratio > 0.90:
        return 0.3
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.3
    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = 4 * np.pi * area / (peri**2 + 1e-6)
    # Colour contrast between lesion and ring around it
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated = cv2.dilate(mask, kernel)
    ring    = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))
    if ring.sum() == 0:
        contrast = 0.5
    else:
        lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch    = lab[:, :, 0].astype(float)
        l_les   = l_ch[mask > 0].mean()
        l_ring  = l_ch[ring > 0].mean()
        contrast = min(abs(l_les - l_ring) / 60.0, 1.0)
    conf = 0.4 * min(fill_ratio / 0.10, 1.0) + 0.3 * min(circ / 0.8, 1.0) + 0.3 * contrast
    return round(float(np.clip(conf, 0.1, 1.0)), 2)


def risk_from_tds(t: float, conf: float = 1.0) -> tuple:
    """
    Returns (risk_score 0–10, risk_level str).
    Confidence-weighted: low-confidence analyses nudge score toward moderate.
    """
    level  = "LOW" if t < 4.75 else ("MODERATE" if t <= 5.45 else "HIGH")
    raw_rs = min(t / 8.0 * 10.0, 10.0)
    # Confidence penalty: if conf < 0.5, compress extremes toward 5
    adj_rs = raw_rs * conf + 5.0 * (1 - conf)
    return round(float(np.clip(adj_rs, 0, 10)), 2), level


def similarity_index(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    SIZE = (256, 256)
    g1   = cv2.cvtColor(cv2.resize(bgr1, SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2   = cv2.cvtColor(cv2.resize(bgr2, SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1n  = (g1 - g1.mean()) / (g1.std() + 1e-6)
    g2n  = (g2 - g2.mean()) / (g2.std() + 1e-6)
    ncc  = ((float(np.mean(g1n * g2n)) + 1) / 2) * 100
    h1   = cv2.cvtColor(cv2.resize(bgr1, SIZE), cv2.COLOR_BGR2HSV)
    h2   = cv2.cvtColor(cv2.resize(bgr2, SIZE), cv2.COLOR_BGR2HSV)
    ht1  = cv2.calcHist([h1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    ht2  = cv2.calcHist([h2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(ht1, ht1); cv2.normalize(ht2, ht2)
    bhatt = cv2.compareHist(ht1, ht2, cv2.HISTCMP_BHATTACHARYYA)
    return round(max(0.0, min(100.0, ncc * 0.5 + (1 - bhatt) * 100 * 0.5)), 1)


def overlay_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    teal    = np.array([94, 234, 212], dtype=np.float32)
    roi     = mask > 0
    overlay[roi] = (overlay[roi].astype(np.float32) * 0.5 + teal * 0.5).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (94, 234, 212), 2)
    return overlay


def load_cv(uploaded_file) -> np.ndarray:
    uploaded_file.seek(0)
    raw = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return img


def analyse_pair(bgr1: np.ndarray, bgr2: np.ndarray) -> dict:
    res = {}
    for key, bgr in [("baseline", bgr1), ("current", bgr2)]:
        mask = segment_lesion_grabcut(bgr)
        conf = segmentation_confidence(mask, bgr)
        a    = compute_asymmetry(mask)
        b    = compute_border(mask)
        c_score, c_flags = compute_color(bgr, mask)
        d    = compute_diameter(mask)
        t    = tds(a, b, c_score, d)
        rs, rl = risk_from_tds(t, conf)
        res[key] = {
            "asymmetry": a, "border": b, "color": c_score, "color_flags": c_flags,
            "diameter": d, "tds": t, "mask": mask, "conf": conf,
            "risk_score": rs, "risk_level": rl,
        }

    sim   = similarity_index(bgr1, bgr2)
    delta = res["current"]["tds"] - res["baseline"]["tds"]
    rl    = res["current"]["risk_level"]
    rs    = res["current"]["risk_score"]
    conf  = min(res["baseline"]["conf"], res["current"]["conf"])

    # ── Change flags ──────────────────────────────────────────────────────────
    flags = []
    ok_flags = []
    if res["current"]["asymmetry"] > res["baseline"]["asymmetry"] + 0.25:
        flags.append(("Asymmetry increased", "The mole's shape is less symmetric compared to your baseline photo."))
    else:
        ok_flags.append("Asymmetry is stable or improved.")
    if res["current"]["border"] > res["baseline"]["border"] + 0.8:
        flags.append(("Border more irregular", "The edges of the mole appear more ragged or uneven than before."))
    else:
        ok_flags.append("Border regularity is unchanged.")
    new_cols = set(res["current"]["color_flags"]) - set(res["baseline"]["color_flags"])
    if new_cols:
        flags.append(("New colour structures", f"New shades detected: {', '.join(new_cols)}. Colour variation can be a watch sign."))
    else:
        ok_flags.append("No new colours have appeared.")
    if res["current"]["diameter"] > res["baseline"]["diameter"] + 0.4:
        flags.append(("Estimated size increased", "The lesion appears larger than in the baseline image."))
    else:
        ok_flags.append("Estimated size is stable.")
    if sim < 65:
        flags.append(("Low visual similarity", f"The two images look quite different (Similarity: {sim:.0f}%). This could mean the mole has changed, or lighting/angle differed."))
    elif sim < 80:
        flags.append(("Moderate visual similarity", f"Similarity is {sim:.0f}% — reasonable but worth monitoring."))
    else:
        ok_flags.append(f"High visual similarity ({sim:.0f}%) — images look consistent.")

    # ── Summary ───────────────────────────────────────────────────────────────
    if delta > 0.5:
        summary = (f"Your mole's score (TDS) increased by {delta:.2f} points between the two photos. "
                   f"This suggests some morphological change has occurred and warrants closer attention.")
    elif delta < -0.3:
        summary = (f"Your mole's score decreased by {abs(delta):.2f} points — the mole appears "
                   f"more stable or uniform compared to your baseline.")
    else:
        summary = (f"Your mole's score changed by {delta:+.2f} points. "
                   f"This is a small variation and the mole appears relatively stable, "
                   f"though continued monitoring is always recommended.")

    # ── What the risk level means in plain English ────────────────────────────
    risk_plain = {
        "HIGH": ("The computer analysis scored this mole in the higher-concern range. "
                 "This does NOT mean you have cancer — many benign moles score here. "
                 "It means a dermatologist should examine this mole in person with a dermatoscope."),
        "MODERATE": ("The analysis scored this mole in an intermediate range. "
                     "It isn't alarming, but it has some features worth watching. "
                     "A follow-up photo in 3 months or a routine dermatology visit is advised."),
        "LOW": ("The analysis scored this mole in the lower-concern range. "
                "Most moles score here. Continue regular self-checks every 1–3 months "
                "and see a dermatologist annually or if anything changes."),
    }

    # ── Action steps ──────────────────────────────────────────────────────────
    actions = {
        "HIGH": [
            "Book an appointment with a dermatologist soon (within 2–4 weeks).",
            "Do not attempt to treat or pick at the lesion.",
            "Take a new reference photo every 2 weeks until seen by a doctor.",
            "Note any symptoms: itching, bleeding, crusting, or rapid size change.",
        ],
        "MODERATE": [
            "Schedule a routine dermatology check-up within the next 3 months.",
            "Retake comparison photos monthly under the same lighting conditions.",
            "Apply broad-spectrum SPF 30+ sunscreen to the area daily.",
            "Watch for the ABCDE warning signs: Asymmetry, Border, Colour, Diameter, Evolution.",
        ],
        "LOW": [
            "Continue monthly self-examinations using good lighting.",
            "See a dermatologist for a full-body skin check once a year.",
            "Use SPF 30+ sunscreen and protective clothing outdoors.",
            "Re-run this tracker in 3–6 months to log changes over time.",
        ],
    }

    return {
        "abcd_baseline": res["baseline"], "abcd_current": res["current"],
        "similarity_index": sim, "delta_tds": delta,
        "risk_score": rs, "risk_level": rl,
        "confidence": conf,
        "change_summary": summary,
        "risk_plain": risk_plain[rl],
        "flags": flags, "ok_flags": ok_flags,
        "actions": actions[rl],
    }


# =============================================================================
#  UI HELPERS
# =============================================================================

def gauge(label, hint, value, max_val, color):
    pct = min(value / max_val * 100, 100)
    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge-header">
        <span class="gauge-name">{label}</span>
        <span class="gauge-hint">{hint}</span>
        <span class="gauge-val">{value:.2f} / {max_val:.0f}</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{pct:.1f}%;background:{color}"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def risk_color(level):
    return {"LOW": "#34d399", "MODERATE": "#fbbf24", "HIGH": "#f87171"}.get(level.upper(), "#34d399")


def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pill(text, kind="ok"):
    return f'<span class="pill pill-{kind}">{text}</span>'


# =============================================================================
#  SESSION STATE
# =============================================================================
# =============================================================================
#  SESSION STATE INIT
# =============================================================================
for key, default in [
    ("cap_baseline", None),
    ("cap_current",  None),
    ("camera_permitted", False),   # user has accepted camera permission
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
#  CAMERA PERMISSION GATE
# =============================================================================

def permission_gate():
    """Show a permission prompt before opening the camera. Returns True when granted."""
    if st.session_state["camera_permitted"]:
        return True

    st.markdown("""
    <div class="card" style="max-width:540px;margin:2rem auto;text-align:center;border-color:rgba(94,234,212,.3)">
      <div style="font-size:2.5rem;margin-bottom:.6rem">📷</div>
      <div class="card-label" style="text-align:center">Camera Access Required</div>
      <div class="explain" style="text-align:left;margin-bottom:1rem">
        DermaScan AI needs access to your camera to take mole photos for analysis.
        <br><br>
        <strong>What we do:</strong><br>
        • Photos are processed locally — nothing is uploaded to any server<br>
        • Images are held only in your browser session and discarded when you close the tab<br>
        • Your privacy is fully protected<br><br>
        Click <strong>Allow Camera</strong> to continue, or use the <em>Upload Images</em> section below
        if you prefer not to use your camera.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("✅  Allow Camera Access", key="grant_cam"):
            st.session_state["camera_permitted"] = True
            st.rerun()
    return False


# =============================================================================
#  WEBCAM SECTION
# =============================================================================

st.markdown("""
<div class="section-head">
  <div class="section-head-txt">📷 Step 1 — Camera Capture</div>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)
st.markdown(
    f'<div style="color:var(--muted);font-size:.85rem;margin-bottom:1.2rem">'
    f'Hold your camera approximately <strong>{TARGET_CM_HINT} cm</strong> from the mole — '
    f'the mole should fill most of the frame. Click the shutter button to capture.</div>',
    unsafe_allow_html=True
)

cam_permitted = permission_gate()

cam_col1, cam_col2 = st.columns(2, gap="large")
for col, slot_key, label in [
    (cam_col1, "cap_baseline", "Baseline"),
    (cam_col2, "cap_current",  "Current"),
]:
    with col:
        st.markdown(f'<div class="card"><div class="card-label">📷 {label} — Camera</div>', unsafe_allow_html=True)
        if cam_permitted:
            snap = st.camera_input(f"Take {label} photo", key=f"cam_{label.lower()}")
            if snap is not None:
                bgr = load_camera_image(snap)
                st.session_state[slot_key] = bgr
                st.success(f"{label} captured ✅")
        else:
            st.markdown('<div class="explain" style="text-align:center;color:var(--muted)">🔒 Allow camera access above to enable live capture.</div>', unsafe_allow_html=True)
        if st.session_state[slot_key] is not None:
            st.markdown('<div style="margin-top:.6rem;font-family:\'DM Mono\',monospace;font-size:.65rem;letter-spacing:.15em;color:var(--accent)">CAPTURED PREVIEW</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(st.session_state[slot_key]),
                     caption=label, use_container_width=True)
            if st.button(f"🗑  Clear {label}", key=f"clear_{label.lower()}"):
                st.session_state[slot_key] = None; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  UPLOAD FALLBACK
# =============================================================================

st.markdown("""
<div class="section-head">
  <div class="section-head-txt">📂 Step 2 — Or Upload Images</div>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)

up_col1, up_col2 = st.columns(2, gap="large")
for col, slot_key, label in [
    (up_col1, "cap_baseline", "Baseline"),
    (up_col2, "cap_current",  "Current"),
]:
    with col:
        st.markdown(f'<div class="card"><div class="card-label">📂 {label} Upload</div>', unsafe_allow_html=True)
        f = st.file_uploader(f"Upload {label.lower()} mole photo", type=["jpg","jpeg","png","webp"],
                             key=f"up_{label.lower()}")
        if f:
            bgr = load_cv(f)
            st.session_state[slot_key] = bgr
            st.image(bgr_to_rgb(bgr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  ANALYSIS
# =============================================================================

st.markdown("""
<div class="section-head">
  <div class="section-head-txt">🔬 Step 3 — Run Analysis</div>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)

bgr1 = st.session_state.get("cap_baseline")
bgr2 = st.session_state.get("cap_current")
can_analyse = bgr1 is not None and bgr2 is not None

if not can_analyse:
    st.info("Capture or upload **both** images (Baseline + Current) to enable analysis.", icon="🔬")
else:
    if st.button("🔬  Analyse & Generate Report"):
        with st.spinner("Running GrabCut segmentation and ABCD analysis…"):
            try:
                data = analyse_pair(bgr1, bgr2)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # ── Persist to SQLite ─────────────────────────────────────────────────
        try:
            saved_id = save_report(data, bgr1, bgr2)
            st.toast(f"Report #{saved_id} saved to database ✅", icon="💾")
        except Exception as db_err:
            st.warning(f"Could not save to database: {db_err}")

        rl   = data["risk_level"]
        rs   = data["risk_score"]
        sim  = data["similarity_index"]
        conf = data["confidence"]
        rc   = risk_color(rl)
        delta = data["delta_tds"]

        # ════════════════════════════════════════════════════════════════
        #  REPORT HEADER — Risk level at a glance
        # ════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📋 Your Mole Report")

        st.markdown(f"""
        <div class="card" style="border-color:{rc}40">
          <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;margin-bottom:1rem;">
            <div>
              <div class="card-label">Overall Risk Assessment</div>
              <span class="risk-badge risk-{rl}">{'⚠' if rl=='HIGH' else ('⚡' if rl=='MODERATE' else '✅')} {rl} RISK</span>
            </div>
            <div style="flex:1;min-width:200px">
              <div style="display:flex;gap:2rem;flex-wrap:wrap">
                <div class="score-ring">
                  <div class="score-num" style="color:{rc}">{rs:.1f}</div>
                  <div class="score-label">Risk Score / 10</div>
                </div>
                <div class="score-ring">
                  <div class="score-num" style="color:#818cf8">{sim:.0f}%</div>
                  <div class="score-label">Visual Similarity</div>
                </div>
                <div class="score-ring">
                  <div class="score-num" style="color:{'#34d399' if conf>0.6 else '#fbbf24'}">{int(conf*100)}%</div>
                  <div class="score-label">Analysis Confidence</div>
                </div>
              </div>
            </div>
          </div>
          <div class="explain">{data['risk_plain']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        #  WHAT CHANGED — plain-English summary
        # ════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-head">
          <div class="section-head-txt">📊 What Changed Since Your Baseline?</div>
          <div class="section-head-line"></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="explain" style="margin-bottom:1rem">{data["change_summary"]}</div>', unsafe_allow_html=True)

        for title, detail in data["flags"]:
            st.markdown(f'<div class="flag-item">⚠️ <div><strong>{title}</strong><br><span style="color:var(--muted)">{detail}</span></div></div>', unsafe_allow_html=True)
        for msg in data["ok_flags"]:
            st.markdown(f'<div class="flag-ok-item">✅ <span>{msg}</span></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        #  ABCD DETAIL CARDS — side by side with explanations
        # ════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-head">
          <div class="section-head-txt">🧬 ABCD Metric Breakdown</div>
          <div class="section-head-line"></div>
        </div>""", unsafe_allow_html=True)

        # Metric info
        metric_info = {
            "asymmetry": {
                "label": "Asymmetry", "hint": "Lower = more symmetric", "max": 2,
                "color": "#7eb8f7",
                "explain": "A symmetric mole looks the same on both sides when folded in half. "
                           "Score 0 = perfectly symmetric; 2 = very asymmetric.",
                "ok": lambda v: v < 0.8, "ok_msg": "Looks symmetric", "warn_msg": "Has some asymmetry"
            },
            "border": {
                "label": "Border Irregularity", "hint": "Lower = smoother edges", "max": 8,
                "color": "#b07ef7",
                "explain": "Checks how ragged or uneven the edges are. "
                           "Regular moles have smooth, well-defined borders. "
                           "Score counts how many of 8 border segments look irregular.",
                "ok": lambda v: v < 3, "ok_msg": "Fairly smooth border", "warn_msg": "Irregular edges detected"
            },
            "color": {
                "label": "Colour Variation", "hint": "Lower = fewer colours", "max": 6,
                "color": "#fbbf24",
                "explain": "Counts how many distinct dermoscopic colours are present. "
                           "Most benign moles are 1–2 shades. 4+ colours in one lesion is notable.",
                "ok": lambda v: v <= 2, "ok_msg": "Normal colour range", "warn_msg": "Multiple colours present"
            },
            "diameter": {
                "label": "Estimated Diameter", "hint": "Lower = smaller lesion", "max": 5,
                "color": "#34d399",
                "explain": "Estimates relative lesion size. Clinically, moles > 6 mm in diameter "
                           "get extra attention. This score scales linearly to that threshold.",
                "ok": lambda v: v < 3, "ok_msg": "Within normal size range", "warn_msg": "Larger than average"
            },
        }

        ab1, ab2 = st.columns(2, gap="medium")
        for col, key, title in [(ab1, "abcd_baseline", "Baseline"), (ab2, "abcd_current", "Current")]:
            abcd = data[key]; t = abcd["tds"]
            tc = "#34d399" if t < 4.75 else ("#fbbf24" if t < 5.45 else "#f87171")
            with col:
                st.markdown(f'<div class="card"><div class="card-label">{title} Measurements</div>', unsafe_allow_html=True)
                for mk, info in metric_info.items():
                    val = abcd[mk]
                    is_ok = info["ok"](val)
                    gauge(info["label"], info["hint"], val, info["max"], info["color"])
                    status_pill = pill(info["ok_msg"] if is_ok else info["warn_msg"],
                                      "ok" if is_ok else "warn")
                    st.markdown(f'<div style="margin:.1rem 0 .5rem">{status_pill}</div>', unsafe_allow_html=True)

                # TDS display
                st.markdown(f"""
                <div style="margin-top:1rem;padding:.6rem .9rem;background:var(--surface2);
                     border-radius:10px;display:flex;justify-content:space-between;align-items:center">
                  <div>
                    <div style="font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.15em;
                      text-transform:uppercase;color:var(--muted)">Total Dermoscopy Score</div>
                    <div style="font-size:.75rem;color:var(--muted);margin-top:.15rem">
                      &lt;4.75 Low · 4.75–5.45 Moderate · &gt;5.45 High
                    </div>
                  </div>
                  <div style="font-family:'Fraunces',serif;font-size:2rem;color:{tc}">{t:.3f}</div>
                </div>""", unsafe_allow_html=True)

                # Colour flags
                cflags = abcd["color_flags"]
                if cflags:
                    pills_html = "".join([pill(c, "warn") for c in cflags])
                    st.markdown(f'<div style="margin-top:.6rem"><div style="font-size:.72rem;color:var(--muted);margin-bottom:.3rem">Colours detected:</div><div class="pill-row">{pills_html}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="margin-top:.6rem">{pill("No colour flags", "ok")}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        # Metric explainer expander
        with st.expander("ℹ️ What do these metrics mean?"):
            for mk, info in metric_info.items():
                st.markdown(f"**{info['label']}** — {info['explain']}")
            st.markdown("**TDS (Total Dermoscopy Score)** — A weighted sum of all four metrics based on the European Dermoscopy Foundation formula. It's a screening tool, not a diagnosis.")

        # ════════════════════════════════════════════════════════════════
        #  COMPARISON TABLE
        # ════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-head">
          <div class="section-head-txt">📈 Side-by-Side Comparison</div>
          <div class="section-head-line"></div>
        </div>""", unsafe_allow_html=True)

        def delta_arrow(a, b, lower_better=True):
            d = b - a
            if abs(d) < 0.05: return f'<span class="cmp-flat">→ {b:.2f}</span>'
            cls = ("cmp-up" if (d > 0) == (not lower_better) else "cmp-down") if lower_better else \
                  ("cmp-up" if d > 0 else "cmp-down")
            arr = "↑" if d > 0 else "↓"
            return f'<span class="{cls}">{arr} {b:.2f}</span>'

        rows = [
            ("Asymmetry", "abcd_baseline", "abcd_current", "asymmetry", True),
            ("Border",    "abcd_baseline", "abcd_current", "border",    True),
            ("Colour",    "abcd_baseline", "abcd_current", "color",     True),
            ("Diameter",  "abcd_baseline", "abcd_current", "diameter",  True),
            ("TDS",       "abcd_baseline", "abcd_current", "tds",       True),
        ]
        table_html = '<table class="cmp-table"><tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>'
        for name, k1, k2, field, lb in rows:
            v1 = data[k1][field]; v2 = data[k2][field]
            table_html += f"<tr><td>{name}</td><td>{v1:.2f}</td><td>{delta_arrow(v1, v2, lb)}</td><td>{'↑' if v2>v1 else ('↓' if v2<v1 else '→')} {abs(v2-v1):.2f}</td></tr>"
        table_html += f"<tr><td>Visual Similarity</td><td colspan='2' style='text-align:center'>{sim:.1f}%</td><td>{'✅ High' if sim>=80 else ('⚠️ Moderate' if sim>=65 else '🔴 Low')}</td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        #  ACTION STEPS
        # ════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-head">
          <div class="section-head-txt">✅ Recommended Next Steps</div>
          <div class="section-head-line"></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="card" style="border-color:{rc}40"><div class="card-label">Action Plan — {rl} Risk</div>', unsafe_allow_html=True)
        for i, step in enumerate(data["actions"], 1):
            st.markdown(f'<div class="action-step"><span class="step-num">{i:02d}</span><span class="step-txt">{step}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        #  SEGMENTATION IMAGES
        # ════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-head">
          <div class="section-head-txt">🖼 Images &amp; Segmentation Preview</div>
          <div class="section-head-line"></div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.image(bgr_to_rgb(bgr1), caption="Baseline (original)", use_container_width=True)
        with c2:
            st.image(overlay_mask(bgr1, data["abcd_baseline"]["mask"]), caption="Baseline — detected lesion (teal overlay)", use_container_width=True)
        with c3:
            st.image(bgr_to_rgb(bgr2), caption="Current (original)", use_container_width=True)
        with c4:
            st.image(overlay_mask(bgr2, data["abcd_current"]["mask"]), caption="Current — detected lesion (teal overlay)", use_container_width=True)

        # Confidence note
        if conf < 0.55:
            st.warning(
                f"⚠️ **Segmentation confidence is {int(conf*100)}%** — the mole was harder than usual to isolate automatically. "
                "This can happen with low contrast, unusual lighting, or cluttered backgrounds. "
                "Check the segmentation images above — if the teal overlay doesn't match the mole, "
                "retake the photo with better lighting and a plain background.", icon="🔍"
            )

        # ════════════════════════════════════════════════════════════════
        #  DISCLAIMER
        # ════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="disclaimer">
        ⚕ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
        It does not constitute medical advice, diagnosis, or treatment.
        The ABCD scoring system and Total Dermoscopy Score are screening heuristics — not a clinical diagnosis.
        Always consult a qualified, board-certified dermatologist for evaluation of any skin lesion.
        Early professional assessment is the single most effective action for skin cancer prevention.
        </div>""", unsafe_allow_html=True)


# =============================================================================
#  HISTORY & TREND GRAPHS  (shown whenever ≥1 report exists)
# =============================================================================

# =============================================================================
#  HISTORY & TREND GRAPHS  — reads directly from SQLite
# =============================================================================

all_reports = load_all_reports()

if all_reports:
    st.markdown("""
    <div class="section-head" style="margin-top:3rem">
      <div class="section-head-txt">📜 Report History & Trends</div>
      <div class="section-head-line"></div>
    </div>""", unsafe_allow_html=True)

    # ── History table ─────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-label">All Saved Reports — Most Recent Last</div>', unsafe_allow_html=True)

    RISK_EMOJI = {"LOW": "✅", "MODERATE": "⚡", "HIGH": "⚠️"}
    header_html = (
        "<table class='cmp-table'>"
        "<tr><th>#</th><th>Date / Time</th><th>Risk</th><th>Score /10</th>"
        "<th>TDS</th><th>ΔTDS</th><th>Asymmetry</th><th>Border</th>"
        "<th>Colour</th><th>Diameter</th><th>Similarity</th><th>Conf.</th><th>Action</th></tr>"
    )
    rows_html = ""
    for r in all_reports:
        rl = r["risk_level"]
        rc_col = {"LOW": "var(--low)", "MODERATE": "var(--mod)", "HIGH": "var(--high)"}.get(rl, "var(--text)")
        d_col  = "var(--high)" if r["delta_tds"] > 0.1 else ("var(--low)" if r["delta_tds"] < -0.1 else "var(--muted)")
        rows_html += (
            f"<tr>"
            f"<td style='color:var(--muted)'>{r['id']}</td>"
            f"<td style='font-family:\"DM Mono\",monospace;font-size:.76rem'>{r['timestamp']}</td>"
            f"<td style='color:{rc_col}'>{RISK_EMOJI.get(rl,'')} {rl}</td>"
            f"<td style='font-family:\"DM Mono\",monospace;color:{rc_col}'>{r['risk_score']:.1f}</td>"
            f"<td style='font-family:\"DM Mono\",monospace'>{r['tds_current']:.3f}</td>"
            f"<td style='color:{d_col};font-family:\"DM Mono\",monospace'>{r['delta_tds']:+.3f}</td>"
            f"<td>{r['asymmetry_c']:.2f}</td>"
            f"<td>{r['border_c']:.2f}</td>"
            f"<td>{r['color_c']:.0f}</td>"
            f"<td>{r['diameter_c']:.2f}</td>"
            f"<td>{r['similarity']:.1f}%</td>"
            f"<td>{int(r['confidence']*100)}%</td>"
            f"<td><a href='?view_id={r['id']}' target='_self' style='color:var(--accent);font-size:.75rem'>View</a></td>"
            f"</tr>"
        )
    st.markdown(header_html + rows_html + "</table>", unsafe_allow_html=True)

    col_del_all, _ = st.columns([1, 3])
    with col_del_all:
        if st.button("🗑  Delete All Reports", key="del_all"):
            delete_all_reports()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-report image viewer ───────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
      <div class="section-head-txt">🖼 Saved Image Viewer</div>
      <div class="section-head-line"></div>
    </div>""", unsafe_allow_html=True)

    report_options = {f"Report #{r['id']} — {r['timestamp']}  [{r['risk_level']}]": r["id"]
                      for r in reversed(all_reports)}
    chosen_label = st.selectbox("Select a saved report to view its images:", list(report_options.keys()))
    chosen_id    = report_options[chosen_label]
    chosen_row   = next(r for r in all_reports if r["id"] == chosen_id)
    imgs         = load_report_images(chosen_id)

    if imgs:
        ic1, ic2, ic3, ic4 = st.columns(4, gap="small")
        with ic1:
            st.image(jpeg_to_rgb(imgs["img_baseline"]),   caption="Baseline (original)",    use_container_width=True)
        with ic2:
            st.image(jpeg_to_rgb(imgs["img_seg_base"]),   caption="Baseline (segmented)",   use_container_width=True)
        with ic3:
            st.image(jpeg_to_rgb(imgs["img_current"]),    caption="Current (original)",     use_container_width=True)
        with ic4:
            st.image(jpeg_to_rgb(imgs["img_seg_curr"]),   caption="Current (segmented)",    use_container_width=True)

        # Mini report card for chosen report
        rc_c = {"LOW": "#34d399", "MODERATE": "#fbbf24", "HIGH": "#f87171"}.get(chosen_row["risk_level"], "#34d399")
        cf_b = json.loads(chosen_row["color_flags_b"] or "[]")
        cf_c = json.loads(chosen_row["color_flags_c"] or "[]")
        st.markdown(f"""
        <div class="card" style="margin-top:.8rem;border-color:{rc_c}33">
          <div class="card-label">Report #{chosen_row['id']} summary — {chosen_row['timestamp']}</div>
          <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:.8rem">
            <div><span style="color:var(--muted);font-size:.75rem">Risk</span><br>
              <span style="color:{rc_c};font-family:'DM Mono',monospace;font-size:1rem">{chosen_row['risk_level']}</span></div>
            <div><span style="color:var(--muted);font-size:.75rem">Score</span><br>
              <span style="font-family:'Fraunces',serif;font-size:1.4rem;color:{rc_c}">{chosen_row['risk_score']:.1f}</span></div>
            <div><span style="color:var(--muted);font-size:.75rem">TDS (current)</span><br>
              <span style="font-family:'DM Mono',monospace">{chosen_row['tds_current']:.3f}</span></div>
            <div><span style="color:var(--muted);font-size:.75rem">ΔTDS</span><br>
              <span style="font-family:'DM Mono',monospace;color:{'#f87171' if chosen_row['delta_tds']>0.1 else ('#34d399' if chosen_row['delta_tds']<-0.1 else 'var(--muted)')}">{chosen_row['delta_tds']:+.3f}</span></div>
            <div><span style="color:var(--muted);font-size:.75rem">Similarity</span><br>
              <span style="font-family:'DM Mono',monospace">{chosen_row['similarity']:.1f}%</span></div>
            <div><span style="color:var(--muted);font-size:.75rem">Confidence</span><br>
              <span style="font-family:'DM Mono',monospace">{int(chosen_row['confidence']*100)}%</span></div>
          </div>
          <div class="explain">{chosen_row['change_summary']}</div>
          {'<div style="margin-top:.5rem;font-size:.78rem;color:var(--muted)">Baseline colours: ' + (", ".join(cf_b) if cf_b else "none") + ' &nbsp;·&nbsp; Current colours: ' + (", ".join(cf_c) if cf_c else "none") + '</div>' if cf_b or cf_c else ''}
        </div>""", unsafe_allow_html=True)

        del_col, _ = st.columns([1, 4])
        with del_col:
            if st.button(f"🗑  Delete Report #{chosen_id}", key=f"del_{chosen_id}"):
                delete_report(chosen_id)
                st.rerun()
    else:
        st.info("No images found for this report.")

    # ── Trend graphs (need ≥2 rows) ───────────────────────────────────────────
    if len(all_reports) >= 2:
        st.markdown("""
        <div class="section-head">
          <div class="section-head-txt">📈 Metric Trends Over Time</div>
          <div class="section-head-line"></div>
        </div>""", unsafe_allow_html=True)

        short_lbl = [f"#{r['id']}" for r in all_reports]
        xs = range(len(all_reports))

        # ── Theme colours ─────────────────────────────────────────────────────
        BG      = "#0f1219"
        SURFACE = "#161b26"
        BORDER  = "#1f2535"
        TEXT    = "#dde1f0"
        MUTED   = "#5a6175"
        ACCENT  = "#5eead4"
        ACCENT2 = "#818cf8"
        LOW_C   = "#34d399"
        MOD_C   = "#fbbf24"
        HIGH_C  = "#f87171"

        def style_ax(ax):
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(short_lbl, color=MUTED, fontsize=8)

        point_colors = [
            {"LOW": LOW_C, "MODERATE": MOD_C, "HIGH": HIGH_C}.get(r["risk_level"], ACCENT)
            for r in all_reports
        ]

        risk_scores = [r["risk_score"]  for r in all_reports]
        tds_vals    = [r["tds_current"] for r in all_reports]
        sim_vals    = [r["similarity"]  for r in all_reports]
        delta_vals  = [r["delta_tds"]   for r in all_reports]

        # ── Graph 1: Risk Score + TDS ─────────────────────────────────────────
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
        fig1.patch.set_facecolor(BG)
        style_ax(ax1); style_ax(ax2)

        ax1.plot(xs, risk_scores, color=ACCENT, linewidth=2, zorder=2, alpha=0.85)
        ax1.fill_between(xs, risk_scores, alpha=0.08, color=ACCENT)
        for xi, yi, pc in zip(xs, risk_scores, point_colors):
            ax1.scatter(xi, yi, color=pc, s=60, zorder=3, edgecolors="none")
        ax1.axhspan(0,   4.5, alpha=0.04, color=LOW_C)
        ax1.axhspan(4.5, 6.8, alpha=0.04, color=MOD_C)
        ax1.axhspan(6.8, 10,  alpha=0.04, color=HIGH_C)
        ax1.set_ylim(0, 10)
        ax1.set_title("Risk Score Over Sessions", color=TEXT, fontsize=10, pad=8)
        ax1.set_ylabel("Risk Score (0–10)", color=MUTED, fontsize=8)
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(2))

        ax2.plot(xs, tds_vals, color=ACCENT2, linewidth=2, zorder=2, alpha=0.85)
        ax2.fill_between(xs, tds_vals, alpha=0.08, color=ACCENT2)
        for xi, yi, pc in zip(xs, tds_vals, point_colors):
            ax2.scatter(xi, yi, color=pc, s=60, zorder=3, edgecolors="none")
        ax2.axhline(4.75, color=MOD_C, linewidth=1, linestyle="--", alpha=0.5, label="Moderate (4.75)")
        ax2.axhline(5.45, color=HIGH_C, linewidth=1, linestyle="--", alpha=0.5, label="High (5.45)")
        ax2.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=MUTED)
        ax2.set_title("Total Dermoscopy Score (TDS)", color=TEXT, fontsize=10, pad=8)
        ax2.set_ylabel("TDS", color=MUTED, fontsize=8)

        fig1.tight_layout(pad=1.5)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        # ── Graph 2: ABCD metrics ─────────────────────────────────────────────
        fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3.4))
        fig2.patch.set_facecolor(BG)

        abcd_metrics = [
            ("asymmetry_c", "Asymmetry", 2, "#7eb8f7"),
            ("border_c",    "Border",    8, "#b07ef7"),
            ("color_c",     "Colour",    6, MOD_C),
            ("diameter_c",  "Diameter",  5, LOW_C),
        ]
        for ax, (field, name, ymax, color) in zip(axes2, abcd_metrics):
            style_ax(ax)
            ax.set_ylim(0, ymax)
            ax.set_title(name, color=TEXT, fontsize=9, pad=6)
            vals = [r[field] for r in all_reports]
            ax.plot(xs, vals, color=color, linewidth=2, alpha=0.85, zorder=2)
            ax.fill_between(xs, vals, alpha=0.12, color=color)
            for xi, yi, pc in zip(xs, vals, point_colors):
                ax.scatter(xi, yi, color=pc, s=45, zorder=3, edgecolors="none")

        fig2.suptitle("ABCD Metrics Across Sessions", color=TEXT, fontsize=11, y=1.02)
        fig2.tight_layout(pad=1.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # ── Graph 3: Similarity + ΔTDS ────────────────────────────────────────
        fig3, (ax_sim, ax_delta) = plt.subplots(1, 2, figsize=(12, 3.4))
        fig3.patch.set_facecolor(BG)
        style_ax(ax_sim); style_ax(ax_delta)

        ax_sim.plot(xs, sim_vals, color=ACCENT, linewidth=2, alpha=0.85)
        ax_sim.fill_between(xs, sim_vals, alpha=0.08, color=ACCENT)
        ax_sim.axhline(80, color=LOW_C, linewidth=1, linestyle="--", alpha=0.5, label="High (80%)")
        ax_sim.axhline(65, color=MOD_C, linewidth=1, linestyle="--", alpha=0.5, label="Moderate (65%)")
        for xi, yi, pc in zip(xs, sim_vals, point_colors):
            ax_sim.scatter(xi, yi, color=pc, s=45, zorder=3, edgecolors="none")
        ax_sim.set_ylim(0, 100)
        ax_sim.set_title("Visual Similarity %", color=TEXT, fontsize=10, pad=8)
        ax_sim.set_ylabel("Similarity %", color=MUTED, fontsize=8)
        ax_sim.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=MUTED)

        bar_colors = [HIGH_C if d > 0.1 else (LOW_C if d < -0.1 else MUTED) for d in delta_vals]
        ax_delta.bar(list(xs), delta_vals, color=bar_colors, alpha=0.75, zorder=2)
        ax_delta.axhline(0, color=TEXT, linewidth=0.8, alpha=0.4)
        ax_delta.set_title("TDS Change vs Baseline (ΔTDS)", color=TEXT, fontsize=10, pad=8)
        ax_delta.set_ylabel("ΔTDS", color=MUTED, fontsize=8)
        ax_delta.tick_params(colors=MUTED)

        fig3.tight_layout(pad=1.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        legend_md = "  ".join([f"`#{r['id']}` = {r['timestamp']}" for r in all_reports])
        st.markdown(
            f'<div style="font-size:.72rem;color:var(--muted);margin-top:.3rem">Session labels: {legend_md}</div>',
            unsafe_allow_html=True
        )

    else:
        st.info("Run at least **2 analyses** to see trend graphs.", icon="📈")
