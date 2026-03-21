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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading

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
  <div class="hero-badge">📡 Distance-Guided Capture · OpenCV · Research Tool</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  CONSTANTS
# =============================================================================
TARGET_CM      = 10
TOLERANCE_CM   = 2.5
REF_AREA_RATIO = 0.18
REF_DISTANCE   = TARGET_CM


# =============================================================================
#  DISTANCE ESTIMATION
# =============================================================================

def estimate_distance_cm(bgr: np.ndarray) -> float:
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo   = np.array([0,  20,  60], np.uint8)
    hi   = np.array([25, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    lo2  = np.array([160, 20, 60], np.uint8)
    hi2  = np.array([180, 255, 255], np.uint8)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    area_px  = float(np.count_nonzero(mask))
    total_px = float(bgr.shape[0] * bgr.shape[1])
    if total_px == 0 or area_px == 0:
        return TARGET_CM
    ratio = area_px / total_px
    dist  = REF_DISTANCE * np.sqrt(REF_AREA_RATIO / (ratio + 1e-6))
    return float(np.clip(dist, 2.0, 50.0))


def distance_status(dist_cm: float) -> tuple:
    diff = dist_cm - TARGET_CM
    if abs(diff) <= TOLERANCE_CM:
        return f"✅  {dist_cm:.1f} cm — GOOD", "#34d399", "dist-ok", True
    elif diff > 0:
        return f"🔴  {dist_cm:.1f} cm — TOO FAR (move closer)", "#f87171", "dist-bad", False
    else:
        return f"🟡  {dist_cm:.1f} cm — TOO CLOSE (move back)", "#fbbf24", "dist-warn", False


# =============================================================================
#  WEBCAM
# =============================================================================

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class MoleWebcam(VideoTransformerBase):
    def __init__(self):
        self.latest_frame   = None
        self.latest_dist_cm = TARGET_CM
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        dist_cm = estimate_distance_cm(bgr)
        _, _, css, ok = distance_status(dist_cm)
        h, w = bgr.shape[:2]
        out  = bgr.copy()
        cx, cy = w // 2, h // 2
        r = min(w, h) // 6
        color = (94, 234, 212) if ok else (248, 113, 113)  # BGR approximation
        for angle in range(0, 360, 20):
            if angle % 40 < 20:
                a1 = np.deg2rad(angle); a2 = np.deg2rad(angle + 18)
                p1 = (int(cx + (r+18)*np.cos(a1)), int(cy + (r+18)*np.sin(a1)))
                p2 = (int(cx + (r+18)*np.cos(a2)), int(cy + (r+18)*np.sin(a2)))
                cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, color, 2, cv2.LINE_AA)
        gap = 12
        cv2.line(out, (cx-r-20, cy), (cx-gap, cy),  color, 1, cv2.LINE_AA)
        cv2.line(out, (cx+gap,   cy), (cx+r+20, cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy-r-20), (cx, cy-gap),   color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy+gap),  (cx, cy+r+20),  color, 1, cv2.LINE_AA)
        blen = 20
        for (bx, by, sx, sy) in [(10,10,1,1),(w-10,10,-1,1),(10,h-10,1,-1),(w-10,h-10,-1,-1)]:
            cv2.line(out, (bx, by), (bx+sx*blen, by), color, 2)
            cv2.line(out, (bx, by), (bx, by+sy*blen), color, 2)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (7, 9, 15), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
        dist_txt = f"Distance: {dist_cm:.1f} cm"
        tgt_txt  = f"Target: {TARGET_CM} cm ±{TOLERANCE_CM} cm"
        cv2.putText(out, dist_txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(out, tgt_txt, (w-260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90,97,117), 1, cv2.LINE_AA)
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h-40), (w, h), (7, 9, 15), -1)
        cv2.addWeighted(overlay2, 0.65, out, 0.35, 0, out)
        status_str = "READY — distance OK" if ok else "ADJUST DISTANCE"
        cv2.putText(out, status_str, (cx-130, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        with self.lock:
            self.latest_frame   = bgr.copy()
            self.latest_dist_cm = dist_cm
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def get_snapshot(self):
        with self.lock:
            return (self.latest_frame.copy() if self.latest_frame is not None else None,
                    self.latest_dist_cm)


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
for key in ("cap_baseline", "cap_current", "dist_baseline", "dist_current"):
    if key not in st.session_state:
        st.session_state[key] = None


# =============================================================================
#  WEBCAM SECTION
# =============================================================================

st.markdown("""
<div class="section-head">
  <div class="section-head-txt">📷 Step 1 — Webcam Capture</div>
  <div class="section-head-line"></div>
</div>
""", unsafe_allow_html=True)
st.markdown(
    f'<div style="color:var(--muted);font-size:.85rem;margin-bottom:1.2rem">'
    f'Hold your camera <strong>{TARGET_CM} cm</strong> from the mole. Capture only when the indicator turns <span style="color:#34d399">green</span>.'
    f'</div>', unsafe_allow_html=True
)

cam_col1, cam_col2 = st.columns(2, gap="large")
for col, slot_key, dist_key, label in [
    (cam_col1, "cap_baseline", "dist_baseline", "Baseline"),
    (cam_col2, "cap_current",  "dist_current",  "Current"),
]:
    with col:
        st.markdown(f'<div class="card"><div class="card-label">📡 {label} — Live Camera</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key=f"webcam_{label.lower()}",
            video_transformer_factory=MoleWebcam,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
        if ctx.video_transformer:
            _, dist_cm = ctx.video_transformer.get_snapshot()
            label_txt, col_hex, css, ok = distance_status(dist_cm)
            st.markdown(
                f'<div style="margin:.5rem 0;padding:.5rem .8rem;background:var(--surface2);'
                f'border-radius:8px;font-family:\'DM Mono\',monospace;font-size:.82rem;">'
                f'<span class="{css}">{label_txt}</span></div>',
                unsafe_allow_html=True
            )
            if st.button(f"📸  Capture {label}", key=f"btn_{label.lower()}", disabled=not ok):
                frame, dist = ctx.video_transformer.get_snapshot()
                if frame is not None:
                    st.session_state[slot_key] = frame
                    st.session_state[dist_key] = dist
                    st.success(f"{label} captured at {dist:.1f} cm ✅")
        if st.session_state[slot_key] is not None:
            st.markdown('<div style="margin-top:.6rem;font-family:\'DM Mono\',monospace;font-size:.65rem;letter-spacing:.15em;color:var(--accent)">CAPTURED PREVIEW</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(st.session_state[slot_key]),
                     caption=f"{label} — {st.session_state[dist_key]:.1f} cm",
                     use_container_width=True)
            if st.button(f"🗑  Clear {label}", key=f"clear_{label.lower()}"):
                st.session_state[slot_key] = None; st.session_state[dist_key] = None; st.rerun()
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

        d1 = st.session_state.get("dist_baseline")
        d2 = st.session_state.get("dist_current")
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.image(bgr_to_rgb(bgr1), caption=f"Baseline original{' — '+str(round(d1,1))+' cm' if d1 else ''}", use_container_width=True)
        with c2:
            st.image(overlay_mask(bgr1, data["abcd_baseline"]["mask"]), caption="Baseline — detected lesion (teal overlay)", use_container_width=True)
        with c3:
            st.image(bgr_to_rgb(bgr2), caption=f"Current original{' — '+str(round(d2,1))+' cm' if d2 else ''}", use_container_width=True)
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
