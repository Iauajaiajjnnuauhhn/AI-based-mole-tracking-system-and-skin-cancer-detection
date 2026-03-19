"""
DermaScan AI – Mole Tracker
────────────────────────────
• Webcam capture with real-time distance estimation (face-landmark scale or
  reference-object fallback) + targeting reticle overlay
• Upload fallback for both images
• Full OpenCV ABCD + Similarity Index analysis
• Rich Streamlit report
"""

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan AI · Mole Tracker",
    page_icon="🔬",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0d0f14;--surface:#13161e;--surface2:#1a1e29;--border:#252836;
      --accent:#4fffb0;--text:#e8eaf2;--muted:#6b7080;
      --low:#4fffb0;--mod:#ffd166;--high:#ff6b6b;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg);color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2rem 4rem;max-width:1300px;}
h1,h2,h3{font-family:'DM Serif Display',serif;letter-spacing:-0.02em;}
.hero{text-align:center;padding:2.5rem 0 1.8rem;
      background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(79,255,176,.08),transparent);
      border-bottom:1px solid var(--border);margin-bottom:2rem;}
.hero-title{font-family:'DM Serif Display',serif;font-size:clamp(2rem,5vw,3.2rem);
            background:linear-gradient(135deg,#e8eaf2 30%,#4fffb0);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .4rem;}
.hero-sub{color:var(--muted);font-size:.95rem;font-weight:300;letter-spacing:.04em;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.4rem;margin-bottom:1.2rem;}
.card-title{font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.18em;
            text-transform:uppercase;color:var(--accent);margin-bottom:.9rem;}
.upload-hint{border:1.5px dashed var(--border);border-radius:10px;padding:1rem;
             text-align:center;color:var(--muted);font-size:.82rem;}
.risk-badge{display:inline-block;padding:.3rem .9rem;border-radius:999px;
            font-family:'DM Mono',monospace;font-size:.85rem;font-weight:500;letter-spacing:.06em;}
.risk-LOW{background:rgba(79,255,176,.12);color:var(--low);border:1px solid rgba(79,255,176,.3);}
.risk-MODERATE{background:rgba(255,209,102,.12);color:var(--mod);border:1px solid rgba(255,209,102,.3);}
.risk-HIGH{background:rgba(255,107,107,.14);color:var(--high);border:1px solid rgba(255,107,107,.35);}
.gauge-row{display:flex;align-items:center;gap:.7rem;margin:.35rem 0;}
.gauge-label{font-size:.75rem;color:var(--muted);width:155px;flex-shrink:0;}
.gauge-track{flex:1;height:5px;background:var(--surface2);border-radius:99px;overflow:hidden;}
.gauge-fill{height:100%;border-radius:99px;}
.gauge-val{font-family:'DM Mono',monospace;font-size:.75rem;color:var(--text);width:38px;text-align:right;}
.rep-section{border-left:2px solid var(--accent);padding-left:.7rem;
             margin:1.1rem 0 .5rem;font-size:.75rem;font-family:'DM Mono',monospace;
             letter-spacing:.12em;text-transform:uppercase;color:var(--accent);}
.rec-box{background:var(--surface2);border-radius:9px;padding:.9rem 1.1rem;
         font-size:.88rem;line-height:1.65;color:var(--text);margin-top:.4rem;}
.dim{color:var(--muted);font-size:.8rem;}
/* Distance badge */
.dist-ok{color:#4fffb0;font-weight:700;}
.dist-warn{color:#ffd166;font-weight:700;}
.dist-bad{color:#ff6b6b;font-weight:700;}
div.stButton>button{background:var(--accent);color:#0d0f14;border:none;border-radius:9px;
                    font-family:'DM Sans',sans-serif;font-weight:600;font-size:.95rem;
                    padding:.65rem 2rem;width:100%;cursor:pointer;transition:opacity .2s;}
div.stButton>button:hover{opacity:.85;}
div.stButton>button:disabled{opacity:.4;cursor:not-allowed;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-title">DermaScan AI</div>
  <div class="hero-sub">ABCD Rule · Similarity Index · Distance-Controlled Webcam Capture · OpenCV</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  DISTANCE ESTIMATION  (skin-patch apparent-size heuristic)
# =============================================================================
# Strategy: detect the largest skin-coloured blob in frame; use its apparent
# pixel-area compared to a reference calibration area to estimate distance.
# At the reference distance (TARGET_CM) the blob occupies REF_AREA_RATIO of
# the frame.  distance ∝ 1/√(blob_area).
# No face / ArUco required – works for close-up skin macro shots.

TARGET_CM      = 10          # desired capture distance in cm
TOLERANCE_CM   = 2.5         # ±2.5 cm is "green"
# Reference: at 10 cm a skin patch covers ~18 % of a 640×480 frame
REF_AREA_RATIO = 0.18
REF_DISTANCE   = TARGET_CM   # cm at which REF_AREA_RATIO was measured


def estimate_distance_cm(bgr: np.ndarray) -> float:
    """
    Estimate camera-to-skin distance from apparent skin-patch area.
    Returns distance in cm (clamped 2–50).
    """
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Skin hue range in HSV
    lo   = np.array([0,  20,  60], np.uint8)
    hi   = np.array([25, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    # Also catch slightly redder skin
    lo2  = np.array([160, 20, 60], np.uint8)
    hi2  = np.array([180, 255, 255], np.uint8)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    area_px  = float(np.count_nonzero(mask))
    total_px = float(bgr.shape[0] * bgr.shape[1])
    if total_px == 0 or area_px == 0:
        return TARGET_CM   # fallback

    ratio    = area_px / total_px
    # distance ∝ 1/√ratio  →  d = REF_DISTANCE * √(REF_AREA_RATIO / ratio)
    dist     = REF_DISTANCE * np.sqrt(REF_AREA_RATIO / (ratio + 1e-6))
    return float(np.clip(dist, 2.0, 50.0))


def distance_status(dist_cm: float) -> tuple:
    """Returns (label, colour_hex, css_class, ok_bool)."""
    diff = dist_cm - TARGET_CM
    if abs(diff) <= TOLERANCE_CM:
        return f"✅  {dist_cm:.1f} cm  — GOOD", "#4fffb0", "dist-ok", True
    elif diff > 0:
        return f"🔴  {dist_cm:.1f} cm  — TOO FAR  (move closer)", "#ff6b6b", "dist-bad", False
    else:
        return f"🟡  {dist_cm:.1f} cm  — TOO CLOSE  (move back)", "#ffd166", "dist-warn", False


# =============================================================================
#  WEBCAM VIDEO TRANSFORMER  (streamlit-webrtc)
# =============================================================================

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class MoleWebcam(VideoTransformerBase):
    """Draws targeting reticle + real-time distance HUD on each video frame."""

    def __init__(self):
        self.latest_frame   : np.ndarray | None = None
        self.latest_dist_cm : float             = TARGET_CM
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")

        dist_cm = estimate_distance_cm(bgr)
        _, _, css, ok = distance_status(dist_cm)

        h, w = bgr.shape[:2]
        out  = bgr.copy()

        # ── Targeting reticle ────────────────────────────────────────────────
        cx, cy = w // 2, h // 2
        r      = min(w, h) // 6          # inner circle radius
        color  = (79, 255, 176) if ok else (100, 100, 255)  # BGR

        # outer dashed circle → draw as arc segments
        for angle in range(0, 360, 20):
            if angle % 40 < 20:
                a1 = np.deg2rad(angle)
                a2 = np.deg2rad(angle + 18)
                p1 = (int(cx + (r + 18) * np.cos(a1)), int(cy + (r + 18) * np.sin(a1)))
                p2 = (int(cx + (r + 18) * np.cos(a2)), int(cy + (r + 18) * np.sin(a2)))
                cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)

        # inner solid circle
        cv2.circle(out, (cx, cy), r, color, 2, cv2.LINE_AA)

        # crosshair lines
        gap = 12
        cv2.line(out, (cx - r - 20, cy), (cx - gap, cy),  color, 1, cv2.LINE_AA)
        cv2.line(out, (cx + gap,    cy), (cx + r + 20, cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy - r - 20), (cx, cy - gap),   color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy + gap),    (cx, cy + r + 20), color, 1, cv2.LINE_AA)

        # corner brackets
        blen = 20
        for (bx, by, sx, sy) in [(10,10,1,1),(w-10,10,-1,1),(10,h-10,1,-1),(w-10,h-10,-1,-1)]:
            cv2.line(out, (bx, by), (bx + sx*blen, by), color, 2)
            cv2.line(out, (bx, by), (bx, by + sy*blen), color, 2)

        # ── HUD overlay ──────────────────────────────────────────────────────
        # semi-transparent bar at top
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (13, 15, 20), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

        dist_txt  = f"Distance: {dist_cm:.1f} cm"
        tgt_txt   = f"Target: {TARGET_CM} cm  ±{TOLERANCE_CM} cm"
        txt_color = (79, 255, 176) if ok else (100, 100, 255)
        cv2.putText(out, dist_txt, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_color, 2, cv2.LINE_AA)
        cv2.putText(out, tgt_txt, (w - 260, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 180), 1, cv2.LINE_AA)

        # status bar at bottom
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h - 40), (w, h), (13, 15, 20), -1)
        cv2.addWeighted(overlay2, 0.65, out, 0.35, 0, out)
        status_str = "READY — distance OK" if ok else "ADJUST DISTANCE"
        cv2.putText(out, status_str, (cx - 130, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_color, 2, cv2.LINE_AA)

        # store for capture
        with self.lock:
            self.latest_frame   = bgr.copy()
            self.latest_dist_cm = dist_cm

        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def get_snapshot(self) -> tuple:
        with self.lock:
            return (self.latest_frame.copy() if self.latest_frame is not None else None,
                    self.latest_dist_cm)


# =============================================================================
#  CV ANALYSIS PIPELINE
# =============================================================================

def load_cv(uploaded_file) -> np.ndarray:
    uploaded_file.seek(0)
    raw = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return img


def segment_lesion(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, _ = cv2.split(lab)
    score = cv2.addWeighted(255 - l, 0.6, a, 0.4, 0)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.uint8(labels == largest) * 255
    return mask


def compute_asymmetry(mask: np.ndarray) -> float:
    h, w = mask.shape
    scores = []
    for axis in [0, 1]:
        if axis == 0:
            h1 = mask[:h // 2, :]
            h2 = cv2.resize(np.flip(mask[h // 2:, :], 0), (h1.shape[1], h1.shape[0]))
        else:
            h1 = mask[:, :w // 2]
            h2 = cv2.resize(np.flip(mask[:, w // 2:], 1), (h1.shape[1], h1.shape[0]))
        union = np.logical_or(h1 > 0, h2 > 0).sum()
        diff  = np.logical_xor(h1 > 0, h2 > 0).sum()
        scores.append(diff / union if union > 0 else 0)
    return round(min((scores[0] + scores[1]) / 2 * 4, 2.0), 2)


def compute_border(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :]
    M   = cv2.moments(mask)
    cx  = M['m10'] / M['m00'] if M['m00'] else mask.shape[1] / 2
    cy  = M['m01'] / M['m00'] if M['m00'] else mask.shape[0] / 2
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    s = []
    for i in range(8):
        lo  = -np.pi + i * (2 * np.pi / 8)
        idx = np.where((angles >= lo) & (angles < lo + 2 * np.pi / 8))[0]
        if len(idx) < 3:
            s.append(0); continue
        r = np.sqrt((pts[idx, 0] - cx)**2 + (pts[idx, 1] - cy)**2)
        s.append(1 if np.std(r) / (np.mean(r) + 1e-6) > 0.12 else 0)
    return float(sum(s))


def compute_color(bgr: np.ndarray, mask: np.ndarray) -> float:
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    roi  = mask > 0
    h, s, v = hsv[:, :, 0][roi], hsv[:, :, 1][roi], hsv[:, :, 2][roi]
    l_ch    = lab[:, :, 0][roi]
    flags   = 0
    if np.mean(v > 200) > 0.05:                                       flags += 1
    if np.mean((h < 10) & (s > 80)) > 0.03:                           flags += 1
    if np.mean((h >= 10) & (h < 25) & (s > 50) & (v > 80)) > 0.05:   flags += 1
    if np.mean((h >= 10) & (h < 30) & (v < 100)) > 0.05:              flags += 1
    if np.mean((h >= 100) & (h < 140) & (s < 80)) > 0.03:             flags += 1
    if np.mean(l_ch < 40) > 0.05:                                      flags += 1
    return float(max(1, flags))


def compute_diameter(mask: np.ndarray, fov_mm: float = 20.0) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    _, (mw, mh), _ = cv2.minAreaRect(cnt)
    diam_mm = max(mw, mh) * fov_mm / mask.shape[1]
    return round(min(diam_mm / 6.0 * 5.0, 5.0), 2)


def tds(a, b, c, d) -> float:
    return round(a * 1.3 + b * 0.1 + c * 0.5 + d * 0.5, 3)


def risk_from_tds(t: float) -> tuple:
    level = "LOW" if t < 4.75 else ("MODERATE" if t <= 5.45 else "HIGH")
    return round(min(t / 8.0 * 10.0, 10.0), 2), level


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
    green   = np.array([0, 220, 120], dtype=np.float32)
    roi     = mask > 0
    overlay[roi] = (overlay[roi].astype(np.float32) * 0.55 + green * 0.45).clip(0, 255).astype(np.uint8)
    return overlay


def analyse_pair(bgr1: np.ndarray, bgr2: np.ndarray) -> dict:
    res = {}
    for key, bgr in [("baseline", bgr1), ("current", bgr2)]:
        mask = segment_lesion(bgr)
        a, b, c, d = (compute_asymmetry(mask), compute_border(mask),
                      compute_color(bgr, mask), compute_diameter(mask))
        res[key] = {"asymmetry": a, "border": b, "color": c,
                    "diameter": d, "tds": tds(a, b, c, d), "mask": mask}

    sim       = similarity_index(bgr1, bgr2)
    delta     = res["current"]["tds"] - res["baseline"]["tds"]
    rs, rl    = risk_from_tds(res["current"]["tds"])

    flags = []
    if res["current"]["asymmetry"] > res["baseline"]["asymmetry"] + 0.3:
        flags.append("Asymmetry has increased since baseline")
    if res["current"]["border"] > res["baseline"]["border"] + 1:
        flags.append("Border irregularity has worsened")
    if res["current"]["color"] > res["baseline"]["color"]:
        flags.append("New colour structures detected in lesion")
    if res["current"]["diameter"] > res["baseline"]["diameter"] + 0.5:
        flags.append("Estimated lesion diameter has grown")
    if sim < 70:
        flags.append("Low similarity index — significant morphological change")

    if delta > 0.5:
        summary = (f"TDS increased by {delta:.2f} pts between visits, indicating morphological "
                   f"progression. Further evaluation is recommended.")
    elif delta < -0.3:
        summary = f"TDS decreased by {abs(delta):.2f} pts — lesion appears more stable than baseline."
    else:
        summary = (f"TDS changed by {delta:+.2f} pts. Lesion appears relatively stable "
                   f"with minor variation across parameters.")

    recs = {
        "HIGH":     ("Current TDS exceeds 5.45. Urgent referral to a board-certified dermatologist "
                     "for dermoscopic evaluation and possible biopsy is strongly advised."),
        "MODERATE": ("TDS falls in the moderate range (4.75–5.45). Schedule follow-up dermoscopy "
                     "within 3 months and monitor closely for further changes."),
        "LOW":      ("TDS is within the low-risk range (<4.75). Continue routine monitoring every "
                     "6–12 months or sooner if new symptoms appear."),
    }
    return {"abcd_baseline": res["baseline"], "abcd_current": res["current"],
            "similarity_index": sim, "delta_tds": delta, "risk_score": rs,
            "risk_level": rl, "change_summary": summary,
            "recommendation": recs[rl], "flags": flags}


# =============================================================================
#  UI HELPERS
# =============================================================================

def gauge(label, value, max_val, color):
    pct = min(value / max_val * 100, 100)
    st.markdown(f"""<div class="gauge-row">
      <div class="gauge-label">{label}</div>
      <div class="gauge-track"><div class="gauge-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
      <div class="gauge-val">{value:.2f}</div></div>""", unsafe_allow_html=True)


def risk_color(level):
    return {"LOW": "#4fffb0", "MODERATE": "#ffd166", "HIGH": "#ff6b6b"}.get(level.upper(), "#4fffb0")


def bgr_to_pil(bgr: np.ndarray):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =============================================================================
#  SESSION STATE
# =============================================================================
for key in ("cap_baseline", "cap_current", "dist_baseline", "dist_current"):
    if key not in st.session_state:
        st.session_state[key] = None


# =============================================================================
#  WEBCAM SECTION
# =============================================================================

st.markdown("## 📷 Webcam Capture")
st.markdown(
    '<div class="dim" style="margin-bottom:1rem">'
    f'Position your camera <strong>{TARGET_CM} cm</strong> from the mole. '
    f'The HUD shows real-time distance. Capture only when the indicator turns <span style="color:#4fffb0">green</span>.'
    '</div>', unsafe_allow_html=True
)

cam_col1, cam_col2 = st.columns(2, gap="large")

for col, slot_key, dist_key, label in [
    (cam_col1, "cap_baseline", "dist_baseline", "Baseline"),
    (cam_col2, "cap_current",  "dist_current",  "Current"),
]:
    with col:
        st.markdown(f'<div class="card"><div class="card-title">📡 {label} — Webcam</div>', unsafe_allow_html=True)

        ctx = webrtc_streamer(
            key=f"webcam_{label.lower()}",
            video_transformer_factory=MoleWebcam,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )

        # Distance indicator beneath video
        if ctx.video_transformer:
            _, dist_cm = ctx.video_transformer.get_snapshot()
            label_txt, col_hex, css, ok = distance_status(dist_cm)
            st.markdown(
                f'<div style="margin:.5rem 0;padding:.5rem .8rem;background:var(--surface2);'
                f'border-radius:8px;font-family:\'DM Mono\',monospace;font-size:.82rem;">'
                f'<span class="{css}">{label_txt}</span></div>',
                unsafe_allow_html=True
            )

            if st.button(f"📸  Capture {label}", key=f"btn_{label.lower()}",
                         disabled=not ok):
                frame, dist = ctx.video_transformer.get_snapshot()
                if frame is not None:
                    st.session_state[slot_key] = frame
                    st.session_state[dist_key] = dist
                    st.success(f"{label} captured at {dist:.1f} cm ✅")

        # Preview captured frame
        if st.session_state[slot_key] is not None:
            st.markdown('<div class="rep-section">Captured Preview</div>', unsafe_allow_html=True)
            st.image(bgr_to_pil(st.session_state[slot_key]),
                     caption=f"{label} — {st.session_state[dist_key]:.1f} cm",
                     use_container_width=True)
            if st.button(f"🗑  Clear {label}", key=f"clear_{label.lower()}"):
                st.session_state[slot_key] = None
                st.session_state[dist_key] = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  UPLOAD FALLBACK
# =============================================================================

st.markdown("---")
st.markdown("## 📂 Or Upload Images")
st.markdown('<div class="dim" style="margin-bottom:1rem">Use uploads instead of (or to replace) webcam captures.</div>', unsafe_allow_html=True)

up_col1, up_col2 = st.columns(2, gap="large")
uploaded = {}
for col, slot_key, label in [
    (up_col1, "cap_baseline", "Baseline"),
    (up_col2, "cap_current",  "Current"),
]:
    with col:
        st.markdown(f'<div class="card"><div class="card-title">📂 {label} Upload</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="upload-hint">Upload {label.lower()} mole photo</div>', unsafe_allow_html=True)
        f = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                             key=f"up_{label.lower()}", label_visibility="collapsed")
        if f:
            bgr = load_cv(f)
            st.session_state[slot_key] = bgr
            st.image(bgr_to_pil(bgr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
#  ANALYSIS
# =============================================================================

st.markdown("---")

bgr1 = st.session_state.get("cap_baseline")
bgr2 = st.session_state.get("cap_current")
can_analyse = bgr1 is not None and bgr2 is not None

if not can_analyse:
    st.info("Capture or upload **both** images (Baseline + Current) to enable analysis.", icon="🔬")
else:
    if st.button("🔬  Run ABCD Analysis & Generate Report"):
        with st.spinner("Segmenting lesion and computing ABCD metrics…"):
            try:
                data = analyse_pair(bgr1, bgr2)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        st.markdown("---")
        st.markdown("## 📋 Analysis Report")

        rl  = data["risk_level"]
        rs  = data["risk_score"]
        sim = data["similarity_index"]
        rc  = risk_color(rl)

        # ── Risk banner ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="card" style="border-color:{rc}33">
          <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
            <div><div class="card-title">Overall Risk</div>
              <span class="risk-badge risk-{rl}">{rl}</span>
            </div>
            <div style="flex:1;min-width:220px"><div class="card-title">Scores</div>
        """, unsafe_allow_html=True)
        gauge("Risk Score (0–10)", rs, 10.0, rc)
        gauge("Similarity Index (0–100)", sim, 100.0, "#4fffb0")
        delta     = data["delta_tds"]
        delta_col = "#ff6b6b" if delta > 0 else "#4fffb0"
        d1 = st.session_state.get("dist_baseline")
        d2 = st.session_state.get("dist_current")
        if d1 and d2:
            st.markdown(
                f'<div class="dim" style="margin-top:.4rem">'
                f'Capture distances — Baseline: <b>{d1:.1f} cm</b> · Current: <b>{d2:.1f} cm</b></div>',
                unsafe_allow_html=True)
        st.markdown(
            f'<div class="dim">TDS Δ: <span style="color:{delta_col};'
            f'font-family:\'DM Mono\',monospace">{delta:+.3f}</span></div>',
            unsafe_allow_html=True)
        st.markdown("</div></div></div>", unsafe_allow_html=True)

        # ── ABCD cards ────────────────────────────────────────────────────────
        ab1, ab2 = st.columns(2, gap="medium")
        for col, key, title in [(ab1, "abcd_baseline", "Baseline ABCD"),
                                (ab2, "abcd_current",  "Current ABCD")]:
            abcd = data[key]; t = abcd["tds"]
            with col:
                st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
                colors = {"asymmetry":"#7eb8f7","border":"#b07ef7","color":"#ffd166","diameter":"#4fffb0"}
                labels = {"asymmetry":"Asymmetry (0–2)","border":"Border (0–8)",
                          "color":"Color (1–6)","diameter":"Diameter (0–5)"}
                maxes  = {"asymmetry":2,"border":8,"color":6,"diameter":5}
                for k in ["asymmetry","border","color","diameter"]:
                    gauge(labels[k], abcd[k], maxes[k], colors[k])
                tc = "#4fffb0" if t < 4.75 else ("#ffd166" if t < 5.45 else "#ff6b6b")
                st.markdown(f"""
                <div style="margin-top:.7rem;padding:.55rem .8rem;background:var(--surface2);
                     border-radius:8px;display:flex;justify-content:space-between;align-items:center">
                  <span style="font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted)">TDS</span>
                  <span style="font-family:'DM Serif Display',serif;font-size:1.35rem;color:{tc}">{t:.3f}</span>
                </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Summary / flags / recommendation ──────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="rep-section">Change Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box">{data["change_summary"]}</div>', unsafe_allow_html=True)
        if data["flags"]:
            st.markdown('<div class="rep-section">⚠ Clinical Flags</div>', unsafe_allow_html=True)
            for f in data["flags"]:
                st.markdown(f"• {f}")
        st.markdown('<div class="rep-section">Clinical Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box" style="border-left:3px solid {rc}">{data["recommendation"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Comparative images ────────────────────────────────────────────────
        st.markdown('<div class="rep-section">Comparative Images &amp; Segmentation</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1: st.image(bgr_to_pil(bgr1), caption="Baseline (original)", use_container_width=True)
        with c2: st.image(overlay_mask(bgr1, data["abcd_baseline"]["mask"]), caption="Baseline (segmented)", use_container_width=True)
        with c3: st.image(bgr_to_pil(bgr2), caption="Current (original)", use_container_width=True)
        with c4: st.image(overlay_mask(bgr2, data["abcd_current"]["mask"]), caption="Current (segmented)", use_container_width=True)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-top:2rem;padding:1rem 1.2rem;background:var(--surface2);border-radius:10px;
             border-left:3px solid #6b7080;font-size:.78rem;color:var(--muted);line-height:1.6">
        ⚕ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
        It does not constitute medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for clinical evaluation of skin lesions.
        </div>""", unsafe_allow_html=True)
