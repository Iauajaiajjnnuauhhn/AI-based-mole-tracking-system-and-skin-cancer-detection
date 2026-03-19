import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan AI · Mole Tracker",
    page_icon="🔬",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #13161e;
    --surface2: #1a1e29;
    --border: #252836;
    --accent: #4fffb0;
    --text: #e8eaf2;
    --muted: #6b7080;
    --low: #4fffb0;
    --mod: #ffd166;
    --high: #ff6b6b;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1280px; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; letter-spacing: -0.02em; }

.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(79,255,176,0.08), transparent);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    background: linear-gradient(135deg, #e8eaf2 30%, #4fffb0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
}
.hero-sub { color: var(--muted); font-size: 1.05rem; font-weight: 300; letter-spacing: 0.04em; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
}
.upload-hint {
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
}
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.06em;
}
.risk-LOW      { background: rgba(79,255,176,0.12);  color: var(--low);  border: 1px solid rgba(79,255,176,0.3); }
.risk-MODERATE { background: rgba(255,209,102,0.12); color: var(--mod);  border: 1px solid rgba(255,209,102,0.3); }
.risk-HIGH     { background: rgba(255,107,107,0.14); color: var(--high); border: 1px solid rgba(255,107,107,0.35); }

.gauge-row   { display: flex; align-items: center; gap: 0.8rem; margin: 0.4rem 0; }
.gauge-label { font-size: 0.78rem; color: var(--muted); width: 160px; flex-shrink: 0; }
.gauge-track { flex: 1; height: 6px; background: var(--surface2); border-radius: 99px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 99px; }
.gauge-val   { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: var(--text); width: 40px; text-align: right; }

.rep-section {
    border-left: 2px solid var(--accent);
    padding-left: 0.8rem;
    margin: 1.2rem 0 0.6rem;
    font-size: 0.8rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
}
.rec-box {
    background: var(--surface2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.65;
    color: var(--text);
    margin-top: 0.5rem;
}
.dim { color: var(--muted); font-size: 0.82rem; }

div.stButton > button {
    background: var(--accent);
    color: #0d0f14;
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.7rem 2.2rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">DermaScan AI</div>
  <div class="hero-sub">ABCD Rule · Similarity Index · Longitudinal Risk Tracking — Powered by OpenCV</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  CV UTILITIES
# =============================================================================

def load_cv(uploaded_file) -> np.ndarray:
    """UploadedFile → BGR numpy array."""
    uploaded_file.seek(0)
    raw = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return img


def segment_lesion(bgr: np.ndarray) -> np.ndarray:
    """
    Segment the mole from skin background using LAB colour space + Otsu thresholding.
    Returns a binary mask (uint8, 0/255).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Combine inverted-L (darkness) + a-channel (redness/brownness)
    score = cv2.addWeighted(255 - l, 0.6, a, 0.4, 0)

    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)

    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.uint8(labels == largest) * 255

    return mask


def compute_asymmetry(mask: np.ndarray) -> float:
    """Score 0–2: folds mask on both axes, measures overlap difference ratio."""
    h, w = mask.shape
    scores = []
    for axis in [0, 1]:
        if axis == 0:
            half1 = mask[:h // 2, :]
            half2 = np.flip(mask[h // 2:, :], axis=0)
            half2 = cv2.resize(half2, (half1.shape[1], half1.shape[0]))
        else:
            half1 = mask[:, :w // 2]
            half2 = np.flip(mask[:, w // 2:], axis=1)
            half2 = cv2.resize(half2, (half1.shape[1], half1.shape[0]))
        union = np.logical_or(half1 > 0, half2 > 0).sum()
        diff  = np.logical_xor(half1 > 0, half2 > 0).sum()
        scores.append(diff / union if union > 0 else 0)
    return round(min((scores[0] + scores[1]) / 2 * 4, 2.0), 2)


def compute_border(mask: np.ndarray) -> float:
    """Score 0–8: counts octants with abrupt/irregular borders."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :]

    M  = cv2.moments(mask)
    cx = M['m10'] / M['m00'] if M['m00'] else mask.shape[1] / 2
    cy = M['m01'] / M['m00'] if M['m00'] else mask.shape[0] / 2

    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    sector_scores = []
    for i in range(8):
        lo  = -np.pi + i * (2 * np.pi / 8)
        hi  = lo + (2 * np.pi / 8)
        idx = np.where((angles >= lo) & (angles < hi))[0]
        if len(idx) < 3:
            sector_scores.append(0)
            continue
        r = np.sqrt((pts[idx, 0] - cx) ** 2 + (pts[idx, 1] - cy) ** 2)
        sector_scores.append(1 if np.std(r) / (np.mean(r) + 1e-6) > 0.12 else 0)
    return float(sum(sector_scores))


def compute_color(bgr: np.ndarray, mask: np.ndarray) -> float:
    """Score 1–6: counts distinct colour structures (white/red/light-brown/dark-brown/blue-grey/black)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    roi = mask > 0

    h, s, v = hsv[:, :, 0][roi], hsv[:, :, 1][roi], hsv[:, :, 2][roi]
    l_ch    = lab[:, :, 0][roi]

    flags = 0
    if np.mean(v > 200) > 0.05:                                             flags += 1  # white
    if np.mean((h < 10) & (s > 80)) > 0.03:                                 flags += 1  # red
    if np.mean((h >= 10) & (h < 25) & (s > 50) & (v > 80)) > 0.05:         flags += 1  # light-brown
    if np.mean((h >= 10) & (h < 30) & (v < 100)) > 0.05:                    flags += 1  # dark-brown
    if np.mean((h >= 100) & (h < 140) & (s < 80)) > 0.03:                   flags += 1  # blue-grey
    if np.mean(l_ch < 40) > 0.05:                                            flags += 1  # black

    return float(max(1, flags))


def compute_diameter(mask: np.ndarray, assumed_fov_mm: float = 20.0) -> float:
    """Score 0–5: estimates physical diameter assuming 20 mm image FOV width."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    _, (mw, mh), _ = cv2.minAreaRect(cnt)
    px_diam  = max(mw, mh)
    mm_per_px = assumed_fov_mm / mask.shape[1]
    diam_mm  = px_diam * mm_per_px
    return round(min(diam_mm / 6.0 * 5.0, 5.0), 2)


def total_dermoscopy_score(a, b, c, d) -> float:
    """TDS = A×1.3 + B×0.1 + C×0.5 + D×0.5"""
    return round(a * 1.3 + b * 0.1 + c * 0.5 + d * 0.5, 3)


def risk_from_tds(t: float) -> tuple:
    """Returns (risk_score 0–10, risk_level string)."""
    if t < 4.75:
        level = "LOW"
    elif t <= 5.45:
        level = "MODERATE"
    else:
        level = "HIGH"
    score = round(min(t / 8.0 * 10.0, 10.0), 2)
    return score, level


def similarity_index(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    """
    Structural + histogram similarity (0–100, 100 = identical).
    Combines normalised cross-correlation on grayscale with Bhattacharyya HSV histogram distance.
    """
    SIZE = (256, 256)
    g1 = cv2.cvtColor(cv2.resize(bgr1, SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(cv2.resize(bgr2, SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)

    g1n = (g1 - g1.mean()) / (g1.std() + 1e-6)
    g2n = (g2 - g2.mean()) / (g2.std() + 1e-6)
    ncc_pct = ((float(np.mean(g1n * g2n)) + 1) / 2) * 100

    h1 = cv2.cvtColor(cv2.resize(bgr1, SIZE), cv2.COLOR_BGR2HSV)
    h2 = cv2.cvtColor(cv2.resize(bgr2, SIZE), cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([h1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([h2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    bhatt    = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    hist_pct = (1 - bhatt) * 100

    return round(max(0.0, min(100.0, ncc_pct * 0.5 + hist_pct * 0.5)), 1)


def overlay_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns RGB image with green segmentation overlay."""
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    green   = np.array([0, 220, 120], dtype=np.float32)
    roi     = mask > 0
    overlay[roi] = (overlay[roi].astype(np.float32) * 0.55 + green * 0.45).clip(0, 255).astype(np.uint8)
    return overlay


def analyse_pair(bgr1: np.ndarray, bgr2: np.ndarray) -> dict:
    """Full ABCD + similarity pipeline on a baseline / current image pair."""
    results = {}
    for key, bgr in [("baseline", bgr1), ("current", bgr2)]:
        mask = segment_lesion(bgr)
        a = compute_asymmetry(mask)
        b = compute_border(mask)
        c = compute_color(bgr, mask)
        d = compute_diameter(mask)
        t = total_dermoscopy_score(a, b, c, d)
        results[key] = {"asymmetry": a, "border": b, "color": c,
                        "diameter": d, "tds": t, "mask": mask}

    sim          = similarity_index(bgr1, bgr2)
    delta_tds    = results["current"]["tds"] - results["baseline"]["tds"]
    risk_score, risk_level = risk_from_tds(results["current"]["tds"])

    # Flags
    flags = []
    if results["current"]["asymmetry"] > results["baseline"]["asymmetry"] + 0.3:
        flags.append("Asymmetry has increased since baseline")
    if results["current"]["border"] > results["baseline"]["border"] + 1:
        flags.append("Border irregularity has worsened")
    if results["current"]["color"] > results["baseline"]["color"]:
        flags.append("New colour structures detected in lesion")
    if results["current"]["diameter"] > results["baseline"]["diameter"] + 0.5:
        flags.append("Estimated lesion diameter has grown")
    if sim < 70:
        flags.append("Low similarity index — significant morphological change between visits")

    # Change summary
    if delta_tds > 0.5:
        change_summary = (f"TDS increased by {delta_tds:.2f} points between visits, indicating "
                          f"notable morphological progression. Further evaluation is recommended.")
    elif delta_tds < -0.3:
        change_summary = (f"TDS decreased by {abs(delta_tds):.2f} points — lesion appears more "
                          f"stable compared to baseline.")
    else:
        change_summary = (f"TDS changed by {delta_tds:+.2f} points. The lesion appears relatively "
                          f"stable with minor variation across parameters.")

    # Recommendation
    if risk_level == "HIGH":
        rec = ("Current TDS exceeds 5.45. Urgent referral to a board-certified dermatologist "
               "for dermoscopic evaluation and possible biopsy is strongly advised.")
    elif risk_level == "MODERATE":
        rec = ("TDS falls in the moderate range (4.75–5.45). Schedule a follow-up dermoscopy "
               "within 3 months and monitor closely for further changes.")
    else:
        rec = ("TDS is within the low-risk range (<4.75). Continue routine monitoring every "
               "6–12 months or sooner if new symptoms appear.")

    return {
        "abcd_baseline":    results["baseline"],
        "abcd_current":     results["current"],
        "similarity_index": sim,
        "delta_tds":        delta_tds,
        "risk_score":       risk_score,
        "risk_level":       risk_level,
        "change_summary":   change_summary,
        "recommendation":   rec,
        "flags":            flags,
    }


# =============================================================================
#  UI HELPERS
# =============================================================================

def gauge(label: str, value: float, max_val: float, color: str):
    pct = min(value / max_val * 100, 100)
    st.markdown(f"""
    <div class="gauge-row">
      <div class="gauge-label">{label}</div>
      <div class="gauge-track"><div class="gauge-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
      <div class="gauge-val">{value:.2f}</div>
    </div>""", unsafe_allow_html=True)


def risk_color(level: str) -> str:
    return {"LOW": "#4fffb0", "MODERATE": "#ffd166", "HIGH": "#ff6b6b"}.get(level.upper(), "#4fffb0")


# =============================================================================
#  LAYOUT
# =============================================================================

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card"><div class="card-title">📂 Baseline Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">Upload the <strong>earlier / reference</strong> mole photo</div>', unsafe_allow_html=True)
    img1 = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], key="img1", label_visibility="collapsed")
    if img1:
        st.image(img1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><div class="card-title">📂 Current Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">Upload the <strong>latest / follow-up</strong> mole photo</div>', unsafe_allow_html=True)
    img2 = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], key="img2", label_visibility="collapsed")
    if img2:
        st.image(img2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

can_analyse = img1 is not None and img2 is not None
if not can_analyse:
    st.info("⬆  Upload both images above to enable analysis.", icon="🔬")

if can_analyse:
    if st.button("🔬  Run ABCD Analysis & Generate Report"):
        with st.spinner("Segmenting lesion and computing ABCD metrics with OpenCV…"):
            bgr1 = load_cv(img1)
            bgr2 = load_cv(img2)
            try:
                data = analyse_pair(bgr1, bgr2)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # ── Report ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📋 Analysis Report")

        risk_level = data["risk_level"]
        risk_score = data["risk_score"]
        sim_idx    = data["similarity_index"]
        rc         = risk_color(risk_level)

        # Risk banner
        st.markdown(f"""
        <div class="card" style="border-color:{rc}33">
          <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
            <div>
              <div class="card-title">Overall Risk</div>
              <span class="risk-badge risk-{risk_level}">{risk_level}</span>
            </div>
            <div style="flex:1;min-width:220px">
              <div class="card-title">Scores</div>
        """, unsafe_allow_html=True)
        gauge("Risk Score (0–10)", risk_score, 10.0, rc)
        gauge("Similarity Index (0–100)", sim_idx, 100.0, "#4fffb0")
        delta     = data["delta_tds"]
        delta_col = "#ff6b6b" if delta > 0 else "#4fffb0"
        st.markdown(
            f'<div class="dim" style="margin-top:0.4rem">TDS Δ: '
            f'<span style="color:{delta_col};font-family:\'DM Mono\',monospace">{delta:+.3f}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown("</div></div></div>", unsafe_allow_html=True)

        # ABCD side-by-side
        ab_col1, ab_col2 = st.columns(2, gap="medium")
        for col, key, title in [
            (ab_col1, "abcd_baseline", "Baseline ABCD"),
            (ab_col2, "abcd_current",  "Current ABCD"),
        ]:
            abcd = data[key]
            t    = abcd["tds"]
            with col:
                st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
                colors = {"asymmetry": "#7eb8f7", "border": "#b07ef7",
                          "color": "#ffd166", "diameter": "#4fffb0"}
                labels = {"asymmetry": "Asymmetry (0–2)", "border": "Border (0–8)",
                          "color": "Color (1–6)", "diameter": "Diameter (0–5)"}
                maxes  = {"asymmetry": 2, "border": 8, "color": 6, "diameter": 5}
                for k in ["asymmetry", "border", "color", "diameter"]:
                    gauge(labels[k], abcd[k], maxes[k], colors[k])
                tds_col = "#4fffb0" if t < 4.75 else ("#ffd166" if t < 5.45 else "#ff6b6b")
                st.markdown(f"""
                <div style="margin-top:0.8rem;padding:0.6rem 0.8rem;background:var(--surface2);
                     border-radius:8px;display:flex;justify-content:space-between;align-items:center">
                  <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:var(--muted)">TDS</span>
                  <span style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:{tds_col}">{t:.3f}</span>
                </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Change summary & flags
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="rep-section">Change Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box">{data["change_summary"]}</div>', unsafe_allow_html=True)

        if data["flags"]:
            st.markdown('<div class="rep-section">⚠ Clinical Flags</div>', unsafe_allow_html=True)
            for f in data["flags"]:
                st.markdown(f"• {f}")

        st.markdown('<div class="rep-section">Clinical Recommendation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="rec-box" style="border-left:3px solid {rc}">{data["recommendation"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Comparative images + segmentation overlays
        st.markdown('<div class="rep-section">Comparative Images &amp; Lesion Segmentation</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4, gap="small")
        img1.seek(0); img2.seek(0)
        with c1:
            st.image(img1, caption="Baseline (original)", use_container_width=True)
        with c2:
            st.image(
                overlay_mask(bgr1, data["abcd_baseline"]["mask"]),
                caption="Baseline (segmented)", use_container_width=True
            )
        with c3:
            st.image(img2, caption="Current (original)", use_container_width=True)
        with c4:
            st.image(
                overlay_mask(bgr2, data["abcd_current"]["mask"]),
                caption="Current (segmented)", use_container_width=True
            )

        # Disclaimer
        st.markdown("""
        <div style="margin-top:2rem;padding:1rem 1.2rem;background:var(--surface2);border-radius:10px;
             border-left:3px solid #6b7080;font-size:0.8rem;color:var(--muted);line-height:1.6">
        ⚕ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
        It does not constitute medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for clinical evaluation of skin lesions.
        </div>""", unsafe_allow_html=True)
