
import streamlit as st
import anthropic
import base64
import json
import re
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────────────────────
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
    --accent2: #ff6b6b;
    --accent3: #ffd166;
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

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1280px; }

h1, h2, h3 { font-family: 'DM Serif Display', serif; letter-spacing: -0.02em; }

/* ── Hero ── */
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
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.04em;
}

/* ── Cards ── */
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

/* ── Upload zones ── */
.upload-hint {
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.06em;
}
.risk-LOW    { background: rgba(79,255,176,0.12); color: var(--low); border: 1px solid rgba(79,255,176,0.3); }
.risk-MODERATE { background: rgba(255,209,102,0.12); color: var(--mod); border: 1px solid rgba(255,209,102,0.3); }
.risk-HIGH   { background: rgba(255,107,107,0.14); color: var(--high); border: 1px solid rgba(255,107,107,0.35); }

/* ── Score gauge bar ── */
.gauge-row { display: flex; align-items: center; gap: 0.8rem; margin: 0.4rem 0; }
.gauge-label { font-size: 0.78rem; color: var(--muted); width: 140px; flex-shrink: 0; }
.gauge-track { flex: 1; height: 6px; background: var(--surface2); border-radius: 99px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 99px; transition: width 0.8s ease; }
.gauge-val   { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: var(--text); width: 36px; text-align: right; }

/* ── Report section headers ── */
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

/* ── Recommendation box ── */
.rec-box {
    background: var(--surface2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.65;
    color: var(--text);
    margin-top: 0.5rem;
}

/* ── Divider ── */
.dim { color: var(--muted); font-size: 0.85rem; }

/* ── Button override ── */
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

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">DermaScan AI</div>
  <div class="hero-sub">ABCD Rule · Similarity Index · Longitudinal Risk Tracking</div>
</div>
""", unsafe_allow_html=True)

# ── Helper: image → base64 ────────────────────────────────────────────────────
def img_to_b64(uploaded_file) -> tuple[str, str]:
    """Return (base64_data, media_type)."""
    raw = uploaded_file.read()
    mime = uploaded_file.type or "image/jpeg"
    return base64.standard_b64encode(raw).decode(), mime

# ── Helper: colour gauge ──────────────────────────────────────────────────────
def gauge(label: str, value: float, max_val: float = 10.0, color: str = "#4fffb0"):
    pct = min(value / max_val * 100, 100)
    st.markdown(f"""
    <div class="gauge-row">
      <div class="gauge-label">{label}</div>
      <div class="gauge-track"><div class="gauge-fill" style="width:{pct}%;background:{color}"></div></div>
      <div class="gauge-val">{value:.1f}</div>
    </div>""", unsafe_allow_html=True)

# ── Helper: risk colour ───────────────────────────────────────────────────────
def risk_color(level: str) -> str:
    return {"LOW": "#4fffb0", "MODERATE": "#ffd166", "HIGH": "#ff6b6b"}.get(level.upper(), "#4fffb0")

# ── Claude analysis ───────────────────────────────────────────────────────────
def analyse_moles(b64_old: str, mime_old: str, b64_new: str, mime_new: str) -> dict:
    """Send both images to Claude and get structured JSON report."""
    client = anthropic.Anthropic()

    system_prompt = """You are an expert dermatology AI assistant specialising in mole/nevus analysis.
You will receive TWO dermoscopy or clinical photographs of the same mole taken at different times
(Image 1 = earlier / baseline, Image 2 = current / follow-up).

Analyse BOTH images using the ABCD rule and return a single JSON object ONLY — no markdown fences,
no preamble, no explanation outside the JSON.

JSON schema:
{
  "abcd_baseline": {
    "asymmetry":  {"score": <0-2, float>, "notes": "<string>"},
    "border":     {"score": <0-8, float>, "notes": "<string>"},
    "color":      {"score": <1-6, float>, "notes": "<string>"},
    "diameter":   {"score": <0-5, float>, "notes": "<string>"},
    "tds":        <float, total dermoscopy score = A*1.3 + B*0.1 + C*0.5 + D*0.5>
  },
  "abcd_current": {
    "asymmetry":  {"score": <0-2, float>, "notes": "<string>"},
    "border":     {"score": <0-8, float>, "notes": "<string>"},
    "color":      {"score": <1-6, float>, "notes": "<string>"},
    "diameter":   {"score": <0-5, float>, "notes": "<string>"},
    "tds":        <float>
  },
  "similarity_index": <0-100, float, 100=identical>,
  "change_summary": "<2-3 sentences describing key morphological changes>",
  "risk_score": <0-10, float>,
  "risk_level": "<LOW|MODERATE|HIGH>",
  "recommendation": "<concrete clinical recommendation>",
  "flags": ["<list of specific concern flags, empty array if none>"]
}

Scoring guidance:
- TDS < 4.75 → LOW risk; 4.75–5.45 → MODERATE; > 5.45 → HIGH
- risk_score maps TDS proportionally: (tds_current / 8) * 10, clamped 0–10
- similarity_index: visual + structural similarity between the two images (100 = no change)
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1800,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1 — Baseline mole photograph:"},
                {"type": "image", "source": {"type": "base64", "media_type": mime_old, "data": b64_old}},
                {"type": "text", "text": "Image 2 — Current mole photograph:"},
                {"type": "image", "source": {"type": "base64", "media_type": mime_new, "data": b64_new}},
                {"type": "text", "text": "Analyse both images and return the JSON report."}
            ]
        }]
    )

    raw = response.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)

# ── UI layout ─────────────────────────────────────────────────────────────────
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

# ── Analyse button ────────────────────────────────────────────────────────────
can_analyse = img1 is not None and img2 is not None
if not can_analyse:
    st.info("⬆  Upload both images above to enable analysis.", icon="🔬")

if can_analyse:
    if st.button("🔬  Run ABCD Analysis & Generate Report"):
        with st.spinner("Analysing mole morphology with Claude AI…"):
            img1.seek(0); img2.seek(0)
            b64_old, mime_old = img_to_b64(img1)
            img1.seek(0)
            b64_new, mime_new = img_to_b64(img2)
            img2.seek(0)
            try:
                data = analyse_moles(b64_old, mime_old, b64_new, mime_new)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # ── Report ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📋 Analysis Report")

        # ── Risk banner ───────────────────────────────────────────────────────
        risk_level = data.get("risk_level", "LOW").upper()
        risk_score = data.get("risk_score", 0.0)
        sim_idx    = data.get("similarity_index", 100.0)

        rc = risk_color(risk_level)
        st.markdown(f"""
        <div class="card" style="border-color:{rc}33">
          <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
            <div>
              <div class="card-title">Overall Risk</div>
              <span class="risk-badge risk-{risk_level}">{risk_level}</span>
            </div>
            <div style="flex:1;min-width:220px">
              <div class="card-title">Risk Score</div>
        """, unsafe_allow_html=True)
        gauge("Risk Score (0–10)", risk_score, 10.0, rc)
        gauge("Similarity Index", sim_idx, 100.0, "#4fffb0")
        st.markdown("</div></div></div>", unsafe_allow_html=True)

        # ── Side-by-side ABCD tables ──────────────────────────────────────────
        ab_col1, ab_col2 = st.columns(2, gap="medium")
        for col, key, title in [
            (ab_col1, "abcd_baseline", "Baseline ABCD"),
            (ab_col2, "abcd_current",  "Current ABCD"),
        ]:
            abcd = data.get(key, {})
            tds  = abcd.get("tds", 0.0)
            with col:
                st.markdown(f'<div class="card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
                colors = {"asymmetry":"#7eb8f7","border":"#b07ef7","color":"#ffd166","diameter":"#4fffb0"}
                labels = {"asymmetry":"Asymmetry (0–2)","border":"Border (0–8)","color":"Color (1–6)","diameter":"Diameter (0–5)"}
                maxes  = {"asymmetry":2,"border":8,"color":6,"diameter":5}
                for k in ["asymmetry","border","color","diameter"]:
                    item = abcd.get(k, {})
                    sc   = item.get("score", 0)
                    note = item.get("notes", "")
                    gauge(labels[k], sc, maxes[k], colors[k])
                    st.markdown(f'<div class="dim" style="margin:-4px 0 6px 150px">{note}</div>', unsafe_allow_html=True)
                tds_col = "#4fffb0" if tds < 4.75 else ("#ffd166" if tds < 5.45 else "#ff6b6b")
                st.markdown(f"""
                <div style="margin-top:0.8rem;padding:0.6rem 0.8rem;background:var(--surface2);
                     border-radius:8px;display:flex;justify-content:space-between;align-items:center">
                  <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:var(--muted)">TDS</span>
                  <span style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:{tds_col}">{tds:.2f}</span>
                </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Change summary & flags ─────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="rep-section">Change Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box">{data.get("change_summary","—")}</div>', unsafe_allow_html=True)

        flags = data.get("flags", [])
        if flags:
            st.markdown('<div class="rep-section">⚠ Clinical Flags</div>', unsafe_allow_html=True)
            for f in flags:
                st.markdown(f"• {f}")

        st.markdown('<div class="rep-section">Clinical Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box" style="border-left:3px solid {rc}">{data.get("recommendation","—")}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Side-by-side images in report ──────────────────────────────────────
        st.markdown('<div class="rep-section">Comparative Images</div>', unsafe_allow_html=True)
        rimg1, rimg2 = st.columns(2, gap="medium")
        with rimg1:
            st.image(img1, caption="Baseline", use_container_width=True)
        with rimg2:
            st.image(img2, caption="Current", use_container_width=True)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-top:2rem;padding:1rem 1.2rem;background:var(--surface2);border-radius:10px;
             border-left:3px solid #6b7080;font-size:0.8rem;color:var(--muted);line-height:1.6">
        ⚕ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
        It does not constitute medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for clinical evaluation of skin lesions.
        </div>""", unsafe_allow_html=True)
