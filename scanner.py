"""pages/scanner.py — Mole capture + ABCD analysis for logged-in patient."""

import streamlit as st
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db
import styles
import analysis as ana

styles.inject(st)

patient = st.session_state.get("patient")
if not patient:
    st.switch_page("app.py")
    st.stop()

styles.nav_bar(st, patient)

# ── Session state for this page ───────────────────────────────────────────────
for k, v in [("cap_baseline", None), ("cap_current", None), ("camera_permitted", False)]:
    if k not in st.session_state:
        st.session_state[k] = v


def load_camera_image(f) -> np.ndarray:
    raw = np.frombuffer(f.getvalue(), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)

def load_uploaded(f) -> np.ndarray:
    f.seek(0)
    raw = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    f.seek(0)
    return img

def bgr_to_rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
def pill(text, kind="ok"): return f'<span class="pill pill-{kind}">{text}</span>'
def risk_col(lvl): return {"LOW":"#34d399","MODERATE":"#fbbf24","HIGH":"#f87171"}.get(lvl,"#34d399")

def gauge(label, hint, value, max_val, color):
    pct = min(value/max_val*100, 100)
    st.markdown(f"""<div class="gauge-wrap">
      <div class="gauge-header">
        <span class="gauge-name">{label}</span>
        <span class="gauge-hint">{hint}</span>
        <span class="gauge-val">{value:.2f}/{max_val:.0f}</span>
      </div>
      <div class="gauge-track"><div class="gauge-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
    </div>""", unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:700">🔬 Run a Scan</div>
  <div style="color:var(--muted);font-size:.88rem">Capture or upload baseline + current photos, then run analysis.</div>
</div>""", unsafe_allow_html=True)

# ── CAMERA PERMISSION GATE ────────────────────────────────────────────────────
if not st.session_state["camera_permitted"]:
    st.markdown("""
    <div class="card" style="max-width:520px;margin:1rem auto;text-align:center;border-color:rgba(94,234,212,.3)">
      <div style="font-size:2.5rem;margin-bottom:.6rem">📷</div>
      <div class="card-label" style="text-align:center">Camera Access</div>
      <div class="explain" style="text-align:left">
        DermaScan needs camera access to take mole photos.<br><br>
        <strong>Privacy:</strong> Photos are processed locally on this server and stored
        only in your personal patient record. Nothing is shared externally.<br><br>
        You can also skip this and use the <strong>Upload</strong> section below.
      </div>
    </div>""", unsafe_allow_html=True)
    gc1, gc2, gc3 = st.columns([1,2,1])
    with gc2:
        if st.button("✅  Allow Camera Access", key="grant_cam"):
            st.session_state["camera_permitted"] = True
            st.rerun()
    st.markdown("---")

# ── Step 1: Capture ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-head-txt">📷 Step 1 — Capture Images</div>
  <div class="section-head-line"></div>
</div>
<div style="color:var(--muted);font-size:.84rem;margin-bottom:1rem">
  Hold the camera ~10 cm from the mole. The mole should fill most of the frame.
</div>""", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
for col, slot, label in [(c1,"cap_baseline","Baseline"), (c2,"cap_current","Current")]:
    with col:
        st.markdown(f'<div class="card"><div class="card-label">📷 {label}</div>', unsafe_allow_html=True)
        if st.session_state["camera_permitted"]:
            snap = st.camera_input(f"Take {label} photo", key=f"cam_{label.lower()}")
            if snap:
                st.session_state[slot] = load_camera_image(snap)
                st.success(f"{label} captured ✅")
        else:
            st.markdown('<div class="explain" style="text-align:center;color:var(--muted)">🔒 Allow camera above to use live capture</div>', unsafe_allow_html=True)
        if st.session_state[slot] is not None:
            st.markdown('<div style="font-size:.65rem;font-family:\'DM Mono\',monospace;color:var(--accent);letter-spacing:.15em;text-transform:uppercase;margin-top:.5rem">Preview</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(st.session_state[slot]), caption=label, use_container_width=True)
            if st.button(f"🗑 Clear", key=f"clr_{label.lower()}"):
                st.session_state[slot] = None; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Step 2: Upload fallback ───────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-head-txt">📂 Step 2 — Or Upload Images</div>
  <div class="section-head-line"></div>
</div>""", unsafe_allow_html=True)

u1, u2 = st.columns(2, gap="large")
for col, slot, label in [(u1,"cap_baseline","Baseline"), (u2,"cap_current","Current")]:
    with col:
        st.markdown(f'<div class="card"><div class="card-label">📂 {label} Upload</div>', unsafe_allow_html=True)
        f = st.file_uploader(f"Upload {label.lower()} photo", type=["jpg","jpeg","png","webp"], key=f"up_{label.lower()}")
        if f:
            bgr = load_uploaded(f)
            st.session_state[slot] = bgr
            st.image(bgr_to_rgb(bgr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Step 3: Analyse ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-head-txt">🧠 Step 3 — Run Analysis</div>
  <div class="section-head-line"></div>
</div>""", unsafe_allow_html=True)

bgr1 = st.session_state.get("cap_baseline")
bgr2 = st.session_state.get("cap_current")

if not bgr1 or not bgr2:
    st.info("Capture or upload **both** images to enable analysis.", icon="🔬")
    st.stop()

if st.button("🔬  Analyse & Save Report", use_container_width=True):
    with st.spinner("Running GrabCut segmentation and ABCD analysis…"):
        try:
            data = ana.analyse_pair(bgr1, bgr2)
        except Exception as e:
            st.error(f"Analysis failed: {e}"); st.stop()

    try:
        rid = db.save_report(patient["patient_id"], data, bgr1, bgr2, ana.overlay_mask)
        st.toast(f"Report #{rid} saved to your record ✅", icon="💾")
    except Exception as e:
        st.warning(f"Could not save to database: {e}")

    rl   = data["risk_level"]
    rs   = data["risk_score"]
    sim  = data["similarity_index"]
    conf = data["confidence"]
    rc   = risk_col(rl)
    delta = data["delta_tds"]

    st.markdown("---")
    st.markdown(f"## 📋 Report for {patient['name']}")

    # Risk banner
    st.markdown(f"""
    <div class="card" style="border-color:{rc}40">
      <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;margin-bottom:1rem">
        <div>
          <div class="card-label">Overall Risk</div>
          <span class="risk-badge risk-{rl}">{'⚠' if rl=='HIGH' else ('⚡' if rl=='MODERATE' else '✅')} {rl} RISK</span>
        </div>
        <div style="display:flex;gap:2rem;flex-wrap:wrap">
          <div class="score-ring"><div class="score-num" style="color:{rc}">{rs:.1f}</div><div class="score-label">Risk /10</div></div>
          <div class="score-ring"><div class="score-num" style="color:#818cf8">{sim:.0f}%</div><div class="score-label">Similarity</div></div>
          <div class="score-ring"><div class="score-num" style="color:{'#34d399' if conf>.6 else '#fbbf24'}">{int(conf*100)}%</div><div class="score-label">Confidence</div></div>
        </div>
      </div>
      <div class="explain">{data['risk_plain']}</div>
    </div>""", unsafe_allow_html=True)

    # Change summary + flags
    st.markdown('<div class="section-head"><div class="section-head-txt">📊 What Changed?</div><div class="section-head-line"></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explain">{data["change_summary"]}</div>', unsafe_allow_html=True)
    for title, detail in data["flags"]:
        st.markdown(f'<div class="flag-item">⚠️ <div><strong>{title}</strong><br><span style="color:var(--muted)">{detail}</span></div></div>', unsafe_allow_html=True)
    for msg in data["ok_flags"]:
        st.markdown(f'<div class="flag-ok-item">✅ <span>{msg}</span></div>', unsafe_allow_html=True)

    # ABCD cards
    st.markdown('<div class="section-head"><div class="section-head-txt">🧬 ABCD Breakdown</div><div class="section-head-line"></div></div>', unsafe_allow_html=True)
    metric_info = {
        "asymmetry": ("Asymmetry","Lower = more symmetric",2,"#7eb8f7",lambda v:v<0.8,"Looks symmetric","Has some asymmetry"),
        "border":    ("Border","Lower = smoother",8,"#b07ef7",lambda v:v<3,"Fairly smooth","Irregular edges"),
        "color":     ("Colour","Lower = fewer colours",6,"#fbbf24",lambda v:v<=2,"Normal range","Multiple colours"),
        "diameter":  ("Diameter","Lower = smaller",5,"#34d399",lambda v:v<3,"Normal size","Larger than average"),
    }
    ab1, ab2 = st.columns(2, gap="medium")
    for col, key, title in [(ab1,"abcd_baseline","Baseline"),(ab2,"abcd_current","Current")]:
        abcd = data[key]; t = abcd["tds"]
        tc = "#34d399" if t<4.75 else ("#fbbf24" if t<5.45 else "#f87171")
        with col:
            st.markdown(f'<div class="card"><div class="card-label">{title}</div>', unsafe_allow_html=True)
            for mk,(lbl,hint,mx,clr,ok_fn,ok_msg,warn_msg) in metric_info.items():
                val = abcd[mk]
                gauge(lbl, hint, val, mx, clr)
                st.markdown(f'<div style="margin:.1rem 0 .45rem">{pill(ok_msg if ok_fn(val) else warn_msg,"ok" if ok_fn(val) else "warn")}</div>', unsafe_allow_html=True)
            st.markdown(f"""<div style="margin-top:.9rem;padding:.6rem .9rem;background:var(--surface2);
              border-radius:10px;display:flex;justify-content:space-between;align-items:center">
              <div><div style="font-family:\'DM Mono\',monospace;font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted)">Total Dermoscopy Score</div>
              <div style="font-size:.72rem;color:var(--muted)">&lt;4.75 Low · 4.75–5.45 Moderate · &gt;5.45 High</div></div>
              <div style="font-family:\'Fraunces\',serif;font-size:2rem;color:{tc}">{t:.3f}</div>
            </div>""", unsafe_allow_html=True)
            cflags = abcd["color_flags"]
            pills_html = "".join([pill(c,"warn") for c in cflags]) if cflags else pill("No colour flags","ok")
            st.markdown(f'<div style="margin-top:.6rem"><div style="font-size:.72rem;color:var(--muted);margin-bottom:.3rem">Colours detected:</div><div class="pill-row">{pills_html}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Action steps
    st.markdown(f'<div class="section-head"><div class="section-head-txt">✅ Next Steps</div><div class="section-head-line"></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card" style="border-color:{rc}40"><div class="card-label">Action Plan — {rl} Risk</div>', unsafe_allow_html=True)
    for i, step in enumerate(data["actions"],1):
        st.markdown(f'<div class="action-step"><span class="step-num">{i:02d}</span><span class="step-txt">{step}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Images
    st.markdown('<div class="section-head"><div class="section-head-txt">🖼 Images & Segmentation</div><div class="section-head-line"></div></div>', unsafe_allow_html=True)
    ic1,ic2,ic3,ic4 = st.columns(4, gap="small")
    with ic1: st.image(bgr_to_rgb(bgr1), caption="Baseline (original)", use_container_width=True)
    with ic2: st.image(ana.overlay_mask(bgr1,data["abcd_baseline"]["mask"]), caption="Baseline (segmented)", use_container_width=True)
    with ic3: st.image(bgr_to_rgb(bgr2), caption="Current (original)", use_container_width=True)
    with ic4: st.image(ana.overlay_mask(bgr2,data["abcd_current"]["mask"]), caption="Current (segmented)", use_container_width=True)

    if conf < 0.55:
        st.warning(f"⚠️ Segmentation confidence is {int(conf*100)}%. Retake the photo with better lighting and a plain background.", icon="🔍")

    st.markdown("""<div class="disclaimer">
    ⚕ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
    It does not constitute medical advice, diagnosis, or treatment.
    Always consult a qualified dermatologist for evaluation of any skin lesion.
    </div>""", unsafe_allow_html=True)
