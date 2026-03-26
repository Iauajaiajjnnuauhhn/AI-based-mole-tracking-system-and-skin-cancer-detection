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

# ── Auth check ────────────────────────────────────────────────────────────────
patient = st.session_state.get("patient")
if not patient:
    st.switch_page("app.py")
    st.stop()

styles.nav_bar(st, patient)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("cap_baseline", None),
    ("cap_current", None),
    ("camera_permitted", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_camera_image(f) -> np.ndarray:
    raw = np.frombuffer(f.getvalue(), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def load_uploaded(f) -> np.ndarray:
    f.seek(0)
    raw = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    f.seek(0)
    return img


def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pill(text, kind="ok"):
    return f'<span class="pill pill-{kind}">{text}</span>'


def risk_col(lvl):
    return {"LOW": "#34d399", "MODERATE": "#fbbf24", "HIGH": "#f87171"}.get(
        lvl, "#34d399"
    )


def gauge(label, hint, value, max_val, color):
    pct = min(value / max_val * 100, 100)
    st.markdown(
        f"""<div class="gauge-wrap">
      <div class="gauge-header">
        <span class="gauge-name">{label}</span>
        <span class="gauge-hint">{hint}</span>
        <span class="gauge-val">{value:.2f}/{max_val:.0f}</span>
      </div>
      <div class="gauge-track"><div class="gauge-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
    </div>""",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="margin-bottom:1.5rem">
  <div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:700">🔬 Run a Scan</div>
  <div style="color:var(--muted);font-size:.88rem">Capture or upload baseline + current photos, then run analysis.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Camera permission ─────────────────────────────────────────────────────────
if not st.session_state["camera_permitted"]:
    st.markdown(
        """
    <div class="card" style="max-width:520px;margin:1rem auto;text-align:center">
      <div style="font-size:2.5rem;margin-bottom:.6rem">📷</div>
      <div class="card-label">Camera Access</div>
      <div class="explain">
        Allow camera to capture mole images or use upload below.
      </div>
    </div>
""",
        unsafe_allow_html=True,
    )

    if st.button("✅ Allow Camera Access"):
        st.session_state["camera_permitted"] = True
        st.rerun()

# ── Step 1: Capture ───────────────────────────────────────────────────────────
st.markdown("## 📷 Step 1 — Capture Images")

c1, c2 = st.columns(2)

for col, slot, label in [
    (c1, "cap_baseline", "Baseline"),
    (c2, "cap_current", "Current"),
]:
    with col:
        st.markdown(f"### {label}")

        if st.session_state["camera_permitted"]:
            snap = st.camera_input(f"Take {label} photo")

            if snap is not None:
                st.session_state[slot] = load_camera_image(snap)
                st.success(f"{label} captured")

        if st.session_state[slot] is not None:
            st.image(
                bgr_to_rgb(st.session_state[slot]),
                caption=label,
                use_container_width=True,
            )

            if st.button(f"Clear {label}", key=f"clr_{label}"):
                st.session_state[slot] = None
                st.rerun()

# ── Step 2: Upload ────────────────────────────────────────────────────────────
st.markdown("## 📂 Step 2 — Upload Images")

u1, u2 = st.columns(2)

for col, slot, label in [
    (u1, "cap_baseline", "Baseline"),
    (u2, "cap_current", "Current"),
]:
    with col:
        f = st.file_uploader(
            f"Upload {label}",
            type=["jpg", "png", "jpeg"],
            key=f"up_{label}",
        )

        if f is not None:
            bgr = load_uploaded(f)
            st.session_state[slot] = bgr
            st.image(bgr_to_rgb(bgr), use_container_width=True)

# ── Step 3: Analyse ───────────────────────────────────────────────────────────
st.markdown("## 🧠 Step 3 — Run Analysis")

bgr1 = st.session_state.get("cap_baseline")
bgr2 = st.session_state.get("cap_current")

if bgr1 is None or bgr2 is None:
    st.info("Upload or capture both images")
    st.stop()

if st.button("🔬 Analyse"):
    with st.spinner("Running analysis..."):
        try:
            data = ana.analyse_pair(bgr1, bgr2)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success("Analysis Complete ✅")

    st.write("### Results")
    st.write("Risk Level:", data["risk_level"])
    st.write("Risk Score:", data["risk_score"])
    st.write("Similarity:", data["similarity_index"])

    st.image(ana.overlay_mask(bgr1, data["abcd_baseline"]["mask"]))
    st.image(ana.overlay_mask(bgr2, data["abcd_current"]["mask"]))
