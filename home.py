"""pages/home.py — Patient home dashboard with greeting and stats."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db
import styles

styles.inject(st)

patient = st.session_state.get("patient")
if not patient:
    st.switch_page("app.py")
    st.stop()

styles.nav_bar(st, patient)

# ── Greeting ──────────────────────────────────────────────────────────────────
import datetime
hour = datetime.datetime.now().hour
greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 18 else "Good evening")
first_name = patient["name"].split()[0]

reports = db.load_patient_reports(patient["patient_id"])
n_reports = len(reports)
last_risk  = reports[-1]["risk_level"] if reports else None
last_ts    = reports[-1]["timestamp"]  if reports else None

risk_emoji = {"LOW": "✅", "MODERATE": "⚡", "HIGH": "⚠️"}
risk_color = {"LOW": "#34d399", "MODERATE": "#fbbf24", "HIGH": "#f87171"}

st.markdown(f"""
<div class="hero" style="padding:2.5rem 1rem 2rem;">
  <div class="hero-title">DermaScan AI</div>
  <div class="greet">{greeting}, {first_name} 👋</div>
  <div class="hero-badge">Patient ID: {patient['patient_id']}</div>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
s1, s2, s3 = st.columns(3, gap="medium")

with s1:
    st.markdown(f"""
    <div class="card" style="text-align:center">
      <div class="card-label">Total Reports</div>
      <div style="font-family:'Fraunces',serif;font-size:3rem;color:var(--accent);line-height:1">{n_reports}</div>
      <div style="font-size:.78rem;color:var(--muted);margin-top:.3rem">scans completed</div>
    </div>""", unsafe_allow_html=True)

with s2:
    if last_risk:
        rc = risk_color[last_risk]
        re = risk_emoji[last_risk]
        st.markdown(f"""
        <div class="card" style="text-align:center;border-color:{rc}33">
          <div class="card-label">Latest Risk Level</div>
          <div style="font-size:2rem">{re}</div>
          <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:{rc};margin-top:.3rem">{last_risk}</div>
          <div style="font-size:.72rem;color:var(--muted);margin-top:.2rem">{last_ts}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center">
          <div class="card-label">Latest Risk Level</div>
          <div style="font-size:2rem">—</div>
          <div style="color:var(--muted);font-size:.82rem;margin-top:.4rem">No scans yet</div>
        </div>""", unsafe_allow_html=True)

with s3:
    member_since = patient.get("created_at","")[:10]
    st.markdown(f"""
    <div class="card" style="text-align:center">
      <div class="card-label">Member Since</div>
      <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:var(--accent2);margin-top:.5rem">{member_since}</div>
      <div style="font-size:.78rem;color:var(--muted);margin-top:.3rem">{patient.get('email','') or 'No email set'}</div>
    </div>""", unsafe_allow_html=True)

# ── Quick actions ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <div class="section-head-txt">⚡ Quick Actions</div>
  <div class="section-head-line"></div>
</div>""", unsafe_allow_html=True)

qa1, qa2, qa3 = st.columns(3, gap="medium")
with qa1:
    if st.button("🔬  Start New Scan", key="qa_scan", use_container_width=True):
        st.switch_page("pages/scanner.py")
with qa2:
    if st.button("📜  View Report History", key="qa_hist", use_container_width=True):
        st.switch_page("pages/history.py")
with qa3:
    if st.button("🚪  Sign Out", key="qa_logout", use_container_width=True):
        st.session_state["patient"] = None
        st.session_state["cap_baseline"] = None
        st.session_state["cap_current"]  = None
        st.session_state["camera_permitted"] = False
        st.rerun()

# ── Feature overview ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head" style="margin-top:2rem">
  <div class="section-head-txt">🧬 What DermaScan Does</div>
  <div class="section-head-line"></div>
</div>
<div class="feat-grid">
  <div class="feat-card">
    <div class="feat-icon">📷</div>
    <div class="feat-title">Photo Capture</div>
    <div class="feat-desc">Use your device camera or upload photos. Compare baseline vs current to track changes over time.</div>
  </div>
  <div class="feat-card">
    <div class="feat-icon">🧠</div>
    <div class="feat-title">ABCD Analysis</div>
    <div class="feat-desc">Automated scoring of Asymmetry, Border irregularity, Colour variation and Diameter using OpenCV.</div>
  </div>
  <div class="feat-card">
    <div class="feat-icon">📊</div>
    <div class="feat-title">Trend Graphs</div>
    <div class="feat-desc">Visualise how your mole's risk score and ABCD metrics change across multiple sessions.</div>
  </div>
  <div class="feat-card">
    <div class="feat-icon">💾</div>
    <div class="feat-title">Persistent History</div>
    <div class="feat-desc">Every report and both images are saved securely to your personal record using SQLite.</div>
  </div>
  <div class="feat-card">
    <div class="feat-icon">🔒</div>
    <div class="feat-title">Private & Secure</div>
    <div class="feat-desc">Your Patient ID and password protect your records. Data is stored only on this server.</div>
  </div>
  <div class="feat-card">
    <div class="feat-icon">⚕️</div>
    <div class="feat-title">Plain-English Reports</div>
    <div class="feat-desc">No medical jargon — each metric is explained in simple language with clear action steps.</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚕ <strong>Medical Disclaimer:</strong> DermaScan AI is a research and educational tool only.
It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified
dermatologist for clinical evaluation of any skin lesion.
</div>""", unsafe_allow_html=True)
