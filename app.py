"""
DermaScan AI — app.py
Login & Registration entry point.
After login, patient is routed via st.navigation to Home / Scanner / History.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import db
import styles

import streamlit as st
import analysis as ana

# Upload baseline and current images
baseline = st.file_uploader("Upload baseline image", type=["jpg","png"])
current  = st.file_uploader("Upload current image", type=["jpg","png"])

if baseline and current:
    bgr1 = cv2.imdecode(np.frombuffer(baseline.read(), np.uint8), cv2.IMREAD_COLOR)
    bgr2 = cv2.imdecode(np.frombuffer(current.read(), np.uint8), cv2.IMREAD_COLOR)
    report = ana.analyse_pair(bgr1, bgr2)
    st.json(report)

    # Example tracking graph (use dummy dates / TDS values)
    dates = ["Day 1","Day 15","Day 30"]
    tds_scores = [report["abcd_baseline"]["tds"], report["abcd_current"]["tds"], report["abcd_current"]["tds"]+0.1]
    fig_path = ana.plot_tracking(dates, tds_scores)
    st.image(fig_path, caption="Mole TDS Tracking")

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
styles.inject(st)

# ── Session state defaults ────────────────────────────────────────────────────
for k, v in [("patient", None), ("auth_tab", "login")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── If already logged in → navigate ──────────────────────────────────────────
if st.session_state["patient"]:
    patient = st.session_state["patient"]

    home_page    = st.Page("pages/home.py",    title="Home",           icon="🏠")
    scanner_page = st.Page("pages/scanner.py", title="Run Scanner",    icon="🔬")
    history_page = st.Page("pages/history.py", title="Report History", icon="📜")

    nav = st.navigation([home_page, scanner_page, history_page])
    nav.run()
    st.stop()

# ── Auth page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">DermaScan AI</div>
  <div class="hero-sub">AI-Powered Mole Tracking · ABCD Analysis · Similarity Index</div>
  <div class="hero-badge">🔬 Powered by OpenCV · SQLite · Streamlit</div>
</div>
""", unsafe_allow_html=True)

col_l, col_m, col_r = st.columns([1, 1.4, 1])
with col_m:
    tab_login, tab_reg = st.tabs(["🔑  Sign In", "✨  Create Account"])

    # ── LOGIN ─────────────────────────────────────────────────────────────────
    with tab_login:
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        pid_in  = st.text_input("Patient ID", placeholder="DS-XXXX", key="li_pid")
        pass_in = st.text_input("Password",   type="password",        key="li_pass")

        if st.button("Sign In", key="btn_login"):
            if not pid_in or not pass_in:
                st.error("Please enter both Patient ID and password.")
            else:
                row = db.login_patient(pid_in, pass_in)
                if row:
                    st.session_state["patient"] = dict(row)
                    st.rerun()
                else:
                    st.error("Invalid Patient ID or password.")

        st.markdown("""
        <div style="margin-top:1.2rem;padding:.8rem;background:var(--surface2);
          border-radius:10px;font-size:.78rem;color:var(--muted);text-align:center">
          Don't have an account? Switch to <strong>Create Account</strong> above.
        </div>""", unsafe_allow_html=True)

    # ── REGISTER ──────────────────────────────────────────────────────────────
    with tab_reg:
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        reg_name  = st.text_input("Full Name",        placeholder="e.g. Amara Silva",  key="reg_name")
        reg_email = st.text_input("Email (optional)", placeholder="you@email.com",      key="reg_email")
        reg_dob   = st.text_input("Date of Birth (optional)", placeholder="YYYY-MM-DD", key="reg_dob")
        reg_pass  = st.text_input("Password",         type="password",                  key="reg_pass")
        reg_pass2 = st.text_input("Confirm Password", type="password",                  key="reg_pass2")

        if st.button("Create Account", key="btn_register"):
            if not reg_name or not reg_pass:
                st.error("Name and password are required.")
            elif reg_pass != reg_pass2:
                st.error("Passwords do not match.")
            elif len(reg_pass) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                try:
                    pid = db.register_patient(reg_name, reg_pass, reg_email, reg_dob)
                    row = db.login_patient(pid, reg_pass)
                    st.session_state["patient"] = dict(row)
                    st.success(f"Account created! Your Patient ID is **{pid}** — save this to log in next time.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Registration failed: {e}")

        st.markdown("""
        <div style="margin-top:1.2rem;padding:.9rem;background:var(--surface2);
          border-radius:10px;font-size:.78rem;color:var(--muted)">
          <strong style="color:var(--accent)">📌 Save your Patient ID!</strong><br>
          After registration, you'll receive a unique Patient ID like <code>DS-4F2A</code>.
          Write it down — you'll need it to log in from any device.
        </div>""", unsafe_allow_html=True)
