"""styles.py — shared CSS injected on every page."""

CSS = """
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
.block-container{padding:2rem 2rem 5rem;max-width:1340px;}
h1,h2,h3{font-family:'Fraunces',serif;}

/* NAV BAR */
.top-nav{display:flex;align-items:center;justify-content:space-between;
  padding:.8rem 1.5rem;background:var(--surface);border-bottom:1px solid var(--border);
  margin-bottom:2rem;border-radius:0 0 14px 14px;}
.nav-brand{font-family:'Fraunces',serif;font-size:1.3rem;font-weight:700;
  background:linear-gradient(135deg,#5eead4,#818cf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.nav-patient{font-family:'DM Mono',monospace;font-size:.75rem;color:var(--muted);}
.nav-pid{color:var(--accent);font-weight:500;}

/* CARDS */
.card{background:var(--surface);border:1px solid var(--border);border-radius:18px;
  padding:1.5rem;margin-bottom:1.2rem;position:relative;overflow:hidden;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(94,234,212,.15),transparent);}
.card-label{font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--accent);margin-bottom:1rem;}

/* HERO */
.hero{text-align:center;padding:3.5rem 1rem 3rem;
  background:radial-gradient(ellipse 70% 50% at 50% 0%,rgba(94,234,212,.07),transparent 70%);
  border-bottom:1px solid var(--border);margin-bottom:2.5rem;}
.hero-title{font-family:'Fraunces',serif;font-size:clamp(2.4rem,5vw,4rem);font-weight:700;
  background:linear-gradient(135deg,#dde1f0 20%,#5eead4 60%,#818cf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:-.03em;margin:0 0 .5rem;}
.hero-sub{color:var(--muted);font-size:.9rem;font-weight:300;letter-spacing:.08em;text-transform:uppercase;}
.hero-badge{display:inline-block;margin-top:.8rem;padding:.3rem 1rem;border-radius:99px;
  background:rgba(94,234,212,.08);border:1px solid rgba(94,234,212,.2);
  font-size:.78rem;font-family:'DM Mono',monospace;color:var(--accent);letter-spacing:.06em;}
.greet{font-family:'Fraunces',serif;font-size:1.5rem;color:var(--accent);
  margin-top:.6rem;font-style:italic;}

/* FEATURE GRID (home page) */
.feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin:2rem 0;}
.feat-card{background:var(--surface2);border:1px solid var(--border);border-radius:14px;
  padding:1.3rem;text-align:center;}
.feat-icon{font-size:2rem;margin-bottom:.5rem;}
.feat-title{font-weight:600;font-size:.95rem;margin-bottom:.3rem;}
.feat-desc{font-size:.8rem;color:var(--muted);line-height:1.5;}

/* RISK */
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
.gauge-fill{height:100%;border-radius:99px;}

/* SCORES */
.score-ring{text-align:center;padding:.8rem;}
.score-num{font-family:'Fraunces',serif;font-size:3.2rem;font-weight:700;line-height:1;}
.score-label{font-size:.72rem;font-family:'DM Mono',monospace;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);margin-top:.3rem;}

/* PILLS */
.pill-row{display:flex;flex-wrap:wrap;gap:.5rem;margin:.6rem 0;}
.pill{padding:.25rem .75rem;border-radius:999px;font-size:.76rem;font-family:'DM Mono',monospace;border:1px solid;}
.pill-ok{background:rgba(52,211,153,.08);color:var(--low);border-color:rgba(52,211,153,.2);}
.pill-warn{background:rgba(251,191,36,.08);color:var(--mod);border-color:rgba(251,191,36,.2);}
.pill-bad{background:rgba(248,113,113,.08);color:var(--high);border-color:rgba(248,113,113,.2);}

/* EXPLAIN / FLAGS */
.explain{background:var(--surface2);border-radius:10px;padding:.9rem 1.1rem;
  font-size:.85rem;line-height:1.7;color:#a8b0c8;margin:.4rem 0;}
.explain strong{color:var(--text);}
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

/* SECTION HEAD */
.section-head{display:flex;align-items:center;gap:.8rem;margin:2rem 0 1rem;}
.section-head-line{flex:1;height:1px;background:var(--border);}
.section-head-txt{font-family:'Fraunces',serif;font-size:1.1rem;color:var(--text);white-space:nowrap;}

/* TABLE */
.cmp-table{width:100%;border-collapse:collapse;font-size:.84rem;}
.cmp-table th{text-align:left;font-family:'DM Mono',monospace;font-size:.65rem;
  letter-spacing:.15em;text-transform:uppercase;color:var(--muted);
  padding:.5rem .7rem;border-bottom:1px solid var(--border);}
.cmp-table td{padding:.5rem .7rem;border-bottom:1px solid var(--border);}
.cmp-up{color:var(--high);}
.cmp-down{color:var(--low);}
.cmp-flat{color:var(--muted);}

/* AUTH CARD */
.auth-card{max-width:420px;margin:3rem auto;}

/* BUTTON */
div.stButton>button{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#07090f;border:none;border-radius:10px;
  font-family:'Outfit',sans-serif;font-weight:600;font-size:.95rem;
  padding:.7rem 2rem;width:100%;cursor:pointer;transition:opacity .2s;}
div.stButton>button:hover{opacity:.85;}
div.stButton>button:disabled{opacity:.35;cursor:not-allowed;}

/* DISCLAIMER */
.disclaimer{margin-top:2.5rem;padding:1.1rem 1.3rem;background:var(--surface);
  border-radius:12px;border-left:3px solid var(--muted);
  font-size:.78rem;color:var(--muted);line-height:1.65;}

/* INPUT DARK STYLE */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select{
  background:var(--surface2) !important;color:var(--text) !important;
  border:1px solid var(--border) !important;border-radius:8px !important;}
</style>
"""


def inject(st):
    st.markdown(CSS, unsafe_allow_html=True)


def nav_bar(st, patient):
    """Render the top nav bar with patient info."""
    st.markdown(f"""
    <div class="top-nav">
      <span class="nav-brand">🔬 DermaScan AI</span>
      <span class="nav-patient">
        Welcome, <strong style="color:var(--text)">{patient['name']}</strong>
        &nbsp;·&nbsp;
        <span class="nav-pid">{patient['patient_id']}</span>
      </span>
    </div>""", unsafe_allow_html=True)
