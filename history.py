"""pages/history.py — Per-patient report history with image viewer and trend graphs."""

import streamlit as st
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db
import styles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

styles.inject(st)

patient = st.session_state.get("patient")
if not patient:
    st.switch_page("app.py")
    st.stop()

styles.nav_bar(st, patient)

st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:700">📜 Report History</div>
  <div style="color:var(--muted);font-size:.88rem">All saved scans for your patient record.</div>
</div>""", unsafe_allow_html=True)

pid      = patient["patient_id"]
reports  = db.load_patient_reports(pid)

if not reports:
    st.info("No reports yet. Go to **Run Scanner** to complete your first analysis.", icon="🔬")
    if st.button("🔬  Start First Scan"):
        st.switch_page("pages/scanner.py")
    st.stop()

# ── Summary stats ─────────────────────────────────────────────────────────────
risk_color = {"LOW":"#34d399","MODERATE":"#fbbf24","HIGH":"#f87171"}
risk_emoji = {"LOW":"✅","MODERATE":"⚡","HIGH":"⚠️"}

total = len(reports)
last  = reports[-1]
high_count = sum(1 for r in reports if r["risk_level"]=="HIGH")
avg_risk   = sum(r["risk_score"] for r in reports) / total

sc1,sc2,sc3,sc4 = st.columns(4, gap="medium")
with sc1:
    st.markdown(f'<div class="card" style="text-align:center"><div class="card-label">Total Scans</div>'
                f'<div style="font-family:\'Fraunces\',serif;font-size:2.5rem;color:var(--accent)">{total}</div></div>',
                unsafe_allow_html=True)
with sc2:
    rc = risk_color[last["risk_level"]]
    st.markdown(f'<div class="card" style="text-align:center;border-color:{rc}33"><div class="card-label">Latest Risk</div>'
                f'<div style="font-size:1.6rem">{risk_emoji[last["risk_level"]]}</div>'
                f'<div style="color:{rc};font-family:\'DM Mono\',monospace">{last["risk_level"]}</div></div>',
                unsafe_allow_html=True)
with sc3:
    st.markdown(f'<div class="card" style="text-align:center"><div class="card-label">Avg Risk Score</div>'
                f'<div style="font-family:\'Fraunces\',serif;font-size:2.5rem;color:var(--accent2)">{avg_risk:.1f}</div>'
                f'<div style="font-size:.72rem;color:var(--muted)">out of 10</div></div>',
                unsafe_allow_html=True)
with sc4:
    hc_col = "#f87171" if high_count > 0 else "#34d399"
    st.markdown(f'<div class="card" style="text-align:center"><div class="card-label">High-Risk Scans</div>'
                f'<div style="font-family:\'Fraunces\',serif;font-size:2.5rem;color:{hc_col}">{high_count}</div></div>',
                unsafe_allow_html=True)

# ── Report table ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-head"><div class="section-head-txt">📋 All Reports</div><div class="section-head-line"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="card"><div class="card-label">Sorted oldest → newest</div>', unsafe_allow_html=True)

hdr = ("<table class='cmp-table'><tr>"
       "<th>#</th><th>Date / Time</th><th>Risk</th><th>Score</th>"
       "<th>TDS</th><th>ΔTDS</th><th>Asym.</th><th>Border</th>"
       "<th>Colour</th><th>Diam.</th><th>Similarity</th><th>Conf.</th></tr>")
rows_html = ""
for r in reports:
    rl = r["risk_level"]
    rc = risk_color.get(rl,"var(--text)")
    dc = "#f87171" if r["delta_tds"]>0.1 else ("#34d399" if r["delta_tds"]<-0.1 else "var(--muted)")
    rows_html += (
        f"<tr>"
        f"<td style='color:var(--muted)'>{r['id']}</td>"
        f"<td style='font-family:\"DM Mono\",monospace;font-size:.76rem'>{r['timestamp']}</td>"
        f"<td style='color:{rc}'>{risk_emoji.get(rl,'')} {rl}</td>"
        f"<td style='color:{rc};font-family:\"DM Mono\",monospace'>{r['risk_score']:.1f}</td>"
        f"<td style='font-family:\"DM Mono\",monospace'>{r['tds_current']:.3f}</td>"
        f"<td style='color:{dc};font-family:\"DM Mono\",monospace'>{r['delta_tds']:+.3f}</td>"
        f"<td>{r['asymmetry_c']:.2f}</td><td>{r['border_c']:.2f}</td>"
        f"<td>{r['color_c']:.0f}</td><td>{r['diameter_c']:.2f}</td>"
        f"<td>{r['similarity']:.1f}%</td><td>{int(r['confidence']*100)}%</td>"
        f"</tr>"
    )
st.markdown(hdr + rows_html + "</table>", unsafe_allow_html=True)

da_col, _ = st.columns([1,4])
with da_col:
    if st.button("🗑  Delete All My Reports", key="del_all"):
        db.delete_all_patient_reports(pid)
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ── Image viewer ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-head"><div class="section-head-txt">🖼 Saved Image Viewer</div><div class="section-head-line"></div></div>', unsafe_allow_html=True)

options = {f"Report #{r['id']} — {r['timestamp']}  [{r['risk_level']}]": r["id"] for r in reversed(reports)}
chosen_label = st.selectbox("Select a report to view:", list(options.keys()))
chosen_id    = options[chosen_label]
chosen_row   = next(r for r in reports if r["id"] == chosen_id)
imgs         = db.load_report_images(chosen_id)

if imgs and imgs["img_baseline"]:
    ic1,ic2,ic3,ic4 = st.columns(4, gap="small")
    with ic1: st.image(db.jpeg_to_rgb(imgs["img_baseline"]),  caption="Baseline (original)",  use_container_width=True)
    with ic2: st.image(db.jpeg_to_rgb(imgs["img_seg_base"]),  caption="Baseline (segmented)", use_container_width=True)
    with ic3: st.image(db.jpeg_to_rgb(imgs["img_current"]),   caption="Current (original)",   use_container_width=True)
    with ic4: st.image(db.jpeg_to_rgb(imgs["img_seg_curr"]),  caption="Current (segmented)",  use_container_width=True)

    rc_c  = risk_color.get(chosen_row["risk_level"],"#34d399")
    cf_b  = json.loads(chosen_row["color_flags_b"] or "[]")
    cf_c  = json.loads(chosen_row["color_flags_c"] or "[]")
    delta_c = "#f87171" if chosen_row["delta_tds"]>0.1 else ("#34d399" if chosen_row["delta_tds"]<-0.1 else "var(--muted)")
    st.markdown(f"""
    <div class="card" style="margin-top:.8rem;border-color:{rc_c}33">
      <div class="card-label">Report #{chosen_row['id']} — {chosen_row['timestamp']}</div>
      <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:.8rem">
        <div><span style="color:var(--muted);font-size:.72rem">Risk</span><br>
          <span style="color:{rc_c};font-family:'DM Mono',monospace">{chosen_row['risk_level']}</span></div>
        <div><span style="color:var(--muted);font-size:.72rem">Score</span><br>
          <span style="font-family:'Fraunces',serif;font-size:1.4rem;color:{rc_c}">{chosen_row['risk_score']:.1f}</span></div>
        <div><span style="color:var(--muted);font-size:.72rem">TDS</span><br>
          <span style="font-family:'DM Mono',monospace">{chosen_row['tds_current']:.3f}</span></div>
        <div><span style="color:var(--muted);font-size:.72rem">ΔTDS</span><br>
          <span style="font-family:'DM Mono',monospace;color:{delta_c}">{chosen_row['delta_tds']:+.3f}</span></div>
        <div><span style="color:var(--muted);font-size:.72rem">Similarity</span><br>
          <span style="font-family:'DM Mono',monospace">{chosen_row['similarity']:.1f}%</span></div>
        <div><span style="color:var(--muted);font-size:.72rem">Confidence</span><br>
          <span style="font-family:'DM Mono',monospace">{int(chosen_row['confidence']*100)}%</span></div>
      </div>
      <div class="explain">{chosen_row['change_summary']}</div>
      <div style="margin-top:.5rem;font-size:.78rem;color:var(--muted)">
        Baseline colours: {", ".join(cf_b) if cf_b else "none"} &nbsp;·&nbsp;
        Current colours: {", ".join(cf_c) if cf_c else "none"}
      </div>
    </div>""", unsafe_allow_html=True)

    d_col, _ = st.columns([1,4])
    with d_col:
        if st.button(f"🗑  Delete Report #{chosen_id}", key=f"del_{chosen_id}"):
            db.delete_report(chosen_id)
            st.rerun()
else:
    st.info("No images stored for this report.")

# ── Trend graphs ──────────────────────────────────────────────────────────────
if len(reports) < 2:
    st.info("Run at least **2 scans** to see trend graphs.", icon="📈")
    st.stop()

st.markdown('<div class="section-head"><div class="section-head-txt">📈 Trends Over Time</div><div class="section-head-line"></div></div>', unsafe_allow_html=True)

BG=  "#0f1219"; SURFACE="#161b26"; BORDER="#1f2535"; TEXT="#dde1f0"
MUTED="#5a6175"; ACCENT="#5eead4"; ACCENT2="#818cf8"
LOW_C="#34d399"; MOD_C="#fbbf24"; HIGH_C="#f87171"

xs        = range(len(reports))
short_lbl = [f"#{r['id']}" for r in reports]
pt_colors = [{"LOW":LOW_C,"MODERATE":MOD_C,"HIGH":HIGH_C}.get(r["risk_level"],ACCENT) for r in reports]

def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(short_lbl, color=MUTED, fontsize=8)

risk_scores = [r["risk_score"]  for r in reports]
tds_vals    = [r["tds_current"] for r in reports]
sim_vals    = [r["similarity"]  for r in reports]
delta_vals  = [r["delta_tds"]   for r in reports]

# Graph 1 — Risk Score + TDS
fig1,(ax1,ax2) = plt.subplots(1,2,figsize=(12,3.8)); fig1.patch.set_facecolor(BG)
style_ax(ax1); style_ax(ax2)
ax1.plot(xs,risk_scores,color=ACCENT,linewidth=2,alpha=.85)
ax1.fill_between(xs,risk_scores,alpha=.08,color=ACCENT)
for xi,yi,pc in zip(xs,risk_scores,pt_colors): ax1.scatter(xi,yi,color=pc,s=60,zorder=3,edgecolors="none")
ax1.axhspan(0,4.5,alpha=.04,color=LOW_C); ax1.axhspan(4.5,6.8,alpha=.04,color=MOD_C); ax1.axhspan(6.8,10,alpha=.04,color=HIGH_C)
ax1.set_ylim(0,10); ax1.set_title("Risk Score",color=TEXT,fontsize=10,pad=8); ax1.set_ylabel("Score (0–10)",color=MUTED,fontsize=8)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(2))
ax2.plot(xs,tds_vals,color=ACCENT2,linewidth=2,alpha=.85)
ax2.fill_between(xs,tds_vals,alpha=.08,color=ACCENT2)
for xi,yi,pc in zip(xs,tds_vals,pt_colors): ax2.scatter(xi,yi,color=pc,s=60,zorder=3,edgecolors="none")
ax2.axhline(4.75,color=MOD_C,linewidth=1,linestyle="--",alpha=.5,label="Moderate (4.75)")
ax2.axhline(5.45,color=HIGH_C,linewidth=1,linestyle="--",alpha=.5,label="High (5.45)")
ax2.legend(fontsize=7,facecolor=SURFACE,edgecolor=BORDER,labelcolor=MUTED)
ax2.set_title("Total Dermoscopy Score",color=TEXT,fontsize=10,pad=8); ax2.set_ylabel("TDS",color=MUTED,fontsize=8)
fig1.tight_layout(pad=1.5); st.pyplot(fig1,use_container_width=True); plt.close(fig1)

# Graph 2 — ABCD metrics
fig2,axes2 = plt.subplots(1,4,figsize=(14,3.4)); fig2.patch.set_facecolor(BG)
abcd_m = [("asymmetry_c","Asymmetry",2,"#7eb8f7"),("border_c","Border",8,"#b07ef7"),("color_c","Colour",6,MOD_C),("diameter_c","Diameter",5,LOW_C)]
for ax,(field,name,ymax,color) in zip(axes2,abcd_m):
    style_ax(ax); ax.set_ylim(0,ymax); ax.set_title(name,color=TEXT,fontsize=9,pad=6)
    vals=[r[field] for r in reports]
    ax.plot(xs,vals,color=color,linewidth=2,alpha=.85)
    ax.fill_between(xs,vals,alpha=.12,color=color)
    for xi,yi,pc in zip(xs,vals,pt_colors): ax.scatter(xi,yi,color=pc,s=45,zorder=3,edgecolors="none")
fig2.suptitle("ABCD Metrics Across Sessions",color=TEXT,fontsize=11,y=1.02)
fig2.tight_layout(pad=1.2); st.pyplot(fig2,use_container_width=True); plt.close(fig2)

# Graph 3 — Similarity + ΔTDS
fig3,(ax_s,ax_d) = plt.subplots(1,2,figsize=(12,3.4)); fig3.patch.set_facecolor(BG)
style_ax(ax_s); style_ax(ax_d)
ax_s.plot(xs,sim_vals,color=ACCENT,linewidth=2,alpha=.85)
ax_s.fill_between(xs,sim_vals,alpha=.08,color=ACCENT)
ax_s.axhline(80,color=LOW_C,linewidth=1,linestyle="--",alpha=.5,label="High (80%)")
ax_s.axhline(65,color=MOD_C,linewidth=1,linestyle="--",alpha=.5,label="Moderate (65%)")
for xi,yi,pc in zip(xs,sim_vals,pt_colors): ax_s.scatter(xi,yi,color=pc,s=45,zorder=3,edgecolors="none")
ax_s.set_ylim(0,100); ax_s.set_title("Visual Similarity %",color=TEXT,fontsize=10,pad=8)
ax_s.set_ylabel("Similarity %",color=MUTED,fontsize=8)
ax_s.legend(fontsize=7,facecolor=SURFACE,edgecolor=BORDER,labelcolor=MUTED)
bar_cols=[HIGH_C if d>0.1 else (LOW_C if d<-0.1 else MUTED) for d in delta_vals]
ax_d.bar(list(xs),delta_vals,color=bar_cols,alpha=.75,zorder=2)
ax_d.axhline(0,color=TEXT,linewidth=.8,alpha=.4)
ax_d.set_title("TDS Change vs Baseline (ΔTDS)",color=TEXT,fontsize=10,pad=8)
ax_d.set_ylabel("ΔTDS",color=MUTED,fontsize=8); ax_d.tick_params(colors=MUTED)
fig3.tight_layout(pad=1.5); st.pyplot(fig3,use_container_width=True); plt.close(fig3)

legend = "  ".join([f"`#{r['id']}` = {r['timestamp']}" for r in reports])
st.markdown(f'<div style="font-size:.72rem;color:var(--muted);margin-top:.3rem">Session labels: {legend}</div>', unsafe_allow_html=True)
