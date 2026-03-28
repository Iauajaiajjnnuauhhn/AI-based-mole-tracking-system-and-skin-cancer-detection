import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from analysis import analyse_pair

st.title("🔬 Mole Scanner")

DATA_FILE = "history.json"

# Load history
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        history = json.load(f)
else:
    history = []

img1_file = st.file_uploader("Upload Baseline Image", type=["jpg","png"])
img2_file = st.file_uploader("Upload Current Image", type=["jpg","png"])

if img1_file and img2_file:

    img1 = cv2.imdecode(np.frombuffer(img1_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_file.read(), np.uint8), cv2.IMREAD_COLOR)

    st.image(img1, caption="Baseline")
    st.image(img2, caption="Current")

    result = analyse_pair(img1, img2)

    # 📊 USER-FRIENDLY REPORT
    st.subheader("🩺 Scan Result")

    st.write(f"**Risk Level:** {result['risk']}")
    st.write(f"**Change Score:** {result['change']}")
    st.info(result["explanation"])

    # 💾 SAVE DATA
    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tds": result["current_tds"],
        "risk": result["risk"]
    }

    history.append(record)

    with open(DATA_FILE, "w") as f:
        json.dump(history, f)

    st.success("✅ Scan saved!")

# 📈 GRAPH
if len(history) > 1:
    st.subheader("📈 Progress Tracking")

    tds_values = [h["tds"] for h in history]

    plt.figure()
    plt.plot(tds_values, marker='o')
    plt.xlabel("Scan Number")
    plt.ylabel("TDS Score")

    st.pyplot(plt)
