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

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ Clear All History"):
        history = []
        with open(DATA_FILE, "w") as f:
            json.dump(history, f)
        st.success("History cleared!")

with col2:
    if st.button("❌ Delete Last Scan"):
        if history:
            history.pop()
            with open(DATA_FILE, "w") as f:
                json.dump(history, f)
            st.success("Last scan deleted!")

# Upload
img1_file = st.file_uploader("Upload Baseline Image", type=["jpg", "png"])
img2_file = st.file_uploader("Upload Current Image", type=["jpg", "png"])

if img1_file and img2_file:

    img1 = cv2.imdecode(np.frombuffer(img1_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_file.read(), np.uint8), cv2.IMREAD_COLOR)

    result = analyse_pair(img1, img2)

    st.subheader("🖼️ Image Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="Baseline")
        st.image(result["overlay_baseline"], caption="Detected Area")

    with col2:
        st.image(img2, caption="Current")
        st.image(result["overlay_current"], caption="Detected Area")

    st.subheader("🩺 Result")

    st.write(f"**Risk Level:** {result['risk']}")
    st.write(f"**Change Score:** {result['delta_tds']}")
    st.write(f"**Percent Change:** {result['percent_change']}%")
    st.write(f"**Similarity Index:** {result['similarity_index']}")
    st.info(result["explanation"])

    # Save
    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tds": result["current_tds"]
    }

    history.append(record)

    with open(DATA_FILE, "w") as f:
        json.dump(history, f)

    st.success("Scan saved!")

# Graph
if len(history) > 1:
    st.subheader("📈 Progress Tracking")

    tds = [h["tds"] for h in history]

    plt.figure()
    plt.plot(tds, marker='o')
    plt.xlabel("Scan")
    plt.ylabel("TDS")
    plt.title("Mole Change Over Time")

    st.pyplot(plt)
