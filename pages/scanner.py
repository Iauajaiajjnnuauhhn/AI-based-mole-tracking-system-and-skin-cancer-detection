import streamlit as st
import cv2
import numpy as np
from analysis import analyse_pair
import matplotlib.pyplot as plt

st.title("🔬 Mole Scanner")

if "history" not in st.session_state:
    st.session_state.history = []

img1_file = st.file_uploader("Upload Baseline Image", type=["jpg","png"])
img2_file = st.file_uploader("Upload Current Image", type=["jpg","png"])

if img1_file and img2_file:

    img1 = cv2.imdecode(np.frombuffer(img1_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_file.read(), np.uint8), cv2.IMREAD_COLOR)

    st.image(img1, caption="Baseline")
    st.image(img2, caption="Current")

    result = analyse_pair(img1, img2)

    st.subheader("📊 ABCD Report")
    st.write(result)

    # Save tracking
    st.session_state.history.append(result["current"]["tds"])

    # Graph
    if len(st.session_state.history) > 1:
        st.subheader("📈 Tracking Graph")

        plt.figure()
        plt.plot(st.session_state.history, marker='o')
        plt.xlabel("Scan Number")
        plt.ylabel("TDS Score")

        st.pyplot(plt)
