import streamlit as st
import matplotlib.pyplot as plt
from analysis import analyse_pair
import cv2
import numpy as np

# Example
if "history" not in st.session_state:
    st.session_state.history = []

uploaded1 = st.file_uploader("Upload Baseline Mole Image", type=["jpg","png"])
uploaded2 = st.file_uploader("Upload Current Mole Image", type=["jpg","png"])

if uploaded1 and uploaded2:
    img1 = cv2.imdecode(np.frombuffer(uploaded1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(uploaded2.read(), np.uint8), cv2.IMREAD_COLOR)

    report = analyse_pair(img1, img2)
    st.write(report)

    # Add to history
    st.session_state.history.append(report["delta_tds"])

    # Plot tracking graph
    if len(st.session_state.history) > 1:
        plt.figure(figsize=(6,3))
        plt.plot(st.session_state.history, marker='o')
        plt.title("Mole TDS Tracking Over Time")
        plt.xlabel("Scan #")
        plt.ylabel("Delta TDS")
        st.pyplot(plt)
