import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Mole Tracker", layout="centered")

st.title("🧠 Mole Tracking System")
st.write("Upload two images to track mole changes over time")

# Upload images
image1 = st.file_uploader("Upload FIRST image", type=["jpg", "png"])
image2 = st.file_uploader("Upload SECOND image", type=["jpg", "png"])

def process_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        # draw contour
        cv2.drawContours(img, [c], -1, (0,255,0), 2)

        return area, img, thresh

    return 0, img, thresh

# FIRST IMAGE
if image1:
    img1 = Image.open(image1)
    st.subheader("First Image")
    st.image(img1)

    area1, processed1, thresh1 = process_image(img1)

    st.image(processed1, caption="Detected Mole")
    st.write(f"📏 Mole Area: {area1:.2f}")

# SECOND IMAGE
if image2:
    img2 = Image.open(image2)
    st.subheader("Second Image")
    st.image(img2)

    area2, processed2, thresh2 = process_image(img2)

    st.image(processed2, caption="Detected Mole")
    st.write(f"📏 Mole Area: {area2:.2f}")

    if image1:
        change = area2 - area1
        st.subheader("📊 Comparison Result")

        st.write(f"Change in size: {change:.2f}")

        if change > 100:
            st.error("⚠️ Mole is growing significantly")
        elif change > 20:
            st.warning("⚠️ Slight increase detected")
        else:
            st.success("✅ No major change")
