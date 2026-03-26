import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import sqlite3
from datetime import datetime
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model("model/skin_model.h5")

# ------------------ DATABASE ------------------
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            prediction TEXT,
            confidence REAL,
            asymmetry REAL,
            border REAL,
            color REAL,
            diameter REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------ PREPROCESS ------------------
def preprocess(img):
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    return np.reshape(img, (1,224,224,3))

# ------------------ SEGMENT ------------------
def segment_mole(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# ------------------ ABCD ------------------
def asymmetry(mask):
    h, w = mask.shape
    left = mask[:, :w//2]
    right = cv2.flip(mask[:, w//2:], 1)
    diff = cv2.absdiff(left, right)
    return np.sum(diff)/(h*w)

def border(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    circularity = (4*np.pi*area)/(peri*peri)
    return 1 - circularity

def color_var(image, mask):
    pixels = image[mask > 0]
    if len(pixels) < 10:
        return 1
    kmeans = KMeans(n_clusters=3).fit(pixels)
    return len(set(kmeans.labels_))

def diameter(mask):
    area = np.sum(mask > 0)
    return np.sqrt(4*area/np.pi)

# ------------------ SAVE ------------------
def save_record(pred, conf, A, B, C, D):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO records (date, prediction, confidence, asymmetry, border, color, diameter)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d"),
        pred, conf, A, B, C, D
    ))
    conn.commit()
    conn.close()

# ------------------ LOAD HISTORY ------------------
def load_history():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT date, diameter FROM records")
    data = cursor.fetchall()
    conn.close()
    return data

# ------------------ UI ------------------
st.title("🧠 Skin Cancer Detection & Tracking")
st.write("Upload or capture a mole image")

uploaded = st.file_uploader("Upload Image", type=["jpg","png"])
camera = st.camera_input("Take Photo")

image = None

# ✅ FIX: check Streamlit objects, not NumPy arrays
if uploaded is not None:
    image = Image.open(uploaded)
elif camera is not None:
    image = Image.open(camera)

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)
    img = np.array(image)

    # Prediction
    processed = preprocess(img)
    pred_val = model.predict(processed)[0][0]
    prediction = "⚠️ Suspicious" if pred_val > 0.5 else "✅ Benign"

    # ABCD Analysis
    mask = segment_mole(img)
    A = asymmetry(mask)
    B = border(mask)
    C = color_var(img, mask)
    D = diameter(mask)
    score = (A*1.3) + (B*0.1) + (C*0.5) + (D*0.05)

    # Save record
    save_record(prediction, float(pred_val), A, B, C, D)

    # Display results
    st.subheader("📊 Result")
    st.write(prediction)
    st.write(f"Confidence: {pred_val:.2f}")

    st.subheader("🧪 ABCD Analysis")
    st.write(f"Asymmetry: {A:.2f}")
    st.write(f"Border: {B:.2f}")
    st.write(f"Color: {C}")
    st.write(f"Diameter: {D:.2f}")

    st.subheader("📈 Risk Score")
    st.write(score)
    if score > 6:
        st.error("🚨 High Risk - Consult a doctor")
    elif score > 4:
        st.warning("⚠️ Moderate Risk")
    else:
        st.success("✅ Low Risk")

    st.image(mask, caption="Segmented Mole")

# ------------------ HISTORY GRAPH ------------------
st.subheader("📈 Mole Growth Tracking")
history = load_history()
if history:
    dates = [h[0] for h in history]
    diameters = [h[1] for h in history]
    plt.plot(dates, diameters)
    plt.xticks(rotation=45)
    plt.title("Diameter Over Time")
    st.pyplot(plt)

# ------------------ DISCLAIMER ------------------
st.warning("⚠️ This is not a medical diagnosis. Consult a dermatologist.")
