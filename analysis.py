"""
analysis.py — Mole segmentation + ABCD analysis + MobileNetV2 prediction + safe loading
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
1. Pre-processing: resize, CLAHE, bilateral filter, hair removal
2. Multi-cue segmentation + GrabCut refinement
3. ABCD metrics computation
4. MobileNetV2 prediction (melanoma vs benign)
5. Combined report
"""

import cv2
import numpy as np
import os

# ── Config ─────────────────────────────────────────────
WORK_SIZE  = 512
MIN_FILL   = 0.01
MAX_FILL   = 0.80
GC_ITERS   = 8
BORDER_PAD = 12
IMG_SIZE   = 224  # MobileNetV2 input

# ── Safe TensorFlow Import ─────────────────────────────
MODEL_PATH = "skin_cancer_mobilenetv2.h5"

model = None
TF_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        TF_AVAILABLE = True
        print("✅ Model loaded successfully")
    else:
        print(f"⚠️ Model file not found: {MODEL_PATH}")

except Exception as e:
    print("⚠️ TensorFlow/model not available:", e)
    model = None

# ── Pre-processing ─────────────────────────────────────
def _resize_pad(bgr):
    h, w = bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), scale

def _enhance(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    bgr_enh = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(bgr_enh, d=9, sigmaColor=75, sigmaSpace=75)

def _inpaint_hair(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3,3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# ── Multi-cue segmentation ─────────────────────────────
# ... (keep all your cue functions here: _cue_lab, _cue_hsv, _cue_excess_red, _cue_sat_drop, _centre_prior)
# ... (keep _ensemble_mask, _clean_mask, _grabcut_refine, segment_lesion as is)

# ── ABCD metrics ─────────────────────────────────────
# ... (keep compute_asymmetry, compute_border, compute_color, compute_diameter, tds, risk_from_tds as is)

# ── MobileNetV2 prediction ─────────────────────────────
def predict_mole(bgr):
    if not TF_AVAILABLE or model is None:
        return "model_not_loaded", 0.0

    try:
        img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        pred = float(model.predict(img, verbose=0)[0][0])
        label = "melanoma" if pred > 0.5 else "benign"
        return label, round(pred, 3)
    except Exception as e:
        print("Prediction error:", e)
        return "error", 0.0

# ── Full analysis ───────────────────────────────────────
def analyse_pair(bgr1, bgr2):
    mask1 = segment_lesion(bgr1)
    mask2 = segment_lesion(bgr2)

    a1 = compute_asymmetry(mask1)
    b1 = compute_border(mask1)
    c1, flags1 = compute_color(bgr1, mask1)
    d1 = compute_diameter(mask1)
    t1 = tds(a1,b1,c1,d1)
    rs1, rl1 = risk_from_tds(t1)

    a2 = compute_asymmetry(mask2)
    b2 = compute_border(mask2)
    c2, flags2 = compute_color(bgr2, mask2)
    d2 = compute_diameter(mask2)
    t2 = tds(a2,b2,c2,d2)
    rs2, rl2 = risk_from_tds(t2)

    label1, conf1 = predict_mole(bgr1)
    label2, conf2 = predict_mole(bgr2)

    delta_tds = t2-t1

    report = {
        "abcd_baseline": {"asymmetry":a1,"border":b1,"color":c1,"color_flags":flags1,"diameter":d1,"tds":t1,"risk_score":rs1,"risk_level":rl1},
        "abcd_current":  {"asymmetry":a2,"border":b2,"color":c2,"color_flags":flags2,"diameter":d2,"tds":t2,"risk_score":rs2,"risk_level":rl2},
        "ml_prediction": {"baseline":{"label":label1,"confidence":conf1},
                          "current":{"label":label2,"confidence":conf2}},
        "delta_tds": delta_tds
    }

    return report
