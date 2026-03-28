import cv2
import numpy as np

def resize_image(bgr, size=256):
    return cv2.resize(bgr, (size, size))

# ── SIMPLE SEGMENTATION ─────────────────────────
def segment_lesion(bgr):
    # Resize for consistency
    img = cv2.resize(bgr, (256, 256))

    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Blur
    blur = cv2.GaussianBlur(enhanced, (7,7), 0)

    # Convert to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Otsu threshold
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Keep largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 255, -1)
        return clean

    return mask

# ── ABCD FUNCTIONS ─────────────────────────
def compute_asymmetry(mask):
    h, w = mask.shape

    flip_h = cv2.flip(mask, 1)
    flip_v = cv2.flip(mask, 0)

    diff_h = np.sum(mask != flip_h)
    diff_v = np.sum(mask != flip_v)

    total = np.sum(mask > 0) + 1e-6

    score = (diff_h + diff_v) / (2 * total)

    return round(score * 2, 3)

def compute_border(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)

    irregularity = 1 - circularity

    return round(irregularity * 10, 3)

def compute_color(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    roi = mask > 0

    if not roi.any():
        return 1, []

    h = hsv[:,:,0][roi]
    s = hsv[:,:,1][roi]
    v = hsv[:,:,2][roi]

    colors = []

    if np.mean(v > 200) > 0.05:
        colors.append("White")
    if np.mean((h < 10) & (s > 100)) > 0.05:
        colors.append("Red")
    if np.mean((h > 10) & (h < 25)) > 0.05:
        colors.append("Brown")
    if np.mean(v < 80) > 0.05:
        colors.append("Dark")

    return len(colors), colors

def compute_diameter(mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)

    diameter = max(w, h)

    return round(diameter / 30, 2)

def tds(a,b,c,d):
    return round(a*1.3 + b*0.1 + c*0.5 + d*0.5, 3)

def risk_from_tds(t):
    if t < 3:
        return "LOW"
    elif t < 6:
        return "MODERATE"
    else:
        return "HIGH"

# ── SAFE ML PLACEHOLDER ─────────────────────────
def predict_mole(bgr):
    return "not_available", 0.0

# ── MAIN ANALYSIS ─────────────────────────
def generate_explanation(risk, delta):
    if risk == "LOW":
        msg = "The mole appears normal. No immediate concern."
    elif risk == "MODERATE":
        msg = "The mole shows some unusual features. Monitoring is recommended."
    else:
        msg = "The mole shows warning signs. Please consult a dermatologist."

    if delta > 0.5:
        msg += " It has changed noticeably over time."
    elif delta < -0.5:
        msg += " It has reduced in severity."
    else:
        msg += " No major change detected."

    return msg


def analyse_pair(img1, img2):

    img1 = resize_image(img1)
    img2 = resize_image(img2)

    mask1 = segment_lesion(img1)
    mask2 = segment_lesion(img2)

    a1 = compute_asymmetry(mask1)
    b1 = compute_border(mask1)
    c1, _ = compute_color(img1, mask1)
    d1 = compute_diameter(mask1)
    t1 = tds(a1,b1,c1,d1)

    a2 = compute_asymmetry(mask2)
    b2 = compute_border(mask2)
    c2, _ = compute_color(img2, mask2)
    d2 = compute_diameter(mask2)
    t2 = tds(a2,b2,c2,d2)

    risk = risk_from_tds(t2)
    delta = round(t2 - t1, 3)

    explanation = generate_explanation(risk, delta)

    return {
        "baseline_tds": t1,
        "current_tds": t2,
        "risk": risk,
        "change": delta,
        "explanation": explanation
    }
