import cv2
import numpy as np

def resize_image(bgr, size=256):
    return cv2.resize(bgr, (size, size))

# ── SIMPLE SEGMENTATION ─────────────────────────
def segment_lesion(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_not(mask)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

# ── ABCD FUNCTIONS ─────────────────────────
def compute_asymmetry(mask):
    h,w = mask.shape
    left = mask[:, :w//2]
    right = cv2.flip(mask[:, w//2:], 1)

    right = cv2.resize(right, (left.shape[1], left.shape[0]))

    diff = np.sum(left != right)
    total = np.sum(left > 0)

    return round(diff / (total + 1e-6), 3)

def compute_border(mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    return round(peri / (area + 1e-6), 3)

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
    if np.mean(v < 80) > 0.05:
        colors.append("Dark")

    return max(1, len(colors)), colors

def compute_diameter(mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    _,_,w,h = cv2.boundingRect(cnt)

    return round(max(w,h)/50, 2)

def tds(a,b,c,d):
    return round(a*1.3 + b*0.1 + c*0.5 + d*0.5, 3)

def risk_from_tds(t):
    if t < 4.75:
        return "LOW"
    elif t <= 5.45:
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
