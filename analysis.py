import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ── Resize ─────────────────────────
def resize_image(bgr, size=256):
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)

# ── Segmentation ───────────────────
def segment_lesion(bgr):
    img = resize_image(bgr)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (7,7), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 255, -1)
        return clean

    return mask

# ── Overlay ────────────────────────
def create_overlay(image, mask):
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = [0, 255, 0]
    return cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

# ── ABCD ───────────────────────────
def compute_asymmetry(mask):
    flip_h = cv2.flip(mask, 1)
    flip_v = cv2.flip(mask, 0)

    diff_h = np.sum(mask != flip_h)
    diff_v = np.sum(mask != flip_v)

    total = np.sum(mask > 0) + 1e-6
    return round(((diff_h + diff_v) / (2 * total)) * 2, 3)

def compute_border(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
    return round((1 - circularity) * 10, 3)

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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return round(max(w, h) / 30, 2)

def tds(a, b, c, d):
    return round(a*1.3 + b*0.1 + c*0.5 + d*0.5, 3)

def risk_from_tds(t):
    if t < 3:
        return "LOW"
    elif t < 6:
        return "MODERATE"
    else:
        return "HIGH"

# ── Similarity ─────────────────────
def compute_similarity(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
    return round(ssim(g1, g2), 3)

# ── MAIN ───────────────────────────
def analyse_pair(img1, img2):

    img1 = resize_image(img1)
    img2 = resize_image(img2)

    mask1 = segment_lesion(img1)
    mask2 = segment_lesion(img2)

    overlay1 = create_overlay(img1, mask1)
    overlay2 = create_overlay(img2, mask2)

    # Baseline
    t1 = tds(
        compute_asymmetry(mask1),
        compute_border(mask1),
        compute_color(img1, mask1)[0],
        compute_diameter(mask1)
    )

    # Current
    t2 = tds(
        compute_asymmetry(mask2),
        compute_border(mask2),
        compute_color(img2, mask2)[0],
        compute_diameter(mask2)
    )

    delta = round(max(t2 - t1, 0), 3)
    percent = round((delta / (t1 + 1e-6)) * 100, 2)
    similarity = compute_similarity(img1, img2)
    risk = risk_from_tds(t2)

    if delta == 0:
        explanation = "No significant change detected."
    elif delta < 0.5:
        explanation = "Minor change detected. Monitor regularly."
    else:
        explanation = "Significant change detected. Consult a doctor."

    return {
        "delta_tds": delta,
        "percent_change": percent,
        "similarity_index": similarity,
        "current_tds": t2,
        "risk": risk,
        "explanation": explanation,
        "overlay_baseline": overlay1,
        "overlay_current": overlay2
    }
