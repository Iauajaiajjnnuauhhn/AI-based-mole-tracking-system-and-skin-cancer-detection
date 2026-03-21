"""
analysis.py — High-accuracy mole segmentation + ABCD analysis pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Segmentation pipeline (5 stages):
  1. Pre-processing  — resize to standard canvas, CLAHE enhancement,
                       bilateral denoising, hair artifact inpainting
  2. Multi-cue mask  — LAB (L+a), HSV darkness, Excess-Red index,
                       saturation-drop cue, Gaussian centre prior
  3. Weighted ensemble + Otsu threshold
  4. GrabCut refine  — eroded FG core / dilated BG ring seeds,
                       GC_ITERS iterations for tight boundary
  5. Post-processing — morphological cleanup, largest-blob selection,
                       convex-hull hole-filling, border smoothing,
                       border-leak sanity gate
"""

import cv2
import numpy as np

WORK_SIZE  = 512
MIN_FILL   = 0.01
MAX_FILL   = 0.80
GC_ITERS   = 8
BORDER_PAD = 12

# ── Pre-processing ─────────────────────────────────────────────────────────────

def _resize_pad(bgr):
    h, w  = bgr.shape[:2]
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
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# ── Multi-cue maps ─────────────────────────────────────────────────────────────

def _cue_lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    a = (lab[:, :, 1] - 128) / 127.0
    return np.clip(0.6 * (1.0 - l) + 0.4 * np.clip(a, 0, 1), 0, 1)

def _cue_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return np.clip(1.0 - hsv[:, :, 2] / 255.0, 0, 1)

def _cue_excess_red(bgr):
    f  = bgr.astype(np.float32) / 255.0
    er = 1.4 * f[:, :, 2] - f[:, :, 1] - 0.5 * f[:, :, 0]
    return np.clip(er, 0, 1)

def _cue_sat_drop(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h   = hsv[:, :, 0]
    s   = hsv[:, :, 1] / 255.0
    v   = hsv[:, :, 2] / 255.0
    skin_hue = (h < 35).astype(np.float32)
    return np.clip(skin_hue * (1.0 - s) * (1.0 - v), 0, 1)

def _centre_prior(h, w, sigma=0.45):
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist   = np.sqrt(((yy - cy) / (cy + 1e-6)) ** 2 + ((xx - cx) / (cx + 1e-6)) ** 2)
    return np.exp(-dist ** 2 / (2 * sigma ** 2))

def _ensemble_mask(bgr):
    h, w   = bgr.shape[:2]
    fused  = (0.35 * _cue_lab(bgr) +
              0.25 * _cue_hsv(bgr) +
              0.20 * _cue_excess_red(bgr) +
              0.10 * _cue_sat_drop(bgr) +
              0.10 * _centre_prior(h, w))
    prior  = _centre_prior(h, w)
    fused  = fused * (0.5 + 0.5 * prior)
    u8     = np.clip(fused * 255, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# ── Morphological clean-up ─────────────────────────────────────────────────────

def _clean_mask(mask, h, w):
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_sm    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask    = np.uint8(labels == largest) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
        eroded_hull = cv2.erode(hull_mask, k_open, iterations=3)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(hull_mask, eroded_hull))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_sm, iterations=2)

def _border_leak(mask, margin=5):
    return bool(mask[:margin, :].any() and mask[-margin:, :].any() and
                mask[:, :margin].any() and mask[:, -margin:].any())

# ── GrabCut refinement ─────────────────────────────────────────────────────────

def _grabcut_refine(bgr, coarse):
    h, w    = bgr.shape[:2]
    k_fg    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_bg    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fg_core = cv2.erode(coarse,  k_fg, iterations=3)
    bg_ring = cv2.dilate(coarse, k_bg, iterations=4)
    gc_mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc_mask[bg_ring > 0] = cv2.GC_PR_BGD
    gc_mask[coarse  > 0] = cv2.GC_PR_FGD
    gc_mask[fg_core > 0] = cv2.GC_FGD
    x, y, bw, bh = cv2.boundingRect(coarse)
    x  = max(0, x  - BORDER_PAD)
    y  = max(0, y  - BORDER_PAD)
    bw = min(w - x, bw + 2 * BORDER_PAD)
    bh = min(h - y, bh + 2 * BORDER_PAD)
    if bw < 10 or bh < 10:
        return coarse
    bgmodel = np.zeros((1, 65), np.float64)
    fgmodel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, (x, y, bw, bh),
                    bgmodel, fgmodel, GC_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse
    fine = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg_before = np.count_nonzero(coarse)
    fg_after  = np.count_nonzero(fine)
    if fg_before == 0 or fg_after < 0.5 * fg_before or fg_after > 3 * fg_before:
        return coarse
    return fine

# ── Main entry point ───────────────────────────────────────────────────────────

def segment_lesion(bgr_orig):
    h_orig, w_orig = bgr_orig.shape[:2]
    bgr_work, _    = _resize_pad(bgr_orig)
    bgr_work       = _inpaint_hair(bgr_work)
    bgr_enh        = _enhance(bgr_work)
    h, w           = bgr_enh.shape[:2]

    coarse = _ensemble_mask(bgr_enh)
    coarse = _clean_mask(coarse, h, w)

    fill = np.count_nonzero(coarse) / (h * w)
    if fill < MIN_FILL or _border_leak(coarse):
        inv = _clean_mask(_ensemble_mask(cv2.bitwise_not(bgr_enh)), h, w)
        fill_inv = np.count_nonzero(inv) / (h * w)
        if MIN_FILL <= fill_inv <= MAX_FILL and not _border_leak(inv):
            coarse = inv

    if np.count_nonzero(coarse) > 0:
        fine = _grabcut_refine(bgr_enh, coarse)
        fine = _clean_mask(fine, h, w)
        if _border_leak(fine) and not _border_leak(coarse):
            fine = coarse
    else:
        fine = coarse

    mask_full = cv2.resize(fine, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, k, iterations=2)

# ── Confidence ────────────────────────────────────────────────────────────────

def segmentation_confidence(mask, bgr):
    h, w       = bgr.shape[:2]
    fill_ratio = np.count_nonzero(mask) / (h * w)
    if fill_ratio < MIN_FILL or fill_ratio > MAX_FILL:
        return 0.25
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.25
    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = 4 * np.pi * area / (peri ** 2 + 1e-6)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring    = cv2.bitwise_and(cv2.dilate(mask, kernel), cv2.bitwise_not(mask))
    if ring.sum() == 0:
        contrast = 0.5
    else:
        l_ch     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
        contrast = min(abs(l_ch[mask > 0].mean() - l_ch[ring > 0].mean()) / 50.0, 1.0)
    hull_area  = cv2.contourArea(cv2.convexHull(cnt))
    solidity   = area / (hull_area + 1e-6)
    leak_pen   = 0.4 if _border_leak(mask) else 0.0
    conf = (0.25 * min(fill_ratio / 0.15, 1.0) +
            0.25 * min(circ / 0.7, 1.0) +
            0.30 * contrast +
            0.20 * min(solidity, 1.0) -
            leak_pen)
    return round(float(np.clip(conf, 0.1, 1.0)), 2)

# ── ABCD metrics ───────────────────────────────────────────────────────────────

def compute_asymmetry(mask):
    h, w = mask.shape
    scores = []
    for axis in [0, 1]:
        if axis == 0:
            half1 = mask[:h // 2, :]
            half2 = cv2.resize(np.flip(mask[h // 2:, :], 0), (half1.shape[1], half1.shape[0]))
        else:
            half1 = mask[:, :w // 2]
            half2 = cv2.resize(np.flip(mask[:, w // 2:], 1), (half1.shape[1], half1.shape[0]))
        union = np.logical_or(half1 > 0, half2 > 0).sum()
        diff  = np.logical_xor(half1 > 0, half2 > 0).sum()
        scores.append(diff / union if union > 0 else 0)
    return round(min((scores[0] + scores[1]) / 2 * 4.0, 2.0), 3)

def compute_border(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :]
    M   = cv2.moments(mask)
    cx  = M['m10'] / M['m00'] if M['m00'] else mask.shape[1] / 2
    cy  = M['m01'] / M['m00'] if M['m00'] else mask.shape[0] / 2
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    scores = []
    for i in range(8):
        lo  = -np.pi + i * (2 * np.pi / 8)
        idx = np.where((angles >= lo) & (angles < lo + 2 * np.pi / 8))[0]
        if len(idx) < 3:
            scores.append(0); continue
        r = np.sqrt((pts[idx, 0] - cx) ** 2 + (pts[idx, 1] - cy) ** 2)
        scores.append(1 if np.std(r) / (np.mean(r) + 1e-6) > 0.10 else 0)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = 4 * np.pi * area / (peri ** 2 + 1e-6)
    return min(float(sum(scores)) + max(0.0, 1.0 - circ / 0.6), 8.0)

def compute_color(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    roi = mask > 0
    h, s, v = hsv[:, :, 0][roi], hsv[:, :, 1][roi], hsv[:, :, 2][roi]
    l_ch    = lab[:, :, 0][roi]
    found   = []
    if np.mean(v > 200) > 0.04:                                        found.append("White")
    if np.mean((h < 10) & (s > 80)) > 0.03:                           found.append("Red")
    if np.mean((h >= 10) & (h < 25) & (s > 40) & (v > 100)) > 0.05:  found.append("Light-brown")
    if np.mean((h >= 10) & (h < 30) & (v < 120) & (s > 30)) > 0.05:  found.append("Dark-brown")
    if np.mean((h >= 95) & (h < 140) & (s < 90)) > 0.03:              found.append("Blue-grey")
    if np.mean(l_ch < 35) > 0.04:                                      found.append("Black")
    return float(max(1, len(found))), found

def compute_diameter(mask, fov_mm=20.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    _, (mw, mh), _ = cv2.minAreaRect(cnt)
    diam_mm = max(mw, mh) * fov_mm / (mask.shape[1] + 1e-6)
    return round(min(diam_mm / 6.0 * 5.0, 5.0), 3)

def tds(a, b, c, d):
    return round(a * 1.3 + b * 0.1 + c * 0.5 + d * 0.5, 3)

def risk_from_tds(t, conf=1.0):
    level  = "LOW" if t < 4.75 else ("MODERATE" if t <= 5.45 else "HIGH")
    raw_rs = min(t / 8.0 * 10.0, 10.0)
    adj_rs = raw_rs * conf + 5.0 * (1 - conf)
    return round(float(np.clip(adj_rs, 0, 10)), 2), level

def similarity_index(bgr1, bgr2):
    SIZE = (256, 256)
    g1   = cv2.cvtColor(cv2.resize(bgr1, SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2   = cv2.cvtColor(cv2.resize(bgr2, SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1n  = (g1 - g1.mean()) / (g1.std() + 1e-6)
    g2n  = (g2 - g2.mean()) / (g2.std() + 1e-6)
    ncc  = ((float(np.mean(g1n * g2n)) + 1) / 2) * 100
    h1   = cv2.cvtColor(cv2.resize(bgr1, SIZE), cv2.COLOR_BGR2HSV)
    h2   = cv2.cvtColor(cv2.resize(bgr2, SIZE), cv2.COLOR_BGR2HSV)
    ht1  = cv2.calcHist([h1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    ht2  = cv2.calcHist([h2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(ht1, ht1); cv2.normalize(ht2, ht2)
    bhatt = cv2.compareHist(ht1, ht2, cv2.HISTCMP_BHATTACHARYYA)
    return round(max(0.0, min(100.0, ncc * 0.5 + (1 - bhatt) * 100 * 0.5)), 1)

# ── Overlay visualisation ──────────────────────────────────────────────────────

def overlay_mask(bgr, mask):
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy().astype(np.float32)
    teal    = np.array([94, 234, 212], dtype=np.float32)
    dist    = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist    = np.clip(dist / 12.0, 0, 1)
    alpha   = dist[:, :, np.newaxis] * 0.45
    overlay = overlay * (1 - alpha) + teal * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (94, 234, 212), 2)
    return overlay

# ── Full analysis ──────────────────────────────────────────────────────────────

def analyse_pair(bgr1, bgr2):
    res = {}
    for key, bgr in [("baseline", bgr1), ("current", bgr2)]:
        mask            = segment_lesion(bgr)
        conf            = segmentation_confidence(mask, bgr)
        a               = compute_asymmetry(mask)
        b               = compute_border(mask)
        c_score, c_flags = compute_color(bgr, mask)
        d               = compute_diameter(mask)
        t               = tds(a, b, c_score, d)
        rs, rl          = risk_from_tds(t, conf)
        res[key] = {"asymmetry": a, "border": b, "color": c_score, "color_flags": c_flags,
                    "diameter": d, "tds": t, "mask": mask, "conf": conf,
                    "risk_score": rs, "risk_level": rl}

    sim   = similarity_index(bgr1, bgr2)
    delta = res["current"]["tds"] - res["baseline"]["tds"]
    rl    = res["current"]["risk_level"]
    rs    = res["current"]["risk_score"]
    conf  = min(res["baseline"]["conf"], res["current"]["conf"])

    flags, ok_flags = [], []
    if res["current"]["asymmetry"] > res["baseline"]["asymmetry"] + 0.25:
        flags.append(["Asymmetry increased", "The mole's shape is less symmetric compared to your baseline photo."])
    else: ok_flags.append("Asymmetry is stable or improved.")
    if res["current"]["border"] > res["baseline"]["border"] + 0.8:
        flags.append(["Border more irregular", "The edges appear more ragged or uneven than before."])
    else: ok_flags.append("Border regularity is unchanged.")
    new_cols = set(res["current"]["color_flags"]) - set(res["baseline"]["color_flags"])
    if new_cols:
        flags.append(["New colour structures", f"New shades detected: {', '.join(new_cols)}."])
    else: ok_flags.append("No new colours have appeared.")
    if res["current"]["diameter"] > res["baseline"]["diameter"] + 0.4:
        flags.append(["Estimated size increased", "The lesion appears larger than in the baseline image."])
    else: ok_flags.append("Estimated size is stable.")
    if sim < 65:
        flags.append(["Low visual similarity", f"Similarity: {sim:.0f}% — significant change or lighting difference."])
    elif sim < 80:
        flags.append(["Moderate visual similarity", f"Similarity is {sim:.0f}% — reasonable but worth monitoring."])
    else: ok_flags.append(f"High visual similarity ({sim:.0f}%) — images look consistent.")

    if delta > 0.5:
        summary = f"Your mole's score (TDS) increased by {delta:.2f} points — some morphological change has occurred."
    elif delta < -0.3:
        summary = f"Your mole's score decreased by {abs(delta):.2f} points — the mole appears more stable."
    else:
        summary = f"Your mole's score changed by {delta:+.2f} points — relatively stable with minor variation."

    risk_plain = {
        "HIGH":     ("The analysis scored this mole in the higher-concern range. This does NOT mean cancer — "
                     "many benign moles score here. A dermatologist should examine it in person."),
        "MODERATE": ("The analysis scored this mole in an intermediate range. It isn't alarming, "
                     "but has features worth watching. A routine dermatology visit is advised."),
        "LOW":      ("The analysis scored this mole in the lower-concern range. "
                     "Continue regular self-checks every 1–3 months."),
    }
    actions = {
        "HIGH":     ["Book a dermatologist appointment within 2–4 weeks.",
                     "Do not treat or pick at the lesion.",
                     "Take a new photo every 2 weeks until seen by a doctor.",
                     "Note symptoms: itching, bleeding, crusting, rapid size change."],
        "MODERATE": ["Schedule a dermatology check-up within 3 months.",
                     "Retake comparison photos monthly under the same lighting.",
                     "Apply broad-spectrum SPF 30+ sunscreen daily.",
                     "Watch for the ABCDE warning signs."],
        "LOW":      ["Continue monthly self-examinations.",
                     "See a dermatologist for a full-body check once a year.",
                     "Use SPF 30+ sunscreen and protective clothing outdoors.",
                     "Re-run this tracker in 3–6 months."],
    }
    return {
        "abcd_baseline": res["baseline"], "abcd_current": res["current"],
        "similarity_index": sim, "delta_tds": delta,
        "risk_score": rs, "risk_level": rl, "confidence": conf,
        "change_summary": summary, "risk_plain": risk_plain[rl],
        "flags": flags, "ok_flags": ok_flags,
        "actions": actions[rl], "recommendation": actions[rl][0],
    }
