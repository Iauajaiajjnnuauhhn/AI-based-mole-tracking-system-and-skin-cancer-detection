"""
analysis.py — Mole segmentation + ABCD analysis + MobileNetV2 prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
1. Pre-processing: resize, CLAHE, bilateral filter, hair removal
2. Multi-cue segmentation + GrabCut refinement
3. ABCD metrics computation
4. MobileNetV2 prediction (melanoma vs benign)
5. Combined report
"""

import cv2
import numpy as np

# ── Optional TensorFlow (SAFE IMPORT) ─────────────────────────
MODEL_PATH = "skincancer_mobilenet.h5"
model = None
TF_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array

    model = load_model(MODEL_PATH)
    TF_AVAILABLE = True
    print("✅ Model loaded successfully")

except Exception as e:
    print("⚠️ TensorFlow/model not available:", e)
    model = None

# ── Config ─────────────────────────────────────────────
WORK_SIZE  = 512
MIN_FILL   = 0.01
MAX_FILL   = 0.80
GC_ITERS   = 8
BORDER_PAD = 12
IMG_SIZE   = 224  # MobileNetV2 input

# Load MobileNetV2 model (must be trained and saved beforehand)
model = load_model("mole_classifier.h5")

# ── Pre-processing ─────────────────────────────────────────────

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

def _border_leak(mask, margin=5):
    return bool(mask[:margin, :].any() and mask[-margin:, :].any() and
                mask[:, :margin].any() and mask[:, -margin:].any())

# ── Multi-cue segmentation ─────────────────────────────────────

def _cue_lab(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:,:,0]/255.0
    a = (lab[:,:,1]-128)/127.0
    return np.clip(0.6*(1-l)+0.4*np.clip(a,0,1),0,1)

def _cue_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return np.clip(1.0 - hsv[:,:,2]/255.0,0,1)

def _cue_excess_red(bgr):
    f = bgr.astype(np.float32)/255.0
    er = 1.4*f[:,:,2] - f[:,:,1] - 0.5*f[:,:,0]
    return np.clip(er,0,1)

def _cue_sat_drop(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:,:,0]; s = hsv[:,:,1]/255.0; v = hsv[:,:,2]/255.0
    skin_hue = (h<35).astype(np.float32)
    return np.clip(skin_hue*(1.0-s)*(1.0-v),0,1)

def _centre_prior(h,w,sigma=0.45):
    cy,cx = h/2,w/2
    yy,xx = np.mgrid[0:h,0:w].astype(np.float32)
    dist = np.sqrt(((yy-cy)/(cy+1e-6))**2 + ((xx-cx)/(cx+1e-6))**2)
    return np.exp(-dist**2/(2*sigma**2))

def _ensemble_mask(bgr):
    h,w = bgr.shape[:2]
    fused = (0.35*_cue_lab(bgr) + 0.25*_cue_hsv(bgr) + 0.20*_cue_excess_red(bgr) +
             0.10*_cue_sat_drop(bgr) + 0.10*_centre_prior(h,w))
    prior = _centre_prior(h,w)
    fused = fused*(0.5+0.5*prior)
    u8 = np.clip(fused*255,0,255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return mask

def _clean_mask(mask, h, w):
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    k_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,k_close,iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,k_open,iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask,connectivity=8)
    if n>1:
        largest = 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        mask = np.uint8(labels==largest)*255
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(max(contours,key=cv2.contourArea))
        hull_mask = np.zeros((h,w),np.uint8)
        cv2.drawContours(hull_mask,[hull],-1,255,-1)
        eroded_hull = cv2.erode(hull_mask,k_open,iterations=3)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(hull_mask,eroded_hull))
    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k_sm,iterations=2)

def _grabcut_refine(bgr, coarse):
    h,w = bgr.shape[:2]
    k_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    fg_core = cv2.erode(coarse,k_fg,iterations=3)
    bg_ring = cv2.dilate(coarse,k_bg,iterations=4)
    gc_mask = np.full((h,w),cv2.GC_BGD,np.uint8)
    gc_mask[bg_ring>0]=cv2.GC_PR_BGD
    gc_mask[coarse>0]=cv2.GC_PR_FGD
    gc_mask[fg_core>0]=cv2.GC_FGD
    x,y,bw,bh = cv2.boundingRect(coarse)
    x=max(0,x-BORDER_PAD); y=max(0,y-BORDER_PAD)
    bw=min(w-x,bw+2*BORDER_PAD); bh=min(h-y,bh+2*BORDER_PAD)
    if bw<10 or bh<10:
        return coarse
    bgmodel = np.zeros((1,65),np.float64)
    fgmodel = np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(bgr,gc_mask,(x,y,bw,bh),bgmodel,fgmodel,GC_ITERS,cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse
    fine = np.where((gc_mask==cv2.GC_FGD)|(gc_mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    return fine

# ── Lesion segmentation ─────────────────────────────────────
def segment_lesion(bgr_orig):
    h_orig, w_orig = bgr_orig.shape[:2]
    bgr_work,_ = _resize_pad(bgr_orig)
    bgr_work = _inpaint_hair(bgr_work)
    bgr_enh = _enhance(bgr_work)
    coarse = _ensemble_mask(bgr_enh)
    coarse = _clean_mask(coarse,*bgr_enh.shape[:2])
    fine = _grabcut_refine(bgr_enh, coarse) if np.count_nonzero(coarse)>0 else coarse
    mask_full = cv2.resize(fine,(w_orig,h_orig),interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    return cv2.morphologyEx(mask_full,cv2.MORPH_CLOSE,k,iterations=2)

# ── ABCD metrics ─────────────────────────────────────────────
def compute_asymmetry(mask):
    h,w = mask.shape
    scores=[]
    for axis in [0,1]:
        if axis==0:
            half1 = mask[:h//2,:]; half2 = cv2.resize(np.flip(mask[h//2:,:],0),(half1.shape[1],half1.shape[0]))
        else:
            half1 = mask[:,:w//2]; half2 = cv2.resize(np.flip(mask[:,w//2:],1),(half1.shape[1],half1.shape[0]))
        union = np.logical_or(half1>0,half2>0).sum()
        diff = np.logical_xor(half1>0,half2>0).sum()
        scores.append(diff/union if union>0 else 0)
    return round(min((scores[0]+scores[1])/2*4.0,2.0),3)

def compute_border(mask):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours: return 0.0
    cnt = max(contours,key=cv2.contourArea)
    pts = cnt[:,0,:]
    M=cv2.moments(mask)
    cx = M['m10']/M['m00'] if M['m00'] else mask.shape[1]/2
    cy = M['m01']/M['m00'] if M['m00'] else mask.shape[0]/2
    angles = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    scores=[]
    for i in range(8):
        lo = -np.pi + i*2*np.pi/8
        idx = np.where((angles>=lo)&(angles<lo+2*np.pi/8))[0]
        if len(idx)<3: scores.append(0); continue
        r = np.sqrt((pts[idx,0]-cx)**2 + (pts[idx,1]-cy)**2)
        scores.append(1 if np.std(r)/(np.mean(r)+1e-6)>0.10 else 0)
    area=cv2.contourArea(cnt); peri=cv2.arcLength(cnt,True)
    circ = 4*np.pi*area/(peri**2+1e-6)
    return min(float(sum(scores))+max(0.0,1.0-circ/0.6),8.0)

def compute_color(bgr,mask):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)
    roi = mask>0
    if not roi.any(): return 1.0,[]
    h=hsv[:,:,0][roi]; s=hsv[:,:,1][roi]; v=hsv[:,:,2][roi]; l_ch=lab[:,:,0][roi]
    found=[]
    if np.mean(v>200)>0.04: found.append("White")
    if np.mean((h<10)&(s>80))>0.03: found.append("Red")
    if np.mean((h>=10)&(h<25)&(s>40)&(v>100))>0.05: found.append("Light-brown")
    if np.mean((h>=10)&(h<30)&(v<120)&(s>30))>0.05: found.append("Dark-brown")
    if np.mean((h>=95)&(h<140)&(s<90))>0.03: found.append("Blue-grey")
    if np.mean(l_ch<35)>0.04: found.append("Black")
    return float(max(1,len(found))),found

def compute_diameter(mask,fov_mm=20.0):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    cnt = max(contours,key=cv2.contourArea)
    _,(mw,mh),_ = cv2.minAreaRect(cnt)
    diam_mm = max(mw,mh)*fov_mm/(mask.shape[1]+1e-6)
    return round(min(diam_mm/6.0*5.0,5.0),3)

def tds(a,b,c,d):
    return round(a*1.3+b*0.1+c*0.5+d*0.5,3)

def risk_from_tds(t,conf=1.0):
    level = "LOW" if t<4.75 else ("MODERATE" if t<=5.45 else "HIGH")
    raw_rs = min(t/8.0*10.0,10.0)
    adj_rs = raw_rs*conf + 5.0*(1-conf)
    return round(float(np.clip(adj_rs,0,10)),2),level

# ── MobileNetV2 prediction ─────────────────────────────
def predict_mole(bgr):
    if model is None:
        return "model_not_loaded", 0.0

    try:
        IMG_SIZE = 224
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
def analyse_pair(bgr1,bgr2):
    mask1 = segment_lesion(bgr1)
    mask2 = segment_lesion(bgr2)

    a1 = compute_asymmetry(mask1)
    b1 = compute_border(mask1)
    c1, flags1 = compute_color(bgr1,mask1)
    d1 = compute_diameter(mask1)
    t1 = tds(a1,b1,c1,d1)
    rs1, rl1 = risk_from_tds(t1)

    a2 = compute_asymmetry(mask2)
    b2 = compute_border(mask2)
    c2, flags2 = compute_color(bgr2,mask2)
    d2 = compute_diameter(mask2)
    t2 = tds(a2,b2,c2,d2)
    rs2, rl2 = risk_from_tds(t2)

    label1, conf1 = predict_mole(bgr1)
    label2, conf2 = predict_mole(bgr2)

    delta_tds = t2-t1

    report = {
        "abcd_baseline":{"asymmetry":a1,"border":b1,"color":c1,"color_flags":flags1,"diameter":d1,"tds":t1,"risk_score":rs1,"risk_level":rl1},
        "abcd_current":{"asymmetry":a2,"border":b2,"color":c2,"color_flags":flags2,"diameter":d2,"tds":t2,"risk_score":rs2,"risk_level":rl2},
        "ml_prediction":{"baseline":{"label":label1,"confidence":conf1},
                         "current":{"label":label2,"confidence":conf2}},
        "delta_tds": delta_tds
    }

    return report
