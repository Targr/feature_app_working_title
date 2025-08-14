import io
import os
import base64
import json
from typing import List, Dict, Tuple, Any

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd
import trackpy as tp
import zipfile

# --- Flask App ---
app = Flask(__name__, static_folder="../static", static_url_path="/")

# --------- Image Utils ---------

def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_gray(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    if invert:
        gray = cv2.bitwise_not(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def hough_multi_circles(gray: np.ndarray) -> List[Tuple[int, int, int]]:
    h, w = gray.shape[:2]
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 4,
        param1=100,
        param2=30,
        minRadius=min(h, w) // 6,
        maxRadius=min(h, w) // 2,
    )
    if circles is None:
        return []
    circles = np.uint16(np.around(circles[0]))
    return [(int(x), int(y), int(r)) for x, y, r in circles]


def detect_features(gray_norm: np.ndarray, diameter: int, minmass: int, separation: int, confidence: int) -> pd.DataFrame:
    img = (gray_norm / 255.0).astype(np.float32)
    try:
        feats = tp.locate(
            img,
            diameter=int(diameter),
            minmass=float(minmass),
            separation=int(separation) if separation else int(diameter),
            percentile=int(confidence),
            invert=False,
        )
    except Exception:
        feats = pd.DataFrame(columns=["x", "y"])  # fail-safe
    if feats is None or feats.empty:
        return pd.DataFrame(columns=["x", "y"]) 
    return feats[["x", "y"]]


def mask_inside_circle(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def annotate_image(base_rgb: np.ndarray, plates: List[Dict[str, Any]], all_features: List[pd.DataFrame]) -> Image.Image:
    pil = Image.fromarray(base_rgb)
    draw = ImageDraw.Draw(pil)
    for idx, plate in enumerate(plates):
        cx, cy, r = plate["center_x"], plate["center_y"], plate["radius"]
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(255, 0, 0), width=3)
        feats = all_features[idx]
        for _, row in feats.iterrows():
            x, y = float(row["x"]), float(row["y"])
            draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], outline=(0, 255, 0), width=2)
        draw.text((cx - r, cy - r - 10), f"Plate {idx+1}: {len(feats)}", fill=(255, 255, 0))
    return pil

# --------- ORB Matching (AI Comparison) ---------

def compute_orb_desc(gray: np.ndarray) -> Tuple[list, Any]:
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def similarity_score(desc1, desc2) -> float:
    if desc1 is None or desc2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    denom = max(1, min(len(desc1), len(desc2)))
    return round(len(good) / denom, 4)


def compare_against_references(plate_gray: np.ndarray, refs: List[Tuple[str, np.ndarray]]) -> Dict[str, float]:
    _, d1 = compute_orb_desc(plate_gray)
    scores = {}
    for ref_name, ref_gray in refs:
        _, d2 = compute_orb_desc(ref_gray)
        scores[ref_name] = similarity_score(d1, d2)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

# --------- Core Processing ---------

def process_single_image(
    image_bytes: bytes,
    params: Dict[str, Any],
    reference_bytes: List[Tuple[str, bytes]]
) -> Dict[str, Any]:
    file_arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image")

    # downscale if very large to prevent memory issues on mobile uploads
    while img_bgr.nbytes > 4_000_000:
        h, w = img_bgr.shape[:2]
        img_bgr = cv2.resize(img_bgr, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    invert = bool(params.get("invert", False))
    diameter = int(params.get("diameter", 15))
    minmass = int(params.get("minmass", 10))
    confidence = int(params.get("confidence", 90))
    separation = int(params.get("separation", diameter))

    norm = normalize_gray(gray, invert=invert)

    # detect all plates
    circles = hough_multi_circles(norm)
    if not circles:
        # fallback: use whole image
        h, w = gray.shape
        circles = [(w // 2, h // 2, min(w, h) // 2)]

    # prepare references
    ref_grays: List[Tuple[str, np.ndarray]] = []
    for ref_name, ref_b in reference_bytes:
        arr = np.asarray(bytearray(ref_b), dtype=np.uint8)
        ref_img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if ref_img is not None:
            ref_grays.append((ref_name, normalize_gray(ref_img)))

    results = []
    all_inside_feats = []

    h, w = gray.shape
    for idx, (cx, cy, r) in enumerate(circles):
        mask = mask_inside_circle(h, w, cx, cy, r)
        feats = detect_features(norm, diameter, minmass, separation, confidence)
        if feats.empty:
            feats_in = feats
        else:
            fx = feats["x"].astype(int).clip(0, w - 1)
            fy = feats["y"].astype(int).clip(0, h - 1)
            inside = mask[fy, fx]
            feats_in = feats[inside]

        # crop plate ROI for comparison
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        roi = norm[y1:y2, x1:x2]

        scores = compare_against_references(roi, ref_grays) if ref_grays else {}
        best_ref = max(scores, key=scores.get) if scores else None

        results.append({
            "plate_id": idx + 1,
            "center_x": int(cx),
            "center_y": int(cy),
            "radius": int(r),
            "feature_count": int(len(feats_in)),
            "best_reference": best_ref,
            "scores": scores,
            "features": feats_in.round(2).to_dict(orient="records"),
        })
        all_inside_feats.append(feats_in)

    annotated = annotate_image(rgb, results, all_inside_feats)
    annotated_b64 = pil_to_base64(annotated)

    return {
        "plates": results,
        "annotated_image_base64": annotated_b64,
    }

# --------- Routes ---------
@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/process")
def api_process():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image_file = request.files["image"]

    # optional reference images
    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    # params
    def to_int(v, dv):
        try:
            return int(v)
        except Exception:
            return dv
    mode = request.form.get("mode", "Plaque")
    invert = True if mode == "Colony" else False

    params = {
        "invert": invert,
        "diameter": to_int(request.form.get("diameter", 15), 15),
        "minmass": to_int(request.form.get("minmass", 10), 10),
        "confidence": to_int(request.form.get("confidence", 90), 90),
        "separation": to_int(request.form.get("separation", 15), 15),
    }

    out = process_single_image(image_file.read(), params, references)
    return jsonify(out)


@app.post("/api/batch")
def api_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    def to_int(v, dv):
        try:
            return int(v)
        except Exception:
            return dv
    mode = request.form.get("mode", "Plaque")
    invert = True if mode == "Colony" else False
    params = {
        "invert": invert,
        "diameter": to_int(request.form.get("diameter", 15), 15),
        "minmass": to_int(request.form.get("minmass", 10), 10),
        "confidence": to_int(request.form.get("confidence", 90), 90),
        "separation": to_int(request.form.get("separation", 15), 15),
    }

    rows = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for f in files:
            try:
                result = process_single_image(f.read(), params, references)
                plates = result["plates"]
                total = int(sum(p["feature_count"] for p in plates))
                rows.append({
                    "image_name": f.filename,
                    "total_features": total,
                    "plates": plates
                })

                # Save annotated image to ZIP
                img_data = base64.b64decode(result["annotated_image_base64"])
                zf.writestr(f.filename.replace(" ", "_"), img_data)

            except Exception as e:
                rows.append({
                    "image_name": f.filename,
                    "error": str(e),
                    "plates": []
                })

    zip_buf.seek(0)
    csv_df = pd.DataFrame(rows)
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")
    zip_b64 = base64.b64encode(zip_buf.read()).decode("utf-8")

    return jsonify({
        "results": rows,
        "csv_base64": csv_b64,
        "zip_base64": zip_b64
    })


# --- Run Server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
