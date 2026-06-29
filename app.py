"""
Crop Disease Prediction — Flask Web App
========================================
Author  : Abhishek Gorakh Borade
College : Sandip University | B.Sc CS (AI, ML & VR) — 2026
Tech    : Python, Flask, TensorFlow/Keras, CNN, OpenCV
"""

import os
import json
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image                          # ← Pillow (more reliable on Windows)
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = os.path.join(app.root_path, "static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH         = os.path.join(os.path.dirname(__file__), "models", "crop_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "results", "class_indices.json")
IMG_SIZE           = (64, 64)

# ──────────────────────────────────────────────────
# DISEASE ADVICE
# ──────────────────────────────────────────────────
DISEASE_INFO = {
    "Healthy": {
        "advice": "No treatment needed. Continue good agricultural practices such as proper watering, fertilization, and regular monitoring.",
        "severity": "none", "icon": "🌿", "color": "#27ae60"
    },
    "Bacterial_Blight": {
        "advice": "Apply copper-based bactericides. Remove and destroy infected plant parts. Avoid overhead irrigation. Practice crop rotation.",
        "severity": "high", "icon": "🦠", "color": "#e74c3c"
    },
    "Leaf_Rust": {
        "advice": "Apply fungicides (e.g., propiconazole or tebuconazole). Remove infected leaves. Ensure proper plant spacing for good airflow.",
        "severity": "medium", "icon": "🍂", "color": "#e67e22"
    },
    "Powdery_Mildew": {
        "advice": "Use sulfur-based fungicides or neem oil spray. Increase plant spacing. Avoid excess nitrogen fertilization.",
        "severity": "medium", "icon": "🌫️", "color": "#f39c12"
    },
    "Early_Blight": {
        "advice": "Apply chlorothalonil or mancozeb fungicide. Mulch around plants. Water at the base of the plant to keep foliage dry.",
        "severity": "medium", "icon": "🍁", "color": "#d35400"
    },
    "Late_Blight": {
        "advice": "Apply systemic fungicides immediately. Remove heavily infected plants. Avoid overhead watering. Act fast — this is a serious disease.",
        "severity": "high", "icon": "⚠️", "color": "#c0392b"
    },
    "Mosaic_Virus": {
        "advice": "No chemical cure. Remove and destroy infected plants. Control aphid vectors. Use virus-resistant varieties in future.",
        "severity": "high", "icon": "🧬", "color": "#8e44ad"
    },
}

# ──────────────────────────────────────────────────
# LOAD MODEL (once at startup)
# ──────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
print(f"Model loaded. Classes: {list(class_indices.keys())}")


# ──────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(image_path):
    """
    Read image with Pillow (works reliably on Windows paths & all formats).
    Convert to RGB, resize, normalise → numpy tensor.
    """
    img = Image.open(image_path).convert("RGB")   # handles RGBA/palette/WEBP too
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)             # shape (1, 64, 64, 3)


def predict(image_path, top_k=5):
    tensor  = preprocess(image_path)
    probs   = model.predict(tensor, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    results = []
    for i in top_idx:
        cls  = idx_to_class[i]
        conf = float(probs[i])
        info = DISEASE_INFO.get(
            cls,
            {"advice": "Consult an agricultural expert.", "severity": "unknown", "icon": "❓", "color": "#95a5a6"}
        )
        results.append({
            "class":      cls,
            "label":      cls.replace("_", " "),
            "confidence": round(conf * 100, 2),
            "advice":     info["advice"],
            "severity":   info["severity"],
            "icon":       info["icon"],
            "color":      info["color"],
        })
    return results


# ──────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, or WEBP"}), 400

    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)

        # Safety check — confirm file actually exists before predicting
        if not os.path.exists(filepath):
            return jsonify({"error": "File save failed. Check write permissions on static/uploads/"}), 500

        results   = predict(filepath, top_k=5)
        image_url = url_for("static", filename=f"uploads/{filename}")

        return jsonify({
            "success":   True,
            "image_url": image_url,
            "results":   results,
            "top":       results[0],
        })

    except Exception as e:
        app.logger.exception("Prediction route failed")
        # Return the actual error in debug mode so you can see what went wrong
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)