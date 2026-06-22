"""
Crop Disease Prediction — CLI Inference Script
===============================================
Author  : Abhishek Gorakh Borade
College : Sandip University | B.Sc CS (AI, ML & VR) — 2026

Usage:
    python predict.py --image path/to/leaf.jpg
    python predict.py --image leaf.jpg --model models/crop_disease_model.h5 --top_k 5
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

DEFAULT_MODEL_PATH  = "models/crop_disease_model.h5"
CLASS_INDICES_PATH  = "results/class_indices.json"
IMG_SIZE            = (64, 64)

DISEASE_ADVICE = {
    "Healthy":          "No treatment needed. Continue good agricultural practices.",
    "Bacterial_Blight": "Apply copper-based bactericides. Remove infected parts. Avoid overhead irrigation.",
    "Leaf_Rust":        "Apply fungicides (propiconazole or tebuconazole). Remove infected leaves.",
    "Powdery_Mildew":   "Use sulfur-based fungicides or neem oil. Increase plant spacing.",
    "Early_Blight":     "Apply chlorothalonil or mancozeb fungicide. Mulch around plants.",
    "Late_Blight":      "Apply systemic fungicides immediately. Remove heavily infected plants.",
    "Mosaic_Virus":     "No chemical cure. Remove infected plants. Control aphid vectors.",
}


def load_model_and_classes(model_path, class_indices_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    model = tf.keras.models.load_model(model_path)
    with open(class_indices_path) as f:
        class_indices = json.load(f)
    return model, {v: k for k, v in class_indices.items()}


def preprocess_image(image_path, img_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    return img_rgb, np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)


def predict_disease(model, img_tensor, idx_to_class, top_k=3):
    probs = model.predict(img_tensor, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(idx_to_class[i], float(probs[i])) for i in top_indices]


def visualise_prediction(orig_img, predictions, save_path="results/prediction_result.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    top_class, top_conf = predictions[0]
    advice = DISEASE_ADVICE.get(top_class, "Consult an agricultural expert.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]})
    axes[0].imshow(orig_img); axes[0].axis("off"); axes[0].set_title("Input Leaf Image", fontsize=13, fontweight="bold")

    labels  = [p[0].replace("_", " ") for p in predictions]
    confs   = [p[1] * 100 for p in predictions]
    colours = ["#E53935" if i == 0 else "#90CAF9" for i in range(len(predictions))]
    bars = axes[1].barh(labels[::-1], confs[::-1], color=colours[::-1], edgecolor="white")
    axes[1].set_xlim(0, 110); axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Top Predictions", fontsize=13, fontweight="bold")
    for bar, c in zip(bars, confs[::-1]):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{c:.1f}%", va="center", fontsize=10)

    color = "#C62828" if top_class != "Healthy" else "#2E7D32"
    fig.suptitle(f"Prediction: {top_class.replace('_',' ')}  |  Confidence: {top_conf*100:.1f}%",
                 fontsize=15, fontweight="bold", color=color)
    fig.text(0.5, 0.01, f"Advice: {advice}", ha="center", fontsize=9, color="#424242")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Result image saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Crop Disease Prediction CLI")
    parser.add_argument("--image",  required=True)
    parser.add_argument("--model",  default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  CROP DISEASE PREDICTION — INFERENCE")
    print("=" * 55)

    model, idx_to_class = load_model_and_classes(args.model, CLASS_INDICES_PATH)
    orig_img, img_tensor = preprocess_image(args.image)
    predictions = predict_disease(model, img_tensor, idx_to_class, top_k=args.top_k)

    print(f"\nImage : {args.image}\nTop {args.top_k} Predictions:\n" + "-" * 35)
    for rank, (cls, conf) in enumerate(predictions, 1):
        bar = "█" * int(conf * 30)
        print(f"  {rank}. {cls:<25} {conf*100:5.1f}%  {bar}")

    top_class, top_conf = predictions[0]
    advice = DISEASE_ADVICE.get(top_class, "Consult an agricultural expert.")
    print(f"\n{'─'*55}\n  DIAGNOSIS : {top_class.replace('_',' ').upper()}")
    print(f"  CONFIDENCE: {top_conf*100:.1f}%\n  ADVICE    : {advice}\n{'─'*55}\n")
    visualise_prediction(orig_img, predictions)


if __name__ == "__main__":
    main()
