"""
Crop Disease Prediction System — Training Script
=================================================
Author  : Abhishek Gorakh Borade
College : Sandip University | B.Sc CS (AI, ML & VR) — 2026
Tech    : Python, TensorFlow/Keras, CNN, MobileNetV2, OpenCV

Usage:
    # Generate demo data and train
    python data_prep.py demo --dest data/ --samples 80
    python train.py

    # Train on your own dataset (PlantVillage)
    python data_prep.py split --source raw_data/ --dest data/ --split 0.70 0.15 0.15
    python train.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ──────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────
CONFIG = {
    "data_dir"      : "data",
    "model_path"    : "models/crop_disease_model.h5",
    "results_dir"   : "results",
    "history_path"  : "results/history.json",
    "img_size"      : (64, 64),          # use (224,224) with MobileNetV2 on real data
    "batch_size"    : 32,
    "epochs"        : 20,
    "learning_rate" : 1e-4,
    "dropout_rate"  : 0.4,
}

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


# ──────────────────────────────────────────────────
# 1. DATA GENERATORS
# ──────────────────────────────────────────────────
def build_data_generators(config):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(config["data_dir"], "train"),
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(config["data_dir"], "val"),
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
    )
    test_gen = val_datagen.flow_from_directory(
        os.path.join(config["data_dir"], "test"),
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ──────────────────────────────────────────────────
# 2. MODEL — Custom CNN (offline, no download needed)
#    For production: swap to MobileNetV2 transfer learning
# ──────────────────────────────────────────────────
def build_model(num_classes, config):
    inputs = tf.keras.Input(shape=(*config["img_size"], 3))
    x = layers.Conv2D(32,  3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64,  3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(config["dropout_rate"])(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(config["dropout_rate"] / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="CropDiseaseCNN")


# ──────────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────────
def train(config):
    print("\n" + "=" * 60)
    print("  CROP DISEASE PREDICTION — TRAINING")
    print("=" * 60)

    train_gen, val_gen, test_gen = build_data_generators(config)
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    print(f"\nClasses ({num_classes}): {class_names}")
    print(f"Train : {train_gen.samples} | Val : {val_gen.samples} | Test : {test_gen.samples}")

    model = build_model(num_classes, config)
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(config["model_path"], save_best_only=True, monitor="val_accuracy", verbose=1),
    ]

    print("\n[Training] Starting ...")
    history = model.fit(
        train_gen, epochs=config["epochs"],
        validation_data=val_gen, callbacks=cbs, verbose=1,
    )

    with open(config["history_path"], "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # Evaluate
    print("\n[Evaluation]")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Accuracy : {acc * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:\n", report)

    with open(os.path.join(config["results_dir"], "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, num_classes), max(6, num_classes - 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Crop Disease Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "confusion_matrix.png"), dpi=150)
    plt.close()

    # Training plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="#FF5722", linestyle="--")
    axes[0].set_title("Model Accuracy"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(history.history["loss"],         label="Train Loss", color="#4CAF50")
    axes[1].plot(history.history["val_loss"],     label="Val Loss",   color="#F44336", linestyle="--")
    axes[1].set_title("Model Loss"); axes[1].legend(); axes[1].grid(alpha=.3)
    plt.suptitle("Training History — Crop Disease CNN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "training_history.png"), dpi=150)
    plt.close()

    with open(os.path.join(config["results_dir"], "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    print(f"\nModel saved → {config['model_path']}")
    print(f"Results    → {config['results_dir']}/")


if __name__ == "__main__":
    train(CONFIG)
