"""
Dataset Preparation Utility
============================
Author : Abhishek Gorakh Borade

Splits a flat class-folder dataset into train / val / test splits and
optionally creates a small synthetic demo dataset for testing the pipeline
without downloading the real PlantVillage dataset.

Usage
-----
    # Split an existing dataset
    python data_prep.py --source raw_data/ --dest data/ --split 0.7 0.15 0.15

    # Create a tiny synthetic demo (random coloured images) and split it
    python data_prep.py --demo --classes Healthy Bacterial_Blight Leaf_Rust \
                        --samples_per_class 50 --dest data/
"""

import os
import shutil
import argparse
import random
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# SPLIT EXISTING DATASET
# ─────────────────────────────────────────────

def split_dataset(source_dir, dest_dir, split=(0.70, 0.15, 0.15), seed=42):
    """
    source_dir/
        ClassA/  img1.jpg  img2.jpg  ...
        ClassB/  ...

    →  dest_dir/
           train/ClassA/  train/ClassB/
           val/ClassA/    val/ClassB/
           test/ClassA/   test/ClassB/
    """
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    random.seed(seed)

    source_path = Path(source_dir)
    dest_path   = Path(dest_dir)

    for split_name in ("train", "val", "test"):
        (dest_path / split_name).mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No sub-folders (classes) found in {source_dir}")

    summary = {}

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        images   = list(cls_dir.glob("*.[jJpPbBgG][pPnNiImM][gGfFpP]*"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * split[0])
        n_val   = int(n * split[1])

        splits = {
            "train": images[:n_train],
            "val"  : images[n_train: n_train + n_val],
            "test" : images[n_train + n_val:],
        }

        summary[cls_name] = {}
        for split_name, files in splits.items():
            target_dir = dest_path / split_name / cls_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_path in files:
                shutil.copy2(img_path, target_dir / img_path.name)
            summary[cls_name][split_name] = len(files)

    print("\n Dataset split complete")
    print(f" {'Class':<25} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print(" " + "-" * 52)
    for cls, counts in summary.items():
        total = sum(counts.values())
        print(f" {cls:<25} {counts['train']:>7} {counts['val']:>7} {counts['test']:>7} {total:>7}")
    print(f"\n Output directory → {dest_dir}")


# ─────────────────────────────────────────────
# SYNTHETIC DEMO DATASET
# ─────────────────────────────────────────────

def create_demo_dataset(classes, samples_per_class, dest_dir, img_size=(224, 224), seed=42):
    """
    Creates random PNG images (one colour tint per class) to let developers
    test the full pipeline without downloading real data.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Using PIL instead.")
        _create_demo_pil(classes, samples_per_class, dest_dir, img_size, seed)
        return

    np.random.seed(seed)
    source_dir = Path(dest_dir) / "_demo_raw"

    # Assign a distinct hue per class
    hues = np.linspace(0, 170, len(classes), dtype=int)

    for cls, hue in zip(classes, hues):
        cls_dir = source_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            # Random HSV image with dominant class hue
            img_hsv = np.zeros((*img_size, 3), dtype=np.uint8)
            img_hsv[:, :, 0] = hue
            img_hsv[:, :, 1] = np.random.randint(150, 255, img_size, dtype=np.uint8)
            img_hsv[:, :, 2] = np.random.randint(100, 255, img_size, dtype=np.uint8)
            # Add noise to make images non-identical
            noise = np.random.randint(0, 30, (*img_size, 3), dtype=np.uint8)
            img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            img_bgr = cv2.add(img_bgr, noise)
            cv2.imwrite(str(cls_dir / f"{cls}_{i:04d}.png"), img_bgr)

    print(f"Demo raw images created → {source_dir}")
    split_dataset(str(source_dir), dest_dir)


def _create_demo_pil(classes, samples_per_class, dest_dir, img_size, seed):
    from PIL import Image
    np.random.seed(seed)
    source_dir = Path(dest_dir) / "_demo_raw"
    colours = [(200, 50, 50), (50, 180, 50), (50, 50, 200),
               (200, 200, 50), (200, 50, 200), (50, 200, 200)]

    for idx, cls in enumerate(classes):
        cls_dir = source_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        base_colour = colours[idx % len(colours)]
        for i in range(samples_per_class):
            arr = np.clip(
                np.random.randint(-30, 30, (*img_size, 3)) +
                np.array(base_colour, dtype=np.int16),
                0, 255
            ).astype(np.uint8)
            Image.fromarray(arr).save(str(cls_dir / f"{cls}_{i:04d}.png"))

    split_dataset(str(source_dir), dest_dir)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation for Crop Disease Prediction")
    sub = parser.add_subparsers(dest="command")

    # split command
    sp = sub.add_parser("split", help="Split an existing flat dataset")
    sp.add_argument("--source", required=True)
    sp.add_argument("--dest",   required=True)
    sp.add_argument("--split",  nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))

    # demo command
    dp = sub.add_parser("demo", help="Create synthetic demo dataset")
    dp.add_argument("--classes", nargs="+",
                    default=["Healthy", "Bacterial_Blight", "Leaf_Rust",
                             "Powdery_Mildew", "Early_Blight"])
    dp.add_argument("--samples", type=int, default=60, dest="samples_per_class")
    dp.add_argument("--dest", default="data/")

    args = parser.parse_args()

    if args.command == "split":
        split_dataset(args.source, args.dest, tuple(args.split))
    elif args.command == "demo":
        create_demo_dataset(args.classes, args.samples_per_class, args.dest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
