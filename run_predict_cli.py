import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

MODEL_PATH = "models/crop_disease_model.h5"
CLASS_INDICES_PATH = "results/class_indices.json"
IMG_PATH = "data/test/Bacterial_Blight/Bacterial_Blight_0000.png"
IMG_SIZE = (64, 64)
TOP_K = 3

if not Path(MODEL_PATH).exists():
    print(f"Model missing: {MODEL_PATH}")
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

img = cv2.imread(IMG_PATH)
if img is None:
    print(f"Cannot read image: {IMG_PATH}")
    raise SystemExit(1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_res = cv2.resize(img_rgb, IMG_SIZE)
tensor = np.expand_dims(img_res.astype(np.float32)/255.0, axis=0)
probs = model.predict(tensor, verbose=0)[0]
top_idx = np.argsort(probs)[::-1][:TOP_K]
out_lines = []
out_lines.append(f"Image: {IMG_PATH}")
out_lines.append(f"Top {TOP_K} predictions:")
for i, idx in enumerate(top_idx, 1):
    cls = idx_to_class[idx]
    conf = probs[idx]
    bar = "█" * int(conf * 30)
    out_lines.append(f" {i}. {cls:<20} {conf*100:5.1f}%  {bar}")
print("\n".join(out_lines))
Path('results').mkdir(exist_ok=True)
with open('results/prediction_cli.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(out_lines) + "\n")

# save visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels = [idx_to_class[i].replace('_',' ') for i in top_idx]
confs = [probs[i]*100 for i in top_idx]
fig, axes = plt.subplots(1,2, figsize=(12,6), gridspec_kw={'width_ratios':[1,1]})
axes[0].imshow(img_rgb); axes[0].axis('off'); axes[0].set_title('Input Image')
axes[1].barh(labels[::-1], confs[::-1], color=['#E53935']+['#90CAF9']*(len(labels)-1))
axes[1].set_xlim(0,110)
axes[1].set_xlabel('Confidence (%)')
plt.tight_layout()
Path('results').mkdir(exist_ok=True)
out='results/prediction_result_cli.png'
plt.savefig(out, dpi=150)
print(f"Saved visualization → {out}")
with open('results/prediction_cli.txt', 'a', encoding='utf-8') as f:
    f.write(f"Saved visualization → {out}\n")
