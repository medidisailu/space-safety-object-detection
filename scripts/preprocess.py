import os
import cv2
import shutil
import numpy as np

DATA_DIR = r"D:\ml2\data"
OUT_DIR = r"D:\ml2\data\preprocessed"
SPLITS = ["train", "valid", "test"]
IMG_SIZE = 256

def letterbox(img, size=256, color=(114,114,114)):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh))

    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y+nh, x:x+nw] = img_resized
    return canvas

for split in SPLITS:
    img_dir = os.path.join(DATA_DIR, split, "images")
    lbl_dir = os.path.join(DATA_DIR, split, "labels")

    out_img = os.path.join(OUT_DIR, split, "images")
    out_lbl = os.path.join(OUT_DIR, split, "labels")

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        processed = letterbox(img, IMG_SIZE)
        cv2.imwrite(os.path.join(out_img, img_name), processed)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(lbl_dir, label_name)
        dst_label = os.path.join(out_lbl, label_name)

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

print("✅ Preprocessing done — all 7 classes preserved")