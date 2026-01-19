"""
YOLOv8 Training Script
Project: ML2 Object Detection (7 Classes)
Optimized for low-end / CPU systems
"""

import torch
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_YAML = "D:/ml2/data/data.yaml"   # Must point to preprocessed dataset
MODEL_NAME = "yolov8s.pt"             # Best balance for accuracy vs speed
IMG_SIZE = 256                        # Same as preprocessing
EPOCHS = 50                           # Safe for your PC
BATCH = 8                             # Reduce to 4 if RAM error
DEVICE = "cpu"                        # Change to 0 if GPU exists

# -----------------------------
# TRAINING
# -----------------------------
def main():
    print("🚀 Starting YOLOv8 Training...")
    print(f"📦 Dataset: {DATA_YAML}")
    print(f"🖼️ Image Size: {IMG_SIZE}")
    print(f"🔁 Epochs: {EPOCHS}")
    print(f"🧠 Device: {DEVICE}")

    # Load model
    model = YOLO(MODEL_NAME)

    # Train
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,

        # Optimizer & LR
        optimizer="Adam",
        lr0=0.001,
        lrf=0.01,

        # Early stopping
        patience=10,

        # Augmentations (safe & effective)
        mosaic=0.8,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        # Regularization
        weight_decay=0.0005,

        # Logging
        verbose=True,
        plots=True
    )

    print("✅ Training completed successfully!")
    print("📁 Check results in: runs/detect/train/")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()