from ultralytics import YOLO

# Load model
# Load model
import os
model_path = "runs/detect/train6/weights/best.pt" 
if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found. Using 'yolov8s.pt' instead.")
    model_path = "yolov8s.pt"

model = YOLO(model_path)

# Run prediction on test images
results = model.predict(
    source="data/test/images",
    imgsz=256,
    conf=0.15,
    augment=True
)

# Print summary
for r in results:
    print(r)