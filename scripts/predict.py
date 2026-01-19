from ultralytics import YOLO

# Load model
model = YOLO("D:/ml2/runs/space_station_safety/weights/best.pt")

# Run prediction on test images
results = model.predict(
    source="D:/ml2/data/test/images",
    imgsz=256,
    conf=0.15,
    augment=True
)

# Print summary
for r in results:
    print(r)