import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

st.title("🚀 Space Station Safety Object Detection")

# Load YOLOv8 model (replace with your custom weights if needed)
model = YOLO("yolov8n.pt")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction (default confidence threshold from YOLO)
    results = model.predict(source=np.array(image), save=False)
    r = results[0]

    # Count detections
    class_counts = Counter()
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = r.names[cls_id]
        class_counts[label] += 1

    # Show annotated image
    annotated = r.plot()
    st.image(annotated, caption="Predictions", use_column_width=True)

    # Detection summary (formatted like [1]: ClassName -> Count)
    st.subheader("Detection Summary")
    for i, (label, count) in enumerate(class_counts.items(), start=1):
        st.write(f"[{i}]: {label} -> {count}")
    st.write(f"Total objects detected: {sum(class_counts.values())}")