import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="Space Station Safety Object Detection",
    layout="centered"
)

st.title("🚀 Space Station Safety Object Detection")

# -----------------------------
# Load YOLO Model (CPU only)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # auto-downloads safely

model = load_model()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path)
    st.image(image, caption=uploaded_file.name, use_container_width=True)

    # -----------------------------
    # Run Detection
    # -----------------------------
    results = model(
        image_path,
        conf=0.5,
        device="cpu"
    )

    boxes = results[0].boxes
    names = results[0].names
    detected_classes = set()

    for box in boxes:
        cls_id = int(box.cls)
        detected_classes.add(names[cls_id])

    # -----------------------------
    # Detection Summary
    # -----------------------------
    st.subheader("Detection Summary")

    if detected_classes:
        for obj in sorted(detected_classes):
            st.success(obj)
        st.info(f"Total objects detected: {len(detected_classes)}")
    else:
        st.warning("No safety objects detected.")
