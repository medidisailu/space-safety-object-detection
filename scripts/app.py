import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Title
st.title("🚀 Space Safety Object Detection")
st.write("Detect safety-related objects (helmets, vests, restricted items) using YOLOv8 + Streamlit.")

# Load YOLOv8 model (pretrained weights auto-download)
model = YOLO("yolov8n.pt")

# Confidence threshold slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model.predict(source=np.array(image), conf=conf_threshold, save=False)

    # Show results
    st.subheader("Detection Results")
    for r in results:
        st.write(r.names)  # class names
        st.write(r.boxes)  # bounding boxes

    # Render annotated image
    annotated_frame = results[0].plot()  # numpy array with boxes drawn
    st.image(annotated_frame, caption="Predictions", use_column_width=True)