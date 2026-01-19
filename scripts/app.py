import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🚀 Space Station Safety Object Detection")

# Load your trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")  # replace with your weights

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model.predict(source=np.array(image), save=False)
    r = results[0]

    # Show annotated image
    annotated = r.plot()
    st.image(annotated, caption="Predictions", use_column_width=True)

    # Detection summary (object names only, no numbers)
    st.subheader("Detection Summary")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = r.names[cls_id]
        st.write(label)