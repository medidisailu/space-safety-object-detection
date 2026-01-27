import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO

# -----------------------------
# Load YOLO model (cached so it doesn't reload every time)
# -----------------------------
@st.cache_resource
def load_model():
    # Adjust path to your trained weights
    return YOLO("runs/detect/train6/weights/best.pt")

model = load_model()

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Space Safety Object Detection", layout="centered")

st.title("🚀 Space Station Safety Object Detection")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path)
    st.image(image, caption=f"{uploaded_file.name}", width=700)

    # -----------------------------
    # Run Detection
    # -----------------------------
    results = model.predict(source=image_path, save=False, conf=0.5)
    boxes = results[0].boxes
    names = results[0].names
    detected_classes = set()

    for box in boxes:
        cls_id = int(box.cls)
        class_name = names[cls_id]
        detected_classes.add(class_name)

    # -----------------------------
    # Detection Summary
    # -----------------------------
    st.subheader("Detection Summary")

    if detected_classes:
        for name in sorted(detected_classes):
            st.write(f"✔ {name}")
        st.write(f"**Total objects detected:** {len(detected_classes)}")
   else:
    st.markdown(
        """
        <div style='color:#3c3c3c; font-size:18px;'>
        ⚠️ Ensure the image clearly shows the object.<br>
        ⚠️ Use high-resolution images with good lighting.<br>
        ⚠️ Confirm the object is part of the model’s training classes.<br>
        </div>
        """,
        unsafe_allow_html=True
    )else:
    st.warning(
        "⚠️ Ensure the image clearly shows the object.\n"
        "⚠️ Use high-resolution images with good lighting.\n"
        "⚠️ Confirm the object is part of the model’s training classes."
    )
