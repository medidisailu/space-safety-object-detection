import streamlit as st
from PIL import Image
import os
import sys

# Debug: Check if system libraries are available
try:
    import ctypes
    ctypes.CDLL('libGL.so.1')
    st.sidebar.success("✅ libGL.so.1 loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ libGL.so.1 not found: {e}")

# Lazy load YOLO to avoid import errors on startup
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    MODEL_PATH = "runs/detect/train6/weights/best.pt"
    return YOLO(MODEL_PATH)

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Space Station Safety Object Detection", layout="centered")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa");
        background-size: cover;
        background-position: center;
        position: relative;
    }
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(0,0,0,0.4);
        z-index: -1;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    model = load_model()
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
    st.markdown("<h2 style='color:#ee82ee;'>Detection Summary</h2>", unsafe_allow_html=True)

    if detected_classes:
        for name in sorted(detected_classes):
            st.markdown(f"<p style='color:#3c3c3c; font-size:28px;'>✔ {name}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#3c3c3c; font-size:24px;'>Total objects detected: {len(detected_classes)}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#3c3c3c; font-size:24px;'>⚠️ No safety objects detected.</p>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#3c3c3c; font-size:18px;'>
        ✅ Ensure the image clearly shows the object.<br>
        ✅ Use high-resolution images with good lighting.<br>
        ✅ Confirm the object is part of the model’s training classes.<br>
        🛠️ Try uploading a known sample (e.g., <code>Oxygen-tank.jpg</code>) for testing.
        </div>
        """, unsafe_allow_html=True)