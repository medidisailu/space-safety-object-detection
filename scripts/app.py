import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import uuid
from pathlib import Path
from collections import Counter

# ================= CONFIG =================
MODEL_PATH = "D:/ml2/runs/detect/train6/weights/best.pt"
OUTPUT_DIR = Path("D:/ml2/runs/predict")

CONF_THRESHOLD = 0.5   # 🔥 clean & strict
IMAGE_SIZE = 640
# =========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Streamlit UI
st.title("🚀 Space Station Safety Object Detection")
st.write("Clean detection output for all safety objects")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Save image
    image_id = str(uuid.uuid4())[:8]
    input_path = OUTPUT_DIR / f"{image_id}.jpg"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run YOLO prediction
    results = model.predict(
        source=str(input_path),
        conf=CONF_THRESHOLD,
        imgsz=IMAGE_SIZE,
        save=False,
        verbose=False
    )

    result = results[0]

    # Read original image
    image = cv2.imread(str(input_path))

    detected_classes = []

    # ===== DRAW CLEAN BOXES =====
    for box in result.boxes:
        confidence = float(box.conf)

        # Extra safety filter
        if confidence < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls)
        class_name = result.names[cls_id]
        detected_classes.append(class_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Green bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Convert to RGB for Streamlit
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(
        Image.fromarray(image_rgb),
        caption="✅ Detection Output",
        width=700
    )

    # ===== SUMMARY =====
    st.subheader("Detection Summary")

    if detected_classes:
        counts = Counter(detected_classes)

        for cls, cnt in counts.items():
            st.success(f"{cls}: {cnt}")

        st.info(f"Total objects detected: {sum(counts.values())}")
    else:
        st.warning(
            "⚠️ No safety objects were detected in this image. "
            "This may happen if the objects are too small, unclear, or not part of the trained classes."
        )
        st.info(
            "💡 Tip: Try uploading  trained classes objects for better results."
        )