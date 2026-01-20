import streamlit as st
from ultralytics import YOLO
import os
import cv2

MODEL_PATH = "runs/detect/train6/weights/best.pt"

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)

def run_inference(model, file_path):
    results = model.predict(source=file_path, conf=0.25, save=True)
    return results

def run_webcam(model):
    st.info("Starting webcam... Press 'q' in the window to stop.")
    results = model.predict(source=0, conf=0.25, show=True, stream=True)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            name = r.names[cls_id]
            st.write(f"✔ {name} ({conf:.2f})")

def main():
    st.title("🔍 YOLO Object Detection App")

    model = load_model()
    if model is None:
        return
    st.success("✅ Model loaded successfully!")

    option = st.radio("Choose input source:", ["Upload Image/Video", "Webcam"])

    if option == "Upload Image/Video":
        uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])
        if uploaded_file is not None:
            temp_path = os.path.join("scripts", "temp_input." + uploaded_file.name.split(".")[-1])
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            st.info("Running detection...")
            results = run_inference(model, temp_path)

            st.subheader("✅ Detection Summary")
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    name = r.names[cls_id]
                    st.write(f"✔ {name} ({conf:.2f})")

            st.image(results[0].plot(), caption="Detections", width=700)

    elif option == "Webcam":
        run_webcam(model)

if __name__ == "__main__":
    main()