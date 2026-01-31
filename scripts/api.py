from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import io
import base64
from PIL import Image

app = FastAPI()

# Enable CORS so your mobile app can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = YOLO("model.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = YOLO("yolov8n.pt") # Fallback to NANO model for Free Tier (Low RAM)

@app.get("/")
def home():
    return {
        "status": "online", 
        "message": "Space Safety Object Detection API. Send POST requests to /detect"
    }

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # 1. Read the image file uploaded by the user
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run Object Detection
    results = model.predict(img, conf=0.25)
    result = results[0]

    # 3. Process Results (Count objects)
    detected_objects = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = result.names[cls_id]
        confidence = float(box.conf[0])
        detected_objects.append({
            "class": class_name,
            "confidence": round(confidence, 2)
        })

    # 4. Generate Annotated Image (Visual Result)
    annotated_img = result.plot() # Draw boxes
    
    # Convert BGR to RGB
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_img_rgb)
    
    # Convert to Base64 to send back to phone
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return {
        "success": True,
        "count": len(detected_objects),
        "detections": detected_objects,
        "image_base64": img_str
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
