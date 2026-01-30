# 🚀 Space Safety Object Detection
Detect safety-related objects (OxygenTank,NitrogenTank ,FirstAidBox ,FireAlarm ,SafetySwitchPanel ,EmergencyPhone ,FireExtinguisher) using YOLOv8 and Streamlit.

## 🔗 Live Demo
[Click here to try the app](https://space-safety-object-detection-ksbwdfdkrr78cpywp6yztb.streamlit.app/)

## Overview
This project demonstrates a **Cloud-Native Space Safety System** using a **Microservices Architecture**.
It detects safety-related equipment (Oxygen Tanks, First Aid, etc.) on space stations using **YOLOv8**, accessible via a **Global Web Dashboard** and a **Mobile App**.

## System Architecture
*   **Backend (Service #1):** FastAPI server deployed on Render (Cloud). Handles AI Inference.
*   **Frontend (Service #2):** Streamlit Web Dashboard deployed on Render. Serves as the Control Center.
*   **Mobile App:** React Native (Expo) app for astronauts/rovers to verify objects in real-time.

## Features
- 🌍 **Global Access**: Accessible on any network (4G/Wi-Fi) without tunneling.
- 📱 **Mobile Integration**: Real-time object detection via Mobile Phone Camera.
- 🚀 **Advanced UI**: Sci-Fi themed dashboard for the web interface.
- ☁️ **Cloud AI**: Heavy lifting done on the server, keeping client devices fast.
- 🔍 **YOLOv8 Power**: High-accuracy detection of 7 safety classes.  

## Project Structure
ML2/
├── data/
│   ├── preprocess/train/
│   ├── preprocessed/
│   ├── test/
│   ├── train/
│   ├── valid/
│   ├── data.yaml
│   ├── README.dataset.txt
│   └── README.roboflow.txt
├── runs/
├── scripts/
│   ├── app.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── yolov8s.pt
├── yolov8n.pt
├── yolov8x.pt
├── .gitattributes
├── LICENSE
├── README.md
└── requirements.txt



## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/medidisailu/space-safety-object-detection.git
cd space-safety-object-detection
pip install -r requirements.txt

## Usage
### Run the Streamlit App
```bash
streamlit run scripts/app.py
- Upload an image
- Adjust confidence threshold
- View predictions and detection results


##Train the Model
python scripts/train.py

##Evaluate the Model
python scripts/evaluate.py

##Run Predictions
python scripts/predict.py --source path/to/image.jpg

##Dataset
- Custom dataset prepared for safety object detection
- Preprocessing scripts included in scripts/preprocess.py
- Supports YOLOv8 annotation format

##Demo
Screenshots or GIFs can be added here to showcase:
- Streamlit interface
- Detection results on sample images

##Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

##Acknowledgments
- Ultralytics YOLOv8 for providing the object detection framework
- Streamlit for enabling an interactive and user-friendly app interface
- OpenCV for image processing utilities
- Hackathon mentors and collaborators for their guidance, feedback, and support
