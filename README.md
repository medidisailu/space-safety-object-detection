# Space Safety Object Detection 🚀

## Overview
This project demonstrates **real-time safety object detection** using **YOLOv8** integrated with a **Streamlit app**.  
It was developed for hackathons and academic demos, focusing on detecting safety-related objects (like helmets, vests, or restricted items) with a polished user interface and reproducible workflow.

## Features
- 🔍 Object detection powered by YOLOv8  
- 🎛️ Adjustable confidence threshold slider  
- 🖼️ Image upload and prediction visualization  
- 📊 Evaluation metrics reporting  
- 🌐 Streamlit app for interactive demo  
- 📂 Organized dataset preprocessing and training scripts  

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
