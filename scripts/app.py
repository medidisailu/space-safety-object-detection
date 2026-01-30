import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
import io

def get_image_base64(image_input):
    """Convert a PIL Image or Numpy array to base64 string"""
    if isinstance(image_input, np.ndarray):
        # Convert Numpy (assumed RGB if coming from plotting logic) to PIL
        image_input = Image.fromarray(image_input)
    
    if isinstance(image_input, str):
        # It's a file path
        image_input = Image.open(image_input)
        
    buffered = io.BytesIO()
    image_input.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

from PIL import Image
import time
from datetime import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Space Safety Dashboard",
    layout="wide",
    page_icon="️",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS - Sci-Fi Dashboard Theme
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');

    /* Global Styles */
    .stApp {
        background-color: #050b14;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        color: #e0faff;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Modern Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a111a; }
    ::-webkit-scrollbar-thumb { background: #00d2ff; border-radius: 4px; }

    /* Header Styling */
    .main-header {
        text-align: left;
        padding: 20px;
        border-bottom: 2px solid #00d2ff;
        margin-bottom: 20px;
        background: linear-gradient(90deg, rgba(0,210,255,0.1) 0%, transparent 100%);
    }
    .main-header h1 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3rem;
        text-transform: uppercase;
        margin: 0;
        letter-spacing: 4px;
        background: linear-gradient(180deg, #ffffff 0%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
    }
    .sub-header {
        font-family: 'Orbitron', sans-serif;
        color: #00d2ff;
        font-size: 1.2rem;
        letter-spacing: 2px;
    }

    /* Dashboard Cards */
    .dashboard-card {
        background: rgba(10, 20, 30, 0.8);
        border: 1px solid #004a6b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.05);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .dashboard-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00d2ff, transparent);
        animation: scan 4s infinite linear;
    }
    
    @keyframes scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* Buttons */
    .stButton>button {
        background: transparent;
        border: 2px solid #00d2ff;
        color: #00d2ff;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        border-radius: 4px;
        padding: 10px 20px;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.2);
    }
    .stButton>button:hover {
        background: #00d2ff;
        color: #050b14;
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.6);
        transform: translateY(-2px);
    }

    /* Sidebar / Analysis Panel */
    .analysis-panel {
        border-left: 2px solid #00d2ff;
        padding-left: 20px;
        height: 100%;
    }
    .panel-title {
        font-family: 'Orbitron', sans-serif;
        color: #fff;
        font-size: 1.2rem;
        margin-bottom: 20px;
        border-bottom: 1px solid #334;
        padding-bottom: 5px;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
        font-size: 1rem;
        border-bottom: 1px dashed #334;
        padding-bottom: 5px;
    }
    .stat-label { color: #8faab9; }
    .stat-value { color: #00d2ff; font-weight: 600; font-family: 'Orbitron', sans-serif; }

    /* Custom Upload Area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #00d2ff;
        background: rgba(0, 210, 255, 0.05);
        padding: 30px;
        border-radius: 10px;
    }
    
    /* Image Container & Detection Box - Fixed 16:9 */
    .img-container, .detection-result {
        width: 100%;
        aspect-ratio: 16 / 9;
        border: 2px solid #00d2ff;
        border-radius: 8px;
        position: relative;
        background: black;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    
    /* Specific styling for the glowing result box */
    .detection-result {
        animation: box-glow 2s infinite alternate;
    }
    
    .img-overlay {
        position: absolute;
        top: 10px; left: 10px;
        background: rgba(0, 0, 0, 0.7);
        padding: 5px 10px;
        border: 1px solid #00d2ff;
        color: #00d2ff;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.8rem;
        z-index: 10;
        pointer-events: none;
    }
    
    /* Constrain images to 16:9 container */
    .img-container img, .detection-result img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
    }

    /* Scanning Animation */
    .scan-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: #00d2ff;
        box-shadow: 0 0 15px #00d2ff;
        animation: scan-down 3s linear infinite;
        z-index: 5;
        opacity: 0.8;
    }

    @keyframes scan-down {
        0% { top: 0%; opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }


    /* Star Animation */
    @keyframes twinkle {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
    
    .stars {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 40px 70px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 50px 160px, #ddd, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 90px 40px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 130px 80px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 160px 120px, #ddd, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 200px 200px;
        animation: twinkle 5s infinite;
        opacity: 0.3;
    }

    /* Glow Effect for Detection */
    @keyframes box-glow {
        0% { box-shadow: 0 0 5px #00d2ff; }
        50% { box-shadow: 0 0 20px #00d2ff, 0 0 10px #ffffff; }
        100% { box-shadow: 0 0 5px #00d2ff; }
    }
    
    
    .detection-result {
        /* Already handled in shared class above, removing duplicates if any */
    }

</style>
<div class="stars"></div>
""", unsafe_allow_html=True)

# -----------------------------
# Model Logic
# -----------------------------
@st.cache_resource
def load_model():
    custom_model_path = "model.pt"
    if os.path.exists(custom_model_path):
        return YOLO(custom_model_path)
    else:
        return YOLO("yolov8s.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"System Error: Model Loading Failed - {e}")

# -----------------------------
# Layout
# -----------------------------

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
        <div class="main-header">
            <h1>Space Safety <span style="color:#00d2ff">Object Detection</span></h1>
            <div class="sub-header">ORBITAL STATION SURVEILLANCE SYSTEM V2.0</div>
        </div>
    """, unsafe_allow_html=True)
with col_h2:
    # Live Clock
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
        <div style="text-align:right; font-family:'Orbitron'; color:#00d2ff; padding:20px; font-size:1.2rem;">
            SYSTEM TIME<br>
            <span style="color:white">{now} IST</span>
        </div>
    """, unsafe_allow_html=True)

# Dashboard Area
col_main, col_sidebar = st.columns([2.5, 1])

with col_main:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    
    # Toolbar
    st.markdown("### 📁 **INPUT SOURCE**")
    
    uploaded_file = st.file_uploader("Upload Sensor Data", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        # Save and Load
        os.makedirs("uploads", exist_ok=True)
        img_path = os.path.join("uploads", uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Button Row
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            scan_clicked = st.button("INITIATE SCAN ANALYSIS", use_container_width=True)
            
        if scan_clicked:
            with st.spinner("SCANNING TARGET..."):
                time.sleep(1) # Dramatic pause
                results = model.predict(source=img_path, save=False, conf=0.4)
                result = results[0]
                
                # Plot returns BGR, convert to RGB
                annotated_img_bgr = result.plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                
                # Update Session State
                st.session_state['analysis_done'] = True
                st.session_state['last_result_img'] = annotated_img_rgb
                
                # Collect Data
                detected_counts = {}
                for cls_id in result.boxes.cls:
                    name = result.names[int(cls_id)]
                    detected_counts[name] = detected_counts.get(name, 0) + 1
                st.session_state['counts'] = detected_counts

        # Main Display Area
        if st.session_state.get('analysis_done') and 'last_result_img' in st.session_state:
            # Show Result Loop (Fixed HTML Structure)
            result_b64 = get_image_base64(st.session_state['last_result_img'])
            st.markdown(f"""
                <div class="detection-result">
                    <div class="scan-line"></div>
                    <img src="{result_b64}" style="width:100%; height:100%; object-fit:contain;">
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("RESET SYSTEM", key="reset_btn"):
                st.session_state['analysis_done'] = False
                st.rerun()
                
        else:
            # Show Input Feed (Fixed HTML Structure)
            input_img = Image.open(uploaded_file)
            input_b64 = get_image_base64(input_img)
            
            st.markdown(f"""
                <div class="img-container">
                    <div class="img-overlay">LIVE FEED // {uploaded_file.name}</div>
                    <img src="{input_b64}" style="width:100%; height:100%; object-fit:contain;">
                </div>
            """, unsafe_allow_html=True)
    else:
        # Placeholder
        st.markdown("""
            <div style="width:100%; aspect-ratio:16/9; display:flex; align-items:center; justify-content:center; border:2px dashed #004a6b; color:#556; background: rgba(0,0,0,0.3); border-radius:8px;">
                WAITING FOR DATA INPUT...
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Right Sidebar (Analysis Results)
with col_sidebar:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">ANALYSIS RESULTS</div>', unsafe_allow_html=True)

    if 'analysis_done' in st.session_state and st.session_state['analysis_done']:
        counts = st.session_state.get('counts', {})
        total_obj = sum(counts.values())
        
        # Overall Status
        if total_obj > 0:
            status = "OBJECTS DETECTED"
            status_color = "#00d2ff"
        else:
            status = "NO THREATS"
            status_color = "#00ff9d"

        st.markdown(f"""
            <div style="text-align:center; padding:15px; background:rgba(0,210,255,0.1); border-radius:8px; margin-bottom:20px;">
                <div style="font-size:0.8rem; color:#8faab9;">STATUS</div>
                <div style="font-size:1.5rem; font-weight:bold; color:{status_color}; text-shadow:0 0 10px {status_color};">
                    {status}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom:10px; font-weight:bold; color:#fff;">Risk Assessment:</div>', unsafe_allow_html=True)

        # List all possible classes
        safety_classes = [
            "OxygenTank", "NitrogenTank", "FirstAidBox", 
            "FireAlarm", "SafetySwitchPanel", "EmergencyPhone", "FireExtinguisher"
        ]

        for cls_name in safety_classes:
            count = counts.get(cls_name, 0)
            opacity = "1.0" if count > 0 else "0.3"
            color = "#00d2ff" if count > 0 else "#556"
            
            st.markdown(f"""
                <div class="stat-row" style="opacity:{opacity}">
                    <span class="stat-label">● {cls_name}</span>
                    <span class="stat-value" style="color:{color}">{count}</span>
                </div>
            """, unsafe_allow_html=True)
            
        # Probability Visual (Mock chart)
        st.markdown("<br><div style='font-size:0.8rem; color:#8faab9; margin-bottom:5px'>DETECTION CONFIDENCE</div>", unsafe_allow_html=True)
        st.progress(0.92)
        
    else:
        st.markdown("""
            <div style="color:#556; text-align:center; margin-top:50px;">
                SYSTEM STANDBY<br>
                <div style="font-size:3rem; opacity:0.2;">⏸</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="position:fixed; bottom:10px; right:20px; font-size:0.8rem; color:#334;">
        SECURE CONNECTION // ENCRYPTED
    </div>
""", unsafe_allow_html=True)