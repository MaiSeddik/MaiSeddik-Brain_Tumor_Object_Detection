import os
import io
import tempfile
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection - YOLO11",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .stAppViewContainer {
        background-color: #F8FAFC;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions & Model Caching
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    """
    Load YOLO model with caching to prevent reload overhead.
    """
    return YOLO(model_path)


def resolve_default_model_path() -> str:
    """
    Find local model weights (best.pt or 'best .pt').
    """
    if os.path.exists("best.pt"):
        return "best.pt"
    elif os.path.exists("best .pt"):
        return "best .pt"
    return "best.pt"


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=70)
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Model Settings")

# Model selection options
model_option = st.sidebar.selectbox(
    "Select Model Weights",
    options=["Fine-tuned (best.pt)", "Pre-trained YOLO11 (yolo11s.pt)", "Upload Custom Weights (.pt)"],
    index=0,
    help="Choose between fine-tuned brain tumor model weights, base pretrained model, or upload custom weights."
)

model_path = None

if model_option == "Fine-tuned (best.pt)":
    default_path = resolve_default_model_path()
    if os.path.exists(default_path):
        model_path = default_path
    else:
        st.sidebar.error(f"⚠️ `{default_path}` not found locally. Falling back to `yolo11s.pt` or upload custom weights.")
        model_path = "yolo11s.pt"

elif model_option == "Pre-trained YOLO11 (yolo11s.pt)":
    model_path = "yolo11s.pt"

elif model_option == "Upload Custom Weights (.pt)":
    uploaded_weights = st.sidebar.file_uploader("Upload PyTorch Model Weights (.pt)", type=["pt"])
    if uploaded_weights is not None:
        temp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        temp_model.write(uploaded_weights.read())
        temp_model.close()
        model_path = temp_model.name
    else:
        st.sidebar.info("Please upload a `.pt` file to proceed.")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Inference Hyperparameters")

# Hyperparameter sliders (as required by prompt)
conf_thresh = st.sidebar.slider(
    "Confidence Threshold (conf)",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.01,
    help="Minimum confidence threshold for detections (default: 0.45)."
)

iou_thresh = st.sidebar.slider(
    "IoU Threshold (iou)",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.01,
    help="Intersection-over-Union threshold for Non-Maximum Suppression (default: 0.70)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Dataset Class Labels")
st.sidebar.markdown("""
- **`label0`**: Brain Tumor Class 0
- **`label1`**: Brain Tumor Class 1
- **`label2`**: Brain Tumor Class 2
""")

# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
st.markdown('<div class="main-header">🧠 Brain Tumor Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated object detection and diagnosis in MRI brain scans using <b>Ultralytics YOLO11</b>.</div>', unsafe_allow_html=True)

# Main container for file upload
st.write("### 📤 Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose a brain MRI scan image (PNG, JPG, JPEG)...",
    type=["jpg", "jpeg", "png"],
    help="Upload an MRI scan image to perform object detection."
)

if uploaded_file is not None:
    try:
        # Load and convert image to RGB
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error opening image file: {e}")
        st.stop()

    if model_path is None:
        st.warning("⚠️ Please select or upload valid model weights in the sidebar.")
        st.stop()

    # Load Model with status spinner
    with st.spinner("⏳ Loading YOLO11 model weights..."):
        try:
            model = load_yolo_model(model_path)
        except Exception as e:
            st.error(f"Failed to load model weights from `{model_path}`: {e}")
            st.stop()

    # Perform Inference
    st.markdown("---")
    st.write("### 🔍 Inference & Visual Detection Results")

    start_time = time.time()
    with st.spinner("Running YOLO11 object detection..."):
        results = model.predict(source=image, conf=conf_thresh, iou=iou_thresh)
    inference_time = (time.time() - start_time) * 1000  # in ms

    res = results[0]
    boxes = res.boxes

    # Generate annotated output image
    annotated_frame = res.plot()  # Returns BGR numpy array
    annotated_image = annotated_frame[..., ::-1]  # BGR to RGB using pure NumPy
    annotated_pil = Image.fromarray(annotated_image)

    # Side-by-Side Image Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original MRI Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Detection Visualization")
        st.image(annotated_pil, use_container_width=True)

    # Summary Metrics
    st.markdown("---")
    st.write("### 📊 Detection Results Summary")

    total_detections = len(boxes) if boxes is not None else 0

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric(label="Total Detections", value=total_detections)
    with metric_col2:
        top_conf = f"{float(boxes.conf.max()):.2%}" if total_detections > 0 else "N/A"
        st.metric(label="Max Confidence", value=top_conf)
    with metric_col3:
        unique_classes = len(set(boxes.cls.cpu().numpy().astype(int))) if total_detections > 0 else 0
        st.metric(label="Unique Tumor Classes", value=unique_classes)
    with metric_col4:
        st.metric(label="Inference Time", value=f"{inference_time:.1f} ms")

    # Detailed Detection Table
    if total_detections > 0:
        st.write("#### 📋 Detailed Detections Breakdown")
        
        detection_data = []
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            class_name = model.names.get(cls_id, f"label{cls_id}")
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            
            detection_data.append({
                "Detection #": idx + 1,
                "Class Label": class_name,
                "Class ID": cls_id,
                "Confidence": f"{confidence:.2%}",
                "Confidence Value": round(confidence, 4),
                "Bounding Box [xmin, ymin, xmax, ymax]": str(xyxy)
            })

        df_detections = pd.DataFrame(detection_data)
        st.dataframe(
            df_detections[["Detection #", "Class Label", "Confidence", "Bounding Box [xmin, ymin, xmax, ymax]"]],
            use_container_width=True
        )

        # Download Buttons
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image",
                data=buf.getvalue(),
                file_name="detected_brain_tumor.png",
                mime="image/png"
            )
        with dl_col2:
            csv_data = df_detections.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name="detection_results.csv",
                mime="text/csv"
            )
    else:
        st.info("ℹ️ No brain tumors detected above the current confidence threshold. Try adjusting `conf` or `iou` sliders in the sidebar.")

else:
    # Display prompt / instructions when no image is uploaded yet
    st.info("👆 Upload a brain MRI scan above to begin object detection inference.")
    
    st.markdown("---")
    st.markdown("""
    #### 💡 Quick Features Guide:
    1. **Upload MRI Images**: Supports `.jpg`, `.jpeg`, and `.png` file formats.
    2. **Model Selection**: Switch between fine-tuned `best.pt` weights or standard pretrained YOLO11 models.
    3. **Interactive Controls**: Fine-tune **Confidence Threshold** (`conf`) and **NMS IoU Threshold** (`iou`) in real time.
    4. **Downloadable Outputs**: Export annotated scans and structured detection reports in CSV format.
    """)
