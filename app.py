import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


#--- Streamlit page configuration
st.set_page_config(
    page_title = "DiaChestXray - Chest X-ray Disease Detection",
    page_icon = "🩻",
    layout = "wide"
)

DEVICE = torch.device("cpu")
ALEX_MODEL_PATH = "models/alex_best.pth"
DENSE_MODEL_PATH = "models/dense_best.pth"
CLASSES_ALEX = ['Covid19', 'Normal', 'Pneumonia'] 
CLASSES_DENSE = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'No finding', 
    'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 
    'Pneumothorax', 'Pulmonary fibrosis'
]

#--- Load Models
@st.cache_resource
def load_models():
    alex = models.alexnet(weights = None)
    alex.classifier[6] = nn.Linear(4096, len(CLASSES_ALEX))

    dense = models.densenet121(weights = None)
    dense.classifier = nn.Linear(1024, len(CLASSES_DENSE))

    try:
        if os.path.exists(ALEX_MODEL_PATH):
            alex.load_state_dict(torch.load(ALEX_MODEL_PATH, map_location=DEVICE))
            alex.eval()
        else: alex = None
        
        if os.path.exists(DENSE_MODEL_PATH):
            dense.load_state_dict(torch.load(DENSE_MODEL_PATH, map_location=DEVICE))
            dense.eval()
        else: dense = None
            
        return alex, dense
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None
    
def preprocess_image(image):
    tf = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf(image).unsqueeze(0)

# --- Heatmap Generation (XAI)
def run_xai(model, input_tensor, original_image, target_layer):
    with torch.set_grad_enabled(True):
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        img_resized = original_image.resize((227, 227))
        rgb_img = np.float32(img_resized) / 255
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return visualization

# --- Main Application
st.sidebar.title("DiaChestXray System")
st.sidebar.info("This application allows you to upload chest X-ray images and get predictions for diseases.")

alex_model, dense_model = load_models()
if alex_model and dense_model:
    st.sidebar.success("Models loaded successfully!")
else:
    st.sidebar.error("Failed to load models. Please check the model paths.")

st.title("🩻 DiaChestXray - Chest X-ray Disease Detection")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image, only support JPG/JPEG/PNG.", type=["png", "jpg", "jpeg"])

if uploaded_file and alex_model and dense_model:
    col_img, col_res = st.columns([1, 1.5])

    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess_image(image)

    with col_img:
        st.subheader("Original Film Image")
        st.image(image, use_column_width=True)
        show_heatmap = st.toggle("🔍 Show Heatmap Explanations (XAI)", value=False)

    with col_res:
        st.subheader("📋 Prediction Results")

        # AlexNet Prediction
        st.markdown("### Filter Covid-19")
        with st.spinner("Predicting with AlexNet..."):
            out_alex = alex_model(input_tensor)
            probs_alex = torch.softmax(out_alex, 1)[0]
            conf, idx = torch.max(probs_alex, 0)
            label_alex = CLASSES_ALEX[idx]

            if label_alex == 'Normal':
                st.success(f"**Prediction:** {label_alex} (Confidence: {conf.item()*100:.2f}%) - No signs of Covid-19 detected.")
            elif label_alex == 'Covid19':
                st.error(f"**⚠️ Warning:** {label_alex} (Confidence: {conf.item()*100:.2f}%) - Signs of Covid-19 detected!")
            else:
                st.warning(f"**⚠️ Warning:** {label_alex} (Confidence: {conf.item()*100:.2f}%) - Possible Pneumonia detected.")
            
            st.progress(float(conf))
        st.divider()

        # DenseNet Prediction
        st.markdown("### Multi-label Disease Detection")
        with st.spinner("Predicting with DenseNet..."):
            out_dense = dense_model(input_tensor)
            probs_dense = torch.sigmoid(out_dense)[0]
            
            results = []
            for i, p in enumerate(probs_dense):
                if p > 0.15: 
                    results.append((CLASSES_DENSE[i], p.item()))
            results.sort(key=lambda x: x[1], reverse=True)
            
            if not results:
                st.info("No diseases detected with high confidence.")
            else:
                for name, score in results:
                    st.write(f"**{name}**")
                    st.progress(score)
    
    # Heatmap Visualization
    if show_heatmap:
        with col_img:
            st.markdown("---")
            st.subheader("Explanation Heatmaps (XAI)")
            
            if label_alex != "Normal":
                heatmap = run_xai(alex_model, input_tensor, image, alex_model.features[-1])
                st.image(heatmap, caption=f"Suspicious Injury ({label_alex})", use_column_width=True)
            elif results:
                heatmap = run_xai(dense_model, input_tensor, image, dense_model.features.denseblock4.denselayer16)
                st.image(heatmap, caption=f"Suspicious Injury ({results[0][0]})", use_column_width=True)
            else:
                st.info("No significant regions detected for heatmap generation.")

else:
    if not uploaded_file:
        st.info("👈 Please upload an image to begin diagnosis.")