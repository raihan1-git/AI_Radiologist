import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import tempfile
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if repo_root not in sys.path:
    sys.path.append(repo_root)


import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityd, CenterSpatialCropd
from neuro_radiologist.src.models.mae_3d import MaskedAutoencoder3D
from neuro_radiologist.src.models.classifier_3d import NeuroRadiologistClassifier
from neuro_radiologist.src.utils import generate_3d_attention_map
from neuro_radiologist.app.llm_agent import MedicalReportGenerator

# 1. Page Configuration (Medical Dark Mode)
st.set_page_config(
    page_title="AI Neuro-Radiologist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: Inject some custom CSS for a cleaner, darker aesthetic
st.markdown("""
    <style>
    .reportview-container { background-color: #0E1117; }
    .sidebar .sidebar-content { background-color: #262730; }
    h1, h2, h3 { color: #FAFAFA; }
    </style>
""", unsafe_allow_html=True)

# 2. Application Header
st.title("🧠 AI Neuro-Radiologist")
st.markdown("*3D Vision Transformer & Masked Autoencoder Diagnostic Assistant*")
st.divider()

# 3. Sidebar for Controls & Upload
with st.sidebar:
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload 3D MRI Scan (.nii.gz)", type=['nii.gz'])
    
    st.divider()
    st.markdown("### Settings")
    # A slider to let the user scroll through the Z-axis of the 3D volume
    slice_index = st.slider("Axial Slice (Z-Axis)", min_value=0, max_value=63, value=32)
    # A slider for the alpha blending math we discussed
    heatmap_alpha = st.slider("Heatmap Transparency", min_value=0.0, max_value=1.0, value=0.5)


def render_slice(volume, heatmap, z_index, alpha):
    """Renders a 2D slice with an overlaid heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0E1117') # Match dark mode background
    
    # 1. Plot the base MRI slice in grayscale
    ax.imshow(volume[z_index, :, :], cmap='gray')
    
    # 2. Overlay the heatmap using the 'jet' colormap
    # We use vmin=0, vmax=1 assuming the heatmap is normalized
    img = ax.imshow(heatmap[z_index, :, :], cmap='jet', alpha=alpha, vmin=0, vmax=1)
    
    ax.axis('off')
    return fig

# --- NEW: Cache the PyTorch Model ---
@st.cache_resource
def load_ai_model():
    # 1. Initialize the MAE with 4 channels to match the BraTS data
    mae = MaskedAutoencoder3D(image_size=64, patch_size=16, in_channels=4, embed_dim=768)
    model = NeuroRadiologistClassifier(mae, num_classes=2)
    
    # 2. Load the trained weights! 
    # (map_location='cpu' is important since your local machine might not have a massive GPU)
    weight_path = "neuro_radiologist/weights/diagnostic_classifier_final.pth" # Update this path if needed
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    
    model.eval() # Lock it for inference
    return model

ai_model = load_ai_model()
llm_agent = MedicalReportGenerator()

# --- NEW: MONAI Inference Pipeline ---
inference_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    # We resize exactly to 64x64x64 for our ViT
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    ScaleIntensityd(keys=["image"]),
    CenterSpatialCropd(keys=["image"], roi_size=(64, 64, 64))
])

# 4. Main Content Columns

tmp_path = None
if uploaded_file is not None:
    # Save the file reliably to the local directory for this session
    tmp_path = "temp_uploaded_mri.nii.gz"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Volumetric Viewer")
    
    if uploaded_file is None or tmp_path is None:
        st.info("👈 Please upload a NIfTI file from the sidebar to begin.")
    else:
        # Pass the uploaded file through our MONAI pipeline to crop it perfectly
        data_dict = {"image": tmp_path}
        processed_data = inference_transforms(data_dict)
        
        # processed_data["image"] shape is [4, 64, 64, 64]. 
        # We extract Channel 0 (FLAIR modality) to display on the screen!
        volume = processed_data["image"][0].numpy()

        # Fetch the REAL Heatmap from Session State
        if 'generated_heatmap' in st.session_state:
            low_res_heatmap = st.session_state['generated_heatmap']
        else:
            low_res_heatmap = np.zeros((4, 4, 4))
        
        # UPSAMPLE the Heatmap
        upscale_factor = 64 / 4
        high_res_heatmap = ndimage.zoom(low_res_heatmap, zoom=upscale_factor, order=1)
        high_res_heatmap = np.clip(high_res_heatmap, 0, 1)

        # Render the Slice
        fig = render_slice(volume, high_res_heatmap, slice_index, heatmap_alpha)
        st.pyplot(fig)

with col2:
    st.subheader("Diagnostic Report")
    st.write("---")
    
    if uploaded_file is None:
        st.write("Awaiting scan data...")
    else:
        if st.button("Run AI Diagnosis", type="primary"):
            with st.spinner("Analyzing 3D volumetric data..."):
                
                # 1. Process data through MONAI
                data_dict = {"image": tmp_path}
                processed_data = inference_transforms(data_dict)
                input_tensor = processed_data["image"].unsqueeze(0) # Add batch dimension -> (1, 1, 64, 64, 64)
                
                # 2. PyTorch Inference Engine
                with torch.no_grad():
                    logits, raw_attention = ai_model(input_tensor, return_attention=True)
                
                # 3. Generate 3D Heatmap
                heatmap_3d = generate_3d_attention_map(raw_attention, grid_size=(4, 4, 4))
                # Store in Streamlit session state so col1 can use it for rendering!
                st.session_state['generated_heatmap'] = heatmap_3d.numpy()[0] 
                
                # 4. Generate ACTUAL LLM Reports (Updates here!)
                clinician_report, patient_report = llm_agent.generate_reports(logits, low_res_heatmap)
                
                # 5. Display the real results
                st.success("Analysis and Report Generation Complete.")
                
                # We use st.info and st.success blocks instead of st.code to read it naturally
                st.markdown("### 🩺 Clinician Report")
                st.info(clinician_report)
                
                st.markdown("### 🫂 Patient Summary")
                st.success(patient_report)
                    
        elif 'generated_heatmap' in st.session_state:
            st.success("Analysis complete. Check col1 for attention map.")