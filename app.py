import sys
import os
import torch
# Patch torch.classes.__path__ for Streamlit's local source watcher compatibility
if hasattr(torch, 'classes') and not hasattr(torch.classes, '__path__'):
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

# Export top-level app and handler for Vercel Serverless Function compatibility
try:
    from api.index import app, handler
except Exception:
    app = None
    handler = None

import json
import io
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection & U-Net Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    /* Dark theme styling */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .title-text {
        font-family: 'Inter', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        color: #A0AAB8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .status-alert-danger {
        background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(185,28,28,0.3) 100%);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #FCA5A5;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .status-alert-success {
        background: linear-gradient(135deg, rgba(16,185,129,0.2) 0%, rgba(4,120,87,0.3) 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #6EE7B7;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Import model & utilities
from model import UNet, predict_mri_image, load_or_create_model, get_device

@st.cache_resource
def get_cached_model():
    weights_path = "model_best_checkpoint.pth"
    device = get_device()
    return load_or_create_model(weights_path=weights_path, device=device)

@st.cache_data
def get_sample_crops(samples_dir="samples"):
    """Caches sample MRI crops as PIL Images to avoid MediaFileHandler missing file warnings."""
    crops = {}
    if os.path.exists(samples_dir):
        s_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.png')])
        for sf in s_files:
            spath = os.path.join(samples_dir, sf)
            try:
                img = Image.open(spath).convert('RGB')
                arr = np.array(img)
                w = arr.shape[1]
                mri_crop = arr[:, :int(w / 3), :]
                crops[sf] = Image.fromarray(mri_crop)
            except Exception as e:
                print(f"Error caching sample {sf}: {e}")
    return crops

def create_color_overlay(mri_gray, mask, opacity=0.4, colormap_name="Jet"):
    """Blends single-channel gray MRI with segmentation mask using color mapping."""
    h, w = mri_gray.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    mri_rgb = cv2.cvtColor(mri_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)

    if colormap_name == "Jet":
        cmap = plt.get_cmap('jet')
    elif colormap_name == "Viridis":
        cmap = plt.get_cmap('viridis')
    elif colormap_name == "Crimson":
        cmap = plt.get_cmap('magma')
    elif colormap_name == "Hot":
        cmap = plt.get_cmap('hot')
    else:
        cmap = plt.get_cmap('jet')

    colored_mask = (cmap(mask.astype(np.float32))[:, :, :3] * 255.0).astype(np.float32)
    blend = mri_rgb.copy()
    mask_indices = mask > 0
    blend[mask_indices] = (1.0 - opacity) * mri_rgb[mask_indices] + opacity * colored_mask[mask_indices]
    return np.clip(blend, 0, 255).astype(np.uint8)

def draw_bounding_boxes(image_rgb, boxes):
    img_copy = image_rgb.copy()
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 50, 50), 2)
        cv2.putText(img_copy, "TUMOR", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)
    return img_copy

def main():
    st.markdown('<div class="title-text">🧠 Brain Tumor Detection & U-Net Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Deep Learning Clinical Decision Support Tool using PyTorch U-Net Model</div>', unsafe_allow_html=True)

    model = get_cached_model()
    sample_crops = get_sample_crops()

    # Sidebar Navigation & Settings
    st.sidebar.header("⚙️ Controls & Input")
    input_mode = st.sidebar.radio("Select Input Source:", ["Sample MRI Scans", "Upload Custom MRI"])

    selected_image = None
    image_name = "MRI_Scan"

    if input_mode == "Sample MRI Scans":
        if sample_crops:
            sample_choice = st.sidebar.selectbox("Choose a Brain MRI Sample:", list(sample_crops.keys()))
            selected_image = sample_crops[sample_choice]
            image_name = sample_choice
        else:
            st.sidebar.warning("No sample scans found.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload MRI Image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_name = uploaded_file.name

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Detection Parameters")
    threshold = st.sidebar.slider("Probability Threshold:", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
    opacity = st.sidebar.slider("Mask Overlay Opacity:", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    colormap_choice = st.sidebar.selectbox("Color Palette:", ["Jet", "Viridis", "Crimson", "Hot"])

    if selected_image is None:
        st.info("👈 Please select a sample MRI scan from the sidebar or upload your own MRI image to begin detection.")
        return

    # Run U-Net Model Inference
    result = predict_mri_image(selected_image, model=model, threshold=threshold)

    has_tumor = result["has_tumor"]
    confidence = result["confidence"]
    area_pct = result["area_percentage"]
    pixel_count = result["pixel_count"]
    mask_128 = result["binary_mask_128"]
    resized_gray = result["resized_gray"]
    boxes = result["bounding_boxes"]

    # Render Status Banner
    if has_tumor:
        st.markdown(f'''
        <div class="status-alert-danger">
            ⚠️ BRAIN TUMOR DETECTED | Confidence: {confidence*100:.1f}% | Estimated Area: {area_pct:.2f}% of MRI region
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="status-alert-success">
            ✅ NO TUMOR DETECTED | Normal Scan Profile | Confidence: {(1.0 - confidence)*100:.1f}%
        </div>
        ''', unsafe_allow_html=True)

    # 3-Panel Visual Display
    c1, c2, c3 = st.columns(3)

    overlay_img = create_color_overlay(resized_gray, mask_128, opacity=opacity, colormap_name=colormap_choice)
    if has_tumor and boxes:
        overlay_img = draw_bounding_boxes(overlay_img, boxes)

    # Convert images to PIL Objects for clean rendering without missing media file warnings
    pil_gray = Image.fromarray(resized_gray)
    pil_mask = Image.fromarray((mask_128 * 255).astype(np.uint8))
    pil_overlay = Image.fromarray(overlay_img)

    with c1:
        st.subheader("1. Preprocessed MRI")
        st.image(pil_gray, use_container_width=True, caption=f"Original ({result['orig_shape'][1]}x{result['orig_shape'][0]}) -> 128x128")

    with c2:
        st.subheader("2. U-Net Tumor Mask")
        st.image(pil_mask, use_container_width=True, caption=f"Binary Mask (Threshold: {threshold})")

    with c3:
        st.subheader("3. Tumor Overlay Visualizer")
        st.image(pil_overlay, use_container_width=True, caption=f"Blended Overlay ({colormap_choice} Palette)")

    st.markdown("---")

    # Interactive Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Report", "🏗️ U-Net Architecture", "🖼️ Sample Gallery Explorer"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Clinical Metrics Summary")
            st.table({
                "Metric": ["Tumor Status", "Detection Confidence", "Tumor Surface Area (%)", "Tumor Surface Area (Pixels)", "Bounding Box Regions"],
                "Value": [
                    "Detected 🚨" if has_tumor else "Normal Clean ✅",
                    f"{confidence*100:.2f}%",
                    f"{area_pct:.2f}%",
                    f"{pixel_count} px",
                    f"{len(boxes)} region(s)"
                ]
            })

        with col_b:
            st.markdown("### Export Diagnostic Data")
            report_data = {
                "image_name": image_name,
                "has_tumor": bool(has_tumor),
                "confidence_score": float(confidence),
                "tumor_area_percent": float(area_pct),
                "tumor_pixel_count": int(pixel_count),
                "threshold_used": float(threshold),
                "bounding_boxes": boxes
            }
            json_report = json.dumps(report_data, indent=2)

            st.download_button(
                label="📥 Download Diagnostic Summary (JSON)",
                data=json_report,
                file_name=f"tumor_report_{image_name}.json",
                mime="application/json"
            )

            buf = io.BytesIO()
            pil_mask.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Segmented Mask (PNG)",
                data=buf.getvalue(),
                file_name=f"mask_{image_name}.png",
                mime="image/png"
            )

    with tab2:
        st.markdown("### PyTorch U-Net Network Topology")
        st.code("""
Input Image Tensor: (Batch, 1, 128, 128)
├── Contracting Path (Encoder):
│   ├── Block 1: Conv2D(1 -> 64)   + BatchNorm + ReLU -> (Batch, 64, 128, 128) [Skip 1]
│   │   └── MaxPool2D(2x2)                         -> (Batch, 64, 64, 64)
│   ├── Block 2: Conv2D(64 -> 128)  + BatchNorm + ReLU -> (Batch, 128, 64, 64)  [Skip 2]
│   │   └── MaxPool2D(2x2)                         -> (Batch, 128, 32, 32)
│   ├── Block 3: Conv2D(128 -> 256) + BatchNorm + ReLU -> (Batch, 256, 32, 32)  [Skip 3]
│   │   └── MaxPool2D(2x2)                         -> (Batch, 256, 16, 16)
│   └── Block 4: Conv2D(512 -> 512) + BatchNorm + ReLU -> (Batch, 512, 16, 16)  [Skip 4]
│       └── MaxPool2D(2x2)                         -> (Batch, 512, 8, 8)
├── Bottleneck:
│   └── Conv2D(512 -> 1024)        + BatchNorm + ReLU -> (Batch, 1024, 8, 8)
├── Expanding Path (Decoder):
│   ├── ConvTranspose2D(1024 -> 512) + Concat(Skip 4) -> Conv2D -> (Batch, 512, 16, 16)
│   ├── ConvTranspose2D(512 -> 256)  + Concat(Skip 3) -> Conv2D -> (Batch, 256, 32, 32)
│   ├── ConvTranspose2D(256 -> 128)  + Concat(Skip 2) -> Conv2D -> (Batch, 128, 64, 64)
│   └── ConvTranspose2D(128 -> 64)   + Concat(Skip 1) -> Conv2D -> (Batch, 64, 128, 128)
└── Final Output Layer:
    └── Conv2D(64 -> 1, kernel=1x1) + Sigmoid        -> (Batch, 1, 128, 128)
        """, language="text")

    with tab3:
        st.markdown("### Quick Sample Gallery Explorer")
        if sample_crops:
            cols = st.columns(3)
            for idx, (sf, crop_pil) in enumerate(sample_crops.items()):
                col = cols[idx % 3]
                res_sample = predict_mri_image(crop_pil, model=model, threshold=threshold)
                t_str = "🚨 Tumor" if res_sample["has_tumor"] else "✅ Clean"
                col.image(crop_pil, caption=f"{sf} - {t_str}", use_container_width=True)

if __name__ == '__main__':
    main()
