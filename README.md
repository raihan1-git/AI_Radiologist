# 🧠 Explainable 3D AI Neuro-Radiologist

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Deep%20Learning-blue.svg)](https://monai.io/)
[![Gemini API](https://img.shields.io/badge/Gemini%202.5-LLM-orange.svg)](https://deepmind.google/technologies/gemini/)

An end-to-end, multimodal 3D Deep Learning architecture designed to process multi-parametric MRI volumes, localize anatomical anomalies using attention heatmaps, and generate dynamic clinical reports via Large Language Models.

---

## 🚀 Live Demo
**[Try the Live Web App Here](https://YOUR-APP-URL.streamlit.app)** *(Note: Please upload standard 4-Channel NIfTI files like those found in the BraTS dataset).*

---

## 🔬 Architecture Overview

Transitioning from 2D pixel grids to 3D voxel spaces introduces massive computational bottlenecks (specifically the $O(N^2)$ complexity of global attention). This project bypasses standard ResNet wrappers by implementing a custom **3D Vision Transformer (ViT)** backbone, trained from scratch.

### 1. Self-Supervised Pre-Training (3D MAE)
Instead of relying on massive labeled datasets, the foundational model is an asymmetric **Masked Autoencoder (MAE)**. 
* It patches $64 \times 64 \times 64$ multi-modal NIfTI volumes (T1, T1c, T2, FLAIR) into $16^3$ cubes.
* Masks 75% of the sequence, forcing the transformer to mathematically learn the underlying spatial anatomy of the human brain by reconstructing the missing voxels.
* **Activations:** Uses `GELU` non-linearities for stable 3D gradient flow.

### 2. Explainable Downstream Fine-Tuning
The pre-trained encoder weights are frozen and attached to a classification head. The system extracts the raw $12 \times 65 \times 65$ attention matrices from the final transformer block, upsamples them via 3D trilinear interpolation, and overlays them onto the original MRI slices. This transforms the network from a "black box" into an explainable diagnostic tool.

### 3. Medical VQA Integration (Gemini 2.5 Flash)
The UI bridges the gap between pure tensor math and clinical utility. The PyTorch inference engine forwards the diagnostic logits, confidence scores, and highest-density attention coordinates $(Z, Y, X)$ to the Gemini API, dynamically generating two distinct readouts:
1. A dense, highly technical Clinical Impression for the attending radiologist.
2. An empathetic, jargon-free summary for the patient.

---

## 🛠️ Tech Stack & Pipeline

* **Data Ingestion:** `MONAI` (Transforms, Spacing, 3D Center Spatial Cropping)
* **Neural Network:** `PyTorch` (Custom ViT, MAE, bfloat16 AMP)
* **Generative AI:** `google-generativeai` (Gemini 2.5 Flash)
* **Frontend:** `Streamlit` (Interactive Volumetric Viewer, Z-Axis Slicing, Alpha Blending)

---

## 💻 Local Setup & Installation

If you wish to run the inference engine locally or train the MAE from scratch:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AI_Radiologist.git](https://github.com/YOUR_USERNAME/AI_Radiologist.git)
   cd AI_Radiologist/neuro_radiologist