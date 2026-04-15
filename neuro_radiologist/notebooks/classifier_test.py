import torch
from neuro_radiologist.src.models.mae_3d import MaskedAutoencoder3D
from neuro_radiologist.src.models.classifier_3d import NeuroRadiologistClassifier
from neuro_radiologist.src.utils import generate_3d_attention_map

# 1. Create the Dummy Data (2 patients, 1 channel, 64x64x64 volume)
dummy_batch = torch.randn(2, 1, 64, 64, 64)

# 2. Initialize the pre-trained MAE
# (In a real workflow, we would use torch.load() to load the weights saved from Phase 3)
pretrained_mae = MaskedAutoencoder3D()

# 3. Initialize the Classifier, passing in the MAE
# Let's assume 2 classes: 0 (Healthy), 1 (Anomaly Detected)
classifier = NeuroRadiologistClassifier(pretrained_mae, num_classes=2)

# Run forward pass, explicitly asking for the attention map
logits, attn_weights = classifier(dummy_batch, return_attention=True)

print(f"Diagnostic Logits Shape: {logits.shape}")
print(f"Raw Attention Weights Shape: {attn_weights.shape}")

# Generate the 3D Heatmap!
heatmap = generate_3d_attention_map(attn_weights, grid_size=(4, 4, 4))

print(f"Final 3D Heatmap Shape: {heatmap.shape}")
print(f"Heatmap Min/Max values: {heatmap.min():.2f} / {heatmap.max():.2f}")