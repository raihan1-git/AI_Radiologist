from neuro_radiologist.src.models.mae_3d import MaskedAutoencoder3D
import torch

# Our original dummy 5D batch: (Batch, Channels, Depth, Height, Width)
dummy_batch = torch.randn(2, 1, 64, 64, 64)

# Initialize the full MAE
mae = MaskedAutoencoder3D()

# Run the forward pass!
loss, pred, mask = mae(dummy_batch, mask_ratio=0.75)

print(f"Final Voxel Prediction Shape: {pred.shape}")
print(f"Calculated MSE Loss: {loss.item():.4f}")