from neuro_radiologist.src.models.vit_3d import PatchEmbedding3D
import torch

# Create a dummy batch of 2 MRI volumes matching our dataloader output
dummy_batch = torch.randn(2, 1, 64, 64, 64)

# Initialize the patch embedding layer
# We use a patch size of 16. So 64/16 = 4 patches per dimension.
patch_embed = PatchEmbedding3D(image_size=64, patch_size=16, in_channels=1, embed_dim=768)

# Pass the dummy batch through
sequence = patch_embed(dummy_batch)

print(f"Input Volume Shape: {dummy_batch.shape}")
print(f"Output Sequence Shape: {sequence.shape}")
print(f"Total Patches (N): {patch_embed.num_patches}")



from neuro_radiologist.src.models.vit_3d import ViT3D
import torch

dummy_batch = torch.randn(2, 1, 64, 64, 64)

# Initialize the main model shell
model = ViT3D(image_size=64, patch_size=16, in_channels=1, embed_dim=768)

# Forward pass through the embedding and positional encoding phase
sequence_with_pos = model(dummy_batch)

print(f"Final Sequence Shape before Encoder: {sequence_with_pos.shape}")