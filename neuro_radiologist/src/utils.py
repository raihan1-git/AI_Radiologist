import torch

def generate_3d_attention_map(attn_weights, grid_size=(4, 4, 4)):
    """
    Converts raw ViT attention weights into a 3D spatial heatmap.
    attn_weights shape: (B, Heads, N_queries, N_keys)
    """
    # 1. Extract CLS token attention (Index 0) to all spatial patches (Index 1 onwards)
    # Shape becomes: (B, Heads, 64)
    cls_attention = attn_weights[:, :, 0, 1:]
    
    # 2. Average across all attention heads
    # Shape becomes: (B, 64)
    consensus_attention = cls_attention.mean(dim=1)
    
    # 3. Normalize the attention scores between 0 and 1 for easier visualization
    # We do this per-patient in the batch
    min_val = consensus_attention.min(dim=1, keepdim=True)[0]
    max_val = consensus_attention.max(dim=1, keepdim=True)[0]
    normalized_attention = (consensus_attention - min_val) / (max_val - min_val + 1e-8)
    
    # 4. Reshape back into the 3D grid
    # Shape becomes: (B, 4, 4, 4)
    heatmap_3d = normalized_attention.view(-1, grid_size[0], grid_size[1], grid_size[2])
    
    return heatmap_3d