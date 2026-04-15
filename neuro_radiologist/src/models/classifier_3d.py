import torch
import torch.nn as nn
from neuro_radiologist.src.models.mae_3d import MaskedAutoencoder3D

class NeuroRadiologistClassifier(nn.Module):
    """
    Downstream diagnostic classifier leveraging a pre-trained 3D MAE Encoder.
    """
    def __init__(self, pretrained_mae: MaskedAutoencoder3D, num_classes=2):
        super().__init__()
        
        # 1. Extract the pre-trained encoder components
        self.patch_embed = pretrained_mae.patch_embed
        self.cls_token = pretrained_mae.cls_token
        self.pos_embed = pretrained_mae.pos_embed
        self.encoder_blocks = pretrained_mae.encoder_blocks
        self.encoder_norm = pretrained_mae.encoder_norm
        
        # 2. The New Classification Head
        # Maps the 768-dimensional CLS token to our diagnostic classes (e.g., Healthy vs. Tumor)
        self.head = nn.Linear(self.patch_embed.embed_dim, num_classes)

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        attn_map = None
        
        # Process through all blocks
        for i, block in enumerate(self.encoder_blocks):
            # If it's the very last block AND we requested attention
            if i == len(self.encoder_blocks) - 1 and return_attention:
                x, attn_map = block(x, return_attention=True)
            else:
                x = block(x) # Normal forward pass for other blocks
            
        x = self.encoder_norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        
        if return_attention:
            return logits, attn_map
        return logits