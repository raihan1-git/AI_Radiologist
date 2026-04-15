import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding3D(nn.Module):
    """
    Converts a 3D volumetric tensor into a sequence of project patch embeddings.
    """
    def __init__(self, image_size=64, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate the total number of patches (N)
        self.num_patches = (image_size // patch_size) ** 3
        
        # The 3D Convolution acts as our patch extractor and linear projection
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        x = self.proj(x) 
        # After proj shape: (B, embed_dim, D/P, H/P, W/P)
        
        # Flatten the spatial dimensions into a single sequence dimension 'N'
        # and swap embed_dim to the last axis.
        x = rearrange(x, 'b e d h w -> b (d h w) e')
        # Final shape: (B, N, embed_dim)
        return x

class ViT3D(nn.Module):
    """
    The main 3D Vision Transformer architecture.
    """
    def __init__(self, image_size=64, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        
        # 1. The Patch Embedder we just built
        self.patch_embed = PatchEmbedding3D(
            image_size=image_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # 2. The Learnable Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. The Learnable Positional Embedding (N patches + 1 CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Standard ViT initialization practice: 
        # Truncated normal distribution for parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Extract patches -> Shape: (B, N, embed_dim)
        x = self.patch_embed(x)
        
        # 2. Expand CLS token to match batch size -> Shape: (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 3. Concatenate CLS token to sequence -> Shape: (B, N+1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4. Add Positional Embeddings -> Shape: (B, N+1, embed_dim)
        x = x + self.pos_embed
        
        return x
    
class TransformerBlock(nn.Module):
    """
    A standard Pre-Norm Transformer block.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Pre-normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Self Attention (batch_first=True because our shape is B, N, D)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Second normalization
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP Block using GELU
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attention=False):
        # We explicitly request the attention weights if the flag is True
        attn_out, attn_weights = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x), 
            need_weights=return_attention, 
            average_attn_weights=False # Keep heads separate for now
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x

# --- UPDATE YOUR ViT3D CLASS ---

class ViT3D(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=1, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # ADDED: The sequence of Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # ADDED: Final LayerNorm (Standard practice before the final head)
        self.norm = nn.LayerNorm(embed_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # ADDED: Pass through the Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # ADDED: Apply final norm
        x = self.norm(x)
        
        return x