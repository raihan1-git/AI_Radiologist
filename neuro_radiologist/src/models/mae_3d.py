import torch
import torch.nn as nn
from einops import rearrange
from neuro_radiologist.src.models.vit_3d import PatchEmbedding3D, TransformerBlock

class MaskedAutoencoder3D(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=1, 
                 embed_dim=768, encoder_depth=12, encoder_heads=12,
                 decoder_embed_dim=384, decoder_depth=4, decoder_heads=6): # ADDED DECODER PARAMS
        super().__init__()
        self.patch_embed = PatchEmbedding3D(image_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # --- Encoder Components (Keep your existing ones) ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.encoder_blocks = nn.ModuleList([TransformerBlock(embed_dim, encoder_heads) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # --- ADDED: Decoder Components ---
        # 1. Project from encoder dimension to smaller decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # 2. The Learnable Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 3. Decoder Positional Embeddings (Needs to know spatial location)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        # 4. Lightweight Decoder Blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_heads) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # 5. Final projection back to raw voxel pixels (16 * 16 * 16 * 1 = 4096)
        self.patch_dim = patch_size ** 3 * in_channels
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)
        
        # Initialize new weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def random_masking(self, sequence, mask_ratio):
        # ... (Keep your existing random_masking code here) ...
        B, N, D = sequence.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=sequence.device)
        ids_shuffle = torch.argsort(noise, dim=1) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones([B, N], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # 1. Embed 3D volumetric patches -> Shape: (B, N, D)
        x = self.patch_embed(x)
        
        # 2. Add positional embeddings to all patches BEFORE masking
        # We slice [:, 1:, :] to skip the CLS token's positional embedding for now
        x = x + self.pos_embed[:, 1:, :]
        
        # 3. Apply masking to isolate visible patches
        x_visible, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # 4. Setup and append the CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_visible.shape[0], -1, -1)
        
        # Final sequence to process: CLS + Visible Patches
        x = torch.cat((cls_tokens, x_visible), dim=1)
        
        # 5. Process through Asymmetric Encoder
        for block in self.encoder_blocks:
            x = block(x)
            
        x = self.encoder_norm(x)
        
        # We return ids_restore and mask because the decoder will need them!
        return x, mask, ids_restore


    def forward_decoder(self, x, ids_restore):
        # 1. Project encoder output to decoder dimension
        x = self.decoder_embed(x)
        
        # 2. Extract CLS token and visible patches
        cls_token = x[:, :1, :]
        x_visible = x[:, 1:, :]
        
        # 3. Create mask tokens for the missing patches
        B = x.shape[0]
        # Number of missing patches = Total - Visible
        num_masked = self.num_patches - x_visible.shape[1] 
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        
        # 4. Concatenate visible and mask tokens (Currently out of spatial order)
        x_ = torch.cat([x_visible, mask_tokens], dim=1)
        
        # 5. UN-SHUFFLE to restore original 3D spatial order using ids_restore
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))
        
        # 6. Re-attach CLS token and add decoder positional embeddings
        x = torch.cat([cls_token, x_], dim=1)
        x = x + self.decoder_pos_embed
        
        # 7. Pass through decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # 8. Project back to voxels
        x = self.decoder_pred(x)
        
        # Remove CLS token from final output, we only care about patches
        return x[:, 1:, :] 

    def forward_loss(self, imgs, pred, mask):
        """
        MSE loss computed ONLY on the masked patches.
        """
        p = self.patch_embed.patch_size
        
        # We rearrange (B, C, D, H, W) into (B, N, raw_patch_dim)
        # where raw_patch_dim = p * p * p * C
        target = rearrange(
            imgs, 
            'b c (d p_d) (h p_h) (w p_w) -> b (d h w) (p_d p_h p_w c)', 
            p_d=p, p_h=p, p_w=p
        )
        
        # Calculate Mean Squared Error
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
        
        # Apply mask: loss is 0 for visible patches, keep loss for hidden patches
        loss = (loss * mask).sum() / mask.sum() 
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        # Full end-to-end pipeline
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask