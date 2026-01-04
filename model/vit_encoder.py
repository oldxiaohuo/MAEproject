import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(1024, 512), patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        x = rearrange(x, 'b c h w -> b (h w) c')  # flatten patches
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=(1024, 512), patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.blocks(x)
        x = self.norm(x)
        return x
