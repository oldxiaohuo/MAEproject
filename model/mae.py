import torch
import torch.nn as nn
from model.vit_encoder import ViTEncoder  # 你已有的 ViT encoder
from einops import rearrange

class MAE(nn.Module):
    def __init__(self, img_size=(1024, 512), patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # ViT Encoder
        self.encoder = ViTEncoder(img_size, patch_size, in_chans, embed_dim)

        # learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Simple Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        )

    def patchify(self, imgs):
        # imgs: [B, 1, H, W] → [B, N, P*P]
        p = self.patch_size
        patches = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return patches

    def unpatchify(self, patches):
        # patches: [B, N, P*P*C] → [B, C, H, W]
        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p, c=self.in_chans)

    def random_masking(self, x, mask_ratio):
        B, N, _ = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # uniform noise
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        return x_masked, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.1):
        # Step 1: Patchify
        x = self.encoder.patch_embed(imgs)  # [B, N, C]

        # Step 2: Random masking
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Step 3: Encoder
        enc_out = self.encoder.blocks(x_masked)  # [B, N_visible, C]

        # Step 4: 插入 mask_token 并还原顺序
        B, N, C = x.shape
        mask_tokens = self.mask_token.expand(B, N - enc_out.shape[1], C)
        x_ = torch.cat([enc_out, mask_tokens], dim=1)  # [B, N, C]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        # Step 5: Decoder
        decoded = self.decoder(x_)  # [B, N, patch_dim]
        pred_imgs = self.unpatchify(decoded)  # [B, C, H, W]

        # Step 6: Target for loss
        target = self.patchify(imgs)  # [B, N, patch_dim]

        return decoded, target, mask
