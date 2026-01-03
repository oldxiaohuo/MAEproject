import torch
import torch.nn.functional as F

def mae_loss(decoded, target, mask, eps=1e-8):
    # decoded, target: [B, N, patch_dim]
    # mask: [B, N] with 1 for masked, 0 for visible
    # 1) 每个 patch 上的 MSE（沿最后维度平均）
    loss_per_patch = (decoded - target).pow(2).mean(dim=-1)   # [B, N]
    # 2) 只保留 mask==1 的 patch，并求平均
    mask_float = mask.float()                                # [B, N]
    loss = (loss_per_patch * mask_float).sum() / (mask_float.sum() + eps)
    return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5, weight_dice=0.5, weight_ce=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        pred: Tensor of shape (B, C, H, W) — raw logits from the model
        target: Tensor of shape (B, H, W) — ground truth class indices
        """
        # -------- CrossEntropy Loss --------
        ce = self.ce_loss(pred, target)

        # -------- Dice Loss --------
        pred_soft = F.softmax(pred, dim=1)  # (B, C, H, W)
        target_onehot = F.one_hot(target, num_classes=self.num_classes)  # (B, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)  # sum over batch and spatial dims
        intersection = torch.sum(pred_soft * target_onehot, dims)
        union = torch.sum(pred_soft, dims) + torch.sum(target_onehot, dims)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        # -------- Combined Loss --------
        total_loss = self.weight_dice * dice_loss + self.weight_ce * ce
        return total_loss
