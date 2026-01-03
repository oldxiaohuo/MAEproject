# ==================== 环境变量要在 import torch 之前设置 ====================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只用第0号GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from utils import *            # 你自定义的工具函数
from mae import MAE            # Masked Autoencoder 模型
from config import mae_config  # 配置文件
from dcmdataset import MHDImageDataset
from loss import mae_loss


# ==================== 打印环境 ====================
print("torch = {}".format(torch.__version__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 配置 ====================
conf = mae_config()
conf.display()


# ==================== 数据集与 DataLoader ====================
train_set = MHDImageDataset(root_dir=r'data\mhd_folder\abnormal\train', transform=None)
val_set   = MHDImageDataset(root_dir=r'data\mhd_folder\abnormal\val',   transform=None)

# 加上num_workers=8, pin_memory=True, persistent_workers=True后报错

train_loader = DataLoader(
    train_set, batch_size=conf.batch_size, shuffle=True
)
val_loader = DataLoader(
    val_set, batch_size=conf.batch_size
)


# ==================== 模型构建 ====================
model = MAE(img_size=(1024, 512), patch_size=16, in_chans=1, embed_dim=768)

# 如果有多张显卡，就用 DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

model = model.to(device)
print("Total CUDA devices: ", torch.cuda.device_count())

# 打印模型结构
# 1.你的 ViTEncoder 必须有 patch_embed 和 blocks，否则 summary 会卡在 forward。
# 2.如果 torchsummary 仍报错（常见于有 rearrange 或 torch.gather 的情况），建议用 torchinfo.summary，它更稳健。
# 3.summary 不支持 mask_ratio 这样的额外参数，默认会用 mask_ratio=0.1。如果想改，可以在 forward 里写默认值，或者在调用 summary 时传 forward_kwargs：

# summary(model, input_size=(1,1,conf.input_H, conf.input_W),forward_kwargs={"mask_ratio": 0.1})

# try:
#     summary(model, (1, conf.input_H, conf.input_W), batch_size=1)
# except:
#     summary(model.module, (1, conf.input_H, conf.input_W), batch_size=1)


# ==================== 优化器 & 学习率调度 ====================
if conf.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
    raise ValueError("Unsupported optimizer type: {}".format(conf.optimizer))

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=int(conf.patience * 0.8), gamma=0.5
)


# ==================== 状态变量 ====================
train_losses, valid_losses = [], []
avg_train_losses, avg_valid_losses = [], []
best_loss = float('inf')
intial_epoch = 0
num_epoch_no_improvement = 0

# 支持恢复训练
if conf.weights is not None:
    checkpoint = torch.load(conf.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    intial_epoch = checkpoint['epoch']
    print(f"Resumed training from epoch {intial_epoch}")


# ==================== AMP 支持（自动混合精度） ====================
#scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))


# ==================== 训练 & 验证 ====================
for epoch in range(intial_epoch, conf.nb_epoch):
    # scheduler.step()
    model.train()
    train_losses = []
    start_time = time.time()

    # ---- 训练循环 ----
    for step, (images, _) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)

        # optimizer.zero_grad(set_to_none=True)

        # with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        #     pred, target, mask = model(images, mask_ratio=0.1)
        #     loss = mae_loss(pred, target, mask)

        # # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.zero_grad()
        decoded, target, mask = model(images, mask_ratio=0.1)  # decoded: [B,N,patch_dim]
        loss = mae_loss(decoded, target, mask)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # 每10步打印一次进度
        if step % 10 == 0:
            avg_loss = np.mean(train_losses)
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{conf.nb_epoch}] "
                  f"Step [{step}/{len(train_loader)}] "
                  f"LR: {lr:.3e} "
                  f"Loss: {avg_loss:.6f}", flush=True)

    # ---- 验证循环 ----
    model.eval()
    valid_losses = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device, non_blocking=True)
            pred, target, mask = model(images, mask_ratio=0.1)
            loss = mae_loss(pred, target, mask)
            valid_losses.append(loss.item())

    train_loss = float(np.mean(train_losses)) if train_losses else float('inf')
    valid_loss = float(np.mean(valid_losses)) if valid_losses else float('inf')
    elapsed = time.time() - start_time

    print(f"Epoch {epoch+1}: val_loss={valid_loss:.4f}, train_loss={train_loss:.4f}, "
          f"time={elapsed:.1f}s", flush=True)

    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    # ---- 保存最优模型 ----
    if valid_loss < best_loss:
        print(f"Validation loss decreases from {best_loss:.4f} to {valid_loss:.4f}. Saving model...")
        best_loss = valid_loss
        num_epoch_no_improvement = 0

        os.makedirs(conf.model_path, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(conf.model_path, "mae.pt"))
    else:
        num_epoch_no_improvement += 1
        print(f"Validation loss did not improve from {best_loss:.4f}. "
              f"no_improve_epochs = {num_epoch_no_improvement}")

    # ---- 学习率调度 (放在epoch末尾) ----
    scheduler.step()

    # ---- 早停 ----
    if num_epoch_no_improvement >= conf.patience:
        print("Early Stopping triggered.")
        break
