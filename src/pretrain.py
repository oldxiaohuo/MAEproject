import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.mae import MAE
from utils import random_masking
from dcmdataset import MHDImageDataset
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((1024, 512)),
    transforms.ToTensor()
])

# dataset = datasets.ImageFolder(root='data/pretrain', transform=transform)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


train_set = MHDImageDataset(root_dir='data/mhd_folder/abnormal/train', transform=None)
val_set = MHDImageDataset(root_dir='data/mhd_folder/abnormal/val', transform=None)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(100):
    model.train()
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        patches = model.encoder.patch_embed(imgs)
        mask = random_masking(patches, mask_ratio=0.75)

        output = model(imgs, mask)
        target = patches[mask].view(output.shape)

        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'mae_pretrained.pth')
