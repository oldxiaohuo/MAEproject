import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model.vit_encoder import ViTEncoder
from dcmdataset import MHDImageDataset

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((1024, 512)),
    transforms.ToTensor()
])

# train_set = datasets.ImageFolder('data/finetune/train', transform=transform)
# val_set = datasets.ImageFolder('data/finetune/val', transform=transform)

train_set = MHDImageDataset(root_dir='data/mhd_folder/abnormal/train', transform=None)
val_set = MHDImageDataset(root_dir='data/mhd_folder/abnormal/val', transform=None)


train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ViTEncoder().to(device)
encoder.load_state_dict(torch.load('mae_pretrained.pth'), strict=False)

classifier = nn.Sequential(
    nn.Linear(768, 1024),
    nn.Linear(1024, 2)
).to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    encoder.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feats = encoder(imgs)[:, 0]  # use cls token,x[:, 0] 是取第一个 token 的值，不是保留它作为一维；所以维度变成 [B, D]，不是 [B, 0, D]。这在 PyTorch/NumPy 中是标准索引行为。
        preds = classifier(feats)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")
