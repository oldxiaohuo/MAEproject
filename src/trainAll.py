import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息
import time
import os
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *          # 自定义函数，如generate_pair
from mae import MAE               # 自定义的UNet3D模型
from config import mae_config
from tqdm import tqdm        # 用于显示循环进度条
from dcmdataset import MHDImageDataset
from torch.utils.data import DataLoader
from loss import mae_loss

print("torch = {}".format(torch.__version__))

# 指定使用第0号GPU。
# 设置碎片显存重分配阈值，避免显存碎片导致Out of Memory。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

conf = mae_config()
conf.display()

'''
x_train = []
for i, fold in enumerate(tqdm(conf.train_fold)):
    file_name = "bat_{}_s_{}x{}x{}_{}.npy".format(conf.scale, conf.input_rows, conf.input_cols, conf.input_deps, fold)
    s = np.load(os.path.join(conf.data, file_name))
    x_train.extend(s)
x_train = np.expand_dims(np.array(x_train), axis=1)  # 插入通道维度：变成 (N, 1, D, H)

x_valid = []
for i,fold in enumerate(tqdm(conf.valid_fold)):
    file_name = "bat_"+str(conf.scale)+"_s_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    s = np.load(os.path.join(conf.data, file_name))
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)

print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
'''

'''
当你设置 pin_memory=True 时，PyTorch 会把加载的数据放到 page-locked memory（锁页内存），
这样从 CPU → GPU 的数据拷贝 会更快。
锁页内存是不能被交换到磁盘的内存，它支持更快的异步传输（尤其用于 cuda() 操作时）。
配合 non_blocking=True 使用更高效
'''

train_set = MHDImageDataset(root_dir='data/mhd_folder/abnormal/train', transform=None)
val_set = MHDImageDataset(root_dir='data/mhd_folder/abnormal/val', transform=None)

train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True,num_workers=8, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=conf.batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model = MAE(img_size=(1024, 512), patch_size=16, in_chans=1, embed_dim=768)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
model.to(device)

print("Total CUDA devices: ", torch.cuda.device_count())

#打印模型结构，设定损失函数为均方误差。
summary(model, (1,conf.input_H,conf.input_W), batch_size=-1)
loss = nn.MSELoss()

if conf.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
    raise

#学习率调度器：每 patience*0.8 个 epoch 学习率减半。
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=int(conf.patience * 0.8), gamma=0.5
)

#best_loss = 100000 是个很大的初始化值，后续验证损失一旦小于它，就更新模型；
#num_epoch_no_improvement：记录连续多少个epoch验证损失没有下降，用于Early Stopping。
train_losses, valid_losses = [], []
avg_train_losses, avg_valid_losses = [], []
best_loss = 100000
intial_epoch = 0
num_epoch_no_improvement = 0

#可恢复训练中断的权重和优化器状态。
if conf.weights != None:
    checkpoint = torch.load(conf.weights)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    intial_epoch = checkpoint['epoch']

#每轮训练前更新学习率，并切换模型为训练模式。
for epoch in range(intial_epoch, conf.nb_epoch):
    scheduler.step(epoch)
    model.train()
    for iteration in range(int(len(train_set)// conf.batch_size)):
        # image, gt = next(training_generator)
        # gt = np.repeat(gt, conf.nb_class, axis=1)  # 多通道复制标签
        # image, gt = torch.from_numpy(image).float().to(device), torch.from_numpy(gt).float().to(device)
        # # pred = model(image)
        for images,_ in train_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            pred, target, mask = model(images, mask_ratio=0.75)
            loss = mae_loss(pred, target, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(round(loss.item(), 2))

        if (iteration + 1) % 10 == 0:
            print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
                .format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses)))
            sys.stdout.flush()		#输出缓冲区

    #验证过程        
    with torch.no_grad():
        model.eval()
        print("validating....")
    
        for images,label in val_loader:
            

            pred, target, mask = model(images, mask_ratio=0.75)
            loss = mae_loss(pred, target, mask)
        valid_losses.append(loss.item())


    #logging
    #train_loss为每个epoch的loss，train_losse为每个batch的loss
    train_loss=np.average(train_losses)
    valid_loss=np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
    train_losses=[]
    valid_losses=[]
    if valid_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0		
        #save model
        torch.save({
            'epoch': epoch+1,
            'state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },os.path.join(conf.model_path, "mae.pt"))
        print("Saving model ",os.path.join(conf.model_path,"mae.pt"))
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
        num_epoch_no_improvement += 1		#如果一次epoch没改进就把num_epoch_no_improvement加1
        
    #将epochs设置为conf.patience，当验证精度或损失停止提高时停止训练：所谓的early stopping
    if num_epoch_no_improvement == conf.patience:
        print("Early Stopping")
        break
    sys.stdout.flush()