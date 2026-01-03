import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
from torchvision import transforms
import pydicom

class MHDImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dcm_path = []
        '''
        os.walk(root_dir) 会遍历指定目录及其所有子目录，每次迭代返回一个三元组：(root, dirs, files)
        root：当前遍历的目录路径（字符串）
        dirs：该目录下的子目录名（列表）
        files：该目录下的文件名（列表）
        忽略 dirs（子目录名列表），因为在这个场景下用不到它。_ 代表“我知道这里有个变量，但我不关心它”，所以用下划线占位。
        '''
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.dcm'):
                    self.dcm_path.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.dcm_path)

    def __getitem__(self, idx):
        dcm_path = self.dcm_path[idx]

        ds = pydicom.dcmread(dcm_path)
        image = ds.pixel_array.astype(np.float32)
        anterior = image[0]
        posterior = image[1]
        anterior = np.squeeze(anterior)
        posterior = np.squeeze(posterior)
        image_array = np.hstack((anterior,posterior))
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # shape: [1, 1024, 512]

        # # 1. 读取图像,SimpleITK 将图像转为 numpy 格式时，默认以 [slice, height, width] 排列
        # image = sitk.ReadImage(dcm_path)
        # image_array = sitk.GetArrayFromImage(image)[0]  # shape: [depth,H, W], 因为是 [1,depth,H, W]

        # anterior = image_array[0]  # shape: (1024, 256)
        # posterior = image_array[1]  # shape: (1024, 256)

        # # 将前位和后位图像按宽度拼接（左前位，右后位）
        # combined = np.hstack((anterior, posterior))  # shape: (1024, 512)
        # # slice_image = combined
        # # 选择某一层，比如中间那层(下面2行代码暂且注释掉，以后可能还有用，勿删)
        # # slice_index = array.shape[0] // 2
        # # slice_image = array[slice_index]

        # image_array = combined.astype(np.float32)

        # #归一化
        # image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.max(image_array))
        # image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # shape: [1, 1024, 512]

        # 2. 应用图像增强
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # 3. 根据路径自动生成标签
        path_lower = dcm_path.lower()
        if 'abnormal' in path_lower:
            label = 0
        elif 'normal' in path_lower:
            label = 1
        else:
            label = 1
            # raise ValueError(f"Path '{dcm_path}' does not contain 'abnormal' or 'normal'")

        return image_tensor, label  # 自监督任务中 label=0 是 dummy 的
    

# transform = transforms.Compose([
# transforms.Resize((1024, 512)),  # 注意这里只能用于 PIL Image 或 torch Tensor
# ])

#调用代码
# dataset = MHDImageDataset(root_dir='data/mhd_folder', transform=None)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)