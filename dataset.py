import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import load_data  # 添加这行导入语句
import torchvision.transforms.functional as F # type: ignore

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        
        # 添加路径验证
        valid_paths = []
        for img_path in root:
            if not os.path.exists(img_path):
                print(f"警告：图片不存在 - {img_path}")
                continue
                
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            if not os.path.exists(gt_path):
                print(f"警告：标注文件不存在 - {gt_path}")
                continue
                
            valid_paths.append(img_path)
        
        if len(valid_paths) == 0:
            raise RuntimeError("没有找到有效的训练数据！请检查数据集路径。")
            
        self.lines = valid_paths
        if shuffle:
            random.shuffle(self.lines)
        self.nSamples = len(self.lines)
        
        print(f"成功加载 {self.nSamples} 个有效样本")
        
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        max_attempts = 3  # 最大重试次数
        attempt = 0
        
        while attempt < max_attempts:
            try:
                img_path = self.lines[index]
                img, target = load_data(img_path, self.train)
                
                if img is None or target is None:
                    print(f"警告：无效的样本 {img_path}")
                    index = (index + 1) % len(self.lines)
                    attempt += 1
                    continue
                
                if self.transform is not None:
                    img = self.transform(img)
                
                # 确保密度图是正确的形状
                target = torch.from_numpy(target).float().unsqueeze(0)
                
                return img, target
                
            except Exception as e:
                print(f"加载样本时出错 {img_path}: {str(e)}")
                index = (index + 1) % len(self.lines)
                attempt += 1
        
        # 如果多次尝试都失败，返回一个有效的默认样本
        print("警告：多次尝试加载样本失败，返回默认样本")
        # 创建一个全零的默认样本
        default_img = torch.zeros(3, 384, 512)  # 使用一个合适的默认尺寸
        default_target = torch.zeros(1, 48, 64)  # 对应的密度图尺寸
        return default_img, default_target