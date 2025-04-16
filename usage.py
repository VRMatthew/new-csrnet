import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import time

from torchvision import datasets, transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_image():
    root = './dataset/'
    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    # path_sets = [part_A_test]
    path_sets = ['./自己的图片/原始图像/test/images']
    # upload dataset
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

def load_model():
    model = CSRNet()
    model = model.cuda()
    checkpoint = torch.load('改进的CSRNet模型跑自己的数据集100代/0model_best33.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

def predict_num(img_path, model):
    torch.cuda.empty_cache()

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])
    img = transform(Image.open(img_path).convert('RGB')).cuda()  # 原本被注释了
    output = model(img.unsqueeze(0))
    pred_num = output.data.cpu().numpy().sum()

    # plt.imshow(output.detach().cpu()[0,0,:,:],cmap = CM.jet)
    # output_img = output.cpu().detach().numpy()
    
    del output
    torch.cuda.empty_cache()  # 清理失活内存
    return pred_num


if __name__ == '__main__':
    model = load_model()
    image = load_image()
    pred_num = predict_num(image, model)

