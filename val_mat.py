import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from sklearn.metrics import mean_squared_error,mean_absolute_error
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])

root = './dataset/'

# now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
path_sets = [part_A_test]
# path_sets = ['./自己的图片/原始图像/test/images']

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

model = CSRNet()
model = model.cuda()
checkpoint = torch.load('partA_model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

pred = []
gt = []
for i in range(39,43):
    torch.cuda.empty_cache()
    # img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    # img[0, :, :] = img[0, :, :] - 92.8207477031
    # img[1, :, :] = img[1, :, :] - 95.2757037428
    # img[2, :, :] = img[2, :, :] - 104.877445883
    # img = img.cuda()
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda() # 原本被注释了
    gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    output_num = output.data.cpu().numpy().sum()
    true_num = np.sum(groundtruth)
    pred.append(output_num)
    gt.append(true_num)


    output_img = output.cpu().detach().numpy()

    plt.figure(figsize = (20,6))
    plt.subplot(1,3,1)
    plt.imshow(Image.open(img_paths[i]))
    plt.title('original crowd image')
    plt.subplot(1,3,2)
    plt.imshow(output.detach().cpu()[0,0,:,:],cmap = CM.jet)
    plt.title('CSRNet-density map prediction _%.3f'%output_num)
    plt.subplot(1,3,3)
    plt.imshow(groundtruth, cmap=CM.jet)
    plt.title('original density map _%.3f'%true_num)
    plt.show()

    del output
    torch.cuda.empty_cache()  # 清理失活内存
    # time.sleep(20)


mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

print('MAE: ',mae)
print('RMSE: ',rmse)


