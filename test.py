import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
# from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
from image import *
import torch
from scipy.ndimage import filters
import scipy.spatial
import pylab
import os
from collections import Counter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

root = './dataset/'
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
path_sets = [part_A_train, part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)



def read_truth_num_ls(img_paths):
    groundtruth_num_ls = []
    for i in range(len(img_paths)):
        gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
        groundtruth = np.asarray(gt_file['density'])
        groundtruth_num = np.sum(groundtruth).round()
        groundtruth_num_ls.append(groundtruth_num)
    picture_num_distribute(groundtruth_num_ls)
    return groundtruth_num_ls

def picture_num_distribute(Ls):
    b = sorted(Counter(Ls).items())
    b_dict = dict(b)
    print(b_dict)
    plt.plot(b_dict.keys(), b_dict.values())
    plt.show()

read_truth_num_ls(img_paths)
gt_file = h5py.File(img_paths[2].replace('.jpg','.h5').replace('images','ground_truth'),'r')
groundtruth = np.asarray(gt_file['density'])
print(np.sum(groundtruth))


plt.subplot(122)
plt.imshow(groundtruth, cmap=CM.jet)



imga = Image.open(img_paths[2])
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
plt.subplot(121)
plt.imshow(imga)
plt.show()