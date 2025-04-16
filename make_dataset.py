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
from model import CSRNet
import torch
from scipy.ndimage import filters
import scipy.spatial

# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density # 如果gt数组每个节点都为0，那么返回对应的每个节点为0的数组

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))) # pts为二维数组，每个子数组标识非空的元素位置
    leafsize = 2048     # 设置子节点的数目
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize) # 第一个参数为位置参数，第二个参数为子节点个数
    # query kdtree
    distances, locations = tree.query(pts, k=4)  # 最近的4个点，locations代表tree中（pts构建的）中点的位置，locations[n]为第n+1最近邻的点。

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


# set the root to the Shanghai dataset you download
root = './dataset/'

# now generate the ShanghaiA's ground truth
# part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
# part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
# part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
# part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
# path_sets = [part_A_train, part_A_test]
# path_sets = [part_B_train, part_B_test]

diy_data_train = os.path.join('./自己的图片/原始图像/train','images')
diy_data_test = os.path.join('./自己的图片/原始图像/test','images')
path_sets = [diy_data_test] # diy_data_train,

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)

    json_path = img_path.replace('.jpg','.json').replace('images','ground_truth')
    with open(json_path,'r') as f:
        mat = json.load(f)
    arr = []
    for item in mat['points']:
        arr.append([item['x'],item['y']])
    gt = np.array(arr)
    # mat = io.loadmat(img_path.replace('.jpg', '.json').replace('images', 'ground_truth'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    # gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter(k)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:
        hf['density'] = k

# now see a sample from ShanghaiA
plt.imshow(Image.open(img_paths[0]))

gt_file = h5py.File(img_paths[0].replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth, cmap=CM.jet)
plt.show()
print(np.sum(groundtruth))  # don't mind this slight variation

# now generate the ShanghaiB's ground truth
# path_sets = [part_B_train, part_B_test]

# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)
#
# for img_path in img_paths:
#     print(img_path)
#     mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
#     img = plt.imread(img_path)
#     k = np.zeros((img.shape[0], img.shape[1]))
#     gt = mat["image_info"][0, 0][0, 0][0]
#     for i in range(0, len(gt)):
#         if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
#             k[int(gt[i][1]), int(gt[i][0])] = 1
#     k = gaussian_filter(k, 15)
#     with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:
#         hf['density'] = k
