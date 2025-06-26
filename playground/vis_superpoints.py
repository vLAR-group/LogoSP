import os
import numpy as np
import torch
import MinkowskiEngine as ME
import warnings
from lib.helper_ply import read_ply, write_ply
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

colormap = []
for _ in range(100):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines

def contin_label(label):
    Encoder = LabelEncoder()
    return Encoder.fit_transform(label)


def voxelize(coords):
    scale = 1 / 0.05
    coords = np.floor(coords * scale)
    coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), return_index=True, return_inverse=True)
    return coords, unique_map, inverse_map


plypath = './data/ScanNet/processed'
sp_path = './data/ScanNet/initial_superpoints'
train_scene_id = read_txt('/home/zihui/SSD/DivSP/data/ScanNet/scannet/scannet_3d/scannetv2_train.txt')#[800:850]

''' collect all voxel/sp feats '''
context, all_sp_feats = {}, []
acc_no = 0
for scene_id in train_scene_id:
    print(scene_id)
    data = read_ply(os.path.join(plypath, scene_id+'.ply'))
    coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
    colors = colors.astype(np.float32)/255-0.5
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)
    grids, unique_map, inv_map = voxelize(coords)
    coords, colors, labels = coords[unique_map], colors[unique_map], labels[unique_map]
    ##
    sp = np.load(os.path.join('/home/zihui/SSD/GOPS/data/scannet/processed/train', scene_id[5:]+'.npy'))[:, 9]#np.load(os.path.join(sp_path, scene_id+'_superpoint.npy'))
    # sp = sp[unique_map]
    ######### modify spuerpoints idx ###############
    sp = torch.from_numpy(sp).long()
    ####################################################
    colors = np.zeros_like(np.vstack((data['x'], data['y'], data['z'])).T)
    for p in range(colors.shape[0]):
        colors[p] = 255 * (colormap[sp[p]])[:3]
    os.makedirs('check_init_sp', exist_ok=True)
    write_ply(os.path.join('check_init_sp', scene_id+'.ply'), [np.vstack((data['x'], data['y'], data['z'])).T, colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
