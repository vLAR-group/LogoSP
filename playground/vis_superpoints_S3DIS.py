import os
import numpy as np
import torch
import MinkowskiEngine as ME
import warnings
from lib.helper_ply import read_ply, write_ply
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from glob import glob
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


plypath = '/home/zihui/SSD/DivSP/data/S3DIS/processed/'
sp_path = '/home/zihui/SSD/DivSP/data/S3DIS/initial_superpoints_rebuild/'

''' collect all voxel/sp feats '''
context, all_sp_feats = {}, []
acc_no = 0

folders = sorted(glob(plypath + '/*.ply'))
for _, file in enumerate(folders):
    plyname = file.replace(plypath, '')
    sp_file = os.path.join(sp_path, plyname[0:-4] + '_rebuild_superpoint.npy')
    sp = np.load(sp_file)

    data = read_ply(file)
    coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
    colors = colors.astype(np.float32)/255-0.5
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)
    grids, unique_map, inv_map = voxelize(coords)
    coords, colors, labels = coords[unique_map], colors[unique_map], labels[unique_map]
    sp = sp[unique_map]
    sp = torch.from_numpy(sp).long()
    ####################################################
    colors = np.zeros_like(colors)
    for p in range(colors.shape[0]):
        colors[p] = 255 * (colormap[sp[p]])[:3]
    os.makedirs('check_init_rebuildsp_S3DIS', exist_ok=True)
    write_ply(os.path.join('check_init_rebuildsp_S3DIS', plyname), [coords, colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
