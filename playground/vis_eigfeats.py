import os
import numpy as np
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
import time
import warnings
from lib.helper_ply import read_ply, write_ply
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torch_scatter import scatter_mean
from scipy.sparse import csgraph
import colorsys
from sklearn.decomposition import PCA
from typing import List, Tuple
import functools
import pickle
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

colormap = []
for _ in range(1000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)


@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))


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
    _, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), return_index=True, return_inverse=True)
    return coords[unique_map], unique_map, inverse_map


def map_color(err, norm=True):
    color = torch.zeros((err.shape[0], 3))
    if norm:
        err = (err-err.min())/(err.max()-err.min()+1e-5)

    # Define the RGB values for dark blue and yellow
    dark_blue = (0, 0, 100)  # RGB for dark blue
    yellow = (255, 255, 0)  # RGB for yellow
    for c in range(err.shape[0]):
        # Interpolate between dark blue and yellow
        r = int(dark_blue[0] + (yellow[0] - dark_blue[0]) * err[c])
        g = int(dark_blue[1] + (yellow[1] - dark_blue[1]) * err[c])
        b = int(dark_blue[2] + (yellow[2] - dark_blue[2]) * err[c])
        color[c,0] = r
        color[c,1] = g
        color[c,2] = b
    return color

def rbf_eig_vector(data, device='cuda', norm_laplacian=True):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)
    d = torch.cdist(data, data, p=2)
    gamma = 1.0 / data.shape[1]
    A = torch.exp(-gamma*d)
    A = A.cpu().numpy()
    ## not sure whether need this normalization
    laplacian, dd = csgraph.laplacian(A, normed=norm_laplacian, return_diag=True)  # only D-A if normalized is False
    ## pytorch cuda computation
    eigvalues, eigvectors = torch.linalg.eigh(torch.from_numpy(laplacian).cuda())
    return eigvectors.cpu()


plypath = './data/ScanNet/processed'
sp_path = './data/ScanNet/initial_superpoints'
feat_path = './data/ScanNet/distillv2_point_feats_s14up4_1e-3poly_grid'
grow_sp_num = 40
segment_num = 20 ## segment number for each scene
feats_dim = 384
# train_scene_id = read_txt('/home/zihui/SSD/DivSP/data/ScanNet/scannet/scannet_3d/scannetv2_train.txt')[100:150]
train_scene_id = ['scene0011_00', 'scene0011_01']
# train_scene_id = ['scene0000_00', 'scene0000_01', 'scene0000_02']

''' collect all voxel/sp feats '''
acc_sp, all_sp, all_sp_feats, all_coords, all_gt, all_raw_coords = [], [], [], [], [], []
acc_no = 0
for scene_id in train_scene_id:
    print(scene_id)
    data = read_ply(os.path.join(plypath, scene_id+'.ply'))
    coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
    colors = colors.astype(np.float32)/255-0.5
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)

    raw_coords = coords.copy()
    raw_labels = data['class']

    grids, unique_map, inv_map = voxelize(coords)
    coords, colors, labels = coords[unique_map], colors[unique_map], labels[unique_map]
    ##
    raw_sp = np.load(os.path.join('/home/zihui/SSD/GOPS/data/scannet/processed/validation', scene_id[5:]+'.npy'))[:, 9]
    # raw_sp = np.load(os.path.join(sp_path, scene_id+'_superpoint.npy'))
    raw_sp_clone = raw_sp.copy()
    raw_sp_clone[raw_labels==-1] = -1
    raw_sp_clone = raw_sp_clone[unique_map][inv_map]
    #sp = np.load(os.path.join(sp_path, scene_id+'_superpoint.npy'))
    # sp = raw_sp_clone[unique_map]
    sp = raw_sp[unique_map]
    ######### modify spuerpoints idx ###############
    sp[labels == -1] = -1

    for q in np.unique(sp):
        mask = q == sp
        if mask.sum() < 30 and q != -1:
            sp[mask] = -1

    valid_region = sp[sp != -1]
    valid_region = contin_label(valid_region)

    sp[sp != -1] = valid_region
    sp = torch.from_numpy(sp).long()
    ####################################################
    with open(os.path.join(feat_path, scene_id+'.pickle'), 'rb') as f:
        feats = pickle.load(f)

    """valid mask"""
    valid_mask = sp!=-1
    sp, coords, labels = sp[valid_mask], coords[valid_mask], labels[valid_mask]
    feats = torch.from_numpy(feats[valid_mask])
    # feats = F.normalize(feats)
    sp_feats = scatter_mean(feats, sp, dim=0)
    #####################33 make growing ###########################
    if sp_feats.shape[0] < grow_sp_num:
        n_segments = sp_feats.shape[0]
    else:
        n_segments = grow_sp_num
    sp_idx = KMeans(n_clusters=n_segments, n_init=10, random_state=0, n_jobs=-1).fit_predict(sp_feats.numpy())
    sp_idx = contin_label(sp_idx)
    sp_idx = torch.from_numpy(sp_idx).long()
    grow_sp = sp_idx[sp]
    growp_sp_feats = scatter_mean(feats, grow_sp, dim=0)
    ##
    sp, sp_feats = grow_sp, growp_sp_feats
    ###############################################################3
    acc_sp.append(sp+acc_no), all_sp_feats.append(sp_feats), all_coords.append(coords), all_sp.append(sp), all_gt.append(labels), all_raw_coords.append(raw_coords[raw_sp_clone!=-1])
    acc_no += len(sp[sp!=-1].unique())

all_labels = np.concatenate(all_gt)

''' conduct GFT '''
time_start = time.time()
all_sp_feats = torch.cat(all_sp_feats)
all_sp_feats = F.normalize(all_sp_feats)
eigvectors = rbf_eig_vector(all_sp_feats.cuda())  # (sp_feats)  ## spec_embedding is numpy, eigvectors are tensor on cpu
## select W
## 1. compute energy to delete invalid W
all_amp_vector = eigvectors.T @ all_sp_feats  ## [N, C]
all_energy = all_amp_vector[1:].pow(2).sum(-1)  ## [N]
eigvectors = eigvectors[:, 1:]
all_amp_vector = all_amp_vector[1:]
sorted_energy, indices = torch.sort(all_energy, descending=True)
acc_energy = 0
valid_indice_list = []
for i, energy in enumerate(sorted_energy):
    acc_energy = acc_energy + energy
    valid_indice_list.append(indices[i])
    if acc_energy > all_energy.sum() * 0.95:
        break
print(i)


valid_eig_indices = torch.tensor(valid_indice_list).long()
valid_eigvectors, valid_amp_vector = eigvectors[:, valid_eig_indices], all_amp_vector[valid_eig_indices]

# 3. do clustering on amp vectors?
group_w_labels = KMeans(n_clusters=10, n_init=10, n_jobs=-1, random_state=0).fit_predict(valid_amp_vector.numpy().astype(np.float32))
group_w_labels = contin_label(group_w_labels)
# merge some colum in valid_eigvectors:
spec_embedding = scatter_mean(valid_eigvectors.T, torch.from_numpy(group_w_labels).long(), dim=0)  ## [K, N]
spec_embedding = spec_embedding.T.numpy()  ## [N, K]
##

folder = 'vis_fused_eigvector'
for idx, scene_id in enumerate(train_scene_id):
    coords, sp, raw_coords = all_coords[idx], all_sp[idx], all_raw_coords[idx]

    for lamda_index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    # for lamda_index in range(len(eigvectors)):
        eig_mask = torch.zeros_like(eigvectors)
        eig_mask[lamda_index, lamda_index] = 1
        ##
        save_path = os.path.join(folder, scene_id)
        os.makedirs(save_path, exist_ok=True)
        ### compute a corresponding of raw to voxelize
        raw2vol_d = torch.cdist(torch.from_numpy(raw_coords)[None, ...], torch.from_numpy(coords)[None, ...]) ## [1, N, n]
        raw2vol = torch.argmin(raw2vol_d.squeeze(0), dim=1).long()
        ###
        all_sp_GFT_feats = eigvectors[:, [lamda_index]] @ all_sp_feats[[lamda_index]]  ## the convert back features for all datasets [N, c]
        sp_GFT_feats = all_sp_GFT_feats[acc_sp[idx].unique().long()]
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(sp_GFT_feats.numpy())
        min_vals = pca_features.min(axis=0)
        max_vals = pca_features.max(axis=0)
        rgb = 255 * (pca_features - min_vals) / (max_vals - min_vals)
        rgb = rgb[sp]
        write_ply(os.path.join(save_path, 'spGFT_feats_' + str(lamda_index) + '_.ply'), [raw_coords, rgb[raw2vol].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])