import os
import numpy as np
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
import time
from sklearn.cluster import KMeans
import warnings
from lib.helper_ply import read_ply, write_ply
from sklearn.preprocessing import LabelEncoder
from torch_scatter import scatter_mean, scatter_max, scatter_min
from scipy.sparse import csgraph
import colorsys
from typing import List, Tuple
import functools
import pickle
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), return_index=True, return_inverse=True)
    return coords, unique_map, inverse_map

## inputs need be numpy array
def compute_mIoU(preds, gt, semantic_num=20):
    mask = np.logical_and(preds != -1, gt!=-1)
    histogram = np.bincount(semantic_num * gt.astype(np.int32)[mask] + preds.astype(np.int32)[mask],
                            minlength=semantic_num ** 2).reshape(semantic_num, semantic_num)  # hungarian matching
    o_Acc = histogram[range(semantic_num), range(semantic_num)].sum() / histogram.sum() * 100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    print('oAcc {:.2f} IoUs'.format(o_Acc) + s)

## inputs need be numpy array
def vote_gt(preds, gt):
    pseudo_gt = -np.ones_like(preds)
    for p in np.unique(preds):
        if p != -1:
            mask = p == preds
            pseudo_gt[mask] = torch.mode(torch.from_numpy(gt[mask])).values.numpy()
    return pseudo_gt

def rbf_eig_vector(data, device='cuda', norm_laplacian=True):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)
    d = torch.cdist(data, data, p=2)
    gamma = 1.0 / data.shape[1]
    A = torch.exp(-gamma*d)
    A = A.cpu().numpy()

    laplacian, dd = csgraph.laplacian(A, normed=norm_laplacian, return_diag=True)  # only D-A if normalized is False
    eigvalues, eigvectors = torch.linalg.eigh(torch.from_numpy(laplacian).cuda())
    spec_embedding = eigvectors.cpu()[:, 1:31].numpy().astype(np.float32)
    return spec_embedding, eigvectors.cpu()


plypath = './data/ScanNet/processed'
sp_path = './data/ScanNet/initial_superpoints'
feat_path = '/home/zihui/SSD/DivSP/data/ScanNet/distillv2_point_feats_s14up4_1e-3poly'
# feat_path = './data/ScanNet/DINO_point_feats_pkl_s8'
grow_sp_num = 20
semantic_num = 20
primitive_num = 20
feats_dim = 384

train_scene_id = read_txt('/home/zihui/SSD/DivSP/data/ScanNet/scannet/scannet_3d/scannetv2_train.txt')#[800:850]
# train_scene_id = ['scene0000_00']

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
    sp = sp[unique_map]
    ######### modify spuerpoints idx ###############
    sp[labels == -1] = -1

    for q in np.unique(sp):
        mask = q == sp
        if mask.sum() < 10 and q != -1:
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
    labels = torch.from_numpy(labels)
    context[scene_id] = {'gt': labels, 'initial_sp': sp, 'coords': coords, 'label':labels}

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
    sp, sp_feats = grow_sp, F.normalize(growp_sp_feats)
    #
    ''' get sp features (initial or growed)'''
    context[scene_id]['final_sp'] = sp
    context[scene_id]['acc_sp'] = sp+acc_no
    all_sp_feats.append(sp_feats)
    acc_no += len(sp[sp!=-1].unique())


''' conduct GFT '''
time_start = time.time()
all_sp_feats = torch.cat(all_sp_feats)
spec_embedding, eigvectors = rbf_eig_vector(all_sp_feats) ## spec_embedding is numpy, eigvectors are tensor on cpu
## select W
## 1. compute energy to delete invalid W
all_amp_vector = eigvectors.T @ all_sp_feats ## [N, C]
all_energy = all_amp_vector[1:].pow(2).sum(-1) ## [N]
eigvectors = eigvectors[:, 1:]

sorted_energy, indices = torch.sort(all_energy, descending=True)
acc_energy = 0
valid_indice_list = []
for i, energy in enumerate(sorted_energy):
    acc_energy = acc_energy + energy
    valid_indice_list.append(indices[i])
    if acc_energy>all_energy.sum()*0.99:
        break

print('ALL energy: {}, remain ratio: {}, valid w has {}'.format(all_energy.sum().item(), acc_energy/all_energy.sum().item(), len(valid_indice_list)))
valid_eig_indices = torch.tensor(valid_indice_list).long()
print('filtered all energy', all_energy[valid_eig_indices])
valid_eigvectors, valid_amp_vector = eigvectors[:, valid_eig_indices], all_amp_vector[valid_eig_indices]

group_w_labels = KMeans(n_clusters=30, n_jobs=-1, random_state=0).fit_predict(valid_amp_vector.numpy().astype(np.float32))
group_w_labels = contin_label(group_w_labels)
# merge some colum in valid_eigvectors:
spec_embedding = scatter_mean(valid_eigvectors.T, torch.from_numpy(group_w_labels).long(), dim=0) ## [K, N]
spec_embedding = spec_embedding.T.numpy()
##
primitive_labels = KMeans(n_clusters=primitive_num, n_jobs=-1).fit_predict(spec_embedding.astype(np.float32))

'''spread primitive labels for each sp(growed sp) to points'''
accumulate_sp_num = 0
all_gt, all_pseudo, all_pseudo_gt = [], [], []
for idx, scene_id in enumerate(train_scene_id):
    labels, initial_sp, final_sp = context[scene_id]['gt'], context[scene_id]['initial_sp'], context[scene_id]['final_sp']
    accumulate_sp_idx = final_sp + accumulate_sp_num

    pseudo_gt = -torch.ones_like(labels)
    pseudo = primitive_labels[accumulate_sp_idx]

    for p in np.unique(accumulate_sp_idx):
        if p != -1:
            mask = p == accumulate_sp_idx
            pseudo_gt[mask] = torch.mode(labels[mask]).values
    accumulate_sp_idx_unique = np.unique(accumulate_sp_idx)
    accumulate_sp_num += len(accumulate_sp_idx_unique[accumulate_sp_idx_unique != -1])
    ##
    all_gt.append(labels), all_pseudo.append(pseudo), all_pseudo_gt.append(pseudo_gt)

all_gt = np.concatenate(all_gt)
all_pseudo = np.concatenate(all_pseudo)
all_pseudo_gt = np.concatenate(all_pseudo_gt)


'''Check Superpoint/Primitive Acc in Training'''
mask = (all_pseudo_gt!=-1)
histogram = np.bincount(semantic_num* all_gt.astype(np.int32)[mask] + all_pseudo_gt.astype(np.int32)[mask], minlength=semantic_num ** 2).reshape(semantic_num, semantic_num)    # hungarian matching
o_Acc = histogram[range(semantic_num), range(semantic_num)].sum()/histogram.sum()*100
tp = np.diag(histogram)
fp = np.sum(histogram, 0) - tp
fn = np.sum(histogram, 1) - tp
IoUs = tp / (tp + fp + fn + 1e-8)
m_IoU = np.nanmean(IoUs)
s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
for IoU in IoUs:
    s += '{:5.2f} '.format(100 * IoU)
print('Superpoints oAcc {:.2f} IoUs'.format(o_Acc) + s)

pseudo_class2gt = -np.ones_like(all_gt)
for i in range(primitive_num):
    mask = all_pseudo==i
    pseudo_class2gt[mask] = torch.mode(torch.from_numpy(all_gt[mask])).values
mask = (pseudo_class2gt!=-1)&(all_gt!=-1)
histogram = np.bincount(semantic_num* all_gt.astype(np.int32)[mask] + pseudo_class2gt.astype(np.int32)[mask], minlength=semantic_num ** 2).reshape(semantic_num, semantic_num)    # hungarian matching
o_Acc = histogram[range(semantic_num), range(semantic_num)].sum()/histogram.sum()*100
tp = np.diag(histogram)
fp = np.sum(histogram, 0) - tp
fn = np.sum(histogram, 1) - tp
IoUs = tp / (tp + fp + fn + 1e-8)
m_IoU = np.nanmean(IoUs)
s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
for IoU in IoUs:
    s += '{:5.2f} '.format(100 * IoU)
print('Primitives oAcc {:.2f} IoUs'.format(o_Acc) + s)
print(f"project  processing time: {time.time() - time_start:.2f} seconds")


### DINOv2+Distill
## mycuda spectral clustering, scannet grow_sp=20, primitive_num=20
# Primitives oAcc 77.89 IoUs| mIoU 30.66 | 67.75 93.62 30.15 78.76 58.25 57.33 46.57 36.92 34.93 38.62  0.00  0.00  0.00  0.00  0.00  0.00 55.99  0.00  0.00 14.26
## no recover by dd:
# Primitives oAcc 77.91 IoUs| mIoU 31.96 | 69.05 92.14 30.57 78.32 57.27 57.98 53.28 36.93 34.47 38.02  0.00  0.00 22.90  0.00  0.00  0.00 54.31  0.00  0.00 13.96
## no recover by dd, 1-21 coloum eigvectors
# Primitives oAcc 78.49 IoUs| mIoU 32.39 | 69.82 92.95 31.92 78.68 57.33 58.08 53.79 37.78 34.71 41.73  0.00  0.00 22.00  0.00  0.00  0.00 55.08  0.00  0.00 13.92
## if use colum 1-51
# Primitives oAcc 78.68 IoUs| mIoU 36.28 | 66.92 92.41 30.75 80.83 58.54 58.40 55.01  0.00 39.36  0.00  0.00 46.89 30.02  0.00  0.00  0.00 71.93  0.00 81.23 13.31
## 1-101
# Primitives oAcc 77.86 IoUs| mIoU 31.03 | 69.92 94.74 24.41 38.44 61.47 55.97 39.55 38.53 41.37  0.00  0.00 47.95 21.26  0.00  0.00  0.00 75.15  0.00  0.00 11.88
## 1-201
# Primitives oAcc 70.67 IoUs| mIoU 24.21 | 59.65 76.23 28.19  0.00 42.39 53.31 52.58  0.00 40.41 54.39  0.00  0.00  0.00  0.00  0.00  0.00 77.01  0.00  0.00  0.00
## 1-31
# Primitives oAcc 79.29 IoUs| mIoU 36.62 | 67.96 95.99 27.31 79.95 60.02 59.53 52.29 37.11 40.62 51.21  0.00  0.00  0.00  0.00  0.00  0.00 59.24  0.00 83.24 17.91
## all valid colum 271 coloums
# Primitives oAcc 69.50 IoUs| mIoU 23.35 | 53.37 86.66 34.81  0.00 49.96 55.05  0.00 40.33 30.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00 34.61  0.00 82.10  0.00
## cluster to 100
# Primitives oAcc 75.00 IoUs| mIoU 33.37 | 58.39 94.28 31.47 73.50 61.05 53.99  0.00 37.09 41.24  0.00  0.00 46.76  0.00  0.00  0.00  0.00 76.30  0.00 81.60 11.78
## cluster to 50
# Primitives oAcc 79.41 IoUs| mIoU 36.60 | 68.04 93.43 32.19 79.93 58.32 58.10 55.38 39.98  0.00 56.72  0.00  0.00 24.06  0.00  0.00  0.00 71.29  0.00 81.29 13.32
## cluster to 30
# Primitives oAcc 78.50 IoUs| mIoU 33.02 | 68.57 94.38 32.90 79.17 59.88 57.89 52.39 37.09 33.53 49.98  0.00  0.00 16.90  0.00  0.00  0.00 63.46  0.00  0.00 14.17
## cluster to 20
# Primitives oAcc 77.20 IoUs| mIoU 31.12 | 68.31 88.58 35.49 80.18 58.35 55.85 51.71 36.25 34.18 37.05  0.00  0.00  0.00  0.00  0.00  0.00 61.52  0.00  0.00 14.83


## drop 10, col50, eneg 0.99
# Primitives oAcc 74.91 IoUs| mIoU 35.96 | 65.30 83.84 33.11 73.23 55.13 54.86 44.90  0.00 45.08 62.28  0.00  0.00  0.00 45.55  0.00  0.00 64.16  0.00 76.29 15.50
## drop 10, col50, eneg 0.95
# Primitives oAcc 74.29 IoUs| mIoU 38.54 | 63.23 83.75 30.95 72.92 55.61 55.11 48.00 34.54 42.30 63.13  0.00 36.28 25.85  0.00  0.00  0.00 67.39  0.00 76.27 15.50
## drop 10, col30, eneg 0.90
# Primitives oAcc 73.62 IoUs| mIoU 32.70 | 65.45 78.64 25.77 72.22 54.90 53.99 45.03 32.97 45.31 58.57  0.00  0.00  0.00 45.48  0.00  0.00 58.11  0.00  0.00 17.65



