import os
import numpy as np
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
import time
from utils_degrowsp import get_fixclassifier
from sklearn.cluster import KMeans, SpectralClustering
import warnings
from lib.helper_ply import read_ply, write_ply
import pickle
from sklearn.preprocessing import LabelEncoder
import colorsys
from typing import List, Tuple
import functools
warnings.filterwarnings('ignore')

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

plypath = '/home/zihui/SSD/DivSP/data/ScanNet/processed'
sp_path = '/home/zihui/SSD/DivSP/data/ScanNet/initial_superpoints'
feat_path = '/home/zihui/SSD/DivSP/data/ScanNet/stego_potsdam_scale_2'
segment_num = 20 ## segment number for each scene
semantic_num = 20
primitive_num = 20
feats_dim = 70#384
rgb_w, xyz_w, norm_w = 1, 0.2, 0.8
colormap = get_evenly_distributed_colors(primitive_num)

train_scene_id = read_txt('/home/zihui/SSD/DivSP/data/ScanNet/scannet/scannet_3d/scannetv2_train.txt')#[800:1100]
time_start = time.time()

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
    normals = colors
    ##
    # sp = np.load(os.path.join(sp_path, scene_id+'_superpoint.npy'))
    # sp = sp[unique_map]
    sp = np.load(os.path.join('/home/zihui/SSD/GOPS/data/scannet/processed/train', scene_id[5:]+'.npy'))[:, 9]#np.load(os.path.join(sp_path, scene_id+'_superpoint.npy'))
    sp = sp[unique_map]
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
    # feats = torch.load(os.path.join(feat_path, scene_id+'.pth'))
    with open(os.path.join(feat_path, scene_id+'.pickle'), 'rb') as f:
        feats = pickle.load(f).astype(np.float32)[unique_map]
    """valid mask"""
    valid_mask = sp!=-1
    coords, colors, normals, labels, feats, sp = coords[valid_mask], colors[valid_mask], normals[valid_mask], labels[valid_mask], feats[valid_mask], sp[valid_mask]
    colors, coords, normals, feats, labels = torch.from_numpy(colors).cuda(), torch.from_numpy(coords).cuda(), torch.from_numpy(normals).cuda(), torch.from_numpy(feats).cuda(), torch.from_numpy(labels)
    context[scene_id] = {'gt': labels, 'initial_sp': sp, 'coords': coords, 'label':labels}
    ''' cluster initial spuerpoints to 20, means one step growing, may not needed'''
    ## average handcrafted features
    sp_num = len(torch.unique(sp))
    sp_corr = torch.zeros(sp.size(0), sp_num)  # ?
    sp_corr.scatter_(1, sp.view(-1, 1), 1)
    sp_corr = sp_corr.cuda()
    per_sp_num = sp_corr.sum(0, keepdims=True).t()

    sp_feats = F.linear(sp_corr.t(), feats.t()) / per_sp_num
    sp_rgb = F.linear(sp_corr.t(), colors.t()) / per_sp_num
    sp_xyz = F.linear(sp_corr.t(), coords.t()) / per_sp_num
    sp_norm = F.linear(sp_corr.t(), normals.t()) / per_sp_num

    sp_feats = F.normalize(sp_feats, dim=-1)

    if sp_feats.size(0) < segment_num:
        n_segments = sp_feats.size(0)
    else:
        n_segments = segment_num
    sp_idx = KMeans(n_clusters=n_segments, n_init=10, random_state=0, n_jobs=-1).fit_predict(sp_feats.cpu().numpy())
    sp_idx = contin_label(sp_idx)
    sp_idx = torch.from_numpy(sp_idx).long()
    sp = sp_idx[sp]
    ''' get new sp idx for the 20 segments'''
    sp_num = len(torch.unique(sp))
    sp_corr = torch.zeros(sp.size(0), sp_num)
    sp_corr.scatter_(1, sp.view(-1, 1), 1)
    sp_corr = sp_corr.cuda()
    per_sp_num = sp_corr.sum(0, keepdims=True).t()
    sp_feats = F.linear(sp_corr.t(), feats.t()) / per_sp_num
    sp_feats = F.normalize(sp_feats, dim=-1)
    ''' get sp features (initial or growed)'''
    context[scene_id]['final_sp'] = sp
    context[scene_id]['acc_sp'] = sp+acc_no
    all_sp_feats.append(sp_feats)
    acc_no += len(sp[sp!=-1].unique())

print('sp features extracted')

all_sp_feats = torch.cat(all_sp_feats)
primitive_labels = KMeans(n_clusters=primitive_num, n_jobs=-1).fit_predict(all_sp_feats.cpu().numpy().astype(np.float32))
# primitive_labels = SpectralClustering(n_clusters=primitive_num, n_jobs=-1).fit_predict(all_sp_feats.cpu().numpy().astype(np.float32))
primitive_labels = contin_label(primitive_labels)
##
for idx, scene_id in enumerate(train_scene_id):
    # print(scene_id)
    coords, acc_sp = context[scene_id]['coords'].cpu().numpy(), context[scene_id]['acc_sp']
    labels = context[scene_id]['label'].cpu().numpy().astype(np.int32)
    color = np.ones_like(coords) * 128
    color_gt = np.ones_like(coords) * 128
    cur_primitive_labels = primitive_labels[acc_sp.long()]
    for cate in range(primitive_num):
        color[cate == cur_primitive_labels] = colormap[cate]
    for cate in range(semantic_num):
        color_gt[cate == labels] = colormap[cate]
    write_ply(os.path.join('distill_clusterto20', scene_id + '_votecate.ply'), [coords, color.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
    write_ply(os.path.join('distill_clusterto20', scene_id + '_gt.ply'), [coords, color_gt.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
###


# '''Compute Primitive Centers'''
# primitive_centers = torch.zeros((primitive_num, feats_dim))
# for cluster_idx in range(primitive_num):
#     indices = primitive_labels == cluster_idx
#     cluster_avg = all_sp_feats[indices].mean(0, keepdims=True)
#     primitive_centers[cluster_idx] = cluster_avg
# primitive_centers = F.normalize(primitive_centers, dim=1)
# classifier = get_fixclassifier(in_channel=feats_dim, centroids_num=primitive_num, centroids=primitive_centers)

'''spread primitive labels for each sp(growed sp) to points'''
accumulate_sp_num = 0
all_gt, all_pseudo, all_pseudo_gt = [], [], []
for scene_id in train_scene_id:
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

### 100 primitives
### Superpoints oAcc 85.02 IoUs| mIoU 61.01 | 75.95 89.21 63.90 73.68 67.19 73.91 64.01 44.70 58.52 79.69  7.07 57.59 58.78 74.48 40.68 69.21 64.10 37.40 59.99 60.19
### Primitives oAcc 71.65 IoUs| mIoU 34.77 | 63.29 85.40 24.37 43.68 41.74 44.96 38.81  4.92 32.15 46.51  0.00 21.16 27.82 45.68  0.00 49.17 53.89 20.12 34.97 16.74
### 300 primitives
# Superpoints oAcc 85.05 IoUs| mIoU 60.96 | 76.00 89.35 63.84 73.18 67.26 74.48 64.95 45.20 58.73 79.63  6.80 56.88 58.66 74.02 39.80 69.07 64.71 35.96 61.16 59.48
# Primitives oAcc 73.57 IoUs| mIoU 38.66 | 65.19 86.08 28.73 52.31 48.43 45.94 40.41 18.51 33.59 50.09  0.00 25.53 28.50 48.23  0.00 50.40 58.62 22.98 47.06 22.54

### directly cluster to 300 primitives
# Superpoints oAcc 92.08 IoUs| mIoU 76.80 | 84.62 92.98 86.89 87.65 84.64 91.60 84.60 65.50 71.40 90.47 15.50 69.35 81.64 85.16 83.67 80.11 74.16 53.50 71.98 80.63
# Primitives oAcc 75.41 IoUs| mIoU 40.95 | 66.67 88.27 33.08 58.97 53.21 52.11 46.21 18.67 35.49 47.60  0.00 24.59 30.73 47.54 20.79 47.91 56.80 19.04 47.72 23.58

### still cluster to 20 then 300, but no handcrafted features ###
# Superpoints oAcc 85.65 IoUs| mIoU 62.71 | 75.95 89.52 65.28 77.38 70.81 77.96 70.38 41.49 57.24 79.33  7.96 57.42 60.55 72.70 48.12 73.58 65.40 36.08 64.31 62.80
# Primitives oAcc 74.60 IoUs| mIoU 41.16 | 66.68 86.21 32.48 54.10 50.88 48.59 43.30 16.11 37.80 46.58  0.00 27.00 29.32 51.40 19.96 52.31 60.65 26.96 48.07 24.71

### still cluster to 30 then 300, but no handcrafted features ###
# Superpoints oAcc 87.93 IoUs| mIoU 67.46 | 78.77 91.02 72.91 81.49 74.70 83.91 75.41 49.74 61.08 83.45  9.26 62.40 66.42 76.15 59.91 74.51 69.35 41.14 68.20 69.36
# Primitives oAcc 75.00 IoUs| mIoU 40.94 | 67.28 86.86 33.74 52.57 53.29 50.10 44.27 15.88 33.50 46.39  0.00 22.98 30.17 48.91 19.52 49.22 62.86 31.22 47.24 22.81

### still cluster to 40 then 300, but no handcrafted features ###
# Superpoints oAcc 89.12 IoUs| mIoU 70.34 | 80.19 91.53 77.14 83.46 77.02 86.22 78.30 54.83 63.92 85.51 10.65 64.26 71.15 78.23 68.78 75.26 71.30 45.60 70.67 72.74
# Primitives oAcc 75.07 IoUs| mIoU 41.36 | 66.88 87.95 32.83 55.47 53.09 47.06 45.95 14.27 36.33 48.11  0.00 26.50 32.26 44.55 23.61 50.84 58.86 29.87 50.59 22.21

### still cluster to 20 then 100, but no handcrafted features ###
# Superpoints oAcc 85.77 IoUs| mIoU 62.88 | 76.11 89.83 64.97 77.04 71.10 78.42 70.43 42.15 57.16 79.25  7.86 56.84 61.22 72.61 49.58 73.63 65.80 35.71 64.99 62.89
# Primitives oAcc 72.86 IoUs| mIoU 35.68 | 64.62 85.79 29.94 47.60 47.51 50.67 42.22  6.86 26.49 45.61  0.00 20.81 26.78 42.88  0.00 50.07 51.02 18.79 37.02 19.00
### still cluster to 20 then 50, but no handcrafted features ###
# Superpoints oAcc 85.77 IoUs| mIoU 62.88 | 76.11 89.83 64.97 77.04 71.10 78.42 70.43 42.15 57.16 79.25  7.86 56.84 61.22 72.61 49.58 73.63 65.80 35.71 64.99 62.89
# Primitives oAcc 70.95 IoUs| mIoU 28.40 | 63.17 86.57 27.23 35.01 43.33 40.74 42.51  0.00 34.71 42.88  0.00  0.00 19.89 35.49  0.00  0.00 49.63  0.00 28.96 17.96
### still cluster to 20 then 20, but no handcrafted features ###
# Superpoints oAcc 85.77 IoUs| mIoU 62.88 | 76.11 89.83 64.97 77.04 71.10 78.42 70.43 42.15 57.16 79.25  7.86 56.84 61.22 72.61 49.58 73.63 65.80 35.71 64.99 62.89
# Primitives oAcc 68.06 IoUs| mIoU 20.82 | 65.10 81.70 21.87 32.82 41.97  0.00 37.44  0.00 33.14 42.67  0.00  0.00  0.00 25.36  0.00  0.00 21.39  0.00  0.00 13.03

### s8
### 300 primitives
# Primitives oAcc 74.17 IoUs| mIoU 38.51 | 65.21 88.12 31.89 49.34 52.43 47.98 45.08  8.07 34.69 44.76  0.00 24.18 31.17 46.24  0.00 44.28 60.36 30.70 44.00 21.63
### 20 primitives
# Primitives oAcc 67.65 IoUs| mIoU 20.96 | 63.07 87.13 18.55 26.53 33.37 29.31 41.21  0.00 30.56 40.32  0.00  0.00  0.00 37.07  0.00  0.00  0.00  0.00  0.00 12.12

### s8up2
### 20 primitives
# Primitives oAcc 65.63 IoUs| mIoU 17.00 | 60.64 85.11 13.91 27.27 30.18  0.00 39.67  0.00 23.73 37.80  0.00  0.00  0.00 21.67  0.00  0.00  0.00  0.00  0.00  0.00


### dinov2: 20 classes
# Superpoints oAcc 92.08 IoUs| mIoU 76.80 | 84.62 92.98 86.89 87.65 84.64 91.60 84.60 65.50 71.40 90.47 15.50 69.35 81.64 85.16 83.67 80.11 74.16 53.50 71.98 80.63
# Primitives oAcc 73.78 IoUs| mIoU 31.25 | 64.13 85.63 32.62 65.48 54.90 53.58 48.20 34.03 42.78 52.86  0.00  0.00 20.14 51.58  0.00  0.00  0.00  0.00  0.00 19.07

### distill dinov2, 20classes, scannet_sp
# Primitives oAcc 80.70 IoUs| mIoU 32.35 | 74.69 94.66 28.68 81.14 58.95 59.14 55.28 37.40 36.35 48.33  0.00  0.00 23.87  0.00  0.00  0.00  0.00  0.00 37.17 11.40
##spec
# Primitives oAcc 78.93 IoUs| mIoU 33.99 | 69.83 92.30 30.84 77.85 58.73 57.93 55.86 39.78 39.37 49.62  0.00  0.00 24.33  0.00  0.00  0.00 70.91  0.00  0.00 12.47
##spec nonormalized graph:, must do normalization
# Primitives oAcc 34.08 IoUs| mIoU  1.78 |  0.00 34.05  0.00  0.66  0.03  0.00  0.00  0.00  0.00  0.04  0.27  0.00  0.00  0.00  0.00  0.00  0.14  0.00  0.00  0.49

### distill dinov2, 20classes, init_sp
# Primitives oAcc 79.94 IoUs| mIoU 32.39 | 70.45 96.78 29.82 71.69 61.03 57.20 53.69 38.52 45.81 48.66  0.00  0.00 23.38 50.67  0.00  0.00  0.00  0.00  0.00  0.00

## distill dino spec 20
# Primitives oAcc 72.44 IoUs| mIoU 20.71 | 65.54 87.42 21.84  0.00 49.29 31.23 44.26  0.00 30.01 21.59  0.00  0.00 21.20  0.00  0.00  0.00 41.92  0.00  0.00  0.00

## STEGO kmeans 20, scannet sp
# Primitives oAcc 72.05 IoUs| mIoU 17.98 | 71.94 88.48 23.79  0.00 28.82 31.63 38.09  0.00 31.12  0.00 36.11  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  9.68
# Primitives oAcc 71.50 IoUs| mIoU 16.55 | 71.73 86.48 21.80 32.09 28.47  0.00 36.25 23.34 30.84  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
## STEGO kmeans 20, initial sp
# Primitives oAcc 72.76 IoUs| mIoU 16.92 | 72.86 88.10 23.20 33.27 31.71  0.00 35.22 25.98 28.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
# Primitives oAcc 71.24 IoUs| mIoU 18.46 | 71.29 83.88 21.53 32.66 28.16  0.00 29.36 24.44 30.56  0.00  0.00  0.00 12.89 34.43  0.00  0.00  0.00  0.00  0.00  0.00

## STEGO kmeans 20, scannet sp, postdam
# Primitives oAcc 70.55 IoUs| mIoU 16.11 | 65.01 86.55 15.91 27.21 21.90 39.21 44.68  0.00 21.66  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
# Primitives oAcc 69.00 IoUs| mIoU 15.89 | 65.70 89.83 15.05 19.02 18.95 31.25 36.64  0.00 13.60  0.00  0.00  0.00  0.00 27.75  0.00  0.00  0.00  0.00  0.00  0.00

