import os
import numpy as np
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
import time
from sklearn.cluster import KMeans
import warnings
from lib.helper_ply import read_ply, write_ply
import open3d as o3d
from glob import glob
import pickle
from sklearn.decomposition import PCA
from models.fpn import Res16FPN18
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

clip_bound = 4

def augment_coords_to_feats(coords, colors):
    coords_center = coords.mean(0, keepdims=True)
    coords_center[0, 2] = 0
    norm_coords = (coords - coords_center)

    feats = norm_coords
    feats = np.concatenate((colors, feats), axis=-1)
    return norm_coords, feats

def clip(coords, center=None):
    bound_min = np.min(coords, 0).astype(float)
    bound_max = np.max(coords, 0).astype(float)
    bound_size = bound_max - bound_min
    if center is None:
        center = bound_min + bound_size * 0.5
    lim = clip_bound

    if isinstance(clip_bound, (int, float)):
        if bound_size.max() < clip_bound:
            return None
        else:
            clip_inds = ((coords[:, 0] >= (-lim + center[0])) & (coords[:, 0] < (lim + center[0])) & \
                         (coords[:, 1] >= (-lim + center[1])) & (coords[:, 1] < (lim + center[1])) & \
                         (coords[:, 2] >= (-lim + center[2])) & (coords[:, 2] < (lim + center[2])))
            return clip_inds

def contin_label(label):
    Encoder = LabelEncoder()
    return Encoder.fit_transform(label)

def voxelize(coords):
    scale = 1 / 0.05
    coords = np.floor(coords * scale)
    coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), return_index=True, return_inverse=True)
    return coords.numpy(), unique_map, inverse_map

plypath = '/home/zihui/SSD/DivSP/data/S3DIS/input_0.010/'
sp_path = '/home/zihui/SSD/DivSP/data/S3DIS/initial_superpoints_SPG_0.05/'
save_feat_path = '/home/zihui/SSD/DivSP/data/S3DIS/distillv2_point_feats_s14up4_1e-3poly_grid10'
# save_feat_path = '/home/zihui/SSD/DivSP/data/S3DIS/growsp_feats/'
segment_num = 12 ## segment number for each scene
semantic_num = 12
primitive_num = 12
feats_dim = 384
rgb_w, xyz_w, norm_w = 1, 0.2, 0.8

# model = Res16FPN18(in_channels=6, out_channels=20, conv1_kernel_size=5).cuda()
# # model.load_state_dict(torch.load('/home/zihui/SSD/DivSP/ckpt_distill/S3DIS/distillv2_s14up4_1e-3poly_grid1/checkpoint_500.tar')['model'])
# model.load_state_dict(torch.load('/home/zihui/SSD2/GrowSP/ckpt/S3DIS/model_1260_checkpoint.pth'))
# model.eval()

time_start = time.time()

context, all_sp_feats = {}, []
scene_name = []
folders = sorted(glob(plypath + '/*.ply'))
for _, file in enumerate(folders):
    plyname = file.replace(plypath, '')
    scene_name.append(plyname[0:-4])
    print(plyname)
    sp_file = os.path.join(sp_path, plyname[0:-4] + '_superpoint.npy')
    sp = np.load(sp_file)

    data = read_ply(file)
    coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
    colors = colors.astype(np.float32)/255-0.5
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)
    labels[labels == 12] = -1

    ##clip
    clip_inds = clip(coords)
    if clip_inds is not None:
        coords, colors, labels, sp = coords[clip_inds], colors[clip_inds], labels[clip_inds], sp[clip_inds]

    grids, unique_map, inv_map = voxelize(coords)

    coords, colors, labels = coords[unique_map], colors[unique_map], labels[unique_map]
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
    ############ prepare the input to mink
    # # not sure use coords or grids
    # grids, input_feats = augment_coords_to_feats(grids, colors)
    # input_grids = torch.cat((torch.ones(grids.shape[0], 1).int()*0, torch.from_numpy(grids).int()), 1).float()
    # input_feats = torch.from_numpy(input_feats).float()
    # in_field = ME.TensorField(input_feats, input_grids, device=0)
    # with torch.no_grad():
    #     feats = model(in_field).cpu().numpy().astype(np.float32)

    # os.makedirs(save_feat_path, exist_ok=True)
    # with open(os.path.join(save_feat_path, plyname[0:-4]+'.pickle'), 'wb') as f:
    #     pickle.dump(feats, f)
    #############
    with open(os.path.join(save_feat_path, plyname[0:-4]+'.pickle'), 'rb') as f:
        feats = pickle.load(f)

    # pca = PCA(n_components=3)
    # pca_features = pca.fit_transform(feats)
    # min_vals = pca_features.min(axis=0)
    # max_vals = pca_features.max(axis=0)
    # voxel_color = 255 * (pca_features - min_vals) / (max_vals - min_vals)
    # voxel_color = voxel_color.astype(np.uint8)
    #
    # write_ply(os.path.join(save_feat_path, plyname[0:-4]+'.ply'), [coords, voxel_color], ['x', 'y', 'z', 'red', 'green', 'blue'])

    """valid mask"""
    valid_mask = sp!=-1
    coords, colors, labels, feats, sp = coords[valid_mask], colors[valid_mask], labels[valid_mask], feats[valid_mask], sp[valid_mask]
    colors, coords, feats, labels = torch.from_numpy(colors).cuda(), torch.from_numpy(coords).cuda(), torch.from_numpy(feats).cuda(), torch.from_numpy(labels)
    context[plyname[0:-4]] = {'gt': labels, 'initial_sp': sp}
    ''' cluster initial spuerpoints to 20, means one step growing, may not needed'''
    ''' get new sp idx for the 20 segments'''
    sp_num = len(torch.unique(sp))
    sp_corr = torch.zeros(sp.size(0), sp_num)
    sp_corr.scatter_(1, sp.view(-1, 1), 1)
    sp_corr = sp_corr.cuda()
    per_sp_num = sp_corr.sum(0, keepdims=True).t()
    sp_feats = F.linear(sp_corr.t(), feats.t()) / per_sp_num
    sp_feats = F.normalize(sp_feats, dim=-1)
    ''' get sp features (initial or growed)'''
    context[plyname[0:-4]]['final_sp'] = sp
    all_sp_feats.append(sp_feats)
print('sp features extracted')

all_sp_feats = torch.cat(all_sp_feats)
primitive_labels = KMeans(n_clusters=primitive_num, n_jobs=-1).fit_predict(all_sp_feats.cpu().numpy().astype(np.float32))
primitive_labels = contin_label(primitive_labels)

'''spread primitive labels for each sp(growed sp) to points'''
accumulate_sp_num = 0
all_gt, all_pseudo, all_pseudo_gt = [], [], []
for scene_id in scene_name:
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


## grid 5
## 1. growsp:
# Primitives oAcc 78.61 IoUs| mIoU 42.72 | 89.09 83.20 62.89  0.00  0.00 61.76 46.35 52.18 58.36  0.00 58.78  0.00
## 2. SPG 0.05:
# Primitives oAcc 81.20 IoUs| mIoU 44.82 | 86.06 88.25 67.98  0.00  0.00 64.00 54.10 56.30 59.63  0.00 61.48  0.00
## 3. SPG 0.08:
# Primitives oAcc 80.50 IoUs| mIoU 44.30 | 87.67 87.29 65.72  0.00  0.00 58.70 53.47 56.09 61.85  0.00 60.83  0.00
## 4. SPG 0.1:
# Primitives oAcc 79.79 IoUs| mIoU 42.95 | 81.69 86.92 66.24  0.00  0.00 53.30 52.50 53.34 60.39  0.00 61.06  0.00

## grid 10
## 1. growsp:
# Primitives oAcc 78.91 IoUs| mIoU 42.94 | 88.83 85.79 62.53  0.00  0.00 63.35 42.00 52.63 61.38  0.00 58.81  0.00
## 2. SPG 0.05:
# Primitives oAcc 80.71 IoUs| mIoU 44.59 | 87.76 87.53 65.77  0.00  0.00 63.69 53.28 55.60 59.63  0.00 61.79  0.00
## 3. SPG 0.08:
# Primitives oAcc 79.73 IoUs| mIoU 43.56 | 84.18 86.13 64.68  0.00  0.00 58.96 48.68 57.65 61.41  0.00 60.99  0.00
## 4. SPG 0.1:
# Primitives oAcc 79.93 IoUs| mIoU 43.45 | 88.13 85.72 64.99  0.00  0.00 53.46 53.59 53.42 61.80  0.00 60.34  0.00

## grid 1
## 1. growsp:
# Primitives oAcc 78.90 IoUs| mIoU 42.86 | 89.03 83.38 63.62  0.00  0.00 61.64 48.44 50.80 58.30  0.00 59.05  0.00
## 2. SPG 0.05:
# Primitives oAcc 81.26 IoUs| mIoU 44.81 | 86.20 87.63 68.33  0.00  0.00 63.18 55.08 55.70 60.16  0.00 61.42  0.00
## 3. SPG 0.08:
# Primitives oAcc 80.01 IoUs| mIoU 43.85 | 81.57 86.43 65.95  0.00  0.00 59.12 51.81 54.10 61.95  0.00 65.23  0.00
## 4. SPG 0.1:
# Primitives oAcc 79.48 IoUs| mIoU 42.61 | 81.19 85.79 66.18  0.00  0.00 53.23 53.49 51.78 59.60  0.00 60.10  0.00

## sup
## 1.growsp:
# Primitives oAcc 92.84 IoUs| mIoU 75.05 | 95.77 95.51 84.39 78.96 63.35 77.40 84.78 86.85 89.45 57.98 86.14  0.00


## growp
## 1. growsp
# Primitives oAcc 80.41 IoUs| mIoU 43.83 | 94.10 89.64 63.83 54.70  0.00  0.00 46.41 61.69 62.35  0.00 53.25  0.00
