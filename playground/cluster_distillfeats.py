import os
import numpy as np
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
import time
from sklearn.cluster import KMeans
import warnings
from lib.helper_ply import read_ply, write_ply
import pickle
from sklearn.decomposition import PCA
from models.fpn import Res16FPN18
from models.pretrain_models import SubModel
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


def augment_coords_to_feats(coords, colors):
    coords_center = coords.mean(0, keepdims=True)
    coords_center[0, 2] = 0
    norm_coords = (coords - coords_center)

    feats = norm_coords
    feats = np.concatenate((colors, feats), axis=-1)
    return norm_coords, feats

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
    return coords.numpy(), unique_map, inverse_map

plypath = '/home/zihui/SSD/LogoSP/data/ScanNet/processed'
sp_path = '/home/zihui/SSD/LogoSP/data/ScanNet/initial_superpoints'
save_feat_path = '/home/zihui/SSD/LogoSP/data/ScanNet/distill_point_feats_s8_1e-3poly_grid'
# save_feat_path = '/home/zihui/SSD/DivSP/data/ScanNet/distillstego_point_feats_s8_1e-3poly_grid'
# save_feat_path = '/home/zihui/SSD/LogoSP/data/ScanNet/distill_SAM'
# save_feat_path = '/home/zihui/SSD/DivSP/data/ScanNet/distilled_tmp'
segment_num = 20 ## segment number for each scene
semantic_num = 20
primitive_num = 20
feats_dim = 384
rgb_w, xyz_w, norm_w = 1, 0.2, 0.8

# model = Res16FPN18(in_channels=6, out_channels=20, conv1_kernel_size=5).cuda()
# model.load_state_dict(torch.load('/home/zihui/SSD/LogoSP/ckpt_distill/ScanNet/distil_SAM_grid/checkpoint_200.tar'))
# model.load_state_dict(torch.load('/home/zihui/SSD/DivSP/ckpt_GFR/ScanNet_corr/distillv2_nodistill/model_100_checkpoint.pth'))
# # model.load_state_dict(torch.load('/home/zihui/SSD/DivSP/ckpt_GFR/ScanNet_final/distillv2_40sp_kmeans_elsaug_t3_col50_enrg0.99/model_200_checkpoint.pth'))
# # model.load_state_dict(torch.load('./ckpt_distill/ScanNet/distill_s8_1e-3poly_grid/checkpoint_200.tar')['model'])
# model.load_state_dict(torch.load('/home/zihui/SSD/LogoSP/ckpt_distill/ScanNet/distil_SAM_grid/checkpoint_200.tar')['model'])
# # model.load_state_dict(torch.load('/home/zihui/SSD/DivSP/ckpt_GFR/ScanNet/PointDC_distillbyseg_reassign/model_20_checkpoint.pth'))
# model.eval()

# submodel = SubModel(70).cuda()
# submodel.load_state_dict(torch.load('/home/zihui/SSD/DivSP/submodule_200_checkpoint.pth'))
# submodel.eval()

train_scene_id = read_txt('/home/zihui/SSD/LogoSP/data/ScanNet/scannet/scannet_3d/scannetv2_val.txt')
# train_scene_id = ['scene0297_00', 'scene0354_00', 'scene0420_00', 'scene0487_00', 'scene0587_00', 'scene0011_00']
time_start = time.time()

context, all_sp_feats = {}, []
for scene_id in train_scene_id:
    print(scene_id)
    data = read_ply(os.path.join(plypath, scene_id+'.ply'))
    coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
    colors = colors.astype(np.float32)/255-0.5
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)
    raw_coords = coords.copy()

    grids, unique_map, inv_map = voxelize(coords)

    coords, colors, labels = coords[unique_map], colors[unique_map], labels[unique_map]
    ##
    sp = np.load(os.path.join(sp_path, scene_id+'_superpoint.npy'))
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
    ############# prepare the input to mink
    # # # not sure use coords or grids
    # grids, input_feats = augment_coords_to_feats(grids, colors)
    # input_grids = torch.cat((torch.ones(grids.shape[0], 1).int()*0, torch.from_numpy(grids).int()), 1).float()
    # input_feats = torch.from_numpy(input_feats).float()
    # in_field = ME.TensorField(input_feats, input_grids, device=0)
    # with torch.no_grad():
    #     # feats = submodel(model(in_field)).cpu().numpy().astype(np.float32)
    #     feats = model(in_field).cpu().numpy().astype(np.float32)
    #
    # os.makedirs(save_feat_path, exist_ok=True)
    # with open(os.path.join(save_feat_path, scene_id+'.pickle'), 'wb') as f:
    #     pickle.dump(feats, f)
    ############
    with open(os.path.join(save_feat_path, scene_id+'.pickle'), 'rb') as f:
        feats = pickle.load(f)

    # pca = PCA(n_components=3)
    # pca_features = pca.fit_transform(feats)
    # min_vals = pca_features.min(axis=0)
    # max_vals = pca_features.max(axis=0)
    # voxel_color = 255 * (pca_features - min_vals) / (max_vals - min_vals)
    # voxel_color = voxel_color.astype(np.uint8)
    # write_ply(os.path.join(save_feat_path, scene_id+'.ply'), [raw_coords, voxel_color[inv_map]], ['x', 'y', 'z', 'red', 'green', 'blue'])

    """valid mask"""
    valid_mask = sp!=-1
    coords, colors, labels, feats, sp = coords[valid_mask], colors[valid_mask], labels[valid_mask], feats[valid_mask], sp[valid_mask]
    colors, coords, feats, labels = torch.from_numpy(colors).cuda(), torch.from_numpy(coords).cuda(), torch.from_numpy(feats).cuda(), torch.from_numpy(labels)
    context[scene_id] = {'gt': labels, 'initial_sp': sp}
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
    context[scene_id]['final_sp'] = sp
    all_sp_feats.append(sp_feats)
print('sp features extracted')

all_sp_feats = torch.cat(all_sp_feats)
primitive_labels = KMeans(n_clusters=primitive_num, n_jobs=-1).fit_predict(all_sp_feats.cpu().numpy().astype(np.float32))
primitive_labels = contin_label(primitive_labels)

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


### 20 primitives
# Primitives oAcc 68.98 IoUs| mIoU 21.28 | 63.25 84.13 17.90 33.79 44.92 32.50 43.25  0.00 27.68 44.35  0.00  0.00  0.00 22.78  0.00  0.00  0.00  0.00  0.00 11.06

### 300 primitives
# Primitives oAcc 76.16 IoUs| mIoU 37.43 | 67.67 90.71 32.30 60.93 59.23 48.86 48.59 16.60 33.60 53.82  0.00 14.06 31.01 41.81  0.00 40.76 37.72  0.00 48.55 22.33

### 300 primitives for poly
# Primitives oAcc 76.32 IoUs| mIoU 40.17 | 67.03 89.77 30.94 56.66 59.91 50.14 51.67 20.52 37.38 50.30  0.00 21.09 30.59 45.80  0.00 42.69 53.11 19.89 52.48 23.39

### 20 primitives for distill dinov2
# Primitives oAcc 73.52 IoUs| mIoU 28.93 | 63.26 88.29 25.28 63.53 55.31 52.42 45.17  0.00 43.22 56.03  0.00  0.00 20.70 47.14  0.00  0.00  0.00  0.00  0.00 18.33
### 20 primitives for distill dinov2 pre
# Primitives oAcc 72.54 IoUs| mIoU 28.64 | 64.33 81.90 34.06 63.70 44.40 53.30 40.98  0.00 42.14 57.28  0.00  0.00 23.46 46.63  0.00  0.00  0.00  0.00  0.00 20.60
### 20 primitives for distill dinov2 grid
# Primitives oAcc 73.78 IoUs| mIoU 29.10 | 63.63 88.93 24.76 62.69 56.58 51.95 44.07  0.00 43.68 55.95  0.00  0.00 20.51 49.79  0.00  0.00  0.00  0.00  0.00 19.56


### 20 primitives for stage1 pointdc with dinov2
# Primitives oAcc 74.47 IoUs| mIoU 28.74 | 65.19 89.62 23.79 65.85 59.62 54.17 46.32  0.00 42.22 61.19  0.00  0.00 21.60 45.20  0.00  0.00  0.00  0.00  0.00  0.00
