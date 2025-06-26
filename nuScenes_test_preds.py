import torch
import torch.nn.functional as F
from datasets.nuScenes import nuScenestest, cfl_collate_fn_test
from datasets.nuScenes import nuScenesval, cfl_collate_fn_val
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18
import warnings
import argparse
import os
import matplotlib.pyplot as plt
from lib.helper_ply import write_ply
warnings.filterwarnings('ignore')
colormap = np.array(
    [[220,220,  0], [119, 11, 32], [0, 60, 100], [0, 0, 250], [230,230,250],
     [0, 0, 230], [220, 20, 60], [250, 170, 30], [200, 150, 0], [0, 0, 110],
     [128, 64, 128], [0,250, 250], [244, 35, 232], [152, 251, 152], [70, 70, 70],
     [107,142, 35]])
###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--test_input_path', type=str, default='./nuScenes_test_data',
                        help='pont cloud data path')
    parser.add_argument('--val_input_path', type=str, default='./data/nuScenes/nuScenes_3d/val',
                        help='pont cloud data path')
    parser.add_argument('--out_path', type=str, default='./nuScenes_online_testing',
                        help='pont cloud data path')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=10, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=16, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=16, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--model_ckpt', type=str, default='./ckpt_seg/nuScenes/distillv2_20sp_kmeans_16_50m_spec0-50/model_35_checkpoint.pth')
    parser.add_argument('--classifier_ckpt', type=str, default='./ckpt_seg/nuScenes/distillv2_20sp_kmeans_16_50m_spec0-50/model_35_checkpoint.pth')
    return parser.parse_args()


args = parse_args()
def test_preds(args, model, classifier, val_loader):
    for data in val_loader:
        with torch.no_grad():
            coords,inverse_map, index, original_coords, scene_name = data
            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)
            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()
            preds = preds[inverse_map.long()]
            ##use val match test
            vectorized_func = np.vectorize(lambda x: match_dict.get(x, x))
            preds = vectorized_func(preds)
            ##
            colors = colormap[preds].astype(np.uint8)
            tes_name = os.path.join(args.out_path, 'vis'+'/'+ scene_name[0]+'.ply')
            write_ply(tes_name, [original_coords.numpy(), colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
            ###
            preds = preds+1
            preds = preds.astype(np.uint8)
            ##
            bin_file_path = f"{os.path.join(args.out_path, 'preds')}/{scene_name[0]}_lidarseg.bin"
            np.array(preds).astype(np.uint8).tofile(bin_file_path)


'''create vis file'''
if not os.path.exists(os.path.join(args.out_path, 'vis')):
    os.makedirs(os.path.join(args.out_path, 'vis'))
if not os.path.exists(os.path.join(args.out_path, 'preds')):
    os.makedirs(os.path.join(args.out_path, 'preds'))


model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args, mode='train').cuda()
model.load_state_dict(torch.load('/home/zihui/SSD/LogoSP/ckpt_seg/nuScenes/distillv2_20sp_kmeans_16_50m_spec0-50/model_5_checkpoint.pth'))
model.eval()

classifier = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
classifier.load_state_dict(torch.load('/home/zihui/SSD/LogoSP/ckpt_seg/nuScenes/distillv2_20sp_kmeans_16_50m_spec0-50/cls_5_checkpoint.pth'))
classifier.eval()


##preds should match label,preds use dict to transform label space.
test_dataset = nuScenestest(args)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=args.cluster_workers, pin_memory=True)

val_dataset = nuScenesval(args)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=args.cluster_workers, pin_memory=True)


all_preds, all_label = [], []
for data in val_loader:
    with torch.no_grad():
        # coords, labels, inverse_map, index, original_coords, scene_name =data
        coords, inverse_map, labels, index = data
        in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
        feats = model(in_field)
        feats = F.normalize(feats, dim=1)
        scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
        preds = torch.argmax(scores, dim=1).cpu()
        preds = preds[inverse_map.long()]
        all_preds.append(preds[labels != args.ignore_label]), all_label.append(labels[[labels != args.ignore_label]])


##matching
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_label).numpy()
'''Unsupervised, Match pred to gt'''
sem_num = args.semantic_class
mask = (all_labels >= 0) & (all_labels < sem_num)
histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num,sem_num)
'''Hungarian Matching'''
m = linear_assignment(histogram.max() - histogram)
match_dict = {m[i, 1]: m[i, 0] for i in range(len(m))}  # {preds:gt}

'''preds'''
test_preds(args, model, classifier, test_loader)











