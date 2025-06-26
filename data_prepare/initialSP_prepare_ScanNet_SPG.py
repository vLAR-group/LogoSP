from scipy import stats, spatial
import os
from os.path import join, exists, dirname, abspath
import sys
from glob import glob
import torch
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append("./partition/cut-pursuit/build/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
import libcp
import libply_c
from partition.graphs import *

from lib.helper_ply import read_ply, write_ply
import time
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from pathlib import Path
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))


colormap = []
for _ in range(10000):
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



input_path = './data/ScanNet/processed/'
sp_save_path = './data/ScanNet/SPG_0.05/'
ignore_label = -1
voxel_size = 0.05

k_nn_geof = 45
k_nn_adj = 10
lambda_edge_weight = 1.
reg_strength = 0.05

vis = True
# if not exists(sp_save_path):
#     os.makedirs(sp_save_path)
# os.system(f"cp {__file__} {sp_save_path}")

def construct_superpoints(path):
    f = Path(path)
    if os.path.exists(sp_save_path + '/' + f.parts[-1][0:-4] + '_superpoint.npy'):
        scene_name = f.parts[-1][0:-4]
        sp2gt = np.load(sp_save_path + '/' + scene_name + '_superpoint.npy')
        data = read_ply(path)
        semantic = data['class'].squeeze()
        out_labels = semantic.squeeze()
    else:
        scene_name = f.parts[-1][0:-4]
        data = read_ply(path)
        semantic = data['class'].squeeze()
        rgb = np.vstack((data['red'], data['green'], data['blue'])).T
        rgb = rgb.astype(np.float32)
        rgb /= 255
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        xyz -= xyz.mean(0)

        time_start = time.time()
        '''Voxelize'''
        scale = 1 / voxel_size
        coords = np.floor(xyz * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), rgb, labels=semantic, ignore_label=-1, return_index=True, return_inverse=True) ### labels is not used later
        # coords = coords.numpy().astype(np.float32)

        # ''' SPG '''
        # ---compute 10 nn graph-------
        xyz = xyz[unique_map].astype(np.float32)
        rgb = rgb[unique_map].astype(np.float32) / 255
        graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
        # ---compute geometric features-------
        geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
        del target_fea
        # --compute the partition------
        features = np.hstack((geof, rgb)).astype('float32')  # add rgb as a feature for partitioning
        features[:, 3] = 2. * features[:, 3]  # increase importance of verticality (heuristic)

        graph_nn["edge_weight"] = np.array(1. / (lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype='float32')
        components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"], graph_nn["edge_weight"], reg_strength)
        superpoint = in_component.astype(np.int32)
        out_sp_labels = superpoint[inverse_map]
        out_coords = np.vstack((data['x'], data['y'], data['z'])).T
        out_labels = semantic.squeeze()

        if not os.path.exists(sp_save_path):
            os.makedirs(sp_save_path)
        np.save(sp_save_path + '/' + scene_name + '_superpoint.npy', out_sp_labels)

        if vis:
            vis_path = sp_save_path +'/vis/'
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            colors = np.zeros_like(out_coords)
            for p in range(colors.shape[0]):
                colors[p] = 255 * (colormap[out_sp_labels[p].astype(np.int32)].squeeze())[:3]
            colors = colors.astype(np.uint8)
            write_ply(vis_path + '/' + scene_name + '.ply', [out_coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

        sp2gt = -np.ones_like(out_labels)
        for sp in np.unique(out_sp_labels):
            if sp != -1:
                sp_mask = sp == out_sp_labels
                sp2gt[sp_mask] = stats.mode(out_labels[sp_mask])[0][0]

        print('completed scene: {}, used time: {:.2f}s'.format(scene_name, time.time() - time_start))
    return (out_labels, sp2gt)


if __name__ == "__main__":
    ignore_label = -1
    scene_name_list = read_txt('data_prepare/ScanNet_splits/scannetv2_trainval.txt')
    print('start constructing initial superpoints')
    result = []
    path_list = sorted(glob(os.path.join(input_path, '*.ply')))
    for path in path_list:
        if path.split('/')[-1] in scene_name_list:
            result.append(construct_superpoints(path))

    print('end constructing initial superpoints')

    all_labels, all_sp2gt = [], []
    for (labels, sp2gt) in result:
        mask = (sp2gt != -1) & (sp2gt != ignore_label)
        labels, sp2gt = labels[mask].astype(np.int32), sp2gt[mask].astype(np.int32)
        all_labels.append(labels), all_sp2gt.append(sp2gt)

    all_labels, all_sp2gt  = np.concatenate(all_labels), np.concatenate(all_sp2gt)
    sem_num = 20
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_sp2gt[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum()
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    print(' Acc: {:.5f}  Test IoU'.format(o_Acc), s)