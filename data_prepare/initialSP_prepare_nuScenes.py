import open3d as o3d
import numpy as np
import torch
from scipy import stats
from os.path import join, exists, dirname, abspath
import sys, glob

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import read_ply, write_ply
import time
import os
import matplotlib.pyplot as plt
from pathlib import Path


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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='../data/nuscenes/nuScenes_3d/train/', help='raw data path')
parser.add_argument('--sp_path', type=str, default='../data/nuScenes/initial_superpoints/')
args = parser.parse_args()

vis = False


def ransac(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    return np.array(inliers)


def construct_superpoints(path):
    f = Path(path)
    data = torch.load(f)
    coords = data[0]
    labels = data[2]
    mask = labels==255
    labels[mask]=-1
    coords = coords.astype(np.float32)
    coords -= coords.mean(0)

    time_start = time.time()
    name = join(f.parts[-2], f.name)

    '''RANSAC'''
    road_index = ransac(coords)
    other_index = []
    for i in range(coords.shape[0]):
        if i not in road_index:
            other_index.append(i)
    other_index = np.array(other_index)
    other_coords = coords[other_index]  # *self.voxel_size
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(other_coords)
    other_region_idx = np.array(pcd.cluster_dbscan(eps=0.2, min_points=1))

    sp_labels = -np.ones_like(labels)
    sp_labels[other_index] = other_region_idx
    sp_labels[road_index] = other_region_idx.max() + 1
    #
    if not os.path.exists(join(args.sp_path, f.parts[-2])):
        os.makedirs(join(args.sp_path, f.parts[-2]))
    np.save(join(args.sp_path, name[:-4]+'_superpoint.npy'), sp_labels)

    if vis:
        vis_path = join(args.sp_path, 'vis', f.parts[-2])
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        colors = np.zeros_like(coords)
        for p in range(colors.shape[0]):
            colors[p] = 255 * (colormap[sp_labels[p].astype(np.int32)])[:3]
        colors = colors.astype(np.uint8)

        out_coords = data[0]
        write_ply(vis_path + '/' + f.name[:-4], [out_coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    sp2gt = -np.ones_like(labels)
    for sp in np.unique(sp_labels):
        if sp != -1:
            sp_mask = sp == sp_labels
            sp2gt[sp_mask] = stats.mode(labels[sp_mask])[0][0]

    print('completed scene: {}, used time: {:.2f}s'.format(name, time.time() - time_start))
    return (labels, sp2gt)


print('start constructing initial superpoints')
train_path_list = []

scene_list = np.sort(os.listdir(args.input_path))
for scene_id in scene_list:
    scene_path = join(args.input_path, scene_id)
    train_path_list.append(scene_path)

#train_path_list = train_path_list[24500:]

for path in train_path_list:
    construct_superpoints(path)


print('end constructing initial superpoints')

all_labels, all_sp2gt = [], []
for (labels, sp2gt) in result:
    mask = (sp2gt != -1)
    labels, sp2gt = labels[mask].astype(np.int32), sp2gt[mask].astype(np.int32)
    all_labels.append(labels), all_sp2gt.append(sp2gt)

all_labels, all_sp2gt  = np.concatenate(all_labels), np.concatenate(all_sp2gt)
sem_num =16
mask = (all_labels != -1) & (all_labels < sem_num)
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
#result = list(pool.map(construct_superpoints, train_path_list))
#print(' Acc: {:.5f}  Test IoU'.format(o_Acc), s)
