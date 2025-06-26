import torch
import torch.nn.functional as F
from datasets.nuScenes import nuScenesvis, cfl_collate_fn_vis
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from utils_degrowsp import get_fixclassifier
import argparse
import os
from lib.helper_ply import write_ply

colormap = np.array(
    [[220,220,  0], [119, 11, 32], [0, 60, 100], [0, 0, 250], [230,230,250],
     [0, 0, 230], [220, 20, 60], [250, 170, 30], [200, 150, 0], [0, 0, 110],
     [128, 64, 128], [0,250, 250], [244, 35, 232], [152, 251, 152], [70, 70, 70],
     [107,142, 35]])

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='../data/nuScenes/nuscenes_3d/train/',
                        help='pont cloud data path')
    parser.add_argument('--val_input_path', type=str, default='../data/nuScenes/nuscenes_3d/val',
                        help='pont cloud data path')
    parser.add_argument('--vis_path', type=str, default='vis_predications/nuScenes/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=16, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=16, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def vis_preds(args, model_path, cls_path):
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.semantic_class, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.semantic_class, bias=False).cuda()
    cls.load_state_dict(torch.load(cls_path))
    cls.eval()

    primitive_centers = cls.weight.data###[300, 128]
    print('merging primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, random_state=0).fit_predict(primitive_centers.cpu().numpy())

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.semantic_class, centroids_num=args.feats_dim, centroids=centroids).cuda()
    classifier.eval()

    val_dataset = nuScenesvis(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    all_full_preds, all_full_labels, all_full_coords = [], [], []
    voxel_preds, voxel_labels = [], []
    for data in val_loader:
        with torch.no_grad():
            coords, inverse_map, labels, index, full_coords, full_labels = data

            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)
            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds_full = preds[inverse_map.long()]
            preds_full, full_labels, full_coords = preds_full[full_labels!=-1], full_labels[full_labels!=-1], full_coords[full_labels!=-1]
            full_mask = np.sqrt(((full_coords) ** 2).sum(-1)) < 40

            voxel_preds.append(preds[labels!=--1]), voxel_labels.append(labels[labels!=--1])
            all_full_preds.append(preds_full[full_mask]), all_full_labels.append(full_labels[full_mask]), all_full_coords.append(full_coords[full_mask])


    preds = np.concatenate(voxel_preds)
    labels = np.concatenate(voxel_labels)
    ##
    sem_num = args.semantic_class
    mask = (labels >= 0) & (labels < sem_num)
    histogram = np.bincount(sem_num * labels[mask] + preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum()
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]
    # get final metrics
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    print('Epoch: {:02d}, Test acc: {:.5f}  Test IoU'.format(epoch, o_Acc), s)

    print('Visualize')
    vis_path_preds = os.path.join(args.vis_path, 'preds')
    vis_path_gt = os.path.join(args.vis_path, 'gt')
    vis_path_input = os.path.join(args.vis_path, 'input')

    os.makedirs(vis_path_preds, exist_ok=True)
    os.makedirs(vis_path_gt, exist_ok=True)
    os.makedirs(vis_path_input, exist_ok=True)

    m_resort = m[np.argsort(m[:,1])]

    all_miou = []
    for i, coords in enumerate(all_full_coords):

        label = all_full_labels[i]
        mask = torch.logical_and(label!=-1, label!=0)
        preds = all_full_preds[i]
        preds = m_resort[preds, 0]

        ## compute mIoU for this scenes
        hist = np.bincount(sem_num * label[mask] + preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
        # get final metrics
        tp = np.diag(hist)
        fp = np.sum(hist, 0) - tp
        fn = np.sum(hist, 1) - tp
        IoUs = tp / (tp + fp + fn + 1e-8)
        m_IoU = np.nanmean(IoUs)
        all_miou.append(m_IoU)
        ##

        coords = coords[mask].numpy()

        colors = colormap[preds]
        colors = colors[mask].astype(np.uint8)

        colors_GT = colormap[label]
        colors_GT = colors_GT[mask].astype(np.uint8)

        colors_input = np.ones_like(colors_GT)*128
        colors_input = colors_input.astype(np.uint8)


        # Save plys
        cloud_name = val_loader.dataset.name[i]

        test_name = os.path.join(vis_path_preds, cloud_name+'preds.ply')
        test_name_gt = os.path.join(vis_path_gt, cloud_name+'gt.ply')
        test_name_input = os.path.join(vis_path_input, cloud_name+'input.ply')

        write_ply(test_name, [coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(test_name_gt, [coords, colors_GT], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(test_name_input, [coords, colors_input], ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Pair the numbers with the strings
    paired_list = list(zip(all_miou, val_loader.dataset.name))
    # Sort the paired list by the numbers in descending order
    sorted_paired_list = sorted(paired_list, key=lambda x: x[0], reverse=True)
    # Unzip the sorted pairs back into two lists
    sorted_numbers, sorted_strings = zip(*sorted_paired_list)
    # Convert the tuples back to lists (if needed)
    sorted_numbers = list(sorted_numbers)
    sorted_strings = list(sorted_strings)
    for i in range(100):
        print(sorted_strings[i], sorted_numbers[i])


if __name__ == '__main__':
    args = parse_args()
    model_path = '../ckpt/nuScenes/seg/model_checkpoint.pth'
    cls_path = '../ckpt/nuScenes/seg/cls_checkpoint.pth'
    vis_preds(args, model_path, cls_path)