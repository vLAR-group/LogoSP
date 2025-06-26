import torch
import torch.nn.functional as F
from datasets.S3DIS import S3DISvis, cfl_collate_fn_vis
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

import matplotlib.pyplot as plt
colormap = []
for k in range(12):
    colormap.append(plt.cm.Set3(k))
colormap.append([0, 0, 0, 0])
colormap = np.array(colormap)[:, 0:3]*255

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='../data/S3DIS/input_0.010/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='../data/S3DIS/initial_superpoints_growsp/',
                        help='initial sp path')
    parser.add_argument('--vis_path', type=str, default='vis_predications/S3DIS/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=12, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=12, help='invalid label')
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

    trainval_dataset = S3DISvis(args)
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_vis(), num_workers=4, pin_memory=True)

    use_sp = False
    all_full_preds, all_full_labels, all_full_coords, all_full_colors = [], [], [], []
    voxel_preds, voxel_labels = [], []
    for data in trainval_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region, full_coords, full_colors, full_labels = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
            #
            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)
                #
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

                region_scores = F.linear(F.normalize(region_feats), F.normalize(classifier.weight))
                region_no = 0
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                        region_no +=1
            else:
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

            preds_full = preds[inverse_map.long()]
            voxel_preds.append(preds), voxel_labels.append(labels)
            all_full_preds.append(preds_full), all_full_labels.append(full_labels), all_full_coords.append(full_coords), all_full_colors.append(full_colors)


    preds = np.concatenate(voxel_preds)
    labels = np.concatenate(voxel_labels)
    ##
    sem_num = args.semantic_class
    mask = (labels >= 0) & (labels < sem_num)
    histogram = np.bincount(sem_num * labels[mask] + preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[0], m[1]].sum() / histogram.sum()
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[1][idx]]
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

        colors_input = all_full_colors[i].numpy()
        colors_input = colors_input[mask].astype(np.uint8)


        # Save plys
        cloud_name = trainval_loader.dataset.name[i]

        test_name = os.path.join(vis_path_preds, cloud_name+'preds.ply')
        test_name_gt = os.path.join(vis_path_gt, cloud_name+'gt.ply')
        test_name_input = os.path.join(vis_path_input, cloud_name+'input.ply')

        write_ply(test_name, [coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(test_name_gt, [coords, colors_GT], ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply(test_name_input, [coords, colors_input], ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Pair the numbers with the strings
    paired_list = list(zip(all_miou, trainval_loader.dataset.name))
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
    model_path = '../ckpt/S3DIS/seg/model_checkpoint.pth'
    cls_path = '../ckpt/S3DIS/seg/cls_checkpoint.pth'
    vis_preds(args, model_path, cls_path)