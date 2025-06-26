import torch
import torch.nn.functional as F
from datasets.nuScenes import nuScenesval, cfl_collate_fn_val
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from utils_degrowsp import get_fixclassifier
from lib.helper_ply import read_ply, write_ply
import argparse
import matplotlib.pyplot as plt
import os
colormap = np.array(
    [[220,220,  0], [119, 11, 32], [0, 60, 100], [0, 0, 250], [230,230,250],
     [0, 0, 230], [220, 20, 60], [250, 170, 30], [200, 150, 0], [0, 0, 110],
     [128, 64, 128], [0,250, 250], [244, 35, 232], [152, 251, 152], [70, 70, 70],
     [107,142, 35]])

classes =['barrier','bicycle','bus','car','construction vehicle','motorcycle','person','traffic cone','trailer',
        'truck','drivable surface','other flat','sidewalk','terrain','manmade','vegetation']

###
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='./data/nuScenes/nuscenes_3d/train/',
                        help='pont cloud data path')
    parser.add_argument('--val_input_path', type=str, default='./data/nuScenes/nuscenes_3d/val',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= './data/nuScenes/initial_superpoints/train/',
                        help='initial sp path')
    parser.add_argument('--point_feats_path', type=str, default='data/ScanNet/distillv2_point_feats_s14up4_1e-3poly/',
                        help='project dino point feature')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=16, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4,help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--batch_size', type=int, default=8, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')  ### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=16, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=16, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def eval_once(args, model, test_loader, classifier):

    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords,inverse_map,labels, index = data

            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            # in_field = ME.TensorField(coords[:, 1:], coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]
            preds = preds[labels!=args.ignore_label]
            labels = labels[labels!=args.ignore_label]
            all_preds.append(preds), all_label.append(labels)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return all_preds, all_label


def eval(epoch, args):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=16, conv1_kernel_size=5, config=args).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()

    primitive_centers = cls.weight.data###[300, 128]
    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, random_state=0).fit_predict(primitive_centers.cpu().numpy())#.astype(np.float64))

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    val_dataset = nuScenesval(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=args.cluster_workers, pin_memory=True)

    preds, labels = eval_once(args, model, val_loader, classifier)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[0], m[1]].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[m[0], m[1]] / histogram.sum(1))*100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[1][idx]]

    # plot_confusion_matrix(cm=hist_new, classes=classes)

    '''Final Metrics'''
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)

    return o_Acc, m_Acc, s


import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(20, 16))  # Increase the figure size
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion.png')



if __name__ == '__main__':

    args = parse_args()
    epoch=-1
    o_Acc, m_Acc, s = eval(epoch, args)
    print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
    # for epoch in range(0, 2):
    #     if epoch==1:
    #         o_Acc, m_Acc, s = eval(epoch, args)
    #         print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)

