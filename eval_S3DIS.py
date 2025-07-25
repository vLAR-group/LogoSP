import torch
import torch.nn.functional as F
from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
# from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from utils_degrowsp import get_fixclassifier
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/S3DIS/input_0.010/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/S3DIS/initial_superpoints_growsp',
                        help='initial superpoint path')
    parser.add_argument('--save_path', type=str, default='ckpt_GFR/S3DIS_selftrain/distillv2_80sp_kmeans_elsaug_spec0-10/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=12, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--drop_threshold', type=int, default=30, help='ignore superpoints with few points')
    return parser.parse_args()


def eval_once(args, model, test_loader, classifier, use_sp=False):

    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

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

            preds = preds[inverse_map.long()]
            all_preds.append(preds[labels!=args.ignore_label]), all_label.append(labels[[labels!=args.ignore_label]])

    return all_preds, all_label



def eval(epoch, args, test_areas = ['Area_6']):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=12, conv1_kernel_size=5, config=args).cuda()
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

    test_dataset = S3DIStest(args, areas=test_areas)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=4, pin_memory=True)

    preds, labels = eval_once(args, model, test_loader, classifier, use_sp=False)
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


if __name__ == '__main__':

    args = parse_args()
    # for epoch in range(10, 1500):
    #     if epoch % 1270 == 0:
    # for epoch in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
    # for epoch in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    for epoch in range(10, 280, 10):
        o_Acc, m_Acc, s = eval(epoch, args)
        print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)