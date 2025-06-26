import torch
import torch.nn.functional as F
from datasets.ScanNet import Scannetval, cfl_collate_fn_val, Scannettest, cfl_collate_fn_test
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
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))

label_inverse_map = {}
for l in range(20):
    label_inverse_map[l] = VALID_CLASS_IDS[l]
label_inverse_map[-1] = -1

colormap = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/ScanNet/processed/',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/ScanNet/initial_superpoints/',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='./ckpt_GFR/ScanNet_final/distillv2_40sp_kmeans_elsaug_t3_col50_enrg0.99/',
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
    parser.add_argument('--primitive_num', type=int, default=20, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=20, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
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
            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)
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



def eval(epoch, args):
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()

    primitive_centers = cls.weight.data
    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, random_state=0).fit_predict(primitive_centers.cpu().numpy())

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

    val_dataset = Scannetval(args)
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



def test_once(matching, model, test_loader, classifier, use_sp=False):

    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, index, region = data

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

            preds_tmp = preds.clone()
            for i in range(len(matching[0])):
                preds[preds_tmp == matching[1][i]] = i

            preds = preds[inverse_map.long()]
            scene_name = test_loader.dataset.name[index[0]]

            pred_ori_labels = np.array([label_inverse_map[x.item()] for x in preds])
            if not os.path.exists('online_testing/'):
                os.mkdir('online_testing/')
            savepath = 'online_testing/' + scene_name + '.txt'
            np.savetxt(savepath, pred_ori_labels.astype(int), fmt='%d')

            if not os.path.exists('online_testing/vis/'):
                os.mkdir('online_testing/vis/')
            colors = np.array([colormap[x] for x in preds])
            colors = np.array(colors*255).astype(np.uint8)
            coords = coords[:, 1:].numpy()*args.voxel_size
            write_ply('online_testing/vis/' + scene_name + '.ply', [coords[inverse_map], colors], ['x', 'y', 'z', 'red', 'green', 'blue'])


def online_test(epoch, args):
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.semantic_class, bias=False)
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()
    # #
    centroids = F.normalize(cls.weight.data.cpu(), dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    test_dataset = Scannettest(args)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=4, pin_memory=True)

    trainval_dataset = Scannetval(args)
    trainval_dataset.path_file = './data_prepare/ScanNet_splits/scannetv2_trainval.txt'
    trainval_loader = DataLoader(trainval_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=4, pin_memory=True)

    preds, labels = eval_once(args, model, trainval_loader, classifier)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    matching = linear_assignment(histogram.max() - histogram)
    test_once(matching, model, test_loader, classifier)



if __name__ == '__main__':

    args = parse_args()
    print(args)
    # for epoch in range(10, 800):
    #     if epoch%930==0:
    # epoch = 30
    # o_Acc, m_Acc, s = eval(epoch, args)
    # print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
    online_test(200, args)
