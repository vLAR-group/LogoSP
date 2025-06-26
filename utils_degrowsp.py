import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.cluster import KMeans, SpectralClustering
import MinkowskiEngine as ME
from sklearn.preprocessing import LabelEncoder

def contin_label(label):
    Encoder = LabelEncoder()
    return Encoder.fit_transform(label)


def get_sp_feature(args, loader,current_sp, model):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    context = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, labels, inverse_map, pseudo_labels, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)

            valid_mask = region != -1
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)  # ?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()  ##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t()) / per_region_num
            if current_sp is not None:
                region_feats = F.normalize(region_feats, dim=-1)
                #
                if region_feats.size(0) < current_sp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_sp
                sp_idx = KMeans(n_clusters=n_segments, random_state=0).fit_predict(region_feats.cpu().numpy())
                sp_idx = contin_label(sp_idx)
                sp_idx = torch.from_numpy(sp_idx).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            if current_sp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)
            context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
        return point_feats_list, point_labels_list, all_sp_index, context


def get_sp_feature_unScenes(args, loader,current_sp, model):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    context = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, labels, inverse_map, pseudo, region, index, scene_name = data

            region = region.squeeze()
            scene_name = scene_name[0]
            gt = labels.clone()
            raw_region = region.clone()

            # if model is not None:
            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            feats = model(in_field)

            valid_mask = region != -1
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)  # ?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()  ##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t()) / per_region_num
            if current_sp is not None:
                region_feats = F.normalize(region_feats, dim=-1)
                if region_feats.size(0) < current_sp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_sp
                sp_idx = KMeans(n_clusters=n_segments, random_state=0).fit_predict(region_feats.cpu().numpy())
                sp_idx = contin_label(sp_idx)
                sp_idx = torch.from_numpy(sp_idx).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            if current_sp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)
            context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
        return point_feats_list, point_labels_list, all_sp_index, context



def get_pseudo(args, context, cluster_pred, all_sp_index=None):
    print('computing pseduo labels...')
    pseudo_label_folder = args.save_path + '/' + args.pseudo_label_path + '/'
    if not os.path.exists(pseudo_label_folder):
        os.makedirs(pseudo_label_folder)
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sp_index[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_file = pseudo_label_folder + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_pseudo_kitti(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_folder = args.pseudo_label_path + '/' + scene_name[0:3]
        if not os.path.exists(pseudo_label_folder):
            os.makedirs(pseudo_label_folder)

        pseudo_label_file = args.pseudo_label_path + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for para in classifier.parameters():
        para.requires_grad = False
    return classifier