import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
import time
import MinkowskiEngine as ME
import faiss
import open3d as o3d
from lib.helper_ply import read_ply, write_ply

import matplotlib.pyplot as plt
colormap = []
for _ in range(10):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)


def get_sp_feature(args, loader, model, current_growsp):
    print('computing point feats ....')
    point_feats_list = [[], []]
    point_labels_list = [[], []]
    all_sub_cluster = []
    context = []
    sp_semantic = []
    raw_rgb = []
    sp_size = []
    time_start = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords0, coords1, features0, features1, normals0, normals1, labels0, labels1, unique_map, pseudo_labels0, pseudo_labels1, region, inds0, inds1, index, inverse_map = data

            scene_name = loader.dataset.name[index[0]]
            gt0, gt1 = labels0.clone(), labels1.clone()
            raw_region = region.clone()

            in_field0, in_field1 = ME.TensorField(features0, coords0, device=0), ME.TensorField(features1, coords1, device=0)
            feats0, feats1 = model(in_field0), model(in_field1)
            feats0, feats1 = F.normalize(feats0[inds0.long()], dim=-1), F.normalize(feats1[inds1.long()], dim=-1)

            valid_mask = (region!=-1)
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            features0, features1 = features0[inds0.long()].cuda(), features1[inds1.long()].cuda()
            features0, features1 = features0[valid_mask], features1[valid_mask]
            normals0, normals1 = normals0[valid_mask].cuda(), normals1[valid_mask].cuda()
            feats0, feats1 = feats0[valid_mask], feats1[valid_mask]
            labels0, labels1 = labels0[valid_mask], labels1[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_rgb0, pc_rgb1 = features0[:, 0:3], features1[:, 0:3]
            pc_xyz0, pc_xyz1 = features0[:, 3:] * args.voxel_size, features1[:, 3:] * args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ####
            region_feats0, region_feats1 = F.linear(region_corr.t(), feats0.t())/per_region_num, F.linear(region_corr.t(), feats1.t())/per_region_num
            if current_growsp is not None:
                region_rgb0, region_rgb1 = F.linear(region_corr.t(), pc_rgb0.t())/per_region_num, F.linear(region_corr.t(), pc_rgb1.t())/per_region_num
                region_xyz0, region_xyz1 = F.linear(region_corr.t(), pc_xyz0.t())/per_region_num, F.linear(region_corr.t(), pc_xyz1.t())/per_region_num
                region_norm0, region_norm1 = F.linear(region_corr.t(), normals0.t())/per_region_num, F.linear(region_corr.t(), normals1.t())/per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats0, region_feats1 = F.normalize(region_feats0, dim=-1), F.normalize(region_feats1, dim=-1)
                region_feats0, region_feats1 = torch.cat((region_feats0, rgb_w*region_rgb0, xyz_w*region_xyz0, norm_w*region_norm0), dim=-1), \
                                               torch.cat((region_feats1, rgb_w*region_rgb1, xyz_w*region_xyz1, norm_w*region_norm1), dim=-1)
                #
                if region_feats0.size(0)<current_growsp:
                    n_segments = region_feats0.size(0)
                else:
                    n_segments = current_growsp

                # _, error, _, cluster_pred = faiss_kmeans(region_feats.cpu().numpy(), centroids_num=n_segments, centroids_dim=region_feats.size(1))
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats0.cpu().numpy())).long()
            else:
                feats0, feats1 = region_feats0, region_feats1
                sp_idx = sp_idx1 = torch.tensor(range(region_feats0.size(0)))

            neural_region = sp_idx[region]
            pfh0, pfh1 = [], []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
            final_rgb0, final_rgb1 = F.linear(neural_region_corr.t(), pc_rgb0.t())/per_neural_region_num, F.linear(neural_region_corr.t(), pc_rgb1.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats0, feats1 = F.linear(neural_region_corr.t(), feats0.t()) / per_neural_region_num, F.linear(neural_region_corr.t(), feats1.t()) / per_neural_region_num
                feats0, feats1 = F.normalize(feats0, dim=-1), F.normalize(feats1, dim=-1)

            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
            #         # print(batch_idx, p, mask.sum())
            #         # threshold = 30000
            #         # if mask.sum()>=threshold:
            #         #     indices = torch.where(mask)[0]
            #         #     idx_to_change = np.random.choice(range(len(indices)), mask.sum().item() - threshold, replace=False)
            #         #     idx_to_change = indices[idx_to_change]
            #         #     mask[idx_to_change] = False
            #         pfh0.append(compute_hist(normals0[mask].cpu()).unsqueeze(0).cuda())
                    sp_semantic.append(torch.mode(labels0[mask]).values.unsqueeze(0))
            #
            # #
            # pfh0, pfh1 = torch.cat(pfh0, dim=0), torch.cat(pfh1, dim=0)
            # # #
            # rgb_w, geo_w = args.c_rgb, args.c_shape
            # feats0, feats1 = torch.cat((feats0, rgb_w*final_rgb0, geo_w*pfh0), dim=-1), torch.cat((feats0, rgb_w*final_rgb1, geo_w*pfh1), dim=-1)
            feats0, feats1 = F.normalize(feats0, dim=-1), F.normalize(feats1, dim=-1)

            raw_rgb.append(final_rgb0)

            sp_size.append(per_neural_region_num)

            point_feats_list[0].append(feats0.cpu()), point_feats_list[1].append(feats1.cpu())
            point_labels_list[0].append(labels0.cpu()), point_labels_list[1].append(labels1.cpu())

            all_sub_cluster.append(neural_region)
            context.append((scene_name, gt0, raw_region, pc_xyz0, pc_rgb0, inverse_map))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
        print(time.time() - time_start)
    return point_feats_list, point_labels_list, all_sub_cluster, context, torch.cat(sp_semantic), torch.cat(raw_rgb, dim=0), torch.cat(sp_size)



def get_pseudo(args, context, cluster_pred0, cluster_pred1, all_sub_cluster=None):
    print('computing pseduo labels...')
    pseudo_label_folder = args.pseudo_label_path + '/'
    if not os.path.exists(pseudo_label_folder + '/0/'):
        os.makedirs(pseudo_label_folder + '/0/')
    if not os.path.exists(pseudo_label_folder + '/1/'):
        os.makedirs(pseudo_label_folder + '/1/')
    all_gt = []
    all_pseudo = [[], []]
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0
    tmp_my = []

    looptime, savetime, totaltime = 0, 0, 0
    totaltime_start = time.time()

    for i in range(len(context)):
        scene_name, gt, region, coords, pc_rgb, inverse_map = context[i]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        valid_mask = (region != -1)

        labels_tmp = gt[valid_mask]
        pseudo_gt = -torch.ones_like(gt)
        tmp_pseudo_gt = pseudo_gt[valid_mask]

        pseudo0, pseudo1 = -np.ones_like(gt.numpy()).astype(np.int32), -np.ones_like(gt.numpy()).astype(np.int32)
        pseudo0[valid_mask], pseudo1[valid_mask] = cluster_pred0[sub_cluster_pred], cluster_pred1[sub_cluster_pred]

        time_start = time.time()
        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                tmp_pseudo_gt[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = tmp_pseudo_gt
        #
        looptime += time.time() - time_start

        pc_no += 1
        tmp = np.unique(sub_cluster_pred)
        region_num += len(tmp[tmp != -1])

        pseudo_label_file0, pseudo_label_file1  = pseudo_label_folder + '/0/' + scene_name + '.npy', pseudo_label_folder + '/1/' + scene_name + '.npy'

        time_start0 = time.time()
        np.save(pseudo_label_file0, pseudo0)#[inverse_map0])
        np.save(pseudo_label_file1, pseudo1)#[inverse_map1])
        savetime += time.time() - time_start0

        all_gt.append(gt)
        all_pseudo[0].append(pseudo0), all_pseudo[1].append(pseudo1)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo[0], all_pseudo[1] = np.concatenate(all_pseudo[0]), np.concatenate(all_pseudo[1])
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    totaltime+= time.time() - totaltime_start
    print('looptime', looptime, 'savetime', savetime, 'totaltime',totaltime)
    return all_pseudo, all_gt, all_pseudo_gt



def faiss_kmeans(feats, centroids_num, centroids_dim=128, use_GPU = True):
    # print('Kmeansing ....')
    feats = feats.astype('float32')
    faiss_module = faiss.IndexFlatL2(centroids_dim)
    if use_GPU:
        faiss_cfg = faiss.GpuIndexFlatConfig()
        faiss_cfg.useFloat16 = False#True
        faiss_cfg.device = 0  # single gpu only
        faiss_module = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), centroids_dim, faiss_cfg)

    faiss_cluster = faiss.Clustering(centroids_dim, centroids_num)

    faiss_cluster.seed = random.randint(0, 2022)
    faiss_cluster.niter = 100
    faiss_cluster.max_points_per_centroid = 1000000  # 3000000
    faiss_cluster.min_points_per_centroid = 1
    faiss_cluster.nredo = 5
    #
    # count_by_cluster = np.zeros(centroids_num, dtype=np.float32)
    faiss_cluster.train(feats, faiss_module)
    centroids = faiss.vector_float_to_array(faiss_cluster.centroids).reshape(centroids_num, centroids_dim)
    # faiss_module.reset()
    # faiss_module.add(centroids)
    D, I = faiss_module.search(feats, 1)

    if len(np.unique(I))<centroids_num:
        print(np.unique(I))
    # for k in np.unique(I): count_by_cluster[k] += len(np.where(I == k)[0])
    return centroids, D.mean(), D, I.squeeze()



def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for para in classifier.parameters():
        para.requires_grad = False
    return classifier


def compute_hist(normal, bins=10, min=-1, max=1):
    ## normal : [N, 3]
    normal = F.normalize(normal)
    relation = torch.mm(normal, normal.t())
    relation = torch.triu(relation, diagonal=0) # top-half matrix
    hist = torch.histc(relation, bins, min, max)
    # hist = torch.histogram(relation, bins, range=(-1, 1))
    hist /= hist.sum()+1

    return hist
