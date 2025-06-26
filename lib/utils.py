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
for _ in range(1000):
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
    point_feats_list = []
    point_labels_list = []
    all_sub_cluster = []
    model.eval()
    context = []
    sp_semantic = []
    raw_rgb = []
    sp_size = []
    time_start = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(features, coords, device=0)

            feats = model(in_field)
            # feats = F.normalize(feats, dim=-1)
            feats = feats[inds.long()]

            valid_mask = (region!=-1)
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            # normals = normals[inds.long()].cuda()
            normals = normals[valid_mask].cuda()
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:] * args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
                region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
                #
                if region_feats.size(0)<current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp

                # _, error, _, cluster_pred = faiss_kmeans(region_feats.cpu().numpy(), centroids_num=n_segments, centroids_dim=region_feats.size(1))
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                # sp_idx = torch.from_numpy(SpectralClustering(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                # sp_idx = torch.from_numpy(AgglomerativeClustering(n_clusters=n_segments).fit_predict(region_feats.cpu().numpy())).long()
                # sp_idx = torch.from_numpy(DBSCAN(eps=0.75, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                #print(sp_idx)
                # sp_idx = torch.from_numpy(MeanShift(bandwidth=0.74, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
            final_rgb = F.linear(neural_region_corr.t(), pc_rgb.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    # print(batch_idx, p, mask.sum())
                    # threshold = 30000
                    # if mask.sum()>=threshold:
                    #     indices = torch.where(mask)[0]
                    #     idx_to_change = np.random.choice(range(len(indices)), mask.sum().item() - threshold, replace=False)
                    #     idx_to_change = indices[idx_to_change]
                    #     mask[idx_to_change] = False
                    pfh.append(compute_hist(normals[mask].cpu()).unsqueeze(0).cuda())
                    sp_semantic.append(torch.mode(labels[mask]).values.unsqueeze(0))
            #
            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            rgb_w, geo_w = args.c_rgb, args.c_shape
            feats = torch.cat((feats, rgb_w*final_rgb, geo_w*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)

            raw_rgb.append(final_rgb)

            sp_size.append(per_neural_region_num)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sub_cluster.append(neural_region)
            context.append((scene_name, gt, raw_region, pc_xyz, pc_rgb))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
        print(time.time() - time_start)
    return point_feats_list, point_labels_list, all_sub_cluster, context, torch.cat(sp_semantic), torch.cat(raw_rgb, dim=0), torch.cat(sp_size)


def get_sp_feature_selection(args, loader, model, current_growsp, old_sp_list):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sub_cluster = []
    model.eval()
    context = []
    avg_sp_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(features, coords, device=0)

            feats = model(in_field)
            # feats = F.normalize(feats, dim=-1)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            # normals = normals[inds.long()].cuda()
            normals = normals[valid_mask].cuda()
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()

            coords = coords[:, 1:]
            coords = coords[inds.long()]
            coords = coords[valid_mask]
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:] * args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
                region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
                #
                if region_feats.size(0)<current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                # _, error, _, cluster_pred = faiss_kmeans(region_feats.cpu().numpy(), centroids_num=n_segments, centroids_dim=region_feats.size(1))
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                # sp_idx = torch.from_numpy(AgglomerativeClustering(n_clusters=n_segments).fit_predict(region_feats.cpu().numpy())).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            pfh = []

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(coords)
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
            # raw_normals = torch.from_numpy(np.array(pcd.normals)).float().cuda()

            # sp_score = 0
            # colors = -torch.ones_like(coords)
            # for p in torch.unique(neural_region):
            #     if p!=-1:
            #         mask = p==neural_region
            #         ''' Computen normal within superpoint'''
            #         region_coords = coords[mask].numpy()
            #         pcd = o3d.geometry.PointCloud()
            #         pcd.points = o3d.utility.Vector3dVector(region_coords)
            #         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
            #         current_region_normals = torch.from_numpy(np.array(pcd.normals)).float().cuda()
            #         raw_region_normals = raw_normals[mask]
            #         sim = (F.normalize(current_region_normals)*F.normalize(raw_region_normals)).sum(-1).abs().mean()
            #         # degree = np.arccos(np.clip(sim.cpu().numpy(), -1, 1))/np.pi*180
            #         # degree = degree.mean()
            #         ### cos(1du)=0.9998, cos(3du)=0.9986, cos(5du)=0.9962, cos(10du)=0.9848, cos(20du)=0.9397
            #         sp_score += sim
            #         print('superpoint normal cosine distance:',sim.item())
            #         if sim<0.9848:
            #             colors[mask] = torch.tensor([255.0, 0, 0])
            #         # if degree>10:
            #         #     colors[mask] = torch.tensor([255.0, 0, 0])
            # sp_score/= len(torch.unique(neural_region))
            # # print('AVG superpoint normal cosine distance:', sp_score)

            # colors = colors.numpy()
            # colors = colors.astype(np.uint8)
            # write_ply('tmp' + scene_name+'.ply', [coords.cpu().numpy(), colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

            ######### local geometry protection mechanism #######
            neural_region_update = neural_region.clone()
            if current_growsp is not None:
                sp_score = 0
                # old_sp = old_sp_list[batch_idx]
                old_sp = region
                for p in torch.unique(neural_region):
                    if p != -1:
                        mask = p == neural_region
                        # color_mask = colors[mask]
                        ''' Computen normal within superpoint'''
                        region_coords = coords[mask].numpy()
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(region_coords)
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
                        neural_region_normals = torch.from_numpy(np.array(pcd.normals)).float().cuda()
                        ### old region access
                        old_region = old_sp[mask]
                        old_region_normals = -torch.ones_like(neural_region_normals)
                        bound_mask = torch.zeros(old_region_normals.size(0)).bool()
                        if len(torch.unique(old_region))>1:
                            for q in torch.unique(old_region):
                                q_mask = q==old_region
                                old_region_coords = coords[mask][q_mask].numpy()
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(old_region_coords)
                                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
                                old_region_normals[q_mask] = torch.from_numpy(np.array(pcd.normals)).float().cuda()

                                old_region_othercoords = coords[mask][~q_mask]
                                dist = (torch.from_numpy(old_region_coords).unsqueeze(1) - old_region_othercoords.unsqueeze(0)).pow(2).sum(-1).sqrt() ###[N. M] N is the point num of current superpoint
                                bound_mask[q_mask] = dist.min(-1)[0] <3 ## True is 1, False is 0
                                # color_mask[bound_mask] = np.array([1, 0, 0])

                            sim = (F.normalize(neural_region_normals[bound_mask]) * F.normalize(old_region_normals[bound_mask])).sum(-1).abs().mean()
                            sp_score += sim
                            # print('superpoint normal cosine distance:', sim.item())
                            ### cos(1du)=0.9998, cos(3du)=0.9986, cos(5du)=0.9962, cos(10du)=0.9848, cos(20du)=0.9397
                            if sim < 0.9848:
                                neural_region_update[mask] = old_sp[mask]+ len(torch.unique(neural_region))

                sp_score /= len(torch.unique(neural_region))

                neural_region = -torch.ones_like(neural_region_update)
                sp_index_update = torch.unique(neural_region_update)
                for i in range(len(sp_index_update)):
                    sp_i = sp_index_update[i]
                    sp_i_mask = sp_i==neural_region_update
                    neural_region[sp_i_mask] = i
            #################################################################################

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
            final_rgb = F.linear(neural_region_corr.t(), pc_rgb.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask].cpu()).unsqueeze(0).cuda())

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            rgb_w, geo_w = args.c_rgb, args.c_shape
            feats = torch.cat((feats, rgb_w*final_rgb, geo_w*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sub_cluster.append(neural_region)
            context.append((scene_name, gt, raw_region))

            avg_sp_num += neural_region_num

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

            # if current_growsp:
            if not os.path.exists(args.save_path + '/'+ str(current_growsp)):
                os.makedirs(args.save_path + '/'+ str(current_growsp))
            colors = np.zeros_like(coords)
            for p in range(colors.shape[0]):
                colors[p] = 255 * (colormap[neural_region[p].numpy().astype(np.int32)])[:3]
            colors = colors.astype(np.uint8)
            write_ply(args.save_path + '/'+ str(current_growsp) + '/'+ scene_name+'.ply', [coords.cpu().numpy(), colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        print('AVG SP num', avg_sp_num/(batch_idx+1))
    return point_feats_list, point_labels_list, all_sub_cluster, context



def get_sp_feature_selection_prog(args, loader, model, current_growsp, old_sp_list):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sub_cluster = []
    model.eval()
    context = []
    avg_sp_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(features, coords, device=0)

            feats = model(in_field)
            # feats = F.normalize(feats, dim=-1)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            # normals = normals[inds.long()].cuda()
            normals = normals[valid_mask].cuda()
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()

            coords = coords[:, 1:]
            coords = coords[inds.long()]
            coords = coords[valid_mask]

            # colors = np.zeros_like(coords)
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:] * args.voxel_size
            ###
            if current_growsp is not None:
                old_sp = old_sp_list[batch_idx]
                # old_sp = region
                region_num = len(torch.unique(old_sp))
                region_corr = torch.zeros(old_sp.size(0), region_num)  # ?
                region_corr.scatter_(1, old_sp.view(-1, 1), 1)
                region_corr = region_corr.cuda()  ##[N, M]
                per_region_num = region_corr.sum(0, keepdims=True).t()
                region_feats = F.linear(region_corr.t(), feats.t()) / per_region_num

                region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
                region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
                #
                if region_feats.size(0)<current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                # _, error, _, cluster_pred = faiss_kmeans(region_feats.cpu().numpy(), centroids_num=n_segments, centroids_dim=region_feats.size(1))
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                # sp_idx = torch.from_numpy(AgglomerativeClustering(n_clusters=n_segments).fit_predict(region_feats.cpu().numpy())).long()
                neural_region = sp_idx[old_sp]

            else:
                region_num = len(torch.unique(region))
                region_corr = torch.zeros(region.size(0), region_num)  # ?
                region_corr.scatter_(1, region.view(-1, 1), 1)
                region_corr = region_corr.cuda()  ##[N, M]
                per_region_num = region_corr.sum(0, keepdims=True).t()
                region_feats = F.linear(region_corr.t(), feats.t()) / per_region_num

                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))
                neural_region = sp_idx[region]

            pfh = []

            ######### local geometry protection mechanism #######
            neural_region_update = neural_region.clone()
            if current_growsp is not None:
                sp_score = 0
                old_sp = old_sp_list[batch_idx]
                # old_sp = region
                for p in torch.unique(neural_region):
                    if p != -1:
                        mask = p == neural_region
                        # color_mask = colors[mask]
                        ''' Computen normal within superpoint'''
                        region_coords = coords[mask].numpy()
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(region_coords)
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
                        neural_region_normals = torch.from_numpy(np.array(pcd.normals)).float().cuda()
                        ### old region access
                        old_region = old_sp[mask]
                        old_region_normals = -torch.ones_like(neural_region_normals)
                        bound_mask = torch.zeros(old_region_normals.size(0)).bool()
                        if len(torch.unique(old_region))>1:
                            for q in torch.unique(old_region):
                                q_mask = q==old_region
                                old_region_coords = coords[mask][q_mask].numpy()
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(old_region_coords)
                                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
                                old_region_normals[q_mask] = torch.from_numpy(np.array(pcd.normals)).float().cuda()

                                old_region_othercoords = coords[mask][~q_mask]
                                dist = (torch.from_numpy(old_region_coords).unsqueeze(1) - old_region_othercoords.unsqueeze(0)).pow(2).sum(-1).sqrt() ###[N. M] N is the point num of current superpoint
                                bound_mask[q_mask] = dist.min(-1)[0] <3 ## True is 1, False is 0
                                # color_mask[bound_mask] = np.array([1, 0, 0])

                            sim = (F.normalize(neural_region_normals[bound_mask]) * F.normalize(old_region_normals[bound_mask])).sum(-1).abs().mean()
                            sp_score += sim
                            # print('superpoint normal cosine distance:', sim.item())
                            ### cos(1du)=0.9998, cos(3du)=0.9986, cos(5du)=0.9962, cos(10du)=0.9848, cos(20du)=0.9397
                            if sim < 0.866:
                                neural_region_update[mask] = old_sp[mask]+ len(torch.unique(neural_region))
                    # colors[mask] = color_mask

                sp_score /= len(torch.unique(neural_region))

                neural_region = -torch.ones_like(neural_region_update)
                sp_index_update = torch.unique(neural_region_update)
                for i in range(len(sp_index_update)):
                    sp_i = sp_index_update[i]
                    sp_i_mask = sp_i==neural_region_update
                    neural_region[sp_i_mask] = i
            #############################################################

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
            final_rgb = F.linear(neural_region_corr.t(), pc_rgb.t()) / per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            for p in torch.unique(neural_region):
                if p != -1:
                    mask = p == neural_region
                    pfh.append(compute_hist(normals[mask].cpu()).unsqueeze(0).cuda())

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            rgb_w, geo_w = args.c_rgb, args.c_shape
            feats = torch.cat((feats, rgb_w * final_rgb, geo_w * pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sub_cluster.append(neural_region)
            context.append((scene_name, gt, raw_region))

            avg_sp_num += neural_region_num

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

            # if current_growsp:
            if not os.path.exists(args.save_path + '/' + str(current_growsp)):
                os.makedirs(args.save_path + '/' + str(current_growsp))
            colors = np.zeros_like(coords)
            for p in range(colors.shape[0]):
                colors[p] = 255 * (colormap[neural_region[p].numpy().astype(np.int32)])[:3]
            # colors *= 255
            colors = colors.astype(np.uint8)
            write_ply(args.save_path + '/' + str(current_growsp) + '/' + scene_name + '.ply',
                      [coords.cpu().numpy(), colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        print('AVG SP num', avg_sp_num / (batch_idx + 1))
    return point_feats_list, point_labels_list, all_sub_cluster, context


def get_kittisp_feature(args, loader, model, current_growsp):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sub_cluster = []
    model.eval()
    context = []
    clustertime, nntime, producttime, pfhtime = 0, 0, 0, 0
    totaltime = 0
    totaltime_start = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, voxel_labels, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt, voxel_gt = labels.clone(), voxel_labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(coords[:, 1:], coords, device=0)

            time_start = time.time()
            feats = model(in_field)
            nntime += time.time()-time_start
            feats = feats[inds.long()]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            voxel_labels = voxel_labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_remission = features[:,-1].squeeze().unsqueeze(-1)
            pc_xyz = coords[inds.long()][valid_mask][:, 1:].cuda()*args.voxel_size
            ##
            time_start0 = time.time()
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                region_feats = torch.cat((region_feats, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
                #
                if region_feats.size(0)<current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                time_start1 = time.time()
                cluster_pred = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                # _, error, _, cluster_pred = faiss_kmeans(region_feats.cpu().numpy(), centroids_num=n_segments, centroids_dim=region_feats.size(1))
                # if len(np.unique(cluster_pred))<n_segments:
                #     print(np.unique(cluster_pred))
                # cluster_pred = torch.from_numpy(cluster_pred)
                clustertimeiter = time.time() - time_start1
                clustertime += clustertimeiter
                cluster_corr = torch.zeros(len(cluster_pred), n_segments)
                cluster_corr.scatter_(1, cluster_pred.view(-1, 1), 1)
                cluster_corr = cluster_corr.cuda()##[N, M]
                per_cluster_num = cluster_corr.sum(0, keepdims=True).t()
                #
                feats = F.linear(cluster_corr.t(), region_feats.t())/per_cluster_num
                feats = feats[:,0:args.centroids_dim]
                feats = F.normalize(feats, dim=-1)
            else:
                feats = region_feats
                cluster_pred = torch.tensor(range(region_feats.size(0)))

            neural_region = cluster_pred[region]
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            final_remission = F.linear(neural_region_corr.t(), pc_remission.t())/per_neural_region_num
            producttime += time.time() - time_start0# - clustertimeiter
            #
            time_start2 = time.time()
            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask]).unsqueeze(0))

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            remission_w, geo_w = args.c_rgb, args.c_shape
            feats = torch.cat((feats, remission_w*final_remission, geo_w*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)
            pfhtime += time.time() - time_start2

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sub_cluster.append(neural_region)
            context.append((scene_name, gt, voxel_gt, raw_region, inverse_map))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    totaltime += time.time() - totaltime_start
    print('nntime', nntime, 'producttime', producttime, 'clustertime',clustertime, 'pfhtime', pfhtime, 'totaltime', totaltime)
    return point_feats_list, point_labels_list, all_sub_cluster, context



def get_kittisp_feature_cylinder(args, loader, model, current_growsp):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sub_cluster = []
    model.eval()
    context = []
    clustertime, nntime, producttime, pfhtime = 0, 0, 0, 0
    totaltime = 0
    totaltime_start = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, voxel_labels, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt, voxel_gt = labels.clone(), voxel_labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(features, coords, device=0)

            time_start = time.time()
            feats = model(in_field)
            nntime += time.time()-time_start
            feats = feats[inds.long()]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            voxel_labels = voxel_labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_remission = features[:,-1].squeeze().unsqueeze(-1)
            pc_xyz = coords[inds.long()][valid_mask][:, 1:].cuda()*args.voxel_size
            ##
            time_start0 = time.time()
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                # region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
                # region_norm = F.linear(region_corr.t(), normals.t())/per_region_num
                #
                # rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                # region_feats = torch.cat((region_feats, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
                #
                if region_feats.size(0)<current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                time_start1 = time.time()
                # cluster_pred = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
                cluster_pred = torch.from_numpy(AgglomerativeClustering(n_clusters=n_segments).fit_predict(region_feats.cpu().numpy())).long()
                # _, error, _, cluster_pred = faiss_kmeans(region_feats.cpu().numpy(), centroids_num=n_segments, centroids_dim=region_feats.size(1))
                # if len(np.unique(cluster_pred))<n_segments:
                #     print(np.unique(cluster_pred))
                # cluster_pred = torch.from_numpy(cluster_pred)
                clustertimeiter = time.time() - time_start1
                clustertime += clustertimeiter
                cluster_corr = torch.zeros(len(cluster_pred), n_segments)
                cluster_corr.scatter_(1, cluster_pred.view(-1, 1), 1)
                cluster_corr = cluster_corr.cuda()##[N, M]
                per_cluster_num = cluster_corr.sum(0, keepdims=True).t()
                #
                feats = F.linear(cluster_corr.t(), region_feats.t())/per_cluster_num
                feats = feats[:,0:args.centroids_dim]
                feats = F.normalize(feats, dim=-1)
            else:
                feats = region_feats
                cluster_pred = torch.tensor(range(region_feats.size(0)))

            neural_region = cluster_pred[region]
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            final_remission = F.linear(neural_region_corr.t(), pc_remission.t())/per_neural_region_num
            producttime += time.time() - time_start0# - clustertimeiter
            #
            time_start2 = time.time()
            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask]).unsqueeze(0))

            pfh = torch.cat(pfh, dim=0)
            # for p in torch.unique(neural_region):
            #     if p!=-1:
            #         mask = p==neural_region
            #         pfh.append(compute_hist2(pc_remission[mask]).unsqueeze(0))

            # pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            remission_w, geo_w = args.c_rgb, args.c_shape
            feats = torch.cat((feats, remission_w*final_remission, geo_w*pfh), dim=-1)
            # feats = torch.cat((feats, geo_w*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)
            pfhtime += time.time() - time_start2

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sub_cluster.append(neural_region)
            context.append((scene_name, gt, voxel_gt, raw_region, inverse_map))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    totaltime += time.time() - totaltime_start
    print('nntime', nntime, 'producttime', producttime, 'clustertime',clustertime, 'pfhtime', pfhtime, 'totaltime', totaltime)
    return point_feats_list, point_labels_list, all_sub_cluster, context



def get_pseudo(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')
    pseudo_label_folder = args.pseudo_label_path + '/'
    if not os.path.exists(pseudo_label_folder):
        os.makedirs(pseudo_label_folder)
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0
    tmp_my = []

    looptime, savetime, totaltime = 0, 0, 0
    totaltime_start = time.time()

    for i in range(len(context)):
        scene_name, gt, region, coords, pc_rgb = context[i]
        # scene_name, gt, region = context[i]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = gt[valid_mask]
        pseudo_gt = -torch.ones_like(gt)
        tmp_pseudo_gt = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(gt.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        time_start = time.time()
        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                tmp_my.append(sub_cluster_gt.unsqueeze(0))
                tmp_pseudo_gt[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = tmp_pseudo_gt
        #
        looptime += time.time() - time_start

        pc_no += 1
        tmp = np.unique(sub_cluster_pred)
        region_num += len(tmp[tmp != -1])

        pseudo_label_file = pseudo_label_folder + '/' + scene_name + '.npy'

        time_start0 = time.time()
        np.save(pseudo_label_file, pseudo)
        savetime += time.time() - time_start0


        ######### vis primitive
        # Save plys
        # from distinctipy import distinctipy
        # colormap = distinctipy.get_colors(500)

        # for p in np.unique(pseudo[valid_mask]):
        #     if (p==pseudo[valid_mask]).sum()>=10:
        #         save_path = 'vis_primitives/' + str(p) + '/'
        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path)
        #         test_name = save_path + scene_name + '.ply'
        #         colors = 255*(pc_rgb.cpu().numpy()+0.5)
        #         colors[pseudo[valid_mask]!=p] = np.array([128, 128, 128])
        #         colors = colors.astype(np.uint8)
        #         write_ply(test_name, [coords.cpu().numpy(), colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # #######


        all_gt.append(gt)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    totaltime+= time.time() - totaltime_start
    print('looptime', looptime, 'savetime', savetime, 'totaltime',totaltime)
    return all_pseudo, all_gt, all_pseudo_gt, torch.cat(tmp_my)


# def get_pseudo(args, context, cluster_pred, all_sub_cluster=None):
#     print('computing pseduo labels...')
#     pseudo_label_folder = args.pseudo_label_path + '/'
#     if not os.path.exists(pseudo_label_folder):
#         os.makedirs(pseudo_label_folder)
#     all_gt = []
#     all_pseudo = []
#     all_pseudo_gt = []
#     pc_no = 0
#     region_num = 0
#     tmp_my = []
#
#     looptime, savetime, totaltime = 0, 0, 0
#     totaltime_start = time.time()
#
#     for i in range(len(context)):
#         scene_name, gt, region, coords, pc_rgb = context[i]
#         # scene_name, gt, region = context[i]
#
#         sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
#         valid_mask = region != -1
#
#         labels_tmp = gt[valid_mask]
#         pseudo_gt = -torch.ones_like(gt)
#         tmp_pseudo_gt = pseudo_gt[valid_mask]
#
#         pseudo = -np.ones_like(gt.numpy()).astype(np.int32)
#         pseudo[valid_mask] = cluster_pred[sub_cluster_pred]
#
#         time_start = time.time()
#         for p in np.unique(sub_cluster_pred):
#             if p != -1:
#                 mask = p == sub_cluster_pred
#                 sub_cluster_gt = torch.mode(labels_tmp[mask]).values
#                 tmp_my.append(sub_cluster_gt.unsqueeze(0))
#                 tmp_pseudo_gt[mask] = sub_cluster_gt
#         pseudo_gt[valid_mask] = tmp_pseudo_gt
#         #
#         looptime += time.time() - time_start
#
#         pc_no += 1
#         tmp = np.unique(sub_cluster_pred)
#         region_num += len(tmp[tmp != -1])
#
#         pseudo_label_file = pseudo_label_folder + '/' + scene_name + '.npy'
#
#         time_start0 = time.time()
#         np.save(pseudo_label_file, pseudo)
#         savetime += time.time() - time_start0
#
#         all_gt.append(gt)
#         all_pseudo.append(pseudo)
#         all_pseudo_gt.append(pseudo_gt)
#
#     all_gt = np.concatenate(all_gt)
#     all_pseudo = np.concatenate(all_pseudo)
#     all_pseudo_gt = np.concatenate(all_pseudo_gt)
#
#     totaltime+= time.time() - totaltime_start
#     print('looptime', looptime, 'savetime', savetime, 'totaltime',totaltime)
#     return all_pseudo, all_gt, all_pseudo_gt, torch.cat(tmp_my)





def get_pseudo2(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')
    pseudo_label_folder = args.pseudo_label_path + '/'
    if not os.path.exists(pseudo_label_folder):
        os.makedirs(pseudo_label_folder)
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    looptime, savetime, totaltime = 0, 0, 0
    totaltime_start = time.time()

    for i in range(len(context)):
        scene_name, gt, region, inverse_map = context[i]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = gt[valid_mask]
        pseudo_gt = -torch.ones_like(gt)
        tmp_pseudo_gt = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(gt.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

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

        pseudo_label_file = pseudo_label_folder + '/' + scene_name + '.npy'

        time_start0 = time.time()
        # np.save(pseudo_label_file, pseudo[inverse_map.long()])
        np.save(pseudo_label_file, pseudo)
        savetime += time.time() - time_start0

        all_gt.append(gt)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    totaltime+= time.time() - totaltime_start
    print('looptime', looptime, 'savetime', savetime, 'totaltime',totaltime)
    return all_pseudo, all_gt, all_pseudo_gt

def get_pseudo_kitti(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')

    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    looptime, savetime, totaltime = 0, 0, 0
    totaltime_start = time.time()

    for i in range(len(context)):
        # scene_name, gt, region = context[i]
        scene_name, gt,  voxel_gt, region, inverse_map = context[i]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        # region = region[inverse_map.long()]
        valid_mask = region != -1

        labels_tmp =  voxel_gt[valid_mask]
        pseudo_gt = -torch.ones_like( voxel_gt)
        tmp_pseudo_gt = pseudo_gt[valid_mask]

        pseudo = -np.ones_like( voxel_gt.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

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

        pseudo_label_folder = args.pseudo_label_path + '/' + scene_name[0:3]
        if not os.path.exists(pseudo_label_folder):
            os.makedirs(pseudo_label_folder)

        pseudo_label_file = args.pseudo_label_path + '/' + scene_name + '.npy'

        time_start0 = time.time()
        np.save(pseudo_label_file, pseudo[inverse_map.long()])
        # pseudo = pseudo[inverse_map.long()]
        # pseudo_gt = pseudo_gt[inverse_map.long()]
        # np.save(pseudo_label_file, pseudo)
        savetime += time.time() - time_start0

        all_gt.append(voxel_gt)#(gt)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    totaltime+= time.time() - totaltime_start
    print('looptime', looptime, 'savetime', savetime, 'totaltime',totaltime)
    counts = torch.tensor(np.bincount(all_pseudo[all_pseudo != -1].astype('int32'), minlength=args.centroids_num)).float()
    weight = (counts.sum() / counts)
    weight = weight/weight.sum()
    # print(weight)

    return all_pseudo, all_gt, all_pseudo_gt, weight


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
    faiss_cluster.nredo = 10
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

# def faiss_kmeans(feats, centroids_num, centroids_dim=128, use_GPU = True):
#     feats = feats.astype('float32')
#     kmeans = faiss.Kmeans(centroids_dim, centroids_num, niter=100, nredo=10, seed=2022, min_points_per_centroid=1, max_points_per_centroid=1000000, gpu=use_GPU)
#     kmeans.train(feats)
#     centroids = kmeans.centroids
#
#     # if feats.shape[0]<= centroids_num:
#     #     '''Do not search nearest index'''
#     #     D, I = np.zeros((feats.shape[0], 1)), np.array(range(feats.shape[0]))
#     # else:
#     D, I = kmeans.assign(feats)
#     return centroids, D.mean(), D, I.squeeze()



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


def compute_hist2(reflection, bins=10, min=0, max=1):
    hist = torch.histc(reflection, bins, min, max)
    hist /= hist.sum()

    return hist
