import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from lib.helper_ply import read_ply, write_ply
from sklearn.cluster import KMeans
import time
import MinkowskiEngine as ME
scannet_colormap = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])

def construct_growing_superpoints(args, loader, model, colormap, current_growsp, epoch):
    print('computing point feats ....')
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            time_start = time.time()
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            cloud_name = loader.dataset.name[index[0]]

            in_field = ME.TensorField(features, coords, device=0)

            feats = model(in_field)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:]*args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
            region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
            region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

            region_feats= F.normalize(region_feats, dim=-1)
            rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
            region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
            #
            if region_feats.size(0)<current_growsp:
                current_growsp = region_feats.size(0)
            grwosp_labels = torch.from_numpy(KMeans(n_clusters=current_growsp, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()

            '''Visualization for Growing Superpoints'''
            inverse_map = inverse_map.long()
            coords = coords[:, 1:].numpy()[inverse_map]
            labels = labels[inverse_map]
            grwosp_labels = grwosp_labels[inverse_map]
            #
            mask = (labels == -1)

            colors = 255 * (np.array(colormap)[grwosp_labels])  # [:, 0:3]
            colors[~mask] = np.zeros(3)
            colors = colors.astype(np.uint8)

            savepath = args.save_path + str(epoch) + '/growsp/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            test_name = savepath + cloud_name + '.ply'
            write_ply(test_name, [coords[mask], colors[mask]], ['x', 'y', 'z', 'red', 'green', 'blue'])
            print('completed scene: {}, used time: {:.2f}s'.format(cloud_name, time.time() - time_start))


def construct_kitti_growing_superpoints(args, loader, model, colormap, current_growsp, epoch):
    print('computing point feats ....')
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            time_start = time.time()
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            cloud_name = loader.dataset.name[index[0]]

            in_field = ME.TensorField(features, coords, device=0)

            feats = model(in_field)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:]*args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
            region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
            region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

            region_feats= F.normalize(region_feats, dim=-1)
            rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
            region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
            #
            if region_feats.size(0)<current_growsp:
                current_growsp = region_feats.size(0)
            grwosp_labels = torch.from_numpy(KMeans(n_clusters=current_growsp, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()

            '''Visualization for Growing Superpoints'''
            inverse_map = inverse_map.long()
            coords = coords[:, 1:].numpy()[inverse_map]
            labels = labels[inverse_map]
            grwosp_labels = grwosp_labels[inverse_map]
            #
            mask = (labels == -1)

            colors = 255 * (np.array(colormap)[grwosp_labels])  # [:, 0:3]
            colors[~mask] = np.zeros(3)
            colors = colors.astype(np.uint8)

            savepath = args.save_path + str(epoch) + '/growsp/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            test_name = savepath + cloud_name + '.ply'
            write_ply(test_name, [coords[mask], colors[mask]], ['x', 'y', 'z', 'red', 'green', 'blue'])
            print('completed scene: {}, used time: {:.2f}s'.format(cloud_name, time.time() - time_start))


def construct_growing_primitive(args, loader, model, colormap, current_growsp, epoch, classifier):
    print('computing point feats ....')
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            time_start = time.time()
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            cloud_name = loader.dataset.name[index[0]]

            in_field = ME.TensorField(features, coords, device=0)

            feats = model(in_field)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            region = region[valid_mask].long()

            labels = labels[valid_mask]
            coords = coords[inds.long()]
            coords = coords[valid_mask]
            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:]*args.voxel_size
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            # region_rgb = F.linear(region_corr.t(), pc_rgb.t())/per_region_num
            # region_xyz = F.linear(region_corr.t(), pc_xyz.t())/per_region_num
            # region_norm = F.linear(region_corr.t(), normals.t())/per_region_num

            region_feats= F.normalize(region_feats, dim=-1)
            rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
            # region_feats = torch.cat((region_feats, rgb_w*region_rgb, xyz_w*region_xyz, norm_w*region_norm), dim=-1)
            #
            # if region_feats.size(0)<current_growsp:
            #     current_growsp = region_feats.size(0)
            # grwosp_labels = torch.from_numpy(KMeans(n_clusters=current_growsp, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()

            neural_region = region#grwosp_labels[region]
            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
            feats = F.normalize(feats, dim=-1)


            # cluster_corr = torch.zeros(len(grwosp_labels), current_growsp)
            # cluster_corr.scatter_(1, grwosp_labels.view(-1, 1), 1)
            # cluster_corr = cluster_corr.cuda()  ##[N, M]
            # per_cluster_num = cluster_corr.sum(0, keepdims=True).t()
            # #
            # feats = F.linear(cluster_corr.t(), region_feats.t()) / per_cluster_num
            # feats = feats[:, 0:args.centroids_dim]
            # feats = F.normalize(feats, dim=-1)
            primitive_scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            primitive_labels = torch.argmax(primitive_scores, dim=1).cpu()[neural_region]

            '''Visualization for Growing Primitive'''
            inverse_map = inverse_map.long()
            coords = coords[:, 1:].numpy()#[inverse_map]
            labels = labels#[inverse_map]
            primitive_labels = primitive_labels#[inverse_map]
            #
            # mask = (labels != 0) & (labels != -1)
            mask = (labels != -1)

            colors = 255 * (np.array(colormap)[primitive_labels])  # [:, 0:3]
            colors[~mask] = np.zeros(3)
            colors = colors.astype(np.uint8)

            sp_colors = 255 * (np.array(colormap)[neural_region])  # [:, 0:3]
            sp_colors[~mask] = np.zeros(3)
            sp_colors = sp_colors.astype(np.uint8)

            GT_colors = (scannet_colormap)[labels.long()]  # [:, 0:3]
            GT_colors[~mask] = np.zeros(3)
            GT_colors = GT_colors.astype(np.uint8)

            save_path = 'ScanNet/supsp_100primitive/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            test_name = save_path + cloud_name + '.ply'
            test_name_sp = save_path + cloud_name + 'SP.ply'
            test_name_GT = save_path + cloud_name + 'SemanticGT.ply'
            write_ply(test_name, [coords[mask], colors[mask]], ['x', 'y', 'z', 'red', 'green', 'blue'])
            write_ply(test_name_sp, [coords[mask], sp_colors[mask]], ['x', 'y', 'z', 'red', 'green', 'blue'])
            write_ply(test_name_GT, [coords[mask], GT_colors[mask]], ['x', 'y', 'z', 'red', 'green', 'blue'])
            print('completed scene: {}, used time: {:.2f}s'.format(cloud_name, time.time() - time_start))



def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = torch.tensor(centroids, requires_grad=False).cuda()
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
    hist /= hist.sum()

    return hist
