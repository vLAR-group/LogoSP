import torch
from glob import glob
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import open3d as o3d
from lib.aug_tools import rota_coords, scale_coords, trans_coords, elastic_coords
from lib.helper_ply import read_ply as read_ply
import os, pickle

class S3DISvis(Dataset):
    def __init__(self, args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']):
        self.args = args
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.name = []
        self.file = []

        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.replace(self.args.data_path, '')
            if plyname[0:6] in areas:
                name = file.replace(self.args.data_path, '')
                self.name.append(name[:-4])
                self.file.append(file)

    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        # feats = np.concatenate((colors, norm_coords), axis=-1)
        feats = colors
        return norm_coords, feats

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats[unique_map], labels[unique_map], unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        full_coords, full_labels, full_colors = coords.copy(), labels.copy(), colors.copy()
        coords -= coords.mean(0)

        coords, colors, _, unique_map, inverse_map = self.voxelize(coords, colors, labels)
        voxel_coords = coords.astype(np.float32)
        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)

        labels[labels == self.args.ignore_label] = -1
        full_labels[full_labels == self.args.ignore_label] = -1
        region[labels == -1] = -1
        region = region[unique_map]
        voxel_labels = labels[unique_map]

        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)
        region[region != -1] = valid_region

        coords, feats = self.augment_coords_to_feats(coords, colors/255-0.5)
        return voxel_coords, feats, inverse_map, voxel_labels, index, region, full_coords, full_labels, full_colors


class cfl_collate_fn_vis:
    def __call__(self, list_data):
        voxel_coords, feats, inverse_map, voxel_labels, index, region, full_coords, full_labels, full_colors = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch = [], [], [], []
        region_batch = []
        full_coors_batch, full_colors_batch, full_labels_batch = [], [], []
        for batch_id, _ in enumerate(voxel_coords):
            num_points = voxel_coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(voxel_coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(inverse_map[batch_id])
            labels_batch.append(torch.from_numpy(voxel_labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])

            full_coors_batch.append(torch.from_numpy(full_coords[batch_id]))
            full_colors_batch.append(torch.from_numpy(full_colors[batch_id]))
            full_labels_batch.append(torch.from_numpy(full_labels[batch_id]))
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)

        full_coors_batch = torch.cat(full_coors_batch, 0).float()
        full_colors_batch = torch.cat(full_colors_batch, 0).float()
        full_labels_batch = torch.cat(full_labels_batch, 0).int()

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch, full_coors_batch, full_colors_batch, full_labels_batch



class cfl_collate_fn:
    def __call__(self, list_data):
        coords, feats, labels, inverse_map, pseudo, region, index = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch, inverse_batch, pseudo_batch = [], [], [], [], []
        region_batch = []
        accm_num = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            inverse_batch.append(inverse_map[batch_id])
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            region_batch.append(torch.from_numpy(region[batch_id])[:,None])
            accm_num += coords[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()#.int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        region_batch = torch.cat(region_batch, 0)
        return coords_batch, feats_batch, labels_batch, inverse_batch, pseudo_batch, region_batch, index


class S3DIStrain(Dataset):
    def __init__(self, args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']):
        self.args = args
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.name = []
        self.mode = 'train'
        self.clip_bound = 4 # 4m
        self.file = []

        ''' Reading Data'''
        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.replace(self.args.data_path, '')
            if plyname[0:6] in areas:
                name = file.replace(self.args.data_path, '')
                self.name.append(name[:-4])
                self.file.append(file)

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50) ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound = (0.9, 1.1))
        self.elastic_coords = elastic_coords(voxel_size=self.args.voxel_size)


    def augs(self, coords, feats):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        # scale = 1 / self.args.voxel_size
        # coords = self.elastic_coords(coords, 6 * scale // 50, 40 * scale / 50)
        # coords = self.elastic_coords(coords, 20 * scale // 50, 160 * scale / 50)
        return coords, feats


    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        # feats = np.concatenate((colors, norm_coords), axis=-1)
        feats = colors
        return norm_coords, feats


    def clip(self, coords, center=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound

        if isinstance(self.clip_bound, (int, float)):
            if bound_size.max() < self.clip_bound:
                return None
            else:
                clip_inds = ((coords[:, 0] >= (-lim + center[0])) & (coords[:, 0] < (lim + center[0])) & \
                             (coords[:, 1] >= (-lim + center[1])) & (coords[:, 1] < (lim + center[1])) & \
                             (coords[:, 2] >= (-lim + center[2])) & (coords[:, 2] < (lim + center[2])))
                return clip_inds

    def voxelize(self, coords, feats, labels):
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        clip_inds = self.clip(coords)
        if clip_inds is not None:
            coords, feats = coords[clip_inds], feats[clip_inds]
            if labels is not None:
                labels = labels[clip_inds]

        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats[unique_map], labels[unique_map], unique_map, clip_inds, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, colors, labels, unique_map, clip_inds, inverse_map = self.voxelize(coords, colors, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + self.name[index] + '_superpoint.npy'
        region = np.load(region_file)

        '''Clip if Scene includes much Points'''
        if clip_inds is not None:
            region = region[clip_inds]
        region = region[unique_map]

        if self.mode == 'train':
            coords, colors = self.augs(coords, colors)

        coords, feats = self.augment_coords_to_feats(coords, colors/255-0.5)
        labels[labels == self.args.ignore_label] = -1

        '''mode must be cluster or train'''
        if self.mode == 'cluster':

            region[labels==-1] = -1

            for q in np.unique(region):
                mask = q == region
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region[mask] = -1

            valid_region = region[region != -1]
            unique_vals = np.unique(valid_region)
            unique_vals.sort()
            valid_region = np.searchsorted(unique_vals, valid_region)
            #
            region[region != -1] = valid_region

            pseudo = -np.ones_like(labels).astype(np.int32)
            labels[region==-1] = -1
        else:
            scene_name = self.name[index]
            file_path = self.args.save_path + '/'+self.args.pseudo_label_path + '/' + scene_name + '.npy'
            pseudo = np.array(np.load(file_path), dtype=np.int32)

        return coords, feats, labels, inverse_map, pseudo, region, index


class S3DIStest(Dataset):
    def __init__(self, args, areas=['Area_5']):
        self.args = args
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.name = []
        self.file = []

        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.replace(self.args.data_path, '')
            if plyname[0:6] in areas:
                name = file.replace(self.args.data_path, '')
                self.name.append(name[:-4])
                self.file.append(file)

    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        # feats = np.concatenate((colors, norm_coords), axis=-1)
        feats = colors
        return norm_coords, feats

    def voxelize(self, coords, feats, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats[unique_map], labels[unique_map], unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, colors, _, unique_map, inverse_map = self.voxelize(coords, colors, labels)
        coords = coords.astype(np.float32)
        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)

        labels[labels == self.args.ignore_label] = -1
        region[labels == -1] = -1
        region = region[unique_map]

        for q in np.unique(region):
            mask = q == region
            if mask.sum() < self.args.drop_threshold and q != -1:
                region[mask] = -1
        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)
        region[region != -1] = valid_region

        coords, feats = self.augment_coords_to_feats(coords, colors/255-0.5)
        return coords, feats, inverse_map, np.ascontiguousarray(labels), index, region



class cfl_collate_fn_test:
    def __call__(self, list_data):
        coords, feats, inverse_map, labels, index, region = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, labels_batch = [], [], [], []
        region_batch = []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(inverse_map[batch_id])
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:,None])
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch


class S3DISdistill(Dataset):
    def __init__(self, args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']):
        self.args = args
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.name = []
        self.mode = 'distill'
        self.clip_bound = 4 # 4m
        self.file = []
        self.dino_file = []

        ''' Reading Data'''
        folders = sorted(glob(self.args.data_path + '/*.ply'))
        for _, file in enumerate(folders):
            plyname = file.replace(self.args.data_path, '')
            dino_file = os.path.join(self.args.feats_path, plyname[0:-4]+'.pickle')
            if plyname[0:6] in areas:
                name = file.replace(self.args.data_path, '')
                if name[0:-4] not in ['Area_2_storage_8', 'Area_3_hallway_5', 'Area_3_storage_2', 'Area_4_hallway_5', 'Area_4_hallway_6'] and 'auditorium' not in name:
                    self.name.append(name[:-4])
                    self.file.append(file)
                    self.dino_file.append(dino_file)

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50) ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound = (0.9, 1.1))
        self.elastic_coords = elastic_coords(voxel_size=self.args.voxel_size)

    def augs(self, coords, feats):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        # scale = 1 / self.args.voxel_size
        # coords = self.elastic_coords(coords, 6 * scale // 50, 40 * scale / 50)
        # coords = self.elastic_coords(coords, 20 * scale // 50, 160 * scale / 50)
        return coords, feats


    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        # feats = np.concatenate((colors, norm_coords), axis=-1)
        feats = colors
        return norm_coords, feats


    def clip(self, coords, center=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound

        if isinstance(self.clip_bound, (int, float)):
            if bound_size.max() < self.clip_bound:
                return None
            else:
                clip_inds = ((coords[:, 0] >= (-lim + center[0])) & (coords[:, 0] < (lim + center[0])) & \
                             (coords[:, 1] >= (-lim + center[1])) & (coords[:, 1] < (lim + center[1])) & \
                             (coords[:, 2] >= (-lim + center[2])) & (coords[:, 2] < (lim + center[2])))
                return clip_inds

    def voxelize(self, coords, feats, labels):
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        clip_inds = self.clip(coords)
        if clip_inds is not None:
            coords, feats = coords[clip_inds], feats[clip_inds]
            if labels is not None:
                labels = labels[clip_inds]

        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats[unique_map], labels[unique_map], unique_map, clip_inds, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T, data['class']
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        ## loading DINO feats
        with open(self.dino_file[index], 'rb') as f:
            dino_context = pickle.load(f)
        dino_feats = torch.from_numpy(dino_context[0].astype(np.float32))
        dino_indicator = torch.from_numpy(dino_context[1].astype(np.float32))

        coords, colors, labels, unique_map, clip_inds, inverse_map = self.voxelize(coords, colors, labels)
        coords = coords.astype(np.float32)

        '''Clip if Scene includes much Points'''
        coords, colors = self.augs(coords, colors)

        coords, feats = self.augment_coords_to_feats(coords, colors / 255 - 0.5)
        labels[labels == self.args.ignore_label] = -1

        return coords, feats, labels, inverse_map, index, self.name[index], dino_feats, dino_indicator


class cfl_collate_fn_distill:
    def __call__(self, list_data):
        coords, feats, labels, inverse_map, index, scenenames, spfeats, indicator = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch, inverse_batch, spfeats_batch, indicator_batch = [], [], [], [], [], []

        accm_num = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id].copy()).int())
            inverse_batch.append(inverse_map[batch_id])
            spfeats_batch.append(spfeats[batch_id])
            indicator_batch.append(indicator[batch_id])
            accm_num += coords[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()#.int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        spfeats_batch = torch.cat(spfeats_batch, 0)
        indicator_batch = torch.cat(indicator_batch, 0)

        return coords_batch, feats_batch, labels_batch, inverse_batch, index, scenenames, spfeats_batch, indicator_batch