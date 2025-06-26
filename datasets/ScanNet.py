import torch
import numpy as np
from lib.helper_ply import read_ply, write_ply
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import os
from lib.aug_tools import rota_coords, scale_coords, trans_coords, elastic_coords
import pickle
from sklearn.preprocessing import LabelEncoder

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


class Scannetvis(Dataset):
    def __init__(self, args):
        self.args = args
        self.path_file = '../data_prepare/ScanNet_splits/scannetv2_train.txt'
        self.label_to_names = {0: 'wall',
                               1: 'floor',
                               2: 'cabinet',
                               3: 'bed',
                               4: 'chair',
                               5: 'sofa',
                               6: 'table',
                               7: 'door',
                               8: 'window',
                               9: 'bookshelf',
                               10: 'picture',
                               11: 'counter',
                               12: 'desk',
                               13: 'curtain',
                               14: 'refridgerator',
                               15: 'shower curtain',
                               16: 'toilet',
                               17: 'sink',
                               18: 'bathtub',
                               19: 'otherfurniture'}
        self.name = []
        self.plypath = read_txt(self.path_file)
        self.file = []

        for plyname in self.plypath:
            file = os.path.join(self.args.data_path, plyname[0:12]+'.ply')
            self.name.append(plyname[0:12])
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

        voxel_coords, feats = self.augment_coords_to_feats(voxel_coords, colors / 255 - 0.5)
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



class Scannetval(Dataset):
    def __init__(self, args):
        self.args = args
        self.path_file = 'data_prepare/ScanNet_splits/scannetv2_val.txt'
        self.label_to_names = {0: 'wall',
                               1: 'floor',
                               2: 'cabinet',
                               3: 'bed',
                               4: 'chair',
                               5: 'sofa',
                               6: 'table',
                               7: 'door',
                               8: 'window',
                               9: 'bookshelf',
                               10: 'picture',
                               11: 'counter',
                               12: 'desk',
                               13: 'curtain',
                               14: 'refridgerator',
                               15: 'shower curtain',
                               16: 'toilet',
                               17: 'sink',
                               18: 'bathtub',
                               19: 'otherfurniture'}
        self.name = []
        self.plypath = read_txt(self.path_file)
        self.file = []

        for plyname in self.plypath:
            file = os.path.join(self.args.data_path, plyname[0:12]+'.ply')
            self.name.append(plyname[0:12])
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
        # region = labels

        labels[labels == self.args.ignore_label] = -1
        region[labels == -1] = -1
        region = region[unique_map]

        valid_region = region[region != -1]
        unique_vals = np.unique(valid_region)
        unique_vals.sort()
        valid_region = np.searchsorted(unique_vals, valid_region)

        region[region != -1] = valid_region

        coords, feats = self.augment_coords_to_feats(coords, colors / 255 - 0.5)
        return coords, feats, inverse_map, np.ascontiguousarray(labels), index, region


class cfl_collate_fn_val:
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
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)
        return coords_batch, feats_batch, inverse_batch, labels_batch, index, region_batch



class Scannettrain(Dataset):
    def __init__(self, args):
        self.args = args
        self.path_file = 'data_prepare/ScanNet_splits/scannetv2_train.txt'
        self.label_to_names = {0: 'wall',
                               1: 'floor',
                               2: 'cabinet',
                               3: 'bed',
                               4: 'chair',
                               5: 'sofa',
                               6: 'table',
                               7: 'door',
                               8: 'window',
                               9: 'bookshelf',
                               10: 'picture',
                               11: 'counter',
                               12: 'desk',
                               13: 'curtain',
                               14: 'refridgerator',
                               15: 'shower curtain',
                               16: 'toilet',
                               17: 'sink',
                               18: 'bathtub',
                               19: 'otherfurniture'}
        self.name = []
        self.mode = 'train'
        self.plypath = read_txt(self.path_file)
        self.file = []
        self.contin_label = LabelEncoder()

        for plyname in self.plypath:
            file = os.path.join(self.args.data_path, plyname[0:12]+'.ply')
            self.name.append(plyname[0:12])
            self.file.append(file)

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
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

        coords, colors, labels, unique_map, inverse_map = self.voxelize(coords, colors, labels)
        coords = coords.astype(np.float32)

        region_file = self.args.sp_path + '/' +self.name[index] + '_superpoint.npy'
        region = np.load(region_file)[unique_map]

        # if self.mode == 'train':
        #     coords, colors = self.augs(coords, colors)

        coords, feats = self.augment_coords_to_feats(coords, colors / 255 - 0.5)
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
            region[region != -1] = valid_region
            pseudo = -np.ones_like(labels).astype(np.int32)
        else:
            scene_name = self.name[index]
            file_path = self.args.save_path + '/'+self.args.pseudo_label_path + '/' + scene_name + '.npy'
            pseudo = np.array(np.load(file_path), dtype=np.int32)

        return coords, feats, labels, inverse_map, pseudo, region, index


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



class Scannetdistill(Dataset):
    def __init__(self, args):
        self.args = args
        self.path_file = 'data_prepare/ScanNet_splits/scannetv2_train.txt'
        self.label_to_names = {0: 'wall',
                               1: 'floor',
                               2: 'cabinet',
                               3: 'bed',
                               4: 'chair',
                               5: 'sofa',
                               6: 'table',
                               7: 'door',
                               8: 'window',
                               9: 'bookshelf',
                               10: 'picture',
                               11: 'counter',
                               12: 'desk',
                               13: 'curtain',
                               14: 'refridgerator',
                               15: 'shower curtain',
                               16: 'toilet',
                               17: 'sink',
                               18: 'bathtub',
                               19: 'otherfurniture'}
        self.name = []
        self.mode = 'distill'
        self.plypath = read_txt(self.path_file)
        self.file = []
        self.dino_file = []

        for plyname in self.plypath:
            file = os.path.join(self.args.data_path, plyname[0:12]+'.ply')
            # dino_file = os.path.join(self.args.feats_path, plyname[0:12]+'.pth')
            dino_file = os.path.join(self.args.feats_path, plyname[0:12]+'.pickle')
            self.name.append(plyname[0:12])
            self.file.append(file)
            self.dino_file.append(dino_file)
            # with open(dino_file, 'rb') as f:
            #     dino_feats = pickle.load(f)
            # self.dino_file.append(torch.from_numpy(dino_feats))

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

    def augs(self, coords, feats):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords, feats

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

        # loading DINO feats
        with open(self.dino_file[index], 'rb') as f:
            dino_feats = pickle.load(f).astype(np.float32)
        dino_feats = torch.from_numpy(dino_feats)

        # ### change density
        # with open(self.dino_file[index], 'rb') as f:
        #     dict = pickle.load(f)
        # dino_feats = torch.from_numpy(dict['feats'].astype(np.float32))
        # select_idx = dict['idx']
        # coords, colors, labels = coords[select_idx], colors[select_idx], labels[select_idx]
        # ###

        coords, colors, _, unique_map, inverse_map = self.voxelize(coords, colors, labels)
        coords = coords.astype(np.float32)
        # dino_feats = dino_feats[unique_map]

        coords, colors = self.augs(coords, colors)

        coords, feats = self.augment_coords_to_feats(coords, colors / 255 - 0.5)
        labels[labels == self.args.ignore_label] = -1


        return coords, feats, labels, inverse_map, index, self.name[index], dino_feats


class cfl_collate_fn_distill:
    def __call__(self, list_data):
        coords, feats, labels, inverse_map, index, scenenames, spfeats = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch, inverse_batch, pseudo_batch, spfeats_batch, region_batch = [], [], [], [], [], [], []

        accm_num = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id].copy()).int())
            inverse_batch.append(inverse_map[batch_id])
            spfeats_batch.append(spfeats[batch_id])
            accm_num += coords[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()#.int()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        spfeats_batch = torch.cat(spfeats_batch, 0)
        return coords_batch, feats_batch, labels_batch, inverse_batch, index, scenenames, spfeats_batch



class Scannettest(Dataset):
    def __init__(self, args):
        self.args = args
        self.path_file = 'data_prepare/ScanNet_splits/scannetv2_test.txt'
        self.label_to_names = {0: 'wall',
                               1: 'floor',
                               2: 'cabinet',
                               3: 'bed',
                               4: 'chair',
                               5: 'sofa',
                               6: 'table',
                               7: 'door',
                               8: 'window',
                               9: 'bookshelf',
                               10: 'picture',
                               11: 'counter',
                               12: 'desk',
                               13: 'curtain',
                               14: 'refridgerator',
                               15: 'shower curtain',
                               16: 'toilet',
                               17: 'sink',
                               18: 'bathtub',
                               19: 'otherfurniture'}

        self.name = []
        self.plypath = read_txt(self.path_file)
        self.file = []

        for plyname in self.plypath:
            file = os.path.join(self.args.data_path, plyname[0:12]+'.ply')
            self.name.append(plyname[0:12])
            self.file.append(file)

    def augment_coords_to_feats(self, coords, colors):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        # feats = np.concatenate((colors, norm_coords), axis=-1)
        feats = colors
        return norm_coords, feats

    def voxelize(self, coords, feats):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feats[unique_map], unique_map, inverse_map

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        data = read_ply(self.file[index])
        coords, colors = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack((data['red'], data['green'], data['blue'])).T
        colors = colors.astype(np.float32)
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, colors, unique_map, inverse_map = self.voxelize(coords, colors)
        coords = coords.astype(np.float32)
        region_file = self.args.sp_path + '/' + self.name[index] + '_superpoint.npy'
        region = np.load(region_file)[unique_map]

        valid_region = region[region != -1]
        uni = np.unique(valid_region)
        num = len(uni)
        for p in range(num):
            if (valid_region == p).sum() == 0:
                if p == 0:
                    valid_region -= uni[0]
                else:
                    offset = uni[p] - uni[p - 1] - 1
                    valid_region[valid_region >= p] -= offset

        region[region != -1] = valid_region

        coords, feats = self.augment_coords_to_feats(coords, colors / 255 - 0.5)
        return coords, feats, inverse_map, index, region


class cfl_collate_fn_test:
    def __call__(self, list_data):
        coords, feats, inverse_map, index, region = list(zip(*list_data))
        coords_batch, feats_batch, inverse_batch, region_batch = [], [], [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            inverse_batch.append(inverse_map[batch_id])
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)

        return coords_batch, feats_batch, inverse_batch, index, region_batch