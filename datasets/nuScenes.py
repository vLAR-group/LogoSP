import torch
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import pickle
import os
from os.path import join
from lib.aug_tools import rota_coords, scale_coords, trans_coords
from tqdm import tqdm

class nuScenesvis(Dataset):
    def __init__(self, args):
        self.args = args
        self.label_to_names = {0: 'barrier',
                               1: 'bicycle',
                               2: 'bus',
                               3: 'car',
                               4: 'construction vehicle',
                               5: 'motorcycle',
                               6: 'person',
                               7: 'traffic cone',
                               8: 'trailer',
                               9: 'truck',
                               10: 'drivable surface',
                               11: 'other flat',
                               12: 'sidewalk',
                               13: 'terrain',
                               14: 'manmade',
                               15: 'vegetation',
                               -1: 'unlabeled'}
        self.name = []
        self.mode = 'val'
        self.file = []

        scene_list = np.sort(os.listdir(args.val_input_path))
        for scene_id in scene_list:
            scene_path = join(args.val_input_path, scene_id)
            self.file.append(scene_path)
            self.name.append(scene_id[0:-4])

    def augment_coords_to_feats(self, coords):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords

    def voxelize(self, coords, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), features=None, labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords, labels, unique_map, inverse_map

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        with open(file, 'rb') as f:
            data = pickle.load(f)
        coords = data['coords']
        labels = data['labels']
        ##
        label_mask = labels == 255
        labels[label_mask] = -1
        coords = coords.astype(np.float32)
        full_coords, full_labels = coords.copy(), labels.copy()
        coords -= coords.mean(0)

        coords, _, unique_map, inverse_map = self.voxelize(coords, labels)
        voxel_labels = labels[unique_map]

        coords = self.augment_coords_to_feats(coords)
        return coords, inverse_map, np.ascontiguousarray(voxel_labels), index, full_coords, full_labels


class cfl_collate_fn_vis:
    def __call__(self, list_data):
        voxel_coords, inverse_map, voxel_labels, index, full_coords, full_labels = list(zip(*list_data))
        coords_batch, inverse_batch, labels_batch = [], [], []
        full_coors_batch, full_labels_batch = [], []
        for batch_id, _ in enumerate(voxel_coords):
            num_points = voxel_coords[batch_id].shape[0]
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(voxel_coords[batch_id]).int()), 1))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(voxel_labels[batch_id]).int())

            full_coors_batch.append(torch.from_numpy(full_coords[batch_id]))
            full_labels_batch.append(torch.from_numpy(full_labels[batch_id]))

        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()

        full_coors_batch = torch.cat(full_coors_batch, 0).float()
        full_labels_batch = torch.cat(full_labels_batch, 0).int()

        return coords_batch, inverse_batch, labels_batch, index, full_coors_batch, full_labels_batch


class cfl_collate_fn:
    def __call__(self, list_data):
        coords, labels, inverse_map, pseudo, region, index, scene_name = list(zip(*list_data))
        coords_batch, labels_batch, inverse_batch, pseudo_batch = [], [], [], []
        region_batch = []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()  # .int()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        region_batch = torch.cat(region_batch, 0)
        return coords_batch, labels_batch, inverse_batch, pseudo_batch, region_batch, index, scene_name


class nuScenestrain(Dataset):
    def __init__(self, args, scene_idx, split='train'):
        self.args = args
        self.label_to_names = {0: 'barrier',
                               1: 'bicycle',
                               2: 'bus',
                               3: 'car',
                               4: 'construction vehicle',
                               5: 'motorcycle',
                               6: 'person',
                               7: 'traffic cone',
                               8: 'trailer',
                               9: 'truck',
                               10: 'drivable surface',
                               11: 'other flat',
                               12: 'sidewalk',
                               13: 'terrain',
                               14: 'manmade',
                               15: 'vegetation',
                               -1: 'unlabeled'}
        self.mode = 'train'
        self.split = split
        self.train_path_list = []

        scene_list = np.sort(os.listdir(args.data_path))
        for scene_id in scene_list:
            scene_path = join(args.data_path, scene_id)
            self.train_path_list.append(scene_path)

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        self.random_select_sample(scene_idx)

    def random_select_sample(self, scene_idx):
        self.name = []
        self.file_selected = []
        for i in scene_idx:
            self.file_selected.append(self.train_path_list[i])
            self.name.append(self.train_path_list[i][0:-4].replace(self.args.data_path, ''))

    def augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords

    def augment_coords_to_feats(self, coords, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, labels

    def voxelize(self, coords, labels):
        # nuscenes feats =None
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), features=None,
                        labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords, labels, unique_map, inverse_map

    def __len__(self):
        return len(self.file_selected)

    def __getitem__(self, index):
        file = self.file_selected[index]
        with open(file, 'rb') as f:
            data = pickle.load(f)
        coords = data['coords']
        labels = data['labels']
        label_mask = labels == 255
        labels[label_mask] = -1
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, labels, unique_map, inverse_map = self.voxelize(coords, labels)
        # coords = coords.numpy().astype(np.float32)

        region_file = self.args.sp_path + '/' + self.name[index] + '_superpoint.npy'
        region = np.load(region_file)
        region = region[unique_map]

        mask = np.sqrt(((coords*self.args.voxel_size)**2).sum(-1))< 50
        coords, region, labels = coords[mask], region[mask], labels[mask]

        if self.mode == 'train':
            coords = self.augs(coords)
        coords, labels = self.augment_coords_to_feats(coords, labels)

        '''mode must be cluster or train'''
        if self.mode == 'cluster':

            region[labels == -1] = -1
            original_region = region.copy()
            for q in np.unique(region):
                mask = q == region
                if mask.sum() < self.args.drop_threshold and q != -1:
                    region[mask] = -1
                    if np.all(region == -1):
                        region = original_region
                        break

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
        return coords, labels, inverse_map, pseudo, region, index, self.name[index]


class nuScenesval(Dataset):
    def __init__(self, args):
        self.args = args
        self.label_to_names = {0: 'barrier',
                               1: 'bicycle',
                               2: 'bus',
                               3: 'car',
                               4: 'construction vehicle',
                               5: 'motorcycle',
                               6: 'person',
                               7: 'traffic cone',
                               8: 'trailer',
                               9: 'truck',
                               10: 'drivable surface',
                               11: 'other flat',
                               12: 'sidewalk',
                               13: 'terrain',
                               14: 'manmade',
                               15: 'vegetation',
                               -1: 'unlabeled'}
        self.name = []
        self.mode = 'val'
        self.file = []

        scene_list = np.sort(os.listdir(args.val_input_path))
        for scene_id in scene_list:
            scene_path = join(args.val_input_path, scene_id)
            self.file.append(scene_path)

    def augment_coords_to_feats(self, coords):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords

    def voxelize(self, coords, labels):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), features=None,
                         labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords, labels, unique_map, inverse_map

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        with open(file, 'rb') as f:
            data = pickle.load(f)
        coords = data['coords']
        labels = data['labels']
        ##
        label_mask = labels == 255
        labels[label_mask] = -1
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, _, unique_map, inverse_map = self.voxelize(coords, labels)
        # coords = coords.numpy().astype(np.float32)

        coords = self.augment_coords_to_feats(coords)
        return coords, np.ascontiguousarray(labels), inverse_map, index


class cfl_collate_fn_val:
    def __call__(self, list_data):
        coords, labels, inverse_map, index = list(zip(*list_data))
        coords_batch, inverse_batch, labels_batch = [], [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()

        return coords_batch, inverse_batch, labels_batch, index


class nuScenesdistill(Dataset):
    def __init__(self, args, scene_idx, split='train'):
        self.args = args
        self.label_to_names = {0: 'barrier',
                               1: 'bicycle',
                               2: 'bus',
                               3: 'car',
                               4: 'construction vehicle',
                               5: 'motorcycle',
                               6: 'person',
                               7: 'traffic cone',
                               8: 'trailer',
                               9: 'truck',
                               10: 'drivable surface',
                               11: 'other flat',
                               12: 'sidewalk',
                               13: 'terrain',
                               14: 'manmade',
                               15: 'vegetation',
                               -1: 'unlabeled'}
        self.split = split
        self.train_path_list = []
        self.feats_list = []
        self.feats_datas = []
        self.points_datas = []
        self.mode = 'distill'

        scene_list = np.sort(os.listdir(args.data_path))
        for scene_id in scene_list:
            scene_path = join(args.data_path, scene_id)
            self.train_path_list.append(scene_path)
            ##
            feat_path = join(args.feats_path, scene_id[:-4] + '.pt')
            self.feats_list.append(feat_path)

        '''Initial Augmentations'''
        self.trans_coords = trans_coords(shift_ratio=50)  ### 50%
        self.rota_coords = rota_coords(rotation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))
        self.random_select_sample(scene_idx)
        self.preload_data()

    def random_select_sample(self, scene_idx):
        self.name = []
        self.file_selected = []
        self.feats_selected = []
        for i in scene_idx:
            self.file_selected.append(self.train_path_list[i])
            self.feats_selected.append(self.feats_list[i])
            self.name.append(self.train_path_list[i][0:-4].replace(self.args.data_path, ''))

    def preload_data(self):
        for featpath, filepath in tqdm(zip(self.feats_selected, self.file_selected), desc='Pre Load Datas(1500)'):
            point_feat = torch.load(featpath)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.feats_datas.append(point_feat)
            self.points_datas.append(data)

    def augs(self, coords):
        coords = self.rota_coords(coords)
        coords = self.trans_coords(coords)
        coords = self.scale_coords(coords)
        return coords

    def augment_coords_to_feats(self, coords, labels=None):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords, labels

    def voxelize(self, coords, labels):
        # nuscenes feats =None
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), features=None,
                                        labels=labels, ignore_label=-1, return_index=True, return_inverse=True)
        return coords, labels, unique_map, inverse_map

    def __len__(self):
        return len(self.file_selected)

    def __getitem__(self, index):
        data = self.points_datas[index]
        coords = data['coords']
        labels = data['labels']

        label_mask = labels == 255
        labels[label_mask] = -1
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)

        coords, labels, unique_map, inverse_map = self.voxelize(coords, labels)
        # coords = coords.numpy().astype(np.float32)

        feats = self.feats_datas[index]
        point_feats = feats['feat']
        mask_chunk = feats['mask_full']
        mask_chunk = torch.from_numpy(mask_chunk)

        '''# calculate the corresponding point features after voxelization '''
        mask = mask_chunk[unique_map]  ## voxelized visible mask for entire point cloud
        mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
        index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
        index1[mask_ind] = mask_ind
        index1 = index1[unique_map]
        chunk_ind = index1[index1 != -1]
        index2 = torch.zeros(mask_chunk.shape[0])
        index2[mask_ind] = 1
        index3 = torch.cumsum(index2, dim=0, dtype=int)
        # get the indices of corresponding masked point features after voxelization
        indices = index3[chunk_ind] - 1
        point_feats = point_feats[indices]  ##点特征的体素化

        coords = self.augs(coords)
        coords, labels = self.augment_coords_to_feats(coords, labels)

        return coords, labels, inverse_map, point_feats, mask, index


class cfl_collate_fn_distill:
    def __call__(self, list_data):
        coords, labels, inverse_map, point_feats, mask, index = list(zip(*list_data))
        coords_batch, labels_batch, inverse_batch, point_feats_batch, mask_batch = [], [], [], [], []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            point_feats_batch.append(point_feats[batch_id])
            mask_batch.append(mask[batch_id])

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()  # .int()
        labels_batch = torch.cat(labels_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        point_feats_batch = torch.cat(point_feats_batch, 0).float()
        mask_batch = torch.cat(mask_batch, 0).bool()

        return coords_batch, labels_batch, inverse_batch, point_feats_batch, mask_batch, index



class nuScenestest(Dataset):
    def __init__(self, args):
        self.args = args
        self.label_to_names = {0: 'barrier',
                               1: 'bicycle',
                               2: 'bus',
                               3: 'car',
                               4: 'construction vehicle',
                               5: 'motorcycle',
                               6: 'person',
                               7: 'traffic cone',
                               8: 'trailer',
                               9: 'truck',
                               10: 'drivable surface',
                               11: 'other flat',
                               12: 'sidewalk',
                               13: 'terrain',
                               14: 'manmade',
                               15: 'vegetation',
                               -1: 'unlabeled'}

        self.mode = 'test'
        self.file = []

        self.name = []
        scene_list = np.sort(os.listdir(args.test_input_path))
        for scene_id in scene_list:
            scene_path = join(args.test_input_path, scene_id)
            name = scene_path.replace(args.test_input_path, '')
            self.name.append(name[0:-4])
            self.file.append(scene_path)


    def augment_coords_to_feats(self, coords):
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = (coords - coords_center)
        return norm_coords

    def voxelize(self, coords):
        scale = 1 / self.args.voxel_size
        coords = np.floor(coords * scale)
        coords,unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords),return_index=True, return_inverse=True)
        return coords, unique_map, inverse_map


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]
        coords = np.load(file)
        ##
        scene_name = self.name[index]
        original_coords =coords.copy()
        coords -= coords.mean(0)
        coords,unique_map, inverse_map = self.voxelize(coords)
        coords = coords.numpy().astype(np.float32)
        coords= self.augment_coords_to_feats(coords)

        return coords,inverse_map.numpy(), index, original_coords, scene_name


class cfl_collate_fn_test:

    def __call__(self, list_data):
        coords, inverse_map,index,original_coords,scene_name= list(zip(*list_data))
        coords_batch, inverse_batch,original_coords_batch,scene_name_batch= [], [], [],[]
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            original_coords_batch.append(torch.from_numpy(original_coords[batch_id]))
            scene_name_batch.append(scene_name[batch_id])
        #
        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        original_coords_batch =torch.cat(original_coords_batch, 0).float()
        return coords_batch,inverse_batch, index,original_coords_batch,scene_name_batch