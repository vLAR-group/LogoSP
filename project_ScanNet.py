import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
import math
from imageio import imread
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from lib.helper_ply import read_ply, write_ply
from torch_scatter import scatter_mean, scatter_max, scatter_min
import MinkowskiEngine as ME
from Point_feature_dataset_ScanNet import PointFeature, cfl_collate_fn
from torch.utils.data import DataLoader
import time
import pickle
from projector import PointCloudToImageMapper

SCANNET_DATA = './data/ScanNet/scannet_2d'
SCANNET_FRAME_ROOT = './data/ScanNet/scannet_2d/{}/{}/'
SCANNET_3d_DATA = './data/ScanNet/scannet_3d'
SCANNET_FRAME_PATH = './data/ScanNet/scannet_2d/{}/'
PointCloud_Data = './data/ScanNet/processed'
voxel_size = 0.05
scatter_fn = scatter_mean


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0][0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1][1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0][2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1][2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


img_dim = (320, 240)
original_img_dim = (640, 480)
intrinsics= make_intrinsic(fx=577.870605, fy=577.870605, mx=319.5, my=239.5)
intrinsics = adjust_intrinsic(intrinsics, original_img_dim, img_dim)

depth_scale = 1000.0
visibility_threshold = 0.25  # threshold for the visibility check
cut_num_pixel_boundary = 10
# calculate image pixel-3D points correspondances
point2img_mapper = PointCloudToImageMapper(image_dim=img_dim, intrinsics=intrinsics, visibility_threshold=visibility_threshold, cut_bound=cut_num_pixel_boundary)


def to_tensor(arr):
    return torch.Tensor(arr).cuda()

def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    return image


def load_image(file, image_dims):
    # print(file)
    image = imread(file)
    # preprocess
    image = resize_crop_image(image, image_dims)
    if len(image.shape) == 3:  # color image
        image = np.transpose(image, [2, 0, 1])  # move feature to front
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(torch.Tensor(image.astype(np.float32) / 255.0))
    elif len(image.shape) == 2:  # label image
        pass
    else:
        raise
    return image


def load_pose(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    return np.asarray(lines).astype(np.float32)


def load_depth(file, image_dims):
    depth_image = imread(file)
    # preprocess
    depth_image = resize_crop_image(depth_image, image_dims)
    depth_image = depth_image.astype(np.float32) / depth_scale
    return depth_image


def get_scene_data(scene_list):
    scene_data = {}
    for scene_id in scene_list:
        # load the original vertices, not the axis-aligned ones
        scene_data[scene_id] = np.load(os.path.join(SCANNET_DATA, scene_id) + "_vert.npy")[:, :3]
    return scene_data


def voxelize(coords):
    scale = 1 / voxel_size
    coords = np.floor(coords * scale)
    coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), return_index=True, return_inverse=True)
    return coords, unique_map, inverse_map


if __name__ == "__main__":

    feat_dim = 384
    feat_save_path = './data/ScanNet/DINOv2_feats_s14up4_voxel_' + str(voxel_size)

    scene_list = sorted(os.listdir("./data/ScanNet/scannet_2d/"))
    # model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').cuda().eval()
    model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14').cuda().eval()

    model.fc = torch.nn.Identity()
    #
    def get_feature_hook(module, input, output):
        global features
        features = output
    hook = model.blocks[-1].register_forward_hook(get_feature_hook)

    print("projecting multiview features to point cloud...")
    for scene_id in scene_list:
        print("processing {}...".format(scene_id))
        try:
            scene = torch.load(os.path.join(SCANNET_3d_DATA, "train", scene_id) + "_vh_clean_2.pth")[0]
        except:
            scene = torch.load(os.path.join(SCANNET_3d_DATA, "val", scene_id) + "_vh_clean_2.pth")[0]

        frame_list = sorted([int(x.split(".")[0]) for x in os.listdir(SCANNET_FRAME_ROOT.format(scene_id, "color/"))])
        scene_depths = np.zeros((len(frame_list), 240, 320))
        scene_poses = np.zeros((len(frame_list), 4, 4))

        start_time = time.time()
        for i, frame_id in enumerate(frame_list):
            scene_depths[i] = load_depth(SCANNET_FRAME_PATH.format(scene_id) + "/depth" + "/{}.png".format(frame_id), [320, 240])
            scene_poses[i] = load_pose(SCANNET_FRAME_PATH.format(scene_id) + "/pose" + "/{}.txt".format(frame_id))
        load_end_time = time.time()
        print(f" loading processing time: {load_end_time - start_time:.2f} seconds")

        # process feature
        point_feature_dataset = PointFeature(scene_id)
        data_loader = DataLoader(point_feature_dataset, batch_size=4, shuffle=False, collate_fn=cfl_collate_fn(), num_workers=8, pin_memory=True)
        frame_start_time = time.time()
        image_embeddings_list = []
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                one_batch_time = time.time()
                color, frame_id, frame_lists, index = data
                # Upsample by a factor of 4
                # color = F.interpolate(color, scale_factor=2, mode='bilinear', align_corners=False)
                ##
                color = color.cuda()
                features = None
                _ = model(color)
                image_embedding = features
                image_embedding = model.norm(image_embedding)
                batch_size = image_embedding.size(0)
                image_embedding = image_embedding[:, 1:, :].reshape(batch_size, 68, 91, feat_dim).permute(0, 3, 1, 2)
                # image_embedding = image_embedding[:, 1:, :].reshape(batch_size, 15 * 2, 20 * 2, feat_dim).permute(0, 3, 1, 2)
                image_embeddings_list.append(image_embedding)
                print(f" one batch processing time: {time.time() - one_batch_time:.2f} seconds")
        ##
        image_embeddings = torch.cat(image_embeddings_list, dim=0) ### image features from all views
        frame_end_time = time.time()
        print(f" image processing time: {frame_end_time - frame_start_time:.2f} seconds")

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        project_time = time.time()

        # project
        n_points = scene.shape[0]
        counter = torch.zeros((n_points, 1)).cuda()
        sum_features = torch.zeros((n_points, feat_dim)).cuda()
        for i in range(len(image_embeddings)):

            feat = image_embeddings[i].cuda()
            feat_2d = F.interpolate(feat[None, ...], (240, 320), mode='bicubic', align_corners=False).squeeze(0)  ## [C, H, W]
            pose, loc_in, depth = scene_poses[i], scene, scene_depths[i]
            # calculate the 3d-2d mapping based on the depth
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, loc_in, depth)
            if mapping[:, 3].sum() == 0:  # no points corresponds to this image, skip
                continue

            mapping = torch.from_numpy(mapping).cuda()
            mask = mapping[:, 3]

            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

            counter[mask != 0] += 1
            sum_features[mask != 0] += feat_2d_3d[mask != 0]

        counter[counter == 0] = 1e-5
        point_features = sum_features / counter

        # save
        os.makedirs(feat_save_path, exist_ok=True)
        # # 储存体素特征来替换点的特征，首先体素化
        data = read_ply(os.path.join(PointCloud_Data, scene_id) + ".ply")
        coords, colors, labels = np.vstack((data['x'], data['y'], data['z'])).T, np.vstack(
            (data['red'], data['green'], data['blue'])).T, data['class']
        coords = coords.astype(np.float32)
        coords -= coords.mean(0)
        point_features = point_features.cpu().numpy().astype(np.float32)

        ### change density
        # select_idx = np.random.choice(len(coords), int(len(coords)*0.125), replace=False)
        # coords, point_features = coords[select_idx], point_features[select_idx]
        # #
        _, unique_map, inv_map = voxelize(coords)
        voxel_features = point_features[unique_map]
        # voxel_coords = (np.vstack((data['x'], data['y'], data['z'])).T)[unique_map].astype(np.float32)
        voxel_coords = coords[unique_map].astype(np.float32)
        with open(os.path.join(feat_save_path, scene_id+'.pickle'), 'wb') as f:
            pickle.dump(voxel_features, f)

        # with open(os.path.join(feat_save_path, scene_id+'.pickle'), 'wb') as f:
        #     pickle.dump({'idx': select_idx, 'feats': voxel_features}, f)

        ##
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(voxel_features)
        min_vals = pca_features.min(axis=0)
        max_vals = pca_features.max(axis=0)
        voxel_color = 255 * (pca_features - min_vals) / (max_vals - min_vals)
        voxel_color = voxel_color.astype(np.uint8)
        ##
        write_ply(os.path.join(feat_save_path, scene_id+'.ply'), [voxel_coords, voxel_color], ['x', 'y', 'z', 'red', 'green', 'blue'])

        # with open(os.path.join(feat_save_path, scene_id+'.pickle'), 'wb') as f:
        #     pickle.dump(point_features, f)
        # pca = PCA(n_components=3)
        # pca_features = pca.fit_transform(point_features)
        # min_vals = pca_features.min(axis=0)
        # max_vals = pca_features.max(axis=0)
        # color = 255 * (pca_features - min_vals) / (max_vals - min_vals)
        # color = color.astype(np.uint8)
        # #
        # write_ply(os.path.join(feat_save_path, scene_id+'.ply'), [coords, color], ['x', 'y', 'z', 'red', 'green', 'blue'])
        print(f"project  processing time: {time.time() - project_time:.2f} seconds")

    print("done!")
