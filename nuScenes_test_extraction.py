from nuscenes import NuScenes
import os
import numpy as np
import argparse

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='nuScene online testing preprocessing')
    parser.add_argument('--input_dir', type=str, default='/home/zihui/HDD/v1.0-test_meta')
    parser.add_argument('--output_dir', type=str, default='./data/nuScenes/nuScenes_3d/test')
    return parser.parse_args()

args = parse_args()

nusc = NuScenes(version='v1.0-test', dataroot=args.input_dir, verbose=True)

sample_data_to_filename = {}
for sample in nusc.sample:
    lidar_top_token = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', lidar_top_token)
    filename = sample_data['filename']
    sample_data_to_filename[lidar_top_token] = filename


def load_pc_from_file(pc_f):
    # nuScenes lidar is 5 digits one line (last one the ring index)
    return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 5])


os.makedirs(args.output_dir, exist_ok=True)
for sample_token, file_path in sample_data_to_filename.items():
    input_file_path = os.path.join(args.input_dir, file_path)
    output_file_path = os.path.join(args.output_dir, f"{sample_token}.npy")

    ##open pointcloud and prepare
    point_cloud = load_pc_from_file(input_file_path)[:,:3]
    np.save(output_file_path, point_cloud)
    print(f"Saved point cloud to {output_file_path}")

print('prepare done!')









