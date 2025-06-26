import open3d as o3d
import numpy as np
import os
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere

scene_names = 'example.ply'
out_foler = os.path.join('objs')

for scene in scene_names:
    data = read_ply(scene)
    print(len(data))
    # data = data[np.random.choice(len(data), len(data)//5, replace=False)]
    points = data[:, 0:3]
    points = points - points.mean(0, keepdims=True)
    colors = data[:, 3:6]

    # mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.02/0.03 radius for ScanNet
    mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.05, resolution=3, with_background=False, default_color=colors)### 0.02/0.03 radius for ScanNet, 0.02/0.01 for s3dis, maybe 0.1 for semkitti?
    # o3d.visualization.draw_geometries([mesh])

    os.makedirs(out_foler, exist_ok=True)
    save_file = os.path.join(out_foler, scene.split('/')[-1][0:-4] + '.obj')
    o3d.io.write_triangle_mesh(save_file, mesh)