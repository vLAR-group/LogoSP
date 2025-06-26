import numpy as np
from scipy.linalg import expm, norm
import scipy
from scipy import ndimage, interpolate

def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class trans_coords:
    def __init__(self, shift_ratio):
        self.ratio = shift_ratio

    def __call__(self, coords):
        shift = (np.random.uniform(0, 1, 3) * self.ratio)
        # make z tans be 0
        # shift[-1] = 0
        ##
        return coords + shift


class rota_coords:
    def __init__(self, rotation_bound = ((-np.pi/32, np.pi/32), (-np.pi/32, np.pi/32), (-np.pi, np.pi))):
        self.rotation_bound = rotation_bound

    def __call__(self, coords):
        rot_mats = []
        for axis_ind, rot_bound in enumerate(self.rotation_bound):
            theta = 0
            axis = np.zeros(3)
            axis[axis_ind] = 1
            if rot_bound is not None:
                theta = np.random.uniform(*rot_bound)
            rot_mats.append(M(axis, theta))
        # Use random order
        np.random.shuffle(rot_mats)
        rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
        return coords.dot(rot_mat)


class scale_coords:
    def __init__(self, scale_bound=(0.8, 1.25)):
        self.scale_bound = scale_bound

    def __call__(self, coords, return_scale=False):
        scale = np.random.uniform(*self.scale_bound)
        if return_scale:
            return coords*scale, scale
        else:
            return coords*scale


class flip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, coords):
        if np.random.random() < self.p:
          coord_max = np.max(coords[:, -1])
          coords[:, -1] = coord_max - coords[:, -1]
        return coords


class elastic_coords:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, coords, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = (np.abs(coords).max(0).astype(np.int32) // gran + 3).astype(np.int32)
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return coords + g(coords) * mag