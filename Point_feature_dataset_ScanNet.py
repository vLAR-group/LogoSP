import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import math
from scipy.ndimage import zoom
import os

class PointFeature(Dataset):
    def __init__(self,scene_id):
        self.scene_id = scene_id
        self.SCANNET_FRAME_ROOT = './data/ScanNet/scannet_2d/{}/{}/'
        self.frame_list = sorted([int(x.split(".")[0]) for x in os.listdir(self.SCANNET_FRAME_ROOT.format(self.scene_id, "color/"))])
        self.SCANNET_FRAME_PATH = './data/ScanNet/scannet_2d/{}/'
        self.scale_factor = 4
        '''loading data frame'''
    def to_tensor(self,arr):
        return torch.Tensor(arr)

    def resize_crop_image(self,image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(
            Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)

        return image

    def aug(self,color):
        color = np.array(color)
        color = zoom(color,(self.scale_factor, self.scale_factor, 1), order=3)
        color = self.resize_crop_image(color, (1274, 952))
        color = np.transpose(color, [2, 0, 1])
        color = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(color.astype(np.float32) / 255.0))
        color = self.to_tensor(color).unsqueeze(0)
        return color

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self,index):
        frame_list = self.frame_list
        frame_id = frame_list[index]

        color = Image.open(self.SCANNET_FRAME_PATH.format(self.scene_id) + "/color" + "/{}.jpg".format(frame_id))
        color = self.aug(color)

        return color, frame_id, frame_list, index


class cfl_collate_fn:
    def __call__(self, list_data):
        color,frame_id,frame_list,index= list(zip(*list_data))
        color_batch = []
        accm_num = 0
        for batch_id, _ in enumerate(color):
            color_batch.append(color[batch_id])
            accm_num += color_batch[batch_id].shape[0]
        color_batch = torch.cat(color_batch, 0).float()

        return color_batch,frame_id,frame_list,index














