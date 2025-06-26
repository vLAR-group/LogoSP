import argparse
import time
import os
import numpy as np
import random
from datasets.nuScenes import nuScenesdistill, cfl_collate_fn_distill
import torch
from glob import glob
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18
import logging
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='./data/nuScenes/nuscenes_3d/train/',
                        help='pont cloud data path')
    parser.add_argument('--val_input_path', type=str, default='./data/nuScenes/nuscenes_3d/val',
                        help='pont cloud data path')
    parser.add_argument('--feats_path', type=str, default='./data/nuScenes/DINOv2_feats_s14up4_voxel_0.15/',
                        help='project dino point feature')
    ###
    parser.add_argument('--save_path', type=str, default='ckpt/nuScenes/distill/',
                        help='model savepath')
    parser.add_argument('--max_epoch', type=int, default=300, help='max epoch')
    parser.add_argument('--max_iter', type=int, default=300*150, help='max iter')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD parameters')
    parser.add_argument('--dampening', type=float, default=0.1, help='SGD parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='SGD parameters')
    parser.add_argument('--workers', type=int, default=8, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--log-interval', type=int, default=50, help='log interval')
    parser.add_argument('--batch_size', type=int, default=10, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')  ### 6 for XYZGB
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--drop_threshold', type=int, default=10, help='ignore superpoints with few points')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--select_num', type=int, default=1500, help='scene number selected in each round')
    return parser.parse_args()


def main(args, logger):

    scene_idx = np.random.choice(28130, args.select_num, replace=False)  ## nuScenens totally has 28130 training samples

    train_set = nuScenesdistill(args, scene_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=cfl_collate_fn_distill(), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))

    '''Prepare Model/Optimizer'''
    model = Res16FPN18(in_channels=args.input_dim, out_channels=20, conv1_kernel_size=args.conv1_kernel_size,
                       config=args)
    logger.info(model)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = PolyLR(optimizer, max_iter=args.max_iter)

    checkpoints = glob(args.save_path + '/*tar')
    if len(checkpoints) == 0:
        print('No checkpoints found at {}'.format(args.save_path))
        epoch = 1
    else:
        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = args.save_path + 'checkpoint_{}.tar'.format(checkpoints[-1])
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    '''Train'''
    for epoch in range(epoch, args.max_epoch + 1):
        '''Take 10 epochs as a round'''
        if (epoch - 1) % 10 == 0:
            scene_idx = np.random.choice(28130, args.select_num, replace=False)
            train_loader.dataset.random_select_sample(scene_idx)
        train(train_loader, logger, model, optimizer, epoch, scheduler)

        if epoch % 10 == 0:
            ckpt_path = args.save_path + 'checkpoint_{}.tar'.format(epoch)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, ckpt_path)


def train(train_loader, logger, model, optimizer, epoch, scheduler):
    model.train()
    loss_display = 0
    time_curr = time.time()
    for batch_idx, data in enumerate(train_loader):
        iteration = (epoch - 1) * len(train_loader) + batch_idx + 1

        coords, features, inverse_map, feats_2d, masks, index = data

        in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)
        feats_3d = model(in_field)[masks]
        loss_sem = (1 - torch.nn.CosineSimilarity()(feats_3d, feats_2d.cuda().detach())).mean()

        loss_display += loss_sem.item()
        optimizer.zero_grad()
        loss_sem.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        if (batch_idx + 1) % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.10f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, (batch_idx + 1), len(train_loader), 100. * (batch_idx + 1) / len(train_loader),
                    iteration, loss_display, optimizer.param_groups[0]['lr'], time_used, args.log_interval))
            time_curr = time.time()
            loss_display = 0


from torch.optim.lr_scheduler import LambdaLR


class LambdaStepLR(LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter=300 * 151, power=0.9, last_step=-1):
        super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1)) ** power, last_step)


def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    args = parse_args()

    '''Setup logger'''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = set_logger(os.path.join(args.save_path, 'train.log'))

    '''Random Seed'''
    seed = args.seed
    set_seed(seed)

    main(args, logger)