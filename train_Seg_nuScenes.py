import argparse
import time
import os
import numpy as np
import random
from datasets.nuScenes import nuScenestrain, cfl_collate_fn
import torch
from scipy.sparse import csgraph
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18
from eval_nuScenes import eval
from utils_degrowsp import get_pseudo, get_fixclassifier, get_sp_feature_unScenes
from sklearn.cluster import KMeans, SpectralClustering
import logging
from torch_scatter import scatter_mean, scatter_max, scatter_min
from sklearn.preprocessing import LabelEncoder
from os.path import join
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='./data/nuScenes/nuScenes_3d/train/',
                        help='pont cloud data path')
    parser.add_argument('--val_input_path', type=str, default='./data/nuScenes/nuScenes_3d/val',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= './data/nuScenes/initial_superpoints/train/',
                        help='initial sp path')
    ###
    parser.add_argument('--save_path', type=str, default='ckpt_seg/nuScenes/seg',
                        help='model savepath')
    parser.add_argument('--max_epoch', type=int, default=50, help='max epoch')
    parser.add_argument('--max_iter', type=int, default=50*500, help='max iter')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD parameters')
    parser.add_argument('--dampening', type=float, default=0.1, help='SGD parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='SGD parameters')
    parser.add_argument('--workers', type=int, default=8, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--log-interval', type=int, default=500, help='log interval')
    parser.add_argument('--batch_size', type=int, default=10, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')  ### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=16, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=16, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384*1, help='output feature dimension')
    parser.add_argument('--pseudo_label_path', default='pseudo_label_unscenes/', type=str, help='pseudo label save path')
    parser.add_argument('--drop_threshold', type=int, default=10, help='ignore superpoints with few points')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--select_num', type=int, default=5000, help='scene number selected in each round')
    parser.add_argument('--col_num', type=int, default=50)
    parser.add_argument('--eneg_ratio', type=float, default=0.99)
    parser.add_argument('--distill_ckpt', type=str, default='./ckpt/nuScenes/distill/checkpoint_300.tar')
    return parser.parse_args()


def main(args, logger):

    scene_idx = np.random.choice(28130, args.select_num, replace=False)  ## nuScenens totally has 28130 training samples

    train_set = nuScenestrain(args, scene_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=cfl_collate_fn(), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))

    '''Prepare Model/Optimizer'''
    model = Res16FPN18(in_channels=args.input_dim, out_channels=16, conv1_kernel_size=args.conv1_kernel_size, config=args)
    model.load_state_dict(torch.load(args.distill_ckpt)['model'])

    logger.info(model)
    model = model.cuda()
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()

    '''cluster'''
    current_sp = 20
    cluster_set = nuScenestrain(args, scene_idx)
    cluster_loader = DataLoader(cluster_set, batch_size=1, shuffle=True, collate_fn=cfl_collate_fn(), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))
    classifier = cluster(args, logger, cluster_loader, current_sp, model)
    ##
    for para in classifier.parameters():
        para.requires_grad = True

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params': classifier.parameters()}], lr=args.lr)
    scheduler = PolyLR(optimizer, max_iter=args.max_iter)
    '''Train'''
    for epoch in range(1, args.max_epoch + 1):
        train(train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(classifier.state_dict(), join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth'))
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch,args)
                logger.info('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s)


def rbf_eig_vector(data, device='cuda', norm_laplacian=True):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    data = data.to(device)
    d = torch.cdist(data, data, p=2)
    gamma = 1.0 / data.shape[1]
    A = torch.exp(-gamma*d)
    A = A.cpu().numpy()
    laplacian, dd = csgraph.laplacian(A, normed=norm_laplacian, return_diag=True)  # only D-A if normalized is False
    eigvalues, eigvectors = torch.linalg.eigh(torch.from_numpy(laplacian).cuda())
    return eigvectors.cpu()

def contin_label(label):
    Encoder = LabelEncoder()
    return Encoder.fit_transform(label)

def cluster(args, logger, cluster_loader, current_sp, model):
    time_start = time.time()
    cluster_loader.dataset.mode = 'cluster'

    '''Extract Superpoints Feature'''
    feats, labels, sp_index, context = get_sp_feature_unScenes(args, cluster_loader, current_sp, model)
    sp_feats = torch.cat(feats, dim=0)
    sp_feats = F.normalize(sp_feats)
    print('clustering ...')
    if sp_feats.shape[0]>30000:
        print(sp_feats.shape)
        sampled_sp_feats = sp_feats[np.random.choice(sp_feats.shape[0], 30000, replace=False)]
    else:
        sampled_sp_feats = sp_feats
    # ####################################################################################################################
    eigvectors = rbf_eig_vector(sampled_sp_feats)
    ## select W
    ## 1. compute energy to delete invalid W
    all_amp_vector = eigvectors.T @ sampled_sp_feats  ## [N, C]
    all_energy = all_amp_vector[1:].pow(2).sum(-1)  ## [N]
    eigvectors = eigvectors[:, 1:]
    all_amp_vector = all_amp_vector[1:]
    sorted_energy, indices = torch.sort(all_energy, descending=True)
    acc_energy = 0
    valid_indice_list = []
    for i, energy in enumerate(sorted_energy):
        acc_energy = acc_energy + energy
        valid_indice_list.append(indices[i])
        if acc_energy > all_energy.sum() * args.eneg_ratio:
            break

    valid_eig_indices = torch.tensor(valid_indice_list).long()
    valid_eigvectors, valid_amp_vector = eigvectors[:, valid_eig_indices], all_amp_vector[valid_eig_indices]
    # spec_embedding = valid_eigvectors[:, 0:args.col_num].numpy()
    ####################################################################################################################
    group_w_labels = KMeans(n_clusters=args.col_num, random_state=0).fit_predict(valid_amp_vector.numpy().astype(np.float32))
    group_w_labels = contin_label(group_w_labels)
    spec_embedding = scatter_mean(valid_eigvectors.T, torch.from_numpy(group_w_labels).long(), dim=0)
    spec_embedding = spec_embedding.T.numpy()

    primitive_labels = KMeans(n_clusters=args.primitive_num, random_state=0).fit_predict(spec_embedding.astype(np.float32))
    primitive_labels = contin_label(primitive_labels)

    '''Compute Primitive Centers'''
    primitive_centers = torch.zeros((args.primitive_num, args.feats_dim))
    for cluster_idx in range(args.primitive_num):
        indices = primitive_labels == cluster_idx
        cluster_avg = sampled_sp_feats[indices].mean(0, keepdims=True)
        primitive_centers[cluster_idx] = cluster_avg
    primitive_centers = F.normalize(primitive_centers, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.primitive_num, centroids=primitive_centers)

    ##
    primitive_labels = (F.normalize(sp_feats) @ primitive_centers.T).max(1).indices.numpy()

    '''Compute and Save Pseudo Labels'''
    all_pseudo, all_gt, all_pseudo_gt = get_pseudo(args, context, primitive_labels, sp_index)
    logger.info('labelled points ratio %.2f clustering time: %.2fs', (all_pseudo!=-1).sum()/all_pseudo.shape[0], time.time() - time_start)

    '''Check Superpoint/Primitive Acc in Training'''
    sem_num = args.semantic_class
    mask = (all_pseudo_gt!=-1)&(all_gt!=-1)
    histogram = np.bincount(sem_num* all_gt.astype(np.int32)[mask] + all_pseudo_gt.astype(np.int32)[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)    # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Superpoints oAcc {:.2f} IoUs'.format(o_Acc) + s)

    pseudo_class2gt = -np.ones_like(all_gt)
    for i in range(args.primitive_num):
        mask = all_pseudo==i
        if mask.sum()>0:
            pseudo_class2gt[mask] = torch.mode(torch.from_numpy(all_gt[mask])).values
    mask = (pseudo_class2gt!=-1)&(all_gt!=-1)
    histogram = np.bincount(sem_num* all_gt.astype(np.int32)[mask] + pseudo_class2gt.astype(np.int32)[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)    # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum()/histogram.sum()*100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('Primitives oAcc {:.2f} IoUs'.format(o_Acc) + s)
    return classifier.cuda()


def train(train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier):
    model.train()
    classifier.train()
    loss_display = 0
    time_curr = time.time()
    for batch_idx, data in enumerate(train_loader):
        iteration = (epoch - 1) * len(train_loader) + batch_idx + 1

        coords, labels, inverse_map, pseudo_labels, region, index, scene_name = data
        in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)
        feats = model(in_field)
        #
        pseudo_labels_comp = pseudo_labels.long().cuda()
        logits = F.linear(F.normalize(feats), F.normalize(classifier.weight))
        loss_sem = loss(logits * 3, pseudo_labels_comp).mean()## 5 is temperature

        loss_display += loss_sem.item()
        optimizer.zero_grad()
        loss_sem.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        if (batch_idx+1) % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.10f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, (batch_idx+1), len(train_loader), 100. * (batch_idx+1) / len(train_loader),
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
  def __init__(self, optimizer, max_iter=50000, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)

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

    os.system(f"cp {__file__} {args.save_path}")
    os.system(f"cp -r {'./models/'} {args.save_path}")
    os.system(f"cp -r {'./datasets/'} {args.save_path}")
    os.system(f"cp -r {'utils_degrowsp.py'} {args.save_path}")

    '''Random Seed'''
    seed = args.seed
    set_seed(seed)

    main(args, logger)