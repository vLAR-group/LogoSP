import torch, os, argparse,random
import torch.nn.functional as F
import torch.nn as nn
from datasets.ScanNet import Scannettrain,cfl_collate_fn
import numpy as np
import MinkowskiEngine as ME
import logging
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
###

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PointDC')
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=8, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=12, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=384, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--drop_threshold', type=int, default=10, help='ignore superpoints with few points')
    return parser.parse_args()


def get_sp_feature(loader, model):
    loader.dataset.mode = 'cluster'
    region_feats_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, _, labels, inverse_map, pseudo_labels, inds, region, index, scenenames = data
            region = region.squeeze()

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)  # 获取points feats
            ##
            valid_mask = region != -1
            feats = feats[valid_mask]
            region = region[valid_mask].long()
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()  ##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            region_feats = F.linear(region_corr.t(), feats.t()) / per_region_num
            region_feats = F.normalize(region_feats, dim=-1)
            region_feats = region_feats.cpu()
            region_feats_list.append(region_feats)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return region_feats_list


def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for para in classifier.parameters():
        para.requires_grad = False
    return classifier



def cluster(args,cluster_loader,model):

    '''Extract Superpoints Feature'''
    sp_feats_list = get_sp_feature(cluster_loader, model)
    sp_feats = torch.cat(sp_feats_list, dim=0)### will do Kmeans with geometric distance
    primitive_labels = KMeans(n_clusters=args.primitive_num, random_state=0).fit_predict(sp_feats.numpy())

    '''Compute Primitive Centers'''
    primitive_centers = torch.zeros((args.primitive_num, args.feats_dim))
    def cluster_center(args,primitive_labels,sp_feats,primitive_centers):
        for cluster_idx in range(args.primitive_num):
            indices = primitive_labels == cluster_idx
            cluster_avg = sp_feats[indices].mean(0, keepdims=True)
            primitive_centers[cluster_idx] = cluster_avg
        return primitive_centers
    primitive_centers =cluster_center(args,primitive_labels,sp_feats,primitive_centers)
    primitive_centers = F.normalize(primitive_centers, dim=1)
    ##
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.primitive_num, centroids=primitive_centers)
    return classifier.cuda()


def eval_once(args, model, test_loader, classifier):
    model.mode = 'train'
    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data
            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)
            scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
            preds = torch.argmax(scores, dim=1).cpu()
            preds = preds[inverse_map.long()]
            all_preds.append(preds[labels != args.ignore_label]), all_label.append(labels[[labels != args.ignore_label]])
    return all_preds, all_label


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

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)

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
    model_path = './ckpt/ScanNet/model_300_checkpoint.pth'

    args = parse_args()
    '''Setup logger'''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = set_logger(os.path.join(args.save_path, 'Scannet_to_S3dis.log'))

    '''Random Seed'''
    seed = args.seed
    set_seed(seed)

    ##load trained scannet model weight
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num,conv1_kernel_size=args.conv1_kernel_size, args=args)
    model.load_state_dict(torch.load(model_path))
    model= model.cuda()
    model.eval()

    trainset = Scannettrain(args)
    cluster_loader = DataLoader(trainset, batch_size=10, shuffle=True, collate_fn=cfl_collate_fn(),num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))
    cluster_loader.dataset.mode = 'cluster'
    classifier = cluster(args, cluster_loader, model)


    ##start eval
    args.ignore_label = 12
    args.data_path = './data/S3DIS/input_0.010/'
    args.sp_path = './data/S3DIS/initial_superpoints'

    test_areas = ['Area_5']
    logger.info(test_areas)
    val_dataset = S3DIStest(args, areas=test_areas)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(),num_workers=args.cluster_workers, pin_memory=True)
    preds, labels = eval_once(args, model, val_loader, classifier)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[0], m[1]].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[m[0], m[1]] / histogram.sum(1))*100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[1][idx]]

    '''Final Metrics'''
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    logger.info('oAcc {:.2f}  mAcc {:.2f} IoUs'.format( o_Acc, m_Acc) + s)