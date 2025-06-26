import torch
import torch.nn as nn
import MinkowskiEngine as ME

from MinkowskiEngine import MinkowskiNetwork
from MinkowskiEngine import MinkowskiReLU
import torch.nn.functional as F

from os.path import dirname, abspath
import sys
sys.path.append(dirname(abspath(__file__)))
from common import ConvType, NormType, conv, conv_tr, get_norm, sum_pool

class SegHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=20):
        super(SegHead, self).__init__()
        self.cluster = torch.nn.parameter.Parameter(data=torch.randn(out_channels, in_channels), requires_grad=True)
    
    def forward(self, feats):
        normed_clusters = F.normalize(self.cluster, dim=1)
        normed_features = F.normalize(feats, dim=1)
        logits = F.linear(normed_features, normed_clusters)
        
        return logits

class alignlayer(MinkowskiNetwork):
    def __init__(self, in_channels=128, out_channels=70, bn_momentum=0.02, norm_layer=True, D=3):
        super(alignlayer, self).__init__(D)        
        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]
        
        self.conv_align = conv(
            in_channels, out_channels, kernel_size=space_n_time_m(1, 1), stride=1, D=D
        )

    def forward(self, feats_tensorfield):
        x = feats_tensorfield.sparse()
        aligned_out = self.conv_align(x).slice(feats_tensorfield).F # 
        
        return aligned_out

class SubModel(MinkowskiNetwork):
    def __init__(self, dim, D=3):
        super(SubModel, self).__init__(D)
        # self.args = args
        self.distill_layer = alignlayer()#(in_channels=dim)

    def forward(self, feats_tensorfield):
        feats_aligned_nonorm = self.distill_layer(feats_tensorfield)

        return feats_aligned_nonorm