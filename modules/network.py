import torch.nn as nn
import torch
from torch.nn.functional import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Variable
from torch.nn import Parameter


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out
    
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

# class Network(nn.Module):
#     def __init__(self, resnet, feature_dim):
#         super(Network, self).__init__()
#         self.resnet = resnet
#         self.feature_dim = feature_dim
#         self.instance_projector = nn.Sequential(
#             nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
#             nn.ReLU(),
#             nn.Linear(self.resnet.rep_dim, self.feature_dim),
#         )
# #         self.cluster_projector = nn.Sequential(
# #             nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
# #             nn.ReLU(),
# #             nn.Linear(self.resnet.rep_dim, self.cluster_num),
# #             nn.Softmax(dim=1)
# #         )

#     def forward(self, x):
#         imgs = []
#         for i in range(len(x)):
#             h_i = self.resnet(x[i])
#             z_i = normalize(self.instance_projector(h_i), dim=1)
#             imgs.append(z_i)
#         return imgs

class Network(nn.Module):
    def __init__(self, resnet, feature_dim):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
#         self.cluster_projector = nn.Sequential(
#             nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
#             nn.ReLU(),
#             nn.Linear(self.resnet.rep_dim, self.cluster_num),
#             nn.Softmax(dim=1)
#         )
        linear_layer = NormedLinear# if normlinear else nn.Linear
        self.groupDis = nn.Sequential(
            linear_layer(self.resnet.rep_dim, feature_dim),
            Normalize(2))
        
    def forward(self, x):
        h_i = self.resnet(x)
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_i_group = self.groupDis(h_i)
        return z_i, z_i_group

    def forward_get_feature(self, x):
        h = self.resnet(x)
        
        return h