import math
import sys

import torch
import torch.nn as nn
import numpy as np
use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}


def KMeans(x, K=10, Niters=10, verbose=False):
    N, D = x.shape

    c = x[:K, :].clone()
    x_i = x[:, None, :]

    for i in range(Niters):
        c_j = c[None, :, :]

        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cl = D_ij.argmin(dim=1).long().view(-1)

        Ncl = cl.view(cl.size(0), 1).expand(-1, D)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)

        labels_count_all = torch.ones([K]).long().cuda()
        labels_count_all[unique_labels[:, 0]] = labels_count
        c = torch.zeros([K, D], dtype=torch.float).cuda().scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c


def IID_loss(x_out, x_tf_out, EPS=sys.float_info.epsilon):
    """
    This code is from IIC github
    """

    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    loss = loss.sum()
    return loss

def compute_joint(x_out, x_tf_out):
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


class AdvModule(object):
    def __init__(self, model, dataparallel, emb_name='resnet.conv1.weight', epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        if dataparallel:
            self.emb_name = 'module.' + emb_name
        else:
            self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name == name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def CAM(features1, features2, T, args):

    criterion = nn.CrossEntropyLoss().to(args.default_device)
    softmax = nn.Softmax(dim=1)

    cluster_label1, centroids1 = KMeans(features1, K=args.clusters, Niters=args.num_iters)
    cluster_label2, centroids2 = KMeans(features2, K=args.clusters, Niters=args.num_iters)

    affnity1 = torch.mm(features1, centroids2.t())
    affnity1_softmax = softmax(affnity1)
    CLD_loss = criterion(affnity1.div_(T), cluster_label2)

    affnity2 = torch.mm(features2, centroids1.t())
    affnity2_softmax = softmax(affnity2)
    CLD_loss = (CLD_loss + criterion(affnity2.div_(T), cluster_label1)) / 2
    iid_loss = IID_loss(affnity1_softmax, affnity2_softmax)

    if args.do_entro:
        ret_loss = CLD_loss + iid_loss
    else:
        ret_loss = CLD_loss
    return ret_loss


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
