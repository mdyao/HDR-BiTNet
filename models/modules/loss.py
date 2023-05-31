import torch
import torch.nn as nn
from torch.nn import functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        elif self.losstype == 'l_log':
            diff = x - target
            eps = 1e-6
            return torch.mean(torch.sum(-torch.log(1-diff.abs()+eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0

class Forw_strcture(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(Forw_strcture, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        elif self.losstype == 'l_log':
            diff = x - target
            eps = 1e-6
            return torch.mean(torch.sum(-torch.log(1-diff.abs()+eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0


class Forw_orth(nn.Module):
    def __init__(self,  eps=1e-3):
        super(Forw_orth, self).__init__()
        self.eps = eps

    def forward(self, proj_a, proj_b):
        N, _, H, W = proj_a.shape
        proj_a = torch.reshape(proj_a, (N, 1, -1))
        proj_b = torch.reshape(proj_b, (N, -1, 1))
        out = torch.bmm(proj_a, proj_b)
        out = torch.abs(torch.mean(out))
        return out


class TV_extractor(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TV_extractor, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.fil = nn.Parameter(torch.ones(1, 1, 3, 3)/9, requires_grad=False)

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
        w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
        h_tv = F.pad(h_tv, [0,0,0,1], "constant", 0)
        w_tv = F.pad(w_tv, [0,1,0,0], "constant", 0)

        h_tv = F.conv2d(h_tv, self.fil, stride=1, padding=1, groups=1)
        w_tv = F.conv2d(w_tv, self.fil, stride=1, padding=1, groups=1)

        # print(h_tv.shape, w_tv.shape)
        tv = torch.abs(h_tv)+torch.abs(w_tv)
        return tv

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
