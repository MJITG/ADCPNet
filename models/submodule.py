from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np
import time


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def l1_distance(fea1, fea2):
    out = torch.norm(fea1 - fea2, 1, 1)
    return out


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def build_concat_volume_chaos(refimg_fea, targetimg_fea, disps):
    B, C, H, W = refimg_fea.shape
    batch_disp = torch.unsqueeze(disps, dim=2).view(-1, 1, H, W)
    # batch_feat_l = refimg_fea[:, None, :, :, :].repeat(1, disps.shape[1], 1, 1, 1).view(-1, C, H, W)
    batch_feat_r = targetimg_fea[:, None, :, :, :].repeat(1, disps.shape[1], 1, 1, 1).view(-1, C, H, W)
    volume = refimg_fea.new_zeros([B, 2 * C, disps.size(1), H, W])
    warped_batch_feat_r = warp(batch_feat_r, batch_disp).view(B, disps.shape[1], C, H, W).permute(0, 2, 1, 3, 4)
    # print('warped_batch_feat_r shape: {}'.format(warped_batch_feat_r.size()))
    # print('warped_batch_feat_r: {}'.format(warped_batch_feat_r))
    for i in range(disps.size(1)):
        volume[:, :C, i, :, :] = refimg_fea
        volume[:, C:, i, :, :] = warped_batch_feat_r[:, :, i, :, :]
    volume = volume.contiguous()
    return volume


def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float()

    # vgrid = Variable(grid)
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    return output


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


def disparityregression_chaos(x, disps):
    out = torch.sum(x * disps, 1, keepdim=True)
    return out


class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        # self.disp = torch.arange(start*stride, end*stride, stride, device='cuda').view(1, 1, -1, 1, 1).float()
        self.disp = torch.arange(start, end, stride, device='cuda').view(1, 1, -1, 1, 1).float()
        # print('self.disp: {}'.format(self.disp))

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], x.size()[1], 1, x.size()[3], x.size()[4]).to(x.device)
        out = torch.sum(x * disp, 2)
        return out



if __name__ == '__main__':
    l = torch.rand(2, 2, 3, 4)
    r = torch.rand(2, 2, 3, 4)
    d = torch.ones(2, 2, 3, 4)
    d[1, :, :, :] *= 2.
    times = []
    for i in range(100):
        st = time.time()
        cv = build_concat_volume_chaos(l, r, d)
        times.append(time.time() - st)

    print('l: {}'.format(l))
    print('r: {}'.format(r))
    print('d: {}'.format(d))
    print('cv: {}'.format(cv))
    print('Mean time: {:6f}'.format(sum(times) / (len(times) - 1)))
