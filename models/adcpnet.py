from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math


class feature_extraction(nn.Module):
    def __init__(self, init_ch):
        super(feature_extraction, self).__init__()

        self.layer11 = nn.Sequential(convbn(3, init_ch * 2, 3, 2, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn(init_ch * 2, init_ch * 2, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn(init_ch * 2, init_ch * 2, 3, 2, 1, 1),
                                     nn.ReLU(inplace=True))
        self.layer21 = self._make_layer(BasicBlock, init_ch * 2, init_ch * 4, 1, 2, 1, 1)
        self.layer31 = self._make_layer(BasicBlock, init_ch * 4, init_ch * 8, 1, 2, 1, 1)
        self.layer12 = nn.Sequential(
            self._make_layer(BasicBlock, init_ch * 2, init_ch * 2, 2, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_ch * 2, init_ch * 2, 3, 1, 1, bias=False)
        )
        self.layer32 = nn.Sequential(
            self._make_layer(BasicBlock, init_ch * 8, init_ch * 8, 2, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_ch * 8, init_ch * 8, 3, 1, 1, bias=False)
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, pad, dilation))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        level11 = self.layer11(x)
        level21 = self.layer21(level11)
        level31 = self.layer31(level21)

        level12 = self.layer12(level11)
        level32 = self.layer32(level31)

        return [level12, level32]  # 1/4 1/16


class ADCPNet(nn.Module):
    def __init__(self, args):
        super(ADCPNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.mag = args.mag
        cand_disp_num = self.mag * 2 + 1
        self.init_ch = args.c
        self.channels_3d = args.channels_3d
        self.growth_rate = args.growth_rate
        self.channels_3d_16x = self.channels_3d * self.growth_rate[0]
        self.channels_3d_4x = self.channels_3d * self.growth_rate[1]

        self.feature_extraction = feature_extraction(self.init_ch)

        self.dres16x_0 = nn.Sequential(
            convbn_3d(self.init_ch * 16, self.channels_3d_16x, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.dres16x_1 = nn.Sequential(
            convbn_3d(self.channels_3d_16x, self.channels_3d_16x, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(self.channels_3d_16x, self.channels_3d_16x, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(self.channels_3d_16x, self.channels_3d_16x, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(self.channels_3d_16x, self.channels_3d_16x, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.channels_3d_16x, 1, 3, 1, 1, bias=False),
        )
        self.classif16x = nn.Conv3d(self.channels_3d_16x, 1, (3, 3, 3), 1, (1, 1, 1), bias=False)

        self.dres4x_0 = nn.Sequential(
            convbn_3d(self.init_ch * 4, self.channels_3d_4x, (3, 3, 3), 1, (1, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.classif4x = nn.Conv3d(self.channels_3d_4x, 1, (3, 3, 3), 1, (1, 1, 1), bias=False)
        self.dres4x_1 = nn.Sequential(
            convbn_3d(self.channels_3d_4x, self.channels_3d_4x, (3, 3, 3), 1, (1, 1, 1)),
            nn.ReLU(inplace=True),
            convbn_3d(self.channels_3d_4x, self.channels_3d_4x, (3, 3, 3), 1, (1, 1, 1)),
            nn.ReLU(inplace=True),
            convbn_3d(self.channels_3d_4x, self.channels_3d_4x, (3, 3, 3), 1, (1, 1, 1)),
            nn.ReLU(inplace=True),
            convbn_3d(self.channels_3d_4x, self.channels_3d_4x, (3, 3, 3), 1, (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.channels_3d_4x, 1, (3, 3, 3), 1, (1, 1, 1), bias=False),
        )

        self.disp_reg = disparityregression2(0, self.maxdisp // 16)
        self.CRP4x = CandidateResidualPredictor(4, self.mag * 2, args.channel_CRP)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        r'''1/16 Disparity Network'''
        concat_volume = build_concat_volume(features_left[1], features_right[1], self.maxdisp // 16)
        cost16x_0 = self.dres16x_0(concat_volume)
        cost16x_1 = self.dres16x_1(cost16x_0)
        disp16x_1 = F.softmax(cost16x_1, 2)
        disp16x_1 = self.disp_reg(disp16x_1)
        disp16x = F.interpolate(disp16x_1 * 4., [features_left[0].size(2), features_left[0].size(3)],
                                   mode='bilinear')
        disp16x_full = F.interpolate(disp16x * 4., [left.size()[2], left.size()[3]], mode='bilinear').squeeze(1)

        r'''1/4 Disparity Network'''
        left_guide4x = F.interpolate(left, [features_left[0].size(2), features_left[0].size(3)],
                                   mode='bilinear')
        residual_guide_4x = torch.cat([left_guide4x, disp16x], dim=1)
        residual_candidates4x = self.CRP4x(residual_guide_4x)
        disp16x = disp16x.repeat(1, self.mag * 2 + 1, 1, 1)
        disp16x[:, :self.mag, :, :] -= residual_candidates4x[:, :self.mag, :, :]
        disp16x[:, (self.mag + 1):, :, :] -= residual_candidates4x[:, self.mag:, :, :]
        cost4x_0 = build_concat_volume_chaos(features_left[0], features_right[0], disp16x)
        cost4x_0 = self.dres4x_0(cost4x_0)
        cost4x = self.dres4x_1(cost4x_0)
        cost4x = torch.squeeze(cost4x, 1)
        disp4x = F.softmax(cost4x, 1)
        disp4x = disparityregression_chaos(disp4x, disp16x)
        disp4x_full = F.interpolate(disp4x * 4., [left.size()[2], left.size()[3]], mode='bilinear')
        disp4x_full = torch.squeeze(disp4x_full, 1)

        if self.training:

            r'''1/16 pre-supervision'''
            disp16x_0 = self.classif16x(cost16x_0)
            disp16x_0 = F.softmax(disp16x_0, 2)
            disp16x_0 = self.disp_reg(disp16x_0)
            disp16x_full_0 = F.interpolate(disp16x_0 * 16., [left.size()[2], left.size()[3]], mode='bilinear')
            disp16x_full_0 = disp16x_full_0.squeeze(1)

            r'''1/4 pre-supervision'''
            disp4x_0 = self.classif4x(cost4x_0).squeeze(1)
            disp4x_0 = F.softmax(disp4x_0, 1)
            disp4x_0 = disparityregression_chaos(disp4x_0, disp16x)
            disp4x_full_0 = F.interpolate(disp4x_0 * 4., [left.size()[2], left.size()[3]], mode='bilinear')
            disp4x_full_0 = disp4x_full_0.squeeze(1)

            return [disp16x_full_0, disp4x_full_0, disp16x_full, disp4x_full]

        else:
            return [disp16x_full, disp4x_full]


class CandidateResidualPredictor(nn.Module):
    def __init__(self, in_channel, candidate_num, channel):
        super(CandidateResidualPredictor, self).__init__()
        self.conv2d_feature = nn.Sequential(
            convbn(in_channel, channel, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 4, 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(
                    channel, channel, stride=1, downsample=None, pad=1, dilation=di))

        self.conv2d_out = nn.Conv2d(channel, candidate_num, kernel_size=3, stride=1, padding=1)

    def forward(self, in_map):
        output = self.conv2d_feature(in_map)
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)
        return self.conv2d_out(output)

