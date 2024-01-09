# -*- coding: utf-8 -*-
# fasternet.py
# author: lm

"""
https://arxiv.org/abs/2303.03667
<<Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks>>
"""

from collections import OrderedDict
from functools import partial
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import DropPath

from fastenet.pconv import PConv2d

class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 act: str = 'ReLU'):
        super(ConvBNLayer, self).__init__()
        assert act in ('ReLU', 'GELU')
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = getattr(nn, act)()

    def _fuse_bn_tensor(self) -> None:
        kernel = self.conv.weight
        bias = self.conv.bias if hasattr(self.conv, 'bias') and self.conv.bias is not None else 0
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        self.conv.weight.data = kernel * t
        self.conv.bias = nn.Parameter(beta - (running_mean - bias) * gamma / std, requires_grad=False)
        self.bn = nn.Identity()
        return self.conv.weight.data, self.conv.bias.data

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class FasterNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inner_channels: int = None,
                 kernel_size: int = 3,
                 bias=False,
                 act: str = 'ReLU',
                 n_div: int = 4,
                 forward: str = 'split_cat',
                 drop_path: float = 0.,
                 ):
        super(FasterNetBlock, self).__init__()
        inner_channels = inner_channels or in_channels * 2
        self.conv1 = PConv2d(in_channels,
                             kernel_size,
                             n_div,
                             forward)
        self.conv2 = ConvBNLayer(in_channels,
                                 inner_channels,
                                 bias=bias,
                                 act=act)
        self.conv3 = nn.Conv2d(inner_channels,
                               in_channels,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        return x + self.drop_path(y)


class FasterNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=10,
                 last_channels=1280,
                 inner_channels: list = [40, 80, 160, 320],
                 blocks: list = [1, 2, 8, 2],
                 bias=False,
                 act='GELU',
                 n_div=4,
                 forward='slicing',
                 drop_path=0.,
                 ):
        super(FasterNet, self).__init__()
        self.embedding = ConvBNLayer(in_channels,
                                     inner_channels[0],
                                     kernel_size=4,
                                     stride=4,
                                     bias=bias)
        self.embedding1 = ConvBNLayer(in_channels,
                                     inner_channels[0],
                                     kernel_size=4,
                                     stride=4,
                                     bias=bias)
        self.stage1 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[0],
                            bias=bias,
                            act=act,
                            n_div=n_div,
                            forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[0])]))
        self.stage11 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[0],
                            bias=bias,
                            act=act,
                            n_div=n_div,
                            forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[0])]))
        self.merging1 = ConvBNLayer(inner_channels[0],
                                    inner_channels[1],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.merging11 = ConvBNLayer(inner_channels[0],
                                    inner_channels[1],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.stage2 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[1],
                            bias=bias,
                            act=act,
                            n_div=n_div,
                            forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[1])]))
        self.stage21 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[1],
                            bias=bias,
                            act=act,
                            n_div=n_div,
                            forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[1])]))
        self.merging2 = ConvBNLayer(inner_channels[1],
                                    inner_channels[2],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.merging21 = ConvBNLayer(inner_channels[1],
                                    inner_channels[2],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.stage3 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[2],
                            bias=bias,
                            act=act,
                            n_div=n_div,
                            forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[2])]))
        self.stage31 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[2],
                            bias=bias,
                            act=act,
                            n_div=n_div,
                            forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[2])]))
        self.merging3 = ConvBNLayer(inner_channels[2],
                                    inner_channels[3],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.merging31 = ConvBNLayer(inner_channels[2],
                                    inner_channels[3],
                                    kernel_size=2,
                                    stride=2,
                                    bias=bias)
        self.stage4 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[3],
                            bias=bias,
                            act=act,
                            n_div=n_div,
                            forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[3])]))
        self.stage41 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             FasterNetBlock(inner_channels[3],
                            bias=bias,
                            act=act,
                            n_div=n_div,
                            forward=forward,
                            drop_path=drop_path)) for idx in range(blocks[3])]))
        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inner_channels[-1], last_channels, 1, bias=False),
            getattr(nn, act)()
        )
        self.avgpool_pre_head1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inner_channels[-1], last_channels, 1, bias=False),
            getattr(nn, act)()
        )
        # CA注意力
        self.ca = CoordAtt(inp=last_channels * 2, oup=last_channels * 2)
        self.head = nn.Linear(last_channels * 2, out_channels) \
            if out_channels > 0 else nn.Identity()

        self.feature_channels = inner_channels

    def fuse_bn_tensor(self):
        for m in self.modules():
            if isinstance(m, ConvBNLayer):
                m._fuse_bn_tensor()


    def forward(self, x: Tensor,nx:Tensor) -> Tensor:
        x1 = self.stage1(self.embedding(x))
        nx1 = self.stage11(self.embedding1(nx))
        x2 = self.stage2(self.merging1(x1))
        nx2 = self.stage21(self.merging11(nx1))
        x3 = self.stage3(self.merging2(x2))
        nx3 = self.stage31(self.merging21(nx2))
        x4 = self.stage4(self.merging3(x3))
        nx4 = self.stage41(self.merging31(nx3))
        # x = self.classifier(x)
        x4 = self.avgpool_pre_head(x4)
        nx4 = self.avgpool_pre_head1(nx4)
        n_x = torch.cat((x4, nx4), 1)
        # 加入ca
        n_x = self.ca(n_x)
        n_x = torch.flatten(n_x, 1)
        n_x = self.head(n_x)
        return n_x
