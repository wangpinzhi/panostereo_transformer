#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models.resnet import BasicBlock

from utilities.misc import NestedTensor
from basic import SphereConv

# NOTE: build feature extraction with sphere conv


def sphereConvbn(in_height, in_width, sphereType, in_planes, out_planes, kernel_size, stride, pad=1, dilation=1):

    return nn.Sequential(
        SphereConv(in_height,
                   in_width,
                   sphereType,
                   in_planes,
                   out_planes,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=dilation if dilation > 1 else pad,
                   dilation=dilation,
                   bias=False),
        nn.BatchNorm2d(out_planes))


def sphereConvbnrelu(in_height, in_width, sphereType, in_planes, out_planes, kernel_size, stride, pad=1, dilation=1):

    return nn.Sequential(
        SphereConv(in_height,
                   in_width,
                   sphereType,
                   in_planes,
                   out_planes,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=dilation if dilation > 1 else pad,
                   dilation=dilation,
                   bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class SphereBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_height, in_width, sphereType, inplanes, planes, stride=1, downsample=None, pad=1, dilation=1):
        super(SphereBasicBlock, self).__init__()

        self.conv1 = SphereConv(in_height, in_width, sphereType, inplanes, planes, kernel_size=3,
                                stride=stride,
                                padding=dilation if dilation > 1 else pad,
                                dilation=dilation,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SphereConv(in_height//stride, in_width//stride, sphereType, planes, planes, kernel_size=3,
                                stride=1,
                                padding=dilation if dilation > 1 else pad,
                                dilation=dilation,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SppBackbone(nn.Module):
    """
    Contracting path of feature descriptor using Spatial Pyramid Pooling,
    SPP followed by PSMNet (https://github.com/JiaRenChang/PSMNet)
    """

    def __init__(self, image_size, use_shereconv=False):
        super(SppBackbone, self).__init__()
        self.inplanes = 32
        self.use_shpereconv = use_shereconv
        self.in_conv = nn.Sequential(
            SphereConv(image_size[0], image_size[1], 'Cassini', 3, 16, kernel_size=3, stride=2, padding=1, bias=False) if self.use_shpereconv else
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),


            SphereConv(image_size[0]//2, image_size[1]//2, 'Cassini', 16, 16, kernel_size=3, stride=1, padding=1, bias=False) if self.use_shpereconv else
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            SphereConv(image_size[0]//2, image_size[1]//2, 'Cassini', 16, 32, kernel_size=3, stride=1, padding=1, bias=False) if self.use_shpereconv else
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 1/2
        if use_shereconv:
            self.resblock_1 = self._make_layer(
                SphereBasicBlock, 64, 3, 2, (image_size[0]//2, image_size[1]//2))  # 1/4
            self.resblock_2 = self._make_layer(
                SphereBasicBlock, 128, 3, 2, (image_size[0]//4, image_size[1]//4))  # 1/8
        else:
            self.resblock_1 = self._make_layer(BasicBlock, 64, 3, 2)  # 1/4
            self.resblock_2 = self._make_layer(BasicBlock, 128, 3, 2)  # 1/8

        self.branch1 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     nn.Conv2d(
                                         128, 32, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     nn.Conv2d(
                                         128, 32, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     nn.Conv2d(
                                         128, 32, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2, 2)),
                                     nn.Conv2d(
                                         128, 32, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

    def _make_layer(self, block, planes, blocks, stride=1, img_size=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        if self.use_shpereconv:
            layers.append(block(
                img_size[0], img_size[1], 'Cassini', self.inplanes, planes, stride, downsample))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if self.use_shpereconv:
                if stride != 1:
                    h = img_size[0]//2
                    w = img_size[1]//2
                    layers.append(
                        block(h, w, 'Cassini', self.inplanes, planes))
            else:
                layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: NestedTensor):
        """
        :param x: NestedTensor
        :return: list containing feature descriptors at different spatial resolution
                0: [2N, 3, H, W]
                1: [2N, C0, H//4, W//4]
                2: [2N, C1, H//8, W//8]
                3: [2N, C2, H//16, W//16]
        """
        _, _, h, w = x.left.shape

        src_stereo = torch.cat([x.left, x.right], dim=0)  # 2NxCxHxW

        # in conv
        output = self.in_conv(src_stereo)  # 1/2
        # res blocks
        output_1 = self.resblock_1(output)  # 1/4
        output_2 = self.resblock_2(output_1)  # 1/8

        # spp
        h_spp, w_spp = math.ceil(h / 16), math.ceil(w / 16)
        spp_1 = self.branch1(output_2)
        spp_1 = F.interpolate(spp_1, size=(h_spp, w_spp),
                              mode='bilinear', align_corners=False)
        spp_2 = self.branch2(output_2)
        spp_2 = F.interpolate(spp_2, size=(h_spp, w_spp),
                              mode='bilinear', align_corners=False)
        spp_3 = self.branch3(output_2)
        spp_3 = F.interpolate(spp_3, size=(h_spp, w_spp),
                              mode='bilinear', align_corners=False)
        spp_4 = self.branch4(output_2)
        spp_4 = F.interpolate(spp_4, size=(h_spp, w_spp),
                              mode='bilinear', align_corners=False)
        output_3 = torch.cat([spp_1, spp_2, spp_3, spp_4], dim=1)  # 1/16

        return [src_stereo, output_1, output_2, output_3]


def build_backbone(args):
    return SppBackbone(use_shereconv=args.use_shpereconv, image_size=(1024, 512))
