# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
import torch
import torch.nn.functional as F
from torch import nn

from asynet_mask_rcnn.layers import FrozenBatchNorm2d
from asynet_mask_rcnn.layers import Conv2d
from asynet_mask_rcnn.layers import DFConv2d
from asynet_mask_rcnn.modeling.make_layers import group_norm

from asynet_mask_rcnn.layers.asyn import (
    asynSparseConvolution2D,
    asynMaxPool, 
    Sites
)


### Asynchronous ResNet Stem definition ###
class AsynBaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(AsynBaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = asynSparseConvolution2D(
            dimension=2, nIn=3, nOut=out_channels, 
            filter_size=7, use_bias=False, first_layer=True
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

        self.mp1 = asynMaxPool(
            dimension=2, filter_size=3, filter_stride=2, 
            padding_mode='valid'
        )

    def forward(self, x):
        x = list(self.conv1.forward(update_location=x[0], 
            feature_map=x[1], active_sites=None, 
            rule_book_input=None, rule_book_output=None))
        x = self.bn1(x)
        x = F.relu_(x)
        changed_locations = (x[2] > Sites.ACTIVE_SITE.value).nonzero()
        x = self.mp1.forward(
                update_location=changed_locations.long())
        return x


class AsynStemWithFixedBatchNorm(AsynBaseStem):
    def __init__(self, cfg):
        super(AsynStemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class AsynStemWithGN(AsynBaseStem):
    def __init__(self, cfg):
        super(AsynStemWithGN, self).__init__(cfg, norm_func=group_norm)


### Asynchronous ResNet BottleNeck Definition ###
class AsynBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(AsynBottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                asynSparseConvolution2D(
                    dimension=2, nIn=in_channels, nOut=out_channels,
                    filter_size=1, use_bias=False, first_layer=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, asynSparseConvolution2D):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        self.conv1 = asynSparseConvolution2D(
            dimension=2, nIn=in_channels, nOut=bottleneck_channels,
            filter_size=1, use_bias=False, first_layer=True
        )
        self.bn1 = norm_func(bottleneck_channels)
        
        self.conv2 = asynSparseConvolution2D(
            dimension=2, nIn=bottleneck_channels, nOut=bottleneck_channels,
            filter_size=3, use_bias=False, first_layer=False
        )

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = asynSparseConvolution2D(
            dimension=2, nIn=bottleneck_channels, nOut=out_channels, 
            filter_size=1, use_bias=False, first_layer=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv2, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out

class AsynBottleneckWithFixedBatchNorm(AsynBottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(AsynBottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )


class AsynBottleneckWithGN(AsynBottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(AsynBottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


