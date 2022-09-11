from functools import partial

import torch
from torch import nn

from .commons import ConvNormAct, ChannelShuffle

Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3, bias=False, dilation=1, padding=1)
Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
DepthWiseConv3x3 = partial(ConvNormAct, kernel_size=3, padding=1, bias=False)


class ResNetReductionBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=2, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResNetReductionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv3X3BnReLU(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3X3BnReLU(planes, planes, stride=1)
        self.bn2 = norm_layer(planes)
        self.stride = stride

        if self.stride > 1:
            self.down_sample = Conv3X3BnReLU(inplanes, planes, stride=stride)
        else:
            self.down_sample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ShuffleUnitReduction(nn.Module):
    """
    ShuffleNet unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    groups : int
        Number of groups in convolution layers.
    downsample : bool
        Whether to down sample.
    ignore_group : bool
        Whether ignore group value in the first convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 downsample=True,
                 ignore_group=True):
        super(ShuffleUnitReduction, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            oc = out_channels - in_channels
            out_channels = oc if oc >= 2 else out_channels

        self.compress_conv1 = Conv1X1BnReLU(
            in_features=in_channels,
            out_features=mid_channels,
            stride=1,
            padding=0
        )
        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=groups)
        self.dw_conv2 = DepthWiseConv3x3(
            in_features=mid_channels, out_features=mid_channels,
            stride=(2 if self.downsample else 1))
        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)
        self.expand_conv3 = Conv1X1BnReLU(
            in_features=mid_channels,
            out_features=out_channels,
            stride=1,
            padding=0

        )
        self.expand_bn3 = nn.BatchNorm2d(num_features=out_channels)
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.expand_identity = Conv3X3BnReLU(
                in_features=in_channels,
                out_features=out_channels,
                stride=1)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.compress_conv1(x)
        x = self.compress_bn1(x)
        x = self.activ(x)
        x = self.c_shuffle(x)
        x = self.dw_conv2(x)
        x = self.dw_bn2(x)
        x = self.expand_conv3(x)
        x = self.expand_bn3(x)
        if self.downsample:
            identity = self.avgpool(identity)
            x = torch.cat((x, identity), dim=1)
        else:
            x = x + self.expand_identity(identity)
        x = self.activ(x)
        return x


if __name__ == '__main__':
    a = ResNetReductionBlock(32, 64)
    b = a(torch.randn(1, 32, 224, 224))
    print(b.shape)
