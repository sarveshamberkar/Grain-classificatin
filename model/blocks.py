from functools import partial

import torch
from torch import nn

from .commons import ConvNormAct, HSwish, Swish, HSigmoid, ChannelShuffle

DepthWiseConv3x3 = partial(ConvNormAct, kernel_size=3, padding=1, bias=False)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3, bias=False, dilation=1, padding=1)
Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv1X1ReLU = partial(ConvNormAct, kernel_size=1)


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None
    ):
        super(ResNetBlock, self).__init__()
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
        self.down_sample = Conv1X1BnReLU(inplanes, planes, stride=1, padding=0)
        self.stride = stride

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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    approx_sigmoid : bool, default False
        Whether to use approximated sigmoid function.
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    """
    activations = {"relu": nn.ReLU, "relu6": nn.ReLU6, "hswish": HSwish(),
                   "hsigmoid": HSigmoid, "swish": Swish}

    def __init__(self,
                 channels,
                 reduction=16,
                 approx_sigmoid=False,
                 activation='relu'):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = Conv1X1ReLU(
            in_features=channels,
            out_features=mid_cannels,
            stride=1,
            padding=1,
            bias=True)
        self.activ = self.activations[activation]()
        self.conv2 = Conv1X1ReLU(
            in_features=mid_cannels,
            out_features=channels,
            stride=1,
            padding=0,
            bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.sigmoid = HSigmoid() if approx_sigmoid else nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.pool1(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class ShuffleUnit(nn.Module):
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
                 downsample=False,
                 ignore_group=True):
        super(ShuffleUnit, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels

        self.compress_conv1 = Conv1X1BnReLU(
            in_features=in_channels,
            out_features=mid_channels,
            stride=1)
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
            stride=1)
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
    a = ResNetBlock(32, 64)
    b = a(torch.randn(1, 32, 224, 224))
    print(b.shape)
