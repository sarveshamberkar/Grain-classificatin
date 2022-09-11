import torch
from torch import functional as F
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: int,
            stride: int,
            padding: int = 0,
            norm: nn.Module = nn.BatchNorm2d,
            act: nn.Module = nn.ReLU,
            **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                **kwargs
            ),
            norm(out_features),
            act(),
        )


class ConvAct(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: int,
            stride: int,
            padding: int = 0,
            act: nn.Module = nn.ReLU,
            **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                **kwargs
            ),
            act(),
        )


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """

    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Swish(nn.Module):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


def channel_shuffle(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.
    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.
    Returns
    -------
    Tensor
        Resulted tensor.
    """
    batch, channels, height, width = x.size()
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """

    def __init__(self,
                 channels,
                 groups):
        super(ChannelShuffle, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)
