from .blocks import Conv1X1BnReLU, Conv3X3BnReLU, ResNetBlock, SEBlock, ShuffleUnit
from .reduction_blocks import ResNetReductionBlock, ShuffleUnitReduction
from .commons import ConvNormAct, Flatten
from functools import partial

Conv7X7BnReLU = partial(ConvNormAct, kernel_size=7, bias=False, padding=1)
FeatureBlocks = {"resnetblock": ResNetBlock, "seblock": SEBlock, "conv": Conv3X3BnReLU, "shuffleunit": ShuffleUnit}
ReductionBlock = {'shuffle_unit': ShuffleUnitReduction, "resnetblock": ResNetReductionBlock, "conv": Conv3X3BnReLU}


def get_feature_block(name):
    if name == 'resnetblock':
        FeatureBlocks[name]()
    elif name == 'seblock':
        FeatureBlocks[name]()
    elif name == 'conv':
        FeatureBlocks[name]()
    elif name == 'shuffleunit':
        FeatureBlocks[name]()


def get_reduction_block(name):
    if name == 'resnetblock':
        ReductionBlock[name]()
    elif name == 'conv':
        ReductionBlock[name]()
    elif name == 'shuffleunit':
        ReductionBlock[name]()

