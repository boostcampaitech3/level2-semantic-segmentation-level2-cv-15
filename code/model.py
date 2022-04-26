import torchvision
import torch.nn as nn
from importlib import import_module


def load_model(model_type, n_classes):

    if model_type not in dir(torchvision.models.segmentation):
        raise Exception(f'No model named {model_type}')
    
    opt_module = getattr(import_module("torchvision.models.segmentation"), model_type)
    model = opt_module(pretrained=True)
    
    in_channels = model.classifier[-1].in_channels
    out_channels = n_classes
    kernel_size = model.classifier[-1].kernel_size
    stride = model.classifier[-1].stride

    model.classifier[-1] = nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride)

    return model