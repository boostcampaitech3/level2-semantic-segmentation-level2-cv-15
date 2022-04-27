import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
from torch.autograd import Variable

class FocalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.FL = smp.losses.FocalLoss(mode = "multiclass")

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0
            weights = [1, 0.2]
            assert len(weights) == len(pred)
            for i in range(len(pred)):
                loss += self.FL(pred[i], target) * weights[i]
            return loss

        else:
            return self.FL(pred, target)

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.DL = smp.losses.DiceLoss(mode = "multiclass")
    
    def forward(self, pred, target):
        return self.DL(pred, target)

_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'dice' : DiceLoss
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion