# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from math import exp
import numpy as np
from torch import Tensor
from functools import partial
from torch.nn.modules.loss import _Loss

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from functools import partial
from typing import Optional, List
from math import exp
import numpy as np
from torch import Tensor


# @weighted_loss
# def dice_loss(pred,
#               target,
#               valid_mask,
#               smooth=1,
#               exponent=2,
#               class_weight=None,
#               ignore_index=255):
#     assert pred.shape[0] == target.shape[0]
#     total_loss = 0
#     num_classes = pred.shape[1]
#     for i in range(num_classes):
#         if i != ignore_index:
#             dice_loss = binary_dice_loss(
#                 pred[:, i],
#                 target[..., i],
#                 valid_mask=valid_mask,
#                 smooth=smooth,
#                 exponent=exponent)
#             if class_weight is not None:
#                 dice_loss *= class_weight[i]
#             total_loss += dice_loss
#     return total_loss / num_classes


# @weighted_loss
# def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwards):
#     assert pred.shape[0] == target.shape[0]
#     pred = pred.reshape(pred.shape[0], -1)
#     target = target.reshape(target.shape[0], -1)
#     valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

#     num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
#     den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

#     return 1 - num / den

# def soft_dice_score(
#     output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
# ) -> torch.Tensor:
#     assert output.size() == target.size()
#     if dims is not None:
#         intersection = torch.sum(output * target, dim=dims)
#         cardinality = torch.sum(output + target, dim=dims)
#     else:
#         intersection = torch.sum(output * target)
#         cardinality = torch.sum(output + target)
#     dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
#     return dice_score

# @LOSSES.register_module()
# class DiceLoss(nn.Module):
#     """DiceLoss.

#     This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
#     Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

#     Args:
#         smooth (float): A float number to smooth loss, and avoid NaN error.
#             Default: 1
#         exponent (float): An float number to calculate denominator
#             value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
#         reduction (str, optional): The method used to reduce the loss. Options
#             are "none", "mean" and "sum". This parameter only works when
#             per_image is True. Default: 'mean'.
#         class_weight (list[float] | str, optional): Weight of each class. If in
#             str format, read them from a file. Defaults to None.
#         loss_weight (float, optional): Weight of the loss. Default to 1.0.
#         ignore_index (int | None): The label index to be ignored. Default: 255.
#         loss_name (str, optional): Name of the loss item. If you want this loss
#             item to be included into the backward graph, `loss_` must be the
#             prefix of the name. Defaults to 'loss_dice'.
#     """
 
#     def __init__(self,
#                  smooth=1,
#                  exponent=2,
#                  reduction='mean',
#                  class_weight=None,
#                  loss_weight=1.0,
#                  ignore_index=255,
#                  loss_name='loss_dice',
#                  **kwards):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.exponent = exponent
#         self.reduction = reduction
#         self.class_weight = get_class_weight(class_weight)
#         self.loss_weight = loss_weight
#         self.ignore_index = ignore_index
#         self._loss_name = loss_name

#     def forward(self,
#                 pred,
#                 target,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwards):
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if self.class_weight is not None:
#             class_weight = pred.new_tensor(self.class_weight)
#         else:
#             class_weight = None

#         # pred = F.softmax(pred, dim=1)
#         pred = pred.log_softmax(dim=1).exp()
#         bs = target.size(0)
#         dims = (0, 2)
#         num_classes = pred.shape[1]
        
#         target = target.view(bs, -1)
#         pred = pred.view(bs, num_classes, -1)
        
#         mask = target != self.ignore_index
#         pred = pred * mask.unsqueeze(1)
        
#         target = F.one_hot((target * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
#         target = target.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
        
#         scores = self.loss_weight * soft_dice_score(pred, target.type_as(pred), smooth=self.smooth, eps=1e-7, dims=dims)
        
#         # loss = self.loss_weight * dice_loss(
#         #     pred,
#         #     one_hot_target,
#         #     valid_mask=valid_mask,
#         #     reduction=reduction,
#         #     avg_factor=avg_factor,
#         #     smooth=self.smooth,
#         #     exponent=self.exponent,
#         #     class_weight=class_weight,
#         #     ignore_index=self.ignore_index)
#         return scores

#     @property
#     def loss_name(self):
#         """Loss Name.

#         This function must be implemented and will return the name of this
#         loss function. This name will be used to combine different loss items
#         by simple sum operation. In addition, if you want this loss item to be
#         included into the backward graph, `loss_` must be the prefix of the
#         name.
#         Returns:
#             str: The name of this loss item.
#         """
#         return self._loss_name

   
def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    # print("="*15, "intersection", intersection.shape,"="*15)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    # print("="*15, "dice_score", dice_score.shape,"="*15)
    return dice_score

@LOSSES.register_module()
class DiceLoss(_Loss):

    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        loss_name: str = 'dice_loss'
    ):
        """Implementation of Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error 
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {"binary", "multilabel", "multiclass"}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != "binary", "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight, ignore_index) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == "multiclass":
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        
        # print("="*15,"y_pred :",y_pred.shape,"y_true :",y_true.shape,"="*15)
        
        bs = y_true.size(0)
        num_classes = y_pred.shape[1]
        dims = (0, 2)
        y_true = torch.clamp(y_true.long(), 0, num_classes - 1)
        
        if self.mode == "binary":
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == "multiclass":
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)
                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                # print("="*20,y_true[0].min(),"<y_true<",y_true[0].max(),":",y_true.shape, "num_classes :", num_classes,"="*20)
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == "multilabel":
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        # print("="*15, "scores:", scores,"="*15)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        
        
        mask = y_true.sum(dims) > 0
        # print("="*15,"mask:", mask.shape, "loss :", loss.shape, "="*15)
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    