# Adapted from UB Teacher
import torch
from torch import nn
from torch.nn import functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class SoftmaxFocalLoss(nn.Module):
    """
    Note: i don't think this alpha value makes sense - originally was supposed
    to be a foreground weighting thing.
    """
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        reduction='sum',
        loss_weight=1.0
    ):
        super(SoftmaxFocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else 1
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, input, target, weight=None, avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # focal loss
        CE = F.cross_entropy(input, target, reduction='none')
        p = torch.exp(-CE)
        loss = self.alpha * (1 - p) ** self.gamma * CE
        loss = self.loss_weight * loss

        return weight_reduce_loss(
            loss, weight=weight, reduction=reduction,
            avg_factor=avg_factor)