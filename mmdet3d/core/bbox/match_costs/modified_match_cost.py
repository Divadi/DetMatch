import torch

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.match_costs import FocalLossCost

@MATCH_COST.register_module()
class DoubleSidedFocalLossCost:
    def __init__(self, **kwargs):
        self.focal_loss_cost = FocalLossCost(**kwargs)

    def __call__(self, cls_pred_1, cls_pred_2):
        """
        Idea is to argmax cls_pred_1 to get label, compute loss vs cls_pred_2,
        and vice versa.
        Args:
            cls_pred_1 (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            cls_pred_2: same as 1
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        assert cls_pred_1.shape[1] == cls_pred_2.shape[1]

        cls_pred_1_sigmoid = cls_pred_1.sigmoid()
        cls_pred_2_sigmoid = cls_pred_2.sigmoid()

        cls_label_1 = cls_pred_1_sigmoid.argmax(dim=1)
        cls_label_2 = cls_pred_2_sigmoid.argmax(dim=1)

        return (self.focal_loss_cost(cls_pred_1, cls_label_2) +
                self.focal_loss_cost(cls_pred_2, cls_label_1).t()) / 2