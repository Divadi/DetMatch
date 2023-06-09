import torch
from torch.nn import functional as F

from mmdet.core.bbox import build_assigner
from mmdet.models import FocalLoss, MSELoss, build_loss
from mmseg.core import add_prefix
from ...builder import SSL_MODULES
from ..bbox_utils import filter_by_nms_2d
from ..utils import mlvl_get, mlvl_getattr

@SSL_MODULES.register_module
class HungarianConsistency():

    def __init__(self,
                 loss_cls_cfg=None,
                 loss_iou_cfg=None,
                 loss_l1_cfg=None,
                 loss_weights_cfg=dict(),
                 in_bboxes_key='stu.3d_bboxes_nms_2d_proj',
                 target_bboxes_key='tea.2d_bboxes_nms_stu_aug_hung_dtch_bboxes', # these two should be matched
                 cls_includes_bg_pred_in=True,
                 cls_includes_bg_pred_target=True,
                 target_img_metas_key=None,
                 name=None):
        
        self.loss_cls_cfg = loss_cls_cfg
        
        self.loss_cls = (build_loss(loss_cls_cfg)
                         if loss_cls_cfg is not None else None)
        self.loss_iou = (build_loss(loss_iou_cfg)
                         if loss_iou_cfg is not None else None)
        self.loss_l1 = (build_loss(loss_l1_cfg)
                        if loss_l1_cfg is not None else None)
        self.loss_weights_cfg = loss_weights_cfg
        self.in_bboxes_key = in_bboxes_key
        self.target_bboxes_key = target_bboxes_key
        self.cls_includes_bg_pred_in= cls_includes_bg_pred_in
        self.cls_includes_bg_pred_target = cls_includes_bg_pred_target
        self.target_img_metas_key = target_img_metas_key
        self.name = name

    def forward(self, ssl_obj, batch_dict):
        in_bboxes = mlvl_get(batch_dict, self.in_bboxes_key)
        target_bboxes = mlvl_get(batch_dict, self.target_bboxes_key)
        target_img_metas = mlvl_get(batch_dict, self.target_img_metas_key)

        losses = dict()
        for batch_idx, (curr_in, curr_target) in \
                enumerate(zip(in_bboxes, target_bboxes)):

            # TODO: Can add additional features here
            curr_in_bboxes = curr_in[0]
            curr_in_scores = curr_in[1]

            curr_target_bboxes = curr_target[0]
            curr_target_scores = curr_target[1]

            if len(curr_in_scores) == 0 or len(curr_target_scores) == 0:
                continue

            # Consider BG element
            if self.cls_includes_bg_pred_in:
                curr_in_scores = curr_in_scores[:, :-1]
            if self.cls_includes_bg_pred_target:
                curr_target_scores = curr_target_scores[:, :-1]
            assert curr_in_scores.shape[1] == curr_target_scores.shape[1] == 3 # hardcoded to 3 classes


            ### Compute Cls Loss
            if self.loss_cls is not None:
                if 'cls_loss' not in losses:
                    losses['cls_loss'] = []
                
                if isinstance(self.loss_cls, MSELoss): # MSE loss of sigmoided probabilities
                    losses['cls_loss'].append(self.loss_cls(curr_in_scores, curr_target_scores))
                elif isinstance(self.loss_cls, FocalLoss):
                    curr_in_logits = torch.logit(curr_in_scores, eps=1e-6)
                    curr_target_labels = torch.argmax(curr_target_scores, dim=1)
                    losses['cls_loss'].append(self.loss_cls(curr_in_logits, curr_target_labels))
                else:
                    raise Exception("Not Yet Implemented")
            
            ### Compute L1 Loss
            if self.loss_l1 is not None:
                if 'l1_loss' not in losses:
                    losses['l1_loss'] = []
                
                img_h, img_w, _ = target_img_metas[batch_idx]['img_shape']
                factor = curr_in_bboxes.new_tensor([img_w, img_h, img_w,
                                                    img_h]).unsqueeze(0)
                loss = self.loss_l1(curr_in_bboxes / factor, curr_target_bboxes / factor)
                losses['l1_loss'].append(loss)
            
            ### Compute gIoU Loss
            if self.loss_iou is not None:
                if 'iou_loss' not in losses:
                    losses['iou_loss'] = []

                loss = self.loss_iou(curr_in_bboxes, curr_target_bboxes)
                losses['iou_loss'].append(loss)

        for k, v in list(losses.items()):
            if len(losses[k]) == 0:
                losses[k] = in_bboxes[0][0].new_tensor(0.0)
            else:
                losses[k] = sum(losses[k]) / len(losses[k])

        losses = ssl_obj._collapse_losses(losses)
        for k, w in self.loss_weights_cfg.items():
            if k in losses.keys():
                losses[k] = losses[k] * w
            else:
                losses[k] = in_bboxes[0][0].new_tensor(0.0, dtype=torch.float32, requires_grad=True)
        losses = add_prefix(losses, self.name)
        batch_dict['ssl_losses'] = \
            ssl_obj._sum_update_losses(batch_dict['ssl_losses'], losses)

        return batch_dict