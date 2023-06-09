import torch

from mmseg.core import add_prefix
from ...builder import SSL_MODULES
from ..utils import mlvl_get, mlvl_getattr


@SSL_MODULES.register_module
class TwoStageSupervised_2D():
    """For one-stage 3D model, calculate SSL supervised loss.

    Very similar to OneStageSupervised_3D, except conducts the entire forward
    process.
    """

    def __init__(self,
                 loss_detach_keys=[],
                 ssl_obj_attr='student',
                 batch_dict_key='stu'):

        self.loss_detach_keys = loss_detach_keys
        self.ssl_obj_attr = ssl_obj_attr
        self.batch_dict_key = batch_dict_key

    def forward(self, ssl_obj, batch_dict):
        """Overall, this is directly copied from two_stage forward train."""
        detector = mlvl_getattr(ssl_obj, self.ssl_obj_attr)
        curr_batch_dict = mlvl_get(batch_dict, self.batch_dict_key)

        losses = detector.forward_train(curr_batch_dict['img'],
                                        curr_batch_dict['img_metas'],
                                        curr_batch_dict['gt_bboxes'],
                                        curr_batch_dict['gt_labels'],
                                        curr_batch_dict.get(
                                            'gt_bboxes_ignore', None))

        ### Clean losses
        for k in self.loss_detach_keys:
            losses.pop(k)

        losses = add_prefix(losses, f'{self.batch_dict_key}')

        if 'sup_losses' in batch_dict:
            batch_dict['sup_losses'] = \
                ssl_obj._sum_update_losses(batch_dict['sup_losses'], losses)
        else:
            batch_dict['ssl_losses'] = \
                ssl_obj._sum_update_losses(batch_dict['ssl_losses'], losses)

        return batch_dict


@SSL_MODULES.register_module
class HardPseudoLabel_2D():

    def __init__(self,
                 score_thr,
                 cls_includes_bg_pred,
                 loss_detach_keys=[],
                 ssl_obj_attr='student',
                 target_bboxes_key='tea.2d_bboxes_nms_stu_aug',
                 target_img_key='stu.img',
                 target_img_metas_key='stu.img_metas',
                 name='hard_pseudo_2d',
                 weight=1):

        self.score_thr = score_thr
        self.cls_includes_bg_pred = cls_includes_bg_pred
        self.loss_detach_keys = loss_detach_keys
        self.ssl_obj_attr = ssl_obj_attr
        self.target_bboxes_key = target_bboxes_key
        self.target_img_key = target_img_key
        self.target_img_metas_key = target_img_metas_key
        self.weight = weight

        self.name = name

    def forward(self, ssl_obj, batch_dict):
        detector = mlvl_getattr(ssl_obj, self.ssl_obj_attr)

        ### Generate pseudo label target
        pseudo_labels = []
        pseudo_bboxes = []
        for batch_idx, (curr_bboxes, curr_scores) in \
                enumerate(mlvl_get(batch_dict, self.target_bboxes_key)):

            if len(curr_scores) == 0:
                curr_labels = curr_scores.new_zeros((0, ), dtype=torch.long)
                curr_bboxes = curr_scores.new_zeros((0, 4))
            else:
                if self.cls_includes_bg_pred:
                    curr_max_scores, curr_labels = \
                        curr_scores[:, :-1].max(dim=1)
                else:
                    curr_max_scores, curr_labels = curr_scores.max(dim=1)

                curr_labels = curr_labels[curr_max_scores > self.score_thr]
                curr_bboxes = curr_bboxes[curr_max_scores > self.score_thr]

            pseudo_labels.append(curr_labels)
            pseudo_bboxes.append(curr_bboxes)

        ### Forward train
        losses = detector.forward_train(mlvl_get(batch_dict,
                                                 self.target_img_key),
                                        mlvl_get(batch_dict,
                                                 self.target_img_metas_key),
                                        pseudo_bboxes,
                                        pseudo_labels)

        ### Clean & weight
        losses = ssl_obj._collapse_losses(losses)
        for k in self.loss_detach_keys:
            losses.pop(k)
        for k in losses.keys():
            if 'acc' not in k:
                losses[k] = losses[k] * self.weight
        losses = add_prefix(losses, self.name)
        batch_dict['ssl_losses'] = \
            ssl_obj._sum_update_losses(batch_dict['ssl_losses'], losses)

        return batch_dict