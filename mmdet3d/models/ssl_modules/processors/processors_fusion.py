import torch

from mmdet.core.bbox import bbox_xyxy_to_cxcywh, build_assigner
from ...builder import SSL_MODULES
from ..bbox_utils import bbox_3d_to_bbox_2d
from ..utils import mlvl_get, mlvl_set


@SSL_MODULES.register_module
class MaxScoreFilter():
    """Simple class that filters bbox list based on max score."""
    def __init__(self,
                 cls_includes_bg_pred,
                 score_thr,
                 in_bboxes_key,
                 out_bboxes_key):

        self.cls_includes_bg_pred = cls_includes_bg_pred
        self.score_thr = score_thr
        self.in_bboxes_key = in_bboxes_key
        self.out_bboxes_key = out_bboxes_key

    def forward(self, ssl_obj, batch_dict):
        """in_bboxes is list[tuple], filter based on max of second element of
        tuple, which should be N x num_classes or N x num_classes + 1."""
        in_bboxes = mlvl_get(batch_dict, self.in_bboxes_key)

        out_bboxes = []
        for batch_idx, curr_in_bboxes in enumerate(in_bboxes):
            if self.cls_includes_bg_pred:
                scores = curr_in_bboxes[1][:, :-1]
            else:
                scores = curr_in_bboxes[1]

            if len(scores) == 0:
                score_mask = scores.new_tensor([]) > self.score_thr
            else:
                max_score = scores.max(dim=1)[0]
                score_mask = max_score > self.score_thr

            curr_out_bboxes = tuple(
                [tmp[score_mask] for tmp in curr_in_bboxes])

            out_bboxes.append(curr_out_bboxes)

        mlvl_set(batch_dict, self.out_bboxes_key, out_bboxes)

        return batch_dict


@SSL_MODULES.register_module
class FusionHungarianMatching():
    """
    TODO: Remember! This expects sigmoid only, for both 2D and 3D

    Before using this, both 3D and 2D bboxes should be "clean"
        This means they should be in coordinate/transforms of a sample without
        any augmentations.
    This is necessary for 3D->2D projection matching 2D bboxes and for
        using ori_shape for normalization.

    cost_thr: Threshold for matching cost to get valid matches.
    img_metas: Purely used to project the 3D bboxes to 2D.
        Reversing transforms and whatnot should be done outside.
    cls_includes_bg_pred_{3d, 2d}: whether the class scores for 3d or 2d
        includes bg at the end.
    in_bboxes_{3d, 2d}_key: boxes to match
    out_bboxes_{3d, 2d}_key: matched boxes only.
        They should be of same length, and should be matching order (so
        1st 3D bbox matched with 1st 2D bbox.)
    """
    def __init__(self,
                 assigner_cfg,
                 cost_thr,
                 img_metas,
                 cls_includes_bg_pred_3d,
                 cls_includes_bg_pred_2d,
                 in_bboxes_3d_key,
                 in_bboxes_2d_key,
                 out_bboxes_3d_key,
                 out_bboxes_2d_key,
                 match_cost_key=None,
                 project_3d_to_2d=True): # added 1/29

        self.assigner = build_assigner(assigner_cfg)
        self.cost_thr = cost_thr
        self.img_metas = img_metas
        self.cls_includes_bg_pred_3d = cls_includes_bg_pred_3d
        self.cls_includes_bg_pred_2d = cls_includes_bg_pred_2d
        self.in_bboxes_3d_key = in_bboxes_3d_key
        self.in_bboxes_2d_key = in_bboxes_2d_key
        self.out_bboxes_3d_key = out_bboxes_3d_key
        self.out_bboxes_2d_key = out_bboxes_2d_key
        self.match_cost_key = match_cost_key
        self.project_3d_to_2d = project_3d_to_2d

    def forward(self, ssl_obj, batch_dict):
        img_metas = mlvl_get(batch_dict, self.img_metas)
        in_bboxes_3d = mlvl_get(batch_dict, self.in_bboxes_3d_key)
        in_bboxes_2d = mlvl_get(batch_dict, self.in_bboxes_2d_key)

        out_bboxes_3d = []
        out_bboxes_2d = []
        match_cost = []
        for batch_idx, (curr_in_3d, curr_in_2d) in \
                enumerate(zip(in_bboxes_3d, in_bboxes_2d)):

            ### First, in_bboxes_{3d, 2d} are list[tuple] with first two
            ### elements of tuple being bbox & scores respectively. However,
            ### there can be other stuff after which we also want to filter.
            curr_in_bboxes_3d = curr_in_3d[0]
            curr_in_scores_3d = curr_in_3d[1]

            curr_in_bboxes_2d = curr_in_2d[0]
            curr_in_scores_2d = curr_in_2d[1]

            ### Considering cls_includes_bg_pred, shave off background score
            if self.cls_includes_bg_pred_3d:
                curr_in_scores_3d = curr_in_scores_3d[:, :-1]
            if self.cls_includes_bg_pred_2d:
                curr_in_scores_2d = curr_in_scores_2d[:, :-1]
            assert curr_in_scores_3d.shape[1] == curr_in_scores_2d.shape[1]

            ### Project 3D bboxes to 2D
            ### TODO: This completely tosses out "valid".
            if self.project_3d_to_2d:
                curr_in_bboxes_3d_proj, _ = bbox_3d_to_bbox_2d(
                    curr_in_bboxes_3d, img_metas[batch_idx]['lidar2img'],
                    img_metas[batch_idx]['ori_shape'])
            else:
                # Added 1/29 because maybe we also want to feed in 2D predicted boxes
                # directly (because want to do 2D NMS, and match the results)
                curr_in_bboxes_3d_proj = curr_in_bboxes_3d

            ### Do Hungarian Assignment
            ## Because I'm retrofitting GT hungarian assignment class, it
            ## expects normalized (cx, cy, w, h) coords for preds, and
            ## unnormalized xyxy for "gt".
            ## Projected 3D will act like preds, Original 2D will act like GT.
            ## So, normalize & re-represent projected 3D.
            img_h, img_w, _ = img_metas[batch_idx]['ori_shape']
            factor = curr_in_bboxes_2d.new_tensor([img_w, img_h, img_w,
                                                   img_h]).unsqueeze(0)
            curr_in_bboxes_3d_proj_norm = \
                bbox_xyxy_to_cxcywh(curr_in_bboxes_3d_proj) / factor

            ## Hungarian (DoubleSidedFocalLossCost) expects logits for both
            # Sanity check that scores provided are indeed prob distributions
            if curr_in_scores_3d.numel() != 0:
                assert (curr_in_scores_3d.min() >= 0 and
                        curr_in_scores_3d.max() <= 1)
            if curr_in_scores_2d.numel() != 0:
                assert (curr_in_scores_2d.min() >= 0 and
                        curr_in_scores_2d.max() <= 1)

            # Convert back to logits - this assumes sigmoid.
            curr_in_logits_3d = torch.logit(curr_in_scores_3d, eps=1e-6)
            curr_in_logits_2d = torch.logit(curr_in_scores_2d, eps=1e-6)

            ## Assign
            assign_results = \
                self.assigner.assign(curr_in_bboxes_3d_proj_norm,
                                     curr_in_logits_3d,
                                     curr_in_bboxes_2d,
                                     curr_in_logits_2d,
                                     dict(img_shape=img_metas
                                          [batch_idx]['ori_shape']))

            ## Optionally, also filter by cost.
            if self.cost_thr is not None and \
                    assign_results.max_overlaps is not None:
                fg_mask = assign_results.gt_inds > 0
                cost_mask = \
                    assign_results.max_overlaps[fg_mask] > self.cost_thr
                assign_results.gt_inds[
                    fg_mask.nonzero(as_tuple=True)[0][cost_mask]] = 0

            ### Get matched 3D and 2D
            mask_3d = assign_results.gt_inds > 0  # 0 means unmatched
            mask_2d = assign_results.gt_inds[mask_3d] - 1  # one-indexed
            # print('mask_2d', mask_2d)

            # curr_in_bboxes_3d_proj_matched = curr_in_bboxes_3d_proj[mask_3d]
            # curr_in_bboxes_2d_matched = curr_in_bboxes_2d[mask_2d]
            # true_ious = bbox_overlaps(curr_in_bboxes_3d_proj_matched,
            #                           curr_in_bboxes_2d_matched,
            #                           mode='iou',
            #                           is_aligned=True)
            # non_aligned_ious = bbox_overlaps(curr_in_bboxes_3d_proj_matched,
            #                           curr_in_bboxes_2d_matched,
            #                           mode='iou',
            #                           is_aligned=False)

            curr_out_3d = [tmp[mask_3d] for tmp in curr_in_3d]
            curr_out_2d = [tmp[mask_2d] for tmp in curr_in_2d]

            # print(img_metas[batch_idx]['sample_idx'])
            # print('curr_in_bboxes_3d_proj', curr_in_bboxes_3d_proj[mask_3d])
            # print('curr_out_3d', curr_out_3d)
            # print('curr_out_2d', curr_out_2d)
            # print('max_overlaps', assign_results.max_overlaps[mask_3d])
            # print('iou_cost', assign_results.iou_cost[mask_3d])
            # print('true_ious', true_ious)
            # print('# of true gt', batch_dict['stu']['gt_labels'][batch_idx])
            # print('non_aligned_ious', non_aligned_ious)

            # if int(img_metas[batch_idx]['sample_idx']) == 6310:
            #     assert False

            out_bboxes_3d.append(tuple(curr_out_3d))
            out_bboxes_2d.append(tuple(curr_out_2d))
            if assign_results.max_overlaps is not None:
                match_cost.append(assign_results.max_overlaps[mask_3d])
            else:
                assert mask_3d.sum() == 0
                match_cost.append(mask_3d.float())

        mlvl_set(batch_dict, self.out_bboxes_3d_key, out_bboxes_3d)
        mlvl_set(batch_dict, self.out_bboxes_2d_key, out_bboxes_2d)
        if self.match_cost_key is not None:
            mlvl_set(batch_dict, self.match_cost_key, match_cost)

        return batch_dict