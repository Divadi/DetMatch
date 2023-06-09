import numpy as np
import torch

from mmdet.core.bbox import bbox_xyxy_to_cxcywh, build_assigner
from ...builder import SSL_MODULES
from ..bbox_utils import (apply_3d_transformation_bboxes, bbox_3d_to_bbox_2d,
                          filter_by_nms)
from ..utils import mlvl_get, mlvl_getattr, mlvl_set


@SSL_MODULES.register_module
class BboxesTransform_3D():
    """Takes 3D bboxes and transforms them either forward or backward based on
    3D augs in passed img_metas.

    forward - akin to applying the augmentations directly
    """
    def __init__(self,
                 reverse,
                 img_metas,
                 in_bboxes_key,
                 out_bboxes_key):

        self.reverse = reverse
        self.img_metas = img_metas
        self.in_bboxes_key = in_bboxes_key
        self.out_bboxes_key = out_bboxes_key

    def forward(self, ssl_obj, batch_dict):
        img_metas = mlvl_get(batch_dict, self.img_metas)
        in_bboxes = mlvl_get(batch_dict, self.in_bboxes_key)

        out_bboxes = []
        for batch_idx, (curr_in_bboxes, img_meta) in \
                enumerate(zip(in_bboxes, img_metas)):

            if isinstance(curr_in_bboxes, tuple):
                curr_in_bboxes_3d = curr_in_bboxes[0]
                rest = curr_in_bboxes[1:]
            else:
                assert isinstance(curr_in_bboxes, torch.Tensor)
                curr_in_bboxes_3d = curr_in_bboxes

            curr_in_bboxes_3d = apply_3d_transformation_bboxes(
                curr_in_bboxes_3d,
                img_meta,
                reverse=self.reverse)

            if isinstance(curr_in_bboxes, tuple):
                out_bboxes.append((curr_in_bboxes_3d, *rest))
            else:
                out_bboxes.append(curr_in_bboxes_3d)

        mlvl_set(batch_dict, self.out_bboxes_key, out_bboxes)

        return batch_dict


@SSL_MODULES.register_module
class DetachBboxes():
    def __init__(self,
                 in_bboxes_key,
                 out_bboxes_key):

        self.in_bboxes_key = in_bboxes_key
        self.out_bboxes_key = out_bboxes_key

    def forward(self, ssl_obj, batch_dict):
        in_bboxes = mlvl_get(batch_dict, self.in_bboxes_key)

        out_bboxes = []
        for batch_idx, curr_in_bboxes in enumerate(in_bboxes):
            curr_out_bboxes = tuple([tmp.detach() for tmp in curr_in_bboxes])
            out_bboxes.append(curr_out_bboxes)

        mlvl_set(batch_dict, self.out_bboxes_key, out_bboxes)

        return batch_dict


@SSL_MODULES.register_module
class Bboxes3DTo2D():
    """reverses img_metas and gets 2D projected bboxes."""
    def __init__(self,
                 img_metas='stu.img_metas',
                 in_bboxes_key='stu.3d_bboxes_nms',
                 out_bboxes_key='stu.3d_bboxes_nms_2d_proj',
                 filter_invalid=True):

        self.img_metas = img_metas
        self.in_bboxes_key = in_bboxes_key
        self.out_bboxes_key = out_bboxes_key
        self.filter_invalid = filter_invalid

    def forward(self, ssl_obj, batch_dict):
        img_metas = mlvl_get(batch_dict, self.img_metas)
        in_bboxes = mlvl_get(batch_dict, self.in_bboxes_key)

        out_bboxes = []
        for batch_idx, (curr_in_bboxes) in enumerate(in_bboxes):

            if isinstance(curr_in_bboxes, tuple):
                curr_in_bboxes_3d = curr_in_bboxes[0]
                rest = curr_in_bboxes[1:]
            else:
                curr_in_bboxes_3d = curr_in_bboxes
                curr_in_bboxes_3d.tensor = curr_in_bboxes_3d.tensor.cuda()

            # curr_in_bboxes_3d = apply_3d_transformation_bboxes(
            #     curr_in_bboxes_3d,
            #     img_meta,
            #     reverse=self.reverse)

            # if isinstance(curr_in_bboxes, tuple):
            #     out_bboxes.append((curr_in_bboxes_3d, *rest))
            # else:
            #     out_bboxes.append(curr_in_bboxes_3d)

            # Reverse augs
            curr_in_bboxes_3d = apply_3d_transformation_bboxes(
                curr_in_bboxes_3d, img_metas[batch_idx], reverse=True)

            # Project to 2D
            # When img scaling is used, additional thought is required.
            # Commented out 8/14 9PM
            # assert img_metas[batch_idx]['ori_shape'] == \
            #        img_metas[batch_idx]['img_shape']
            curr_bboxes_2d, curr_bboxes_2d_valid = bbox_3d_to_bbox_2d(
                curr_in_bboxes_3d, img_metas[batch_idx]['lidar2img'],
                img_metas[batch_idx]['ori_shape'])

            if isinstance(curr_in_bboxes, tuple):
                if self.filter_invalid:
                    filtered_rest = [tmp[curr_bboxes_2d_valid] for tmp in rest]
                    out_bboxes.append(
                        (curr_bboxes_2d[curr_bboxes_2d_valid], *filtered_rest))
                else:
                    out_bboxes.append((curr_bboxes_2d, *rest))
            else:
                if self.filter_invalid:
                    out_bboxes.append(curr_bboxes_2d[curr_bboxes_2d_valid])
                else:
                    out_bboxes.append(curr_bboxes_2d)

            # if self.filter_invalid:
            #     curr_bboxes_2d = curr_bboxes_2d[curr_bboxes_2d_valid]
            #     curr_scores_2d = curr_scores[curr_bboxes_2d_valid]
            # else:
            #     curr_bboxes_2d = curr_bboxes_2d
            #     curr_scores_2d = curr_scores

            # out_bboxes.append((curr_bboxes_2d, curr_scores_2d))

        mlvl_set(batch_dict, self.out_bboxes_key, out_bboxes)

        return batch_dict