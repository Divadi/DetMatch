import torch
from collections import defaultdict

# from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ...builder import SSL_MODULES
from ..utils import mlvl_get


@SSL_MODULES.register_module
class NumPreds():

    def __init__(self, bboxes_key, out_name):
        self.bboxes_key = bboxes_key
        self.out_name = out_name

    def forward(self, ssl_obj, batch_dict):
        bboxes = mlvl_get(batch_dict, self.bboxes_key)

        num = sum([s[0].shape[0] if isinstance(s, tuple)
                   else s.shape[0] for s in bboxes]) / len(bboxes)
        batch_dict['ssl_losses']['metrics.' + self.out_name] = \
            torch.tensor(num, device=bboxes[0][0].device, dtype=torch.float)

        return batch_dict
