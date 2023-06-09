"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch

from ...utils import common_utils
from . import iou3d_nms_cuda


def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    # keep = torch.zeros(boxes.size(0), dtype=torch.long)
    # print(boxes, keep, thresh, boxes.sum(), boxes.shape)
    # # if boxes.shape[0] == 1024:
    # #     import pickle; pickle.dump((boxes, keep, thresh), open("/home/msc-auto/src/new_code/DetMatch/outputs/nms_good.pkl", "wb+"))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    # print(keep[:num_out])
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None



# from mmdet3d.ops.iou3d import iou3d_cuda


# def xywhr2xyxyr(boxes_xywhr):
#     """Convert a rotated boxes in XYWHR format to XYXYR format.

#     Args:
#         boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

#     Returns:
#         torch.Tensor: Converted boxes in XYXYR format.
#     """
#     boxes = torch.zeros_like(boxes_xywhr)
#     half_w = boxes_xywhr[:, 2] / 2
#     half_h = boxes_xywhr[:, 3] / 2

#     boxes[:, 0] = boxes_xywhr[:, 0] - half_w
#     boxes[:, 1] = boxes_xywhr[:, 1] - half_h
#     boxes[:, 2] = boxes_xywhr[:, 0] + half_w
#     boxes[:, 3] = boxes_xywhr[:, 1] + half_h
#     boxes[:, 4] = boxes_xywhr[:, 4]
#     return boxes

# def nms_bev_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None, **kwargs):
#     """Nms function with gpu implementation.

#     Args:
#         boxes (torch.Tensor): Input boxes with the shape of [N, 5]
#             ([x1, y1, x2, y2, ry]).
#         scores (torch.Tensor): Scores of boxes with the shape of [N].
#         thresh (int): Threshold.
#         pre_maxsize (int): Max size of boxes before nms. Default: None.
#         post_maxsize (int): Max size of boxes after nms. Default: None.

#     Returns:
#         torch.Tensor: Indexes after nms.
#     """
#     boxes = xywhr2xyxyr(boxes[:, [0, 1, 3, 4, 6]])

#     order = scores.sort(0, descending=True)[1]

#     if pre_maxsize is not None:
#         order = order[:pre_maxsize]
#     boxes = boxes[order].contiguous()

#     keep = torch.zeros(boxes.size(0), dtype=torch.long)
#     num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
#     keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
#     if post_max_size is not None:
#         keep = keep[:post_max_size]
#     return keep, None