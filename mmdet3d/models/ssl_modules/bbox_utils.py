import torch
from functools import partial

from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.core.bbox import LiDARInstance3DBoxes
# from mmdet.core import multiclass_nms

import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


def modified_multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False,
                   fixed_return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.
        fixed_return_inds: Added by david. If using score_thr, shouldn't
            keep returned be w.r.t tensors before score_thr?

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        elif fixed_return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], keep
    elif fixed_return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]

def apply_3d_transformation_bboxes(bbox, img_meta, reverse=False):
    """This version is very closely modeled off of apply_3d_transformation from
    coord_transform.py.

    Args:
        bbox (some mmdet 3D box type): The bboxes to be transformed.
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not.

    Note:
        The elements in img_meta['transformation_3d_flow']:
        "T" stands for translation;
        "S" stands for scale;
        "R" stands for rotation;
        "HF" stands for horizontal flip;
        "VF" stands for vertical flip.

    Returns:
        torch.Tensor: The transformed point cloud.
    """
    assert isinstance(img_meta, dict)

    dtype = bbox.tensor.dtype
    device = bbox.tensor.device

    bbox_rotate_mat = (
        torch.tensor(img_meta['pcd_rotation'], dtype=dtype, device=device)
        if 'pcd_rotation' in img_meta else torch.eye(
            3, dtype=dtype, device=device))
    # bbox_rotate_scalar = (
    #     img_meta['pcd_rotation_scalar']
    #     if 'pcd_rotation' in img_meta else 0.)

    bbox_scale_factor = (
        img_meta['pcd_scale_factor'] if 'pcd_scale_factor' in img_meta else 1.)

    bbox_trans_factor = (
        torch.tensor(img_meta['pcd_trans'], dtype=dtype, device=device)
        if 'pcd_trans' in img_meta else torch.zeros(
            (3), dtype=dtype, device=device))

    bbox_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
        img_meta else False

    bbox_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
        img_meta else False

    flow = img_meta['transformation_3d_flow'] \
        if 'transformation_3d_flow' in img_meta else []

    bbox = bbox.clone()  # prevent inplace modification

    horizontal_flip_func = partial(bbox.flip, bev_direction='horizontal') \
        if bbox_horizontal_flip else lambda: None
    vertical_flip_func = partial(bbox.flip, bev_direction='vertical') \
        if bbox_vertical_flip else lambda: None
    if reverse:
        scale_func = partial(bbox.scale, scale_factor=1.0 / bbox_scale_factor)
        translate_func = partial(
            bbox.translate, trans_vector=-bbox_trans_factor)
        # bbox_rotate_mat @ bbox_rotate_mat.inverse() is not
        # exactly an identity matrix
        # use angle to create the inverse rot matrix neither.
        # rotate_func = partial(bbox.rotate, angle=-bbox_rotate_scalar)
        rotate_func = partial(bbox.rotate, angle=bbox_rotate_mat.inverse())

        # reverse the pipeline
        flow = flow[::-1]
    else:
        scale_func = partial(bbox.scale, scale_factor=bbox_scale_factor)
        translate_func = partial(
            bbox.translate, trans_vector=bbox_trans_factor)
        # rotate_func = partial(bbox.rotate, angle=bbox_rotate_scalar)
        rotate_func = partial(bbox.rotate, angle=bbox_rotate_mat)

    flow_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'HF': horizontal_flip_func,
        'VF': vertical_flip_func
    }
    for op in flow:
        assert op in flow_mapping, f'This 3D data '\
            f'transformation op ({op}) is not supported'
        func = flow_mapping[op]
        func()

    return bbox


def filter_by_nms(raw_bbox_list,
                  nms_cfg,
                  use_sigmoid_cls,
                  return_labels=False):
    """
    Args:
        raw_bbox_list: a list of bbox, scores tuples. bbox is a box class,
            scores is # of boxes x num_classes, already sigmoid or softmaxed.
        return_labels:
            multi-class NMS != argmax the resulting scores vector. This returns
                the specific label as well.
    Returns:
        res_bbox_list: a list of tuples of:
            bbox_pred_selected: bbox class of bboxes after nms
            scores_selected: prob distribution of selected bboxes

    """
    res_bbox_list = []
    for batch_idx, (bboxes, scores) in enumerate(raw_bbox_list):
        assert bboxes.tensor.shape[0] == scores.shape[0], \
            '{} vs {}'.format(bboxes.tensor.shape[0], scores.shape[0])
        assert len(scores.shape) == 2

        box_type_3d = bboxes.__class__
        bbox_pred = bboxes.tensor

        ### NMS Pre
        nms_pre = nms_cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            if use_sigmoid_cls:
                max_scores, _ = scores.max(dim=1)
            else:
                max_scores, _ = scores[:, :-1].max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]

        ### Prepare for NMS
        bbox_pred_for_nms = xywhr2xyxyr(
            box_type_3d(bbox_pred, box_dim=bbox_pred.shape[-1]).bev)

        if use_sigmoid_cls:
            # Add a dummy background class to the end when using sigmoid
            padding = scores.new_zeros(scores.shape[0], 1)
            scores_for_nms = torch.cat([scores, padding], dim=1)

        ## The NMS function does not return a list of selected indices.
        ## So, kinda hack it by sending in a list of indices as attr.
        indices = torch.arange(
            len(bbox_pred), dtype=torch.float32, device=bbox_pred.device)

        ### Run NMS
        score_thr = nms_cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(
            bbox_pred,
            bbox_pred_for_nms,
            scores_for_nms,
            score_thr,
            nms_cfg.max_num,
            nms_cfg,
            mlvl_attr_scores=indices)
        bbox_pred_selected, _, bbox_labels_selected, selected_indices = results
        selected_indices = selected_indices.long()

        assert torch.allclose(bbox_pred_selected, bbox_pred[selected_indices])
        scores_selected = scores[selected_indices]

        bbox_pred_selected = box_type_3d(
            bbox_pred_selected, box_dim=bbox_pred_selected.shape[-1])

        if not return_labels:
            res_bbox_list.append((bbox_pred_selected, scores_selected))
        else:
            res_bbox_list.append((
                bbox_pred_selected, scores_selected, bbox_labels_selected))

    return res_bbox_list


def filter_by_nms_2d(bbox_list,
                     nms_cfg,
                     use_sigmoid_cls,
                     return_indices=False):
    """
    Args:
        bbox_list - a list of bbox, scores tuples. bbox is Nx4,
            scores is # of boxes x num_classes, already sigmoid or softmaxed.
    """
    assert not return_indices, 'May not return something expected, as' \
        'as it is, return_indices returns indices into post nms_pre boxes.' \
        'Further, if bbox_list has separate boxes for each class, does not' \
        'work as intended'

    res_bbox_list = []
    for batch_idx, (bboxes, scores) in enumerate(bbox_list):
        assert bboxes.shape[0] == scores.shape[0], \
            '{} vs {}'.format(bboxes.shape[0], scores.shape[0])
        assert len(scores.shape) == 2

        ### NMS Pre
        nms_pre = nms_cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            if use_sigmoid_cls:
                max_scores, _ = scores.max(dim=1)
            else:
                max_scores, _ = scores[:, :-1].max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            bboxes = bboxes[topk_inds, :]
            scores = scores[topk_inds, :]

        ### Prepare for NMS
        if use_sigmoid_cls:
            # Add a dummy background class to the end when using sigmoid
            padding = scores.new_zeros(scores.shape[0], 1)
            scores_for_nms = torch.cat([scores, padding], dim=1)
        else:
            scores_for_nms = scores

        ## 2D multiclass_nms has an explicit "return_indices" option

        ### Run NMS
        ## batched_nms for 2D does not want nms_pre or score_thr as part of
        ## nms_cfg (they have a kwargs), so pop it and put it back later
        score_thr = nms_cfg.pop('score_thr', None)
        nms_pre = nms_cfg.pop('nms_pre', None)
        results = modified_multiclass_nms(
            bboxes,
            scores_for_nms,
            score_thr if score_thr is not None else 0,
            nms_cfg,
            nms_cfg.max_num,
            fixed_return_inds=True)
        if score_thr is not None:
            nms_cfg['score_thr'] = score_thr
        if nms_pre is not None:
            nms_cfg['nms_pre'] = nms_pre
        bboxes_selected, _, selected_indices = results
        bboxes_selected = bboxes_selected[:, :4]  # comes attached with score
        selected_indices = selected_indices.long()
        ## Selected indices here is not what is normally expected.
        ## intuitively, it represents indexing into a flattened score tensor,
        ## flattening the N and num_class dimensions, EXCLUDING the background
        ## What we want is an indexing into just the N dimension.
        selected_indices_class_agnostic = \
            selected_indices // (scores_for_nms.shape[1] - 1)

        if bboxes.shape[1] == 4:
            # Class agnostic boxes
            assert torch.allclose(bboxes_selected,
                                  bboxes[selected_indices_class_agnostic])
        else:
            # Per-class boxes
            assert torch.allclose(bboxes_selected,
                                  bboxes.reshape(-1, 4)[selected_indices])

        scores_selected = scores[selected_indices_class_agnostic]

        if not return_indices:
            res_bbox_list.append((bboxes_selected, scores_selected))
        else:
            assert False
            res_bbox_list.append(
                (bboxes_selected,
                 scores_selected,
                 selected_indices_class_agnostic))

    return res_bbox_list


def bbox_3d_to_bbox_2d(bboxes_3d, lidar2img, img_shape):
    """
    Args:
        bboxes_3d: LiDARInstance3DBoxes
        lidar2img: 4x4 kitti_dataset-esque lidar2img
        img_shape:
    Returns:
        xyxy: num_bboxes x 4,
        final_valid_mask: num_bboxes bool tensor, true where valid bboxes 2d
    """
    assert isinstance(bboxes_3d, LiDARInstance3DBoxes)
    assert lidar2img.shape == (4, 4), '{}'.format(lidar2img.shape)

    num_bboxes = len(bboxes_3d.tensor)
    img_vert_size, img_hori_size = img_shape[0], img_shape[1]
    lidar2img = bboxes_3d.tensor.new_tensor(lidar2img)

    if num_bboxes == 0:
        xyxy = torch.empty((0, 4),
                           dtype=bboxes_3d.tensor.dtype,
                           device=bboxes_3d.tensor.device)
        final_valid_mask = torch.empty((0, ),
                                       dtype=torch.bool,
                                       device=bboxes_3d.tensor.device)
        return xyxy, final_valid_mask

    ### Project corners to 2D
    bboxes_3d_corners = bboxes_3d.corners  # [N, 8, 3]
    bboxes_3d_corners = bboxes_3d_corners.reshape(-1, 3)
    bboxes_3d_corners_hom = torch.cat([
        bboxes_3d_corners,
        bboxes_3d_corners.new_ones(size=(num_bboxes * 8, 1))
    ],
                                      dim=-1)  # [N * 8, 4]
    bboxes_3d_corners_2d = bboxes_3d_corners_hom @ lidar2img.t()

    bboxes_3d_corners_2d[:, 2] = torch.clamp(
        bboxes_3d_corners_2d[:, 2].clone(), min=1e-5)
    bboxes_3d_corners_2d[:, 0] = (bboxes_3d_corners_2d[:, 0].clone() /
                                  bboxes_3d_corners_2d[:, 2].clone())
    bboxes_3d_corners_2d[:, 1] = (bboxes_3d_corners_2d[:, 1].clone() /
                                  bboxes_3d_corners_2d[:, 2].clone())

    ### Now, figure out what boxes are valid
    bboxes_xyd = bboxes_3d_corners_2d[:, :3]
    valid = (bboxes_xyd[:, 0] >= 0) & (bboxes_xyd[:, 0] < img_hori_size) & \
            (bboxes_xyd[:, 1] >= 0) & (bboxes_xyd[:, 1] < img_vert_size) & \
            (bboxes_xyd[:, 2] > 0)
    bboxes_xyd = bboxes_xyd.reshape(-1, 8, 3)
    valid = valid.reshape(-1, 8)

    ## Boxes must have at least 3 corners in the image, and center of box must
    ## be at least 0.5m depth
    MIN_VALID_CORNERS = 3
    MIN_CENTER_DEPTH = 0.5
    valid_mask = valid.sum(dim=1) >= MIN_VALID_CORNERS
    # average depth of 8 corners is depth of center
    center_depth = bboxes_xyd[:, :, 2].mean(dim=1)
    center_depth_mask = center_depth >= MIN_CENTER_DEPTH

    ## Clip boxes to be within img boundaries
    final_valid_mask = valid_mask & center_depth_mask
    xmin = torch.clip(bboxes_xyd[:, :, 0].min(dim=1)[0], 0, img_hori_size)
    ymin = torch.clip(bboxes_xyd[:, :, 1].min(dim=1)[0], 0, img_vert_size)
    xmax = torch.clip(bboxes_xyd[:, :, 0].max(dim=1)[0], 0, img_hori_size)
    ymax = torch.clip(bboxes_xyd[:, :, 1].max(dim=1)[0], 0, img_vert_size)

    ## Get xyxy of valid boxes
    xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)  # N x 4

    return xyxy, final_valid_mask