import numpy as np
import pickle
import torch
from canvas import Canvas

from mmdet3d.models.fusion_layers.coord_transform import \
    apply_3d_transformation
from ...builder import SSL_MODULES
from ..bbox_utils import apply_3d_transformation_bboxes, bbox_3d_to_bbox_2d, filter_by_nms
from ..utils import mlvl_get, mlvl_getattr

import torch
import numpy as np
import cv2
import os
import os.path as osp

from mmdet3d.models.fusion_layers.coord_transform import bbox_2d_transform
from mmseg.core import add_prefix
from ...builder import SSL_MODULES


def draw_boxes_on_image(img, boxes, labels, color=(255, 0, 0)):
    img = img.copy()
    for box, label in zip(np.array(boxes).astype(np.int32).tolist(), 
                          np.array(labels).astype(str).tolist()):

        x1, y1, x2, y2 = box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = cv2.putText(img, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, color, 2, cv2.LINE_AA)
        
    return img

def add_prefix_no_dot(input_dict, prefix):
    return {f'{prefix}{k}': v for k, v in input_dict.items()}


@SSL_MODULES.register_module
class Vis3D():
    """Goal of this module is to use my canvas class to visualize student,
    teacher and GT. They are saved back into the output under a new key "vis".

    Expected usage is with wandb, which consumes this "vis" key.
    """
    def __init__(self,
                 class_names,
                 batch_dict_key,
                 tea_bboxes_key,
                 stu_bboxes_key,
                 use_multiclass_nms_label=False,
                 vis_idxs=None,
                 vis_idxs_interval=1,
                 out_name_prefix='',
                 canvas_init_kwargs=dict(canvas_shape=(900, 1000),
                                         canvas_x_range=(-10, 80),
                                         canvas_y_range=(-50, 50),
                                         box_line_thickness=1,
                                         box_text_size=0.5),
                 canvas_draw_lines_kwargs=dict(x=[-5, 70.4], y=[-40, 40])):

        self.class_names = np.array(list(map(lambda s: s[:3], class_names)))
        self.batch_dict_key = batch_dict_key
        self.use_multiclass_nms_label = use_multiclass_nms_label
        self.stu_bboxes_key = stu_bboxes_key
        self.tea_bboxes_key = tea_bboxes_key

        self.vis_idxs = vis_idxs
        self.vis_idxs_interval = vis_idxs_interval
        if self.vis_idxs is not None:
            if 'nuscenes' in vis_idxs:
                self.vis_idxs = [
                    tmp['token']
                    for tmp in pickle.load(open(self.vis_idxs, 'rb'))['infos']]
            else:
                self.vis_idxs = [
                    tmp['image']['image_idx']
                    for tmp in pickle.load(open(self.vis_idxs, 'rb'))]
            self.vis_idxs = set(self.vis_idxs[::self.vis_idxs_interval])

        self.out_name_prefix = out_name_prefix

        self.canvas_init_kwargs = canvas_init_kwargs
        self.canvas_draw_lines_kwargs = canvas_draw_lines_kwargs

    def forward(self, ssl_obj, batch_dict):
        vis_dict = dict()

        with torch.no_grad():
            curr_batch_dict = mlvl_get(batch_dict, self.batch_dict_key)
            tea_bboxes = mlvl_get(batch_dict, self.tea_bboxes_key)
            if self.stu_bboxes_key is not None:
                stu_bboxes = mlvl_get(batch_dict, self.stu_bboxes_key)

            points = curr_batch_dict['points']
            sample_idxs = [img_meta['sample_idx']
                           for img_meta in curr_batch_dict['img_metas']]
            img_metas = curr_batch_dict['img_metas']

            for batch_idx, (curr_points, curr_sample_idx) in \
                    enumerate(zip(points, sample_idxs)):

                if (self.vis_idxs is not None and
                        curr_sample_idx not in self.vis_idxs):
                    continue

                ### Reverse Student Img Metas
                curr_points = apply_3d_transformation(
                    curr_points[:, :3],
                    'LIDAR',
                    img_metas[batch_idx],
                    reverse=True)
                if 'gt_bboxes' in curr_batch_dict:
                    gt_bboxes = apply_3d_transformation_bboxes(
                        curr_batch_dict['gt_bboxes_3d'][batch_idx],
                        img_metas[batch_idx],
                        reverse=True)
                curr_tea_bboxes = apply_3d_transformation_bboxes(
                    tea_bboxes[batch_idx][0],
                    img_metas[batch_idx],
                    reverse=True)
                if self.stu_bboxes_key is not None:
                    curr_stu_bboxes = apply_3d_transformation_bboxes(
                        stu_bboxes[batch_idx][0],
                        img_metas[batch_idx],
                        reverse=True)

                ### Set up canvas
                canvas = Canvas(**self.canvas_init_kwargs)
                canvas.draw_lines(**self.canvas_draw_lines_kwargs)

                ### Draw points
                canvas_xy, valid_mask = \
                    canvas.get_canvas_coords(curr_points.cpu().numpy())
                canvas.draw_canvas_points(canvas_xy[valid_mask], 'Spectral')

                ### Draw GT boxes in green
                if 'gt_bboxes' in curr_batch_dict:
                    gt_bboxes = gt_bboxes.tensor.cpu().numpy()
                    gt_labels = \
                        curr_batch_dict['gt_labels_3d'][batch_idx].cpu().numpy()
                    gt_bboxes = gt_bboxes[gt_labels != -1]
                    gt_labels = gt_labels[gt_labels != -1]  # remove ignred classes
                    gt_labels = self.class_names[gt_labels]

                    canvas.draw_boxes(gt_bboxes[:, :7], (0, 200, 0), gt_labels)

                ### Optionally draw student predictions in blue
                if self.stu_bboxes_key is not None:
                    curr_stu_bboxes = curr_stu_bboxes.tensor.cpu().numpy()
                    curr_stu_scores = stu_bboxes[batch_idx][1].cpu().numpy()

                    ## Generate labels and scores
                    if not self.use_multiclass_nms_label:
                        labels_idxs = curr_stu_scores[
                            :, :len(self.class_names)].argmax(axis=1)
                    else:
                        labels_idxs = stu_bboxes[batch_idx][2].cpu().numpy()
                    labels = self.class_names[labels_idxs]
                    scores = curr_stu_scores[
                        np.arange(len(labels_idxs)), labels_idxs]
                    texts = ['{}_{}'.format(label, round(float(score) * 100))
                             for label, score in zip(labels, scores)]
                    # print(curr_sample_idx, list(zip(curr_stu_scores, labels)))

                    canvas.draw_boxes(
                        curr_stu_bboxes[:, :7], (0, 200, 200), texts)

                ### Draw teacher predictions
                curr_tea_bboxes = curr_tea_bboxes.tensor.cpu().numpy()
                curr_tea_scores = tea_bboxes[batch_idx][1].cpu().numpy()

                ## Generate labels and scores
                if not self.use_multiclass_nms_label:
                    labels_idxs = curr_tea_scores[:, :len(self.class_names)]\
                        .argmax(axis=1)
                else:
                    labels_idxs = tea_bboxes[batch_idx][2].cpu().numpy()
                labels = self.class_names[labels_idxs]
                scores = curr_tea_scores[
                    np.arange(len(labels_idxs)), labels_idxs]
                texts = ['{}_{}'.format(label, round(float(score) * 100))
                         for label, score in zip(labels, scores)]

                ## Draw boxes, scores, labels. Teacher preds are red
                canvas.draw_boxes(curr_tea_bboxes[:, :7], (200, 0, 0), texts)

                ### Put into dict
                vis_dict[str(curr_sample_idx)] = canvas.canvas

        if 'vis' not in batch_dict:
            batch_dict['vis'] = dict()
        batch_dict['vis'].update(
            add_prefix_no_dot(vis_dict, self.out_name_prefix + '/'))

        return batch_dict



@SSL_MODULES.register_module
class Vis2D_Kitti():
    def __init__(self,
                 class_names,
                 batch_dict_key,
                 tea_bboxes_key,
                 stu_bboxes_key,
                 gt_bboxes_key='gt_bboxes',
                 gt_labels_key='gt_labels',
                 vis_idxs=None,
                 vis_idxs_interval=1,
                 out_name_prefix=''):

        self.class_names = np.array(list(map(lambda s: s[:3], class_names)))
        self.batch_dict_key = batch_dict_key
        self.tea_bboxes_key = tea_bboxes_key
        self.stu_bboxes_key = stu_bboxes_key
        self.gt_bboxes_key = gt_bboxes_key
        self.gt_labels_key = gt_labels_key

        self.vis_idxs = vis_idxs
        self.vis_idxs_interval = vis_idxs_interval
        if self.vis_idxs is not None:
            self.vis_idxs = [
                tmp['image']['image_idx']
                for tmp in pickle.load(open(self.vis_idxs, 'rb'))]
            self.vis_idxs = set(self.vis_idxs[::self.vis_idxs_interval])

        self.out_name_prefix = out_name_prefix

    def forward(self, ssl_obj, batch_dict):
        vis_dict = dict()
        curr_batch_dict = mlvl_get(batch_dict, self.batch_dict_key)

        img = curr_batch_dict['img'] # N x 3 x H x W
        img_metas = curr_batch_dict['img_metas'] # list of length N
        sample_idxs = [img_meta['sample_idx'] for img_meta in img_metas]
        batch_size = img.shape[0]

        if self.gt_bboxes_key is not None and self.gt_labels_key is not None and self.gt_bboxes_key in curr_batch_dict:
            gt_bboxes = curr_batch_dict[self.gt_bboxes_key] # list of lidarboxes
            gt_labels = curr_batch_dict[self.gt_labels_key] # list of arrays

        if self.tea_bboxes_key is not None:
            tea_bboxes = mlvl_get(batch_dict, self.tea_bboxes_key)
        if self.stu_bboxes_key is not None:
            stu_bboxes = mlvl_get(batch_dict, self.stu_bboxes_key)

        for batch_idx in range(len(img)):
            if (self.vis_idxs is not None and
                    sample_idxs[batch_idx] not in self.vis_idxs):
                continue

            curr_img_metas = img_metas[batch_idx]

            curr_img_view = img[batch_idx]
            curr_img_view = curr_img_view.detach().cpu().numpy().transpose(1, 2, 0) # H x W x 3
            curr_img_view = curr_img_view - curr_img_view.min(axis=(0, 1), keepdims=True)
            curr_img_view = curr_img_view / curr_img_view.max(axis=(0, 1), keepdims=True)
            curr_img_view = (curr_img_view * 255).astype(np.uint8)[..., ::-1] # flip channels

            ## Draw GT Boxes Green
            if self.gt_bboxes_key is not None and self.gt_labels_key is not None and self.gt_bboxes_key in curr_batch_dict:
                curr_gt_bboxes = gt_bboxes[batch_idx]
                curr_gt_labels = gt_labels[batch_idx]
                curr_view_gt_bboxes = curr_gt_bboxes.detach().cpu().numpy()
                curr_view_gt_labels = self.class_names[curr_gt_labels.cpu().numpy()]
                curr_img_view = draw_boxes_on_image(curr_img_view, 
                                                    curr_view_gt_bboxes,
                                                    curr_view_gt_labels,
                                                    (0, 200, 0))

            ## Draw Teacher Boxes Red
            if self.tea_bboxes_key is not None:
                curr_view_tea_bboxes = tea_bboxes[batch_idx][0].detach().cpu().numpy()
                curr_view_tea_scores = tea_bboxes[batch_idx][1].detach().cpu().numpy()
                curr_view_tea_labels = curr_view_tea_scores[:, :len(self.class_names)].argmax(axis=1)
                curr_view_tea_scores = curr_view_tea_scores[
                    np.arange(len(curr_view_tea_labels)), curr_view_tea_labels]
                curr_view_tea_labels = self.class_names[curr_view_tea_labels]

                texts = ['{}_{}'.format(label, round(float(score) * 100))
                            for label, score in zip(curr_view_tea_labels, curr_view_tea_scores)]
                curr_img_view = draw_boxes_on_image(curr_img_view, 
                                                    curr_view_tea_bboxes,
                                                    texts,
                                                    (200, 0, 0))
            
            ## Draw Student Boxes Teal
            if self.stu_bboxes_key is not None:
                curr_view_stu_bboxes = stu_bboxes[batch_idx][0].detach().cpu().numpy()
                curr_view_stu_scores = stu_bboxes[batch_idx][1].detach().cpu().numpy()
                curr_view_stu_labels = curr_view_stu_scores[:, :len(self.class_names)].argmax(axis=1)
                curr_view_stu_scores = curr_view_stu_scores[
                    np.arange(len(curr_view_stu_labels)), curr_view_stu_labels]
                curr_view_stu_labels = self.class_names[curr_view_stu_labels]
                
                texts = ['{}_{}'.format(label, round(float(score) * 100))
                            for label, score in zip(curr_view_stu_labels, curr_view_stu_scores)]
                curr_img_view = draw_boxes_on_image(curr_img_view, 
                                                    curr_view_stu_bboxes,
                                                    texts,
                                                    (0, 200, 200))

            vis_dict[str(sample_idxs[batch_idx])] = curr_img_view

        if 'vis' not in batch_dict:
            batch_dict['vis'] = dict()
        batch_dict['vis'].update(
            add_prefix_no_dot(vis_dict, self.out_name_prefix + '/'))

        return batch_dict