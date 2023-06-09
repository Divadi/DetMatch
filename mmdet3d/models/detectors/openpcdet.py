import copy
import numpy as np
import torch
# from collections import OrderedDict
from mmcv.runner import force_fp32
try:
    from pcdet.models import build_network
except:
    pass
from torch.nn import functional as F

from mmdet3d.core import LiDARInstance3DBoxes, bbox3d2result
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .base import Base3DDetector


def limit_period(val, offset=0.5, period=np.pi):
    ans = val - torch.floor(val / period + offset) * period
    return ans


@DETECTORS.register_module()
class OpenPCDetDetector(Base3DDetector):
    def __init__(self,
                 dataset_fields,
                 voxel_layer,
                 pcdet_model,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(OpenPCDetDetector, self).__init__()

        # self.dummy_dataset = DummyDataset(dataset_fields)

        dataset_fields = copy.deepcopy(dataset_fields)

        # for k, v in list(dataset_fields.items()):
        #     if isinstance(v, list):
        #         dataset_fields[k] = np.array(dataset_fields.pop(k))
        dataset_fields['point_cloud_range'] = np.array(
            dataset_fields['point_cloud_range'])

        dataset_fields.grid_size = \
            ((dataset_fields.point_cloud_range[3:6] -
              dataset_fields.point_cloud_range[0:3])
             / np.array(dataset_fields.voxel_size))
        dataset_fields.grid_size = np.round(
            dataset_fields.grid_size).astype(np.int64)

        self.voxel_layer = Voxelization(**voxel_layer)
        self.model = build_network(model_cfg=pcdet_model,
                                   num_class=len(dataset_fields.class_names),
                                   dataset=dataset_fields)

        self.num_classes = len(dataset_fields.class_names)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()  # should I?
    def train_to_openpcdet(self,
                           points,
                           img_metas,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           gt_bboxes_ignore=None):

        """Basically, a pipeline to transform things to OpenPCDet format."""
        assert gt_bboxes_ignore is None
        res = dict()

        ### Voxelize
        voxels, num_points, coors = self.voxelize(points)
        batch_size = coors[-1, 0].item() + 1
        res['batch_size'] = batch_size
        res['voxels'] = voxels
        res['voxel_num_points'] = num_points
        res['voxel_coords'] = coors

        ### Transform GT boxes & labels and combine them
        # print(gt_bboxes_3d)
        gt_bboxes_3d_trans = []
        for curr_gt_bboxes_3d, curr_gt_labels_3d \
                in zip(gt_bboxes_3d, gt_labels_3d):
            ## GT boxes should instead be gravity centered.
            curr_gt_bboxes_3d_grav_cnt = curr_gt_bboxes_3d.clone()
            curr_gt_bboxes_3d_grav_cnt.tensor[:, :3] \
                = curr_gt_bboxes_3d_grav_cnt.gravity_center
            curr_gt_bboxes_3d_grav_cnt = curr_gt_bboxes_3d_grav_cnt.tensor

            ## Change the rotation angle following L108 here
            ## Spconv-OpenPCDet/pcdet/utils/box_utils.py
            curr_gt_bboxes_3d_grav_cnt[:, 6] = \
                (-(curr_gt_bboxes_3d_grav_cnt[:, 6] + np.pi / 2) +
                 (4 * np.pi / 2))

            ## Spconv-OpenPCDet/pcdet/datasets/augmentor/data_augmentor.py L115
            curr_gt_bboxes_3d_grav_cnt[:, 6] = limit_period(
                curr_gt_bboxes_3d_grav_cnt[:, 6], offset=0.5, period=2 * np.pi)

            ## Flip 3 and 4
            tmp = curr_gt_bboxes_3d_grav_cnt[:, 3].clone()
            curr_gt_bboxes_3d_grav_cnt[:, 3] = curr_gt_bboxes_3d_grav_cnt[:, 4]
            curr_gt_bboxes_3d_grav_cnt[:, 4] = tmp

            ## Mask away clases outside of 0 ~ num_classes
            mask = ((0 <= curr_gt_labels_3d) &
                    (curr_gt_labels_3d < self.num_classes))
            curr_gt_bboxes_3d_grav_cnt = curr_gt_bboxes_3d_grav_cnt[mask]
            curr_gt_labels_3d_filt = curr_gt_labels_3d[mask]

            ## Append class label to box, 1-indexed.
            curr_gt_bboxes_3d_trans = torch.cat([
                curr_gt_bboxes_3d_grav_cnt.cpu().float(), # added .cpu().float() on 1/13 for Opd_HardPseudoLabel_3D
                curr_gt_labels_3d_filt[:, None].cpu().float() + 1], dim=1)

            gt_bboxes_3d_trans.append(curr_gt_bboxes_3d_trans)
        # print(gt_bboxes_3d_trans)

        ### To collate, put together all boxes, give padding label "0"
        max_gt = max([len(x) for x in gt_bboxes_3d_trans])
        batch_gt_boxes3d = torch.zeros(
            (batch_size, max_gt, 7 + 1), dtype=torch.float32)
        for k in range(batch_size):
            batch_gt_boxes3d[k, :gt_bboxes_3d_trans[k].__len__(), :] = \
                gt_bboxes_3d_trans[k]
        res['gt_boxes'] = batch_gt_boxes3d.cuda()

        ### Pad points
        points_batch = []
        for k in range(batch_size):
            points_pad = F.pad(points[k], (1, 0), mode='constant', value=k)
            points_batch.append(points_pad)
        points_batch = torch.cat(points_batch, dim=0)
        res['points'] = points_batch

        ### Extras
        # print(img_metas)
        res['frame_id'] = np.array(
            [tmp['sample_idx'] for tmp in img_metas])
        # res['image_shape'] = torch.tensor(
        #     [tmp['img_shape'] for tmp in img_metas]).int().cuda()
        # print(res['gt_boxes'])
        return res

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):

        openpcdet_batch = self.train_to_openpcdet(
            points,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_bboxes_ignore)
        ret_dict, tb_dict, disp_dict = self.model(openpcdet_batch)
        
        loss = ret_dict['loss'].mean()
        
        return dict(loss=loss)

        # return tb_dict

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        openpcdet_batch = dict()

        ### Voxelize
        voxels, num_points, coors = self.voxelize(points)
        batch_size = coors[-1, 0].item() + 1
        openpcdet_batch['batch_size'] = batch_size
        openpcdet_batch['voxels'] = voxels
        openpcdet_batch['voxel_num_points'] = num_points
        openpcdet_batch['voxel_coords'] = coors

        ### Pad points
        points_batch = []
        for k in range(batch_size):
            points_pad = F.pad(points[k], (1, 0), mode='constant', value=k)
            points_batch.append(points_pad)
        points_batch = torch.cat(points_batch, dim=0)
        openpcdet_batch['points'] = points_batch

        ### Extras
        # print(img_metas)
        openpcdet_batch['frame_id'] = np.array(
            [tmp['sample_idx'] for tmp in img_metas])

        ### Send to OpenPCDet
        pred_dicts, _ = self.model(openpcdet_batch)

        ### Convert OpenPCDet boxes to mm3d boxes TODO: This is nonsensical for batch_size > 1 :)
        for pred_dict in pred_dicts:
            # print(pred_dict)
            tmp = pred_dict['pred_boxes'][:, 3].clone()
            pred_dict['pred_boxes'][:, 3] = pred_dict['pred_boxes'][:, 4]
            pred_dict['pred_boxes'][:, 4] = tmp

            pred_dict['pred_boxes'][:, 6] = (
                -pred_dict['pred_boxes'][:, 6] - np.pi / 2)
            # just to deal with the yaw hack in mm3d that openpcdet doesnt have
            # pred_dict['pred_boxes'][:, 6] += np.pi

            pred_dict['pred_boxes'] = LiDARInstance3DBoxes(
                pred_dict['pred_boxes'],
                origin=(0.5, 0.5, 0.5))

        ### Post process
        bbox_results = [
            bbox3d2result(pred_dict['pred_boxes'],
                          pred_dict['pred_scores'],
                          pred_dict['pred_labels'] - 1)
            for pred_dict in pred_dicts
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        raise Exception('To be implemented by child classes')

    def extract_feat(self, points, img_metas=None):
        raise Exception('Irrelevant')

    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        raise Exception('Irrelevant')