import copy
import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.points import LiDARPoints
from mmdet3d.datasets import SSL_Dataset
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.models.utils import apply_3d_transformation_bboxes


def generate_ssl_dataset_config():
    data_root = 'tests/data/kitti/'
    ann_file = 'tests/data/kitti/kitti_infos_train.pkl'
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    pts_prefix = 'velodyne_reduced'
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    input_modality = dict(use_lidar=True, use_camera=False)
    split = 'training'

    db_sampler = dict(
        data_root=data_root,
        info_path=data_root + 'kitti_dbinfos_train.pkl',
        rate=1.0,
        prepare=dict(
            filter_by_difficulty=[-1],
            filter_by_min_points=dict(Pedestrian=10)),
        classes=class_names,
        sample_groups=dict(Pedestrian=6))

    file_client_args = dict(backend='disk')

    labeled_shared_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=file_client_args),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            file_client_args=file_client_args),
        dict(type='ObjectSample', db_sampler=db_sampler),
        dict(
            type='ObjectNoise',
            num_try=100,
            translation_std=[1.0, 1.0, 0.5],
            global_rot_range=[0.0, 0.0],
            rot_range=[-0.78539816, 0.78539816]),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    ]
    labeled_student_pipeline = [
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05]),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    labeled_teacher_pipeline = [
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points'])
    ]

    ##########################################################################
    unlabeled_shared_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=file_client_args
        ),  # No DB Sampling for unlabeled like 3DIoUMatch & ONCE
        # dict(
        #     type='LoadAnnotations3D',
        #     with_bbox_3d=True,
        #     with_label_3d=True,
        #     file_client_args=file_client_args),
        dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    ]
    unlabeled_student_pipeline = [
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05]),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        # dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points'])
    ]
    unlabeled_teacher_pipeline = [
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        # dict(type='PointShuffle'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points'])
    ]

    return (data_root, ann_file, class_names, pts_prefix,
            (labeled_shared_pipeline, labeled_student_pipeline,
             labeled_teacher_pipeline, unlabeled_shared_pipeline,
             unlabeled_student_pipeline, unlabeled_teacher_pipeline),
            input_modality, split)


def test_aug_transforms():
    """Load GT annotations through teacher path too,"""
    np.random.seed(1)
    (data_root, ann_file, class_names, pts_prefix,
     (labeled_shared_pipeline, labeled_student_pipeline,
      labeled_teacher_pipeline, unlabeled_shared_pipeline,
      unlabeled_student_pipeline, unlabeled_teacher_pipeline), input_modality,
     split) = generate_ssl_dataset_config()

    # Just remove GlobalRotScaleTrans
    labeled_teacher_pipeline_with_gt = labeled_student_pipeline[1:]

    dataset = SSL_Dataset(
        labeled_dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=labeled_shared_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR'),
        unlabeled_dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=unlabeled_shared_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=True,
            box_type_3d='LiDAR'),
        labeled_student_pipeline=labeled_student_pipeline,
        labeled_teacher_pipeline=labeled_teacher_pipeline_with_gt,
        unlabeled_student_pipeline=unlabeled_student_pipeline,
        unlabeled_teacher_pipeline=unlabeled_teacher_pipeline,
    )

    data = dataset[0]  # just single sample, not batch
    lab_stu, lab_tea = data['lab_stu'], data['lab_tea']

    for k in lab_stu.keys():
        lab_stu[k] = lab_stu[k]._data

    for k in lab_tea.keys():
        lab_tea[k] = lab_tea[k]._data

    # print(lab_stu['img_metas']['pcd_rotation_scalar'])

    # print('teacher:', lab_tea['img_metas'])
    # print('student:', lab_stu['img_metas'])
    # print('original teacher', lab_tea['gt_bboxes_3d'])

    reversed_tea_boxes = apply_3d_transformation_bboxes(
        lab_tea['gt_bboxes_3d'], lab_tea['img_metas'], reverse=True)
    # print(reversed_tea_boxes)

    reversed_tea_boxes = apply_3d_transformation_bboxes(
        reversed_tea_boxes, lab_stu['img_metas'], reverse=False)

    # print('final reversed teacher:', reversed_tea_boxes)
    # print('student:', lab_stu['gt_bboxes_3d'])

    torch.allclose(reversed_tea_boxes.tensor[:, :6],
                   lab_stu['gt_bboxes_3d'].tensor[:, :6])

    # Angles can be flipped, etc, just compare both sin and cos
    torch.allclose(
        torch.sin(reversed_tea_boxes.tensor[:, 6]),
        torch.sin(lab_stu['gt_bboxes_3d'].tensor[:, 6]))

    torch.allclose(
        torch.cos(reversed_tea_boxes.tensor[:, 6]),
        torch.cos(lab_stu['gt_bboxes_3d'].tensor[:, 6]))

    ### Investigate unlab
    unlab_stu = data['unlab_stu'][0]
    unlab_tea = data['unlab_tea'][0]
    # print(unlab_stu)
    # print(unlab_tea)

    for k in unlab_stu.keys():
        unlab_stu[k] = unlab_stu[k]._data

    for k in unlab_tea.keys():
        unlab_tea[k] = unlab_tea[k]._data

    print(unlab_stu)
    print(unlab_tea)

    reversed_unlab_tea = apply_3d_transformation(
        unlab_tea['points'][:, :3],
        'LIDAR',
        unlab_tea['img_metas'],
        reverse=True)

    reversed_unlab_tea = apply_3d_transformation(
        reversed_unlab_tea, 'LIDAR', unlab_stu['img_metas'], reverse=False)

    print('reversed_unlab_tea', reversed_unlab_tea)
    print("unlab_stu['points']", unlab_stu['points'])

    # new_unlab_stu = dict()
    # for k in data['unlab_stu'][0].keys():
    #     new_unlab_stu[k] = [
    #         sample for batch in data['unlab_stu'] for sample in batch[k]
    #     ]
    # data['unlab_stu'] = new_unlab_stu

    # new_unlab_tea = dict()
    # for k in data['unlab_tea'][0].keys():
    #     new_unlab_tea[k] = [
    #         sample for batch in data['unlab_tea'] for sample in batch[k]
    #     ]
    # data['unlab_tea'] = new_unlab_tea


def test_unlab():
    """Load GT annotations through teacher path too,"""
    np.random.seed(0)
    (data_root, ann_file, class_names, pts_prefix,
     (labeled_shared_pipeline, labeled_student_pipeline,
      labeled_teacher_pipeline, unlabeled_shared_pipeline,
      unlabeled_student_pipeline, unlabeled_teacher_pipeline), input_modality,
     split) = generate_ssl_dataset_config()

    # Just remove GlobalRotScaleTrans
    labeled_teacher_pipeline_with_gt = labeled_student_pipeline[1:]

    dataset = SSL_Dataset(
        labeled_dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=labeled_shared_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR'),
        unlabeled_dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=unlabeled_shared_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=True,
            box_type_3d='LiDAR'),
        labeled_student_pipeline=labeled_student_pipeline,
        labeled_teacher_pipeline=labeled_teacher_pipeline_with_gt,
        unlabeled_student_pipeline=unlabeled_student_pipeline,
        unlabeled_teacher_pipeline=unlabeled_teacher_pipeline,
    )

    data = dataset[0]  # just single sample, not batch

    # new_unlab_stu = dict()
    # for k in data['unlab_stu'][0].keys():
    #     new_unlab_stu[k] = [
    #         sample for batch in data['unlab_stu'] for sample in batch[k]
    #     ]
    # data['unlab_stu'] = new_unlab_stu

    # new_unlab_tea = dict()
    # for k in data['unlab_tea'][0].keys():
    #     new_unlab_tea[k] = [
    #         sample for batch in data['unlab_tea'] for sample in batch[k]
    #     ]
    # data['unlab_tea'] = new_unlab_tea
    # print(data)
    unlab_stu = data['unlab_stu'][0]
    unlab_tea = data['unlab_tea'][0]
    # print(unlab_stu)
    # print(unlab_tea)

    for k in unlab_stu.keys():
        unlab_stu[k] = unlab_stu[k]._data

    for k in unlab_tea.keys():
        unlab_tea[k] = unlab_tea[k]._data

    print("unlab_tea['points']", unlab_tea['points'])

    reversed_unlab_tea = apply_3d_transformation(
        unlab_tea['points'][:, :3],
        'LIDAR',
        unlab_tea['img_metas'],
        reverse=True)

    reversed_unlab_tea = apply_3d_transformation(
        reversed_unlab_tea, 'LIDAR', unlab_stu['img_metas'], reverse=False)

    print('reversed_unlab_tea', reversed_unlab_tea)
    print("unlab_stu['points']", unlab_stu['points'])


def test_rotate():
    np.random.seed(1)
    pts = LiDARPoints(np.random.random((3, 4)), points_dim=4)
    pts_copy = copy.deepcopy(pts)
    boxes = LiDARInstance3DBoxes(np.random.random((2, 7)))

    print(pts.rotate(-0.4497296087225684))
    print(boxes.rotate(-0.4497296087225684, pts_copy))

    print(pts)
    print(pts_copy)


if __name__ == '__main__':
    test_aug_transforms()
    # test_rotate()