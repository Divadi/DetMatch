outputs_dir = 'outputs/detmatch/'

split_folder = 'ssl_splits'
split_num = 1
split_frac = 0.02
pretrained_split_frac = {0.01: '001', 0.02: '002', 0.20: '020'}[split_frac]

work_dir = outputs_dir + '{}/pretrain_frcnn/split_{}'.format(pretrained_split_frac, split_num)

_base_ = [
    '../../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../../_base_/default_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117-6eda6d92.pth'

batch_size = 8
dataset_repeat_multiplier = 10

## Dataset Settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']

train_info_path = data_root + '{}/kitti_infos_train_proj_3d_lab_{}_{}.pkl'.format(split_folder, split_frac, split_num)

################################################################################
################################################################################

model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=3,
            loss_cls=dict(
                _delete_=True,
                type='FocalLoss',
                use_sigmoid=True,
                loss_weight=1.0,
                gamma=2.0,
                alpha=0.5,
                reduction='mean')
        )))

################################################################################
################################################################################

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

input_modality = dict(use_lidar=True, use_camera=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='Resize',
        img_scale=[(640, 192), (2560, 768)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    ### UBteacher Augs ########################################################
    dict(type='TVToPILImage'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='TVColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)],
        p=0.8),
    dict(type='TVRandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur', # use re-implemented version
                sigma_min=0.1,
                sigma_max=2.0)],
        p=0.5),
    dict(type='TVToTensor'),
    dict(
        type='TVRandomErasing',
        p=0.7,
        scale=(0.05, 0.2),
        ratio=(0.3, 3.3),
        value='random'),
    dict(
        type='TVRandomErasing',
        p=0.5,
        scale=(0.02, 0.2),
        ratio=(0.1, 6),
        value='random'),
    dict(
        type='TVRandomErasing',
        p=0.3,
        scale=(0.02, 0.2),
        ratio=(0.05, 8),
        value='random'),
    dict(type='TVToPILImage'),
    dict(type='ToNumpy'),
    ###########################################################################
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2 * dataset_repeat_multiplier,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=train_info_path,
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            filter_empty_gt=True, # added by me
            completely_remove_other_classes=True)), # added by me
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

## Train Settings
lr = 0.02 / 2 * batch_size
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 10])

runner = dict(type='EpochBasedRunner', max_epochs=12)

evaluation = dict(interval=12)
checkpoint_config = dict(interval=12)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

