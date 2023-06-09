outputs_dir = 'outputs/detmatch/'

split_folder = 'ssl_splits'
split_num = 2
split_frac = 0.20
pretrained_split_frac = {0.01: '001', 0.02: '002', 0.20: '020'}[split_frac]

work_dir = outputs_dir + '{}/pretrain_pvrcnn/split_{}'.format(pretrained_split_frac, split_num)

batch_size = 8
dataset_repeat_multiplier = 5

load_from = None

dataset_type = 'KittiDataset'
data_root = 'data/kitti/'

train_info_path = data_root + '{}/kitti_infos_train_proj_3d_lab_{:.02f}_{}.pkl'.format(split_folder, split_frac, split_num)
db_info_path = data_root + '{}/kitti_dbinfos_train_lab_{:.02f}_{}.pkl'.format(split_folder, split_frac, split_num)


class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
###############################################################################
voxel_size = [0.05, 0.05, 0.1]

model = dict(
    type='OpenPCDetDetector',
    dataset_fields=dict(
        class_names=class_names,
        point_feature_encoder=dict(num_point_features=4),
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        depth_downsample_factor=None),
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    pcdet_model=dict(
        NAME='PVRCNN',
        VFE=dict(NAME='MeanVFE'),
        BACKBONE_3D=dict(NAME='VoxelBackBone8x'),
        MAP_TO_BEV=dict(NAME='HeightCompression', NUM_BEV_FEATURES=256),
        BACKBONE_2D=dict(
            NAME='BaseBEVBackbone',
            LAYER_NUMS=[5, 5],
            LAYER_STRIDES=[1, 2],
            NUM_FILTERS=[128, 256],
            UPSAMPLE_STRIDES=[1, 2],
            NUM_UPSAMPLE_FILTERS=[256, 256]),
        DENSE_HEAD=dict(
            NAME='AnchorHeadSingle',
            CLASS_AGNOSTIC=False,
            USE_DIRECTION_CLASSIFIER=True,
            DIR_OFFSET=0.78539,
            DIR_LIMIT_OFFSET=0,
            NUM_DIR_BINS=2,
            ANCHOR_GENERATOR_CONFIG=[
                dict(
                    class_name='Pedestrian',
                    anchor_sizes=[[0.8, 0.6, 1.73]],
                    anchor_rotations=[0, 1.57],
                    anchor_bottom_heights=[-0.6],
                    align_center=False,
                    feature_map_stride=8,
                    matched_threshold=0.5,
                    unmatched_threshold=0.35),
                dict(
                    class_name='Cyclist',
                    anchor_sizes=[[1.76, 0.6, 1.73]],
                    anchor_rotations=[0, 1.57],
                    anchor_bottom_heights=[-0.6],
                    align_center=False,
                    feature_map_stride=8,
                    matched_threshold=0.5,
                    unmatched_threshold=0.35),
                dict(
                    class_name='Car',
                    anchor_sizes=[[3.9, 1.6, 1.56]],
                    anchor_rotations=[0, 1.57],
                    anchor_bottom_heights=[-1.78],
                    align_center=False,
                    feature_map_stride=8,
                    matched_threshold=0.6,
                    unmatched_threshold=0.45),
            ],
            TARGET_ASSIGNER_CONFIG=dict(
                NAME='AxisAlignedTargetAssigner',
                POS_FRACTION=-1,
                SAMPLE_SIZE=512,
                NORM_BY_NUM_EXAMPLES=False,
                MATCH_HEIGHT=False,
                BOX_CODER='ResidualCoder'),
            LOSS_CONFIG=dict(
                LOSS_WEIGHTS=dict(
                    cls_weight=1,
                    loc_weight=2,
                    dir_weight=0.2,
                    code_weights=[1, 1, 1, 1, 1, 1, 1]))),
        PFE=dict(
            NAME='VoxelSetAbstraction',
            POINT_SOURCE='raw_points',
            NUM_KEYPOINTS=2048,
            NUM_OUTPUT_FEATURES=128,
            SAMPLE_METHOD='FPS',
            FEATURES_SOURCE=[
                'bev', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4', 'raw_points'
            ],
            SA_LAYER=dict(
                raw_points=dict(
                    MLPS=[[16, 16], [16, 16]],
                    POOL_RADIUS=[0.4, 0.8],
                    NSAMPLE=[16, 16]),
                x_conv1=dict(
                    DOWNSAMPLE_FACTOR=1,
                    MLPS=[[16, 16], [16, 16]],
                    POOL_RADIUS=[0.4, 0.8],
                    NSAMPLE=[16, 16]),
                x_conv2=dict(
                    DOWNSAMPLE_FACTOR=2,
                    MLPS=[[32, 32], [32, 32]],
                    POOL_RADIUS=[0.8, 1.2],
                    NSAMPLE=[16, 32]),
                x_conv3=dict(
                    DOWNSAMPLE_FACTOR=4,
                    MLPS=[[64, 64], [64, 64]],
                    POOL_RADIUS=[1.2, 2.4],
                    NSAMPLE=[16, 32]),
                x_conv4=dict(
                    DOWNSAMPLE_FACTOR=8,
                    MLPS=[[64, 64], [64, 64]],
                    POOL_RADIUS=[2.4, 4.8],
                    NSAMPLE=[16, 32]))),
        POINT_HEAD=dict(
            NAME='PointHeadSimple',
            CLS_FC=[256, 256],
            CLASS_AGNOSTIC=True,
            USE_POINT_FEATURES_BEFORE_FUSION=True,
            TARGET_CONFIG=dict(GT_EXTRA_WIDTH=[0.2, 0.2, 0.2]),
            LOSS_CONFIG=dict(
                LOSS_REG='smooth-l1', LOSS_WEIGHTS=dict(point_cls_weight=1))),
        ROI_HEAD=dict(
            NAME='PVRCNNHead',
            CLASS_AGNOSTIC=True,
            SHARED_FC=[256, 256],
            CLS_FC=[256, 256],
            REG_FC=[256, 256],
            DP_RATIO=0.3,
            NMS_CONFIG=dict(
                TRAIN=dict(
                    NMS_TYPE='nms_gpu',
                    MULTI_CLASSES_NMS=False,
                    NMS_PRE_MAXSIZE=9000,
                    NMS_POST_MAXSIZE=512,
                    NMS_THRESH=0.8),
                TEST=dict(
                    NMS_TYPE='nms_gpu',
                    MULTI_CLASSES_NMS=False,
                    NMS_PRE_MAXSIZE=1024,
                    NMS_POST_MAXSIZE=100,
                    NMS_THRESH=0.7)),
            ROI_GRID_POOL=dict(
                GRID_SIZE=6,
                MLPS=[[64, 64], [64, 64]],
                POOL_RADIUS=[0.8, 1.6],
                NSAMPLE=[16, 16],
                POOL_METHOD='max_pool'),
            TARGET_CONFIG=dict(
                BOX_CODER='ResidualCoder',
                ROI_PER_IMAGE=128,
                FG_RATIO=0.5,
                SAMPLE_ROI_BY_EACH_CLASS=True,
                CLS_SCORE_TYPE='roi_iou',
                CLS_FG_THRESH=0.75,
                CLS_BG_THRESH=0.25,
                CLS_BG_THRESH_LO=0.1,
                HARD_BG_RATIO=0.8,
                REG_FG_THRESH=0.55),
            LOSS_CONFIG=dict(
                CLS_LOSS='BinaryCrossEntropy',
                REG_LOSS='smooth-l1',
                CORNER_LOSS_REGULARIZATION=True,
                LOSS_WEIGHTS=dict(
                    rcnn_cls_weight=1,
                    rcnn_reg_weight=1,
                    rcnn_corner_weight=1,
                    code_weights=[1, 1, 1, 1, 1, 1, 1]))),
        POST_PROCESSING=dict(
            RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
            SCORE_THRESH=0.1,
            OUTPUT_RAW_SCORE=False,
            EVAL_METRIC='kitti',
            NMS_CONFIG=dict(
                MULTI_CLASSES_NMS=False,
                NMS_TYPE='nms_gpu',
                NMS_THRESH=0.1,
                NMS_PRE_MAXSIZE=4096,
                NMS_POST_MAXSIZE=500))))

###############################################################################
# dataset settings
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=db_info_path,
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    use_road_plane=True,
    limit_whole_scene=False,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10))

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

train_pipeline = [
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
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=1,
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
            completely_remove_other_classes=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
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

###############################################################################
lr = 0.001 / 2 * batch_size
optimizer = dict(type='AdamW', lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

runner = dict(type='EpochBasedRunner', max_epochs=40)

checkpoint_config = dict(interval=40)
evaluation = dict(interval=40)

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
resume_from = None
find_unused_parameters = False
workflow = [('train', 1)]
