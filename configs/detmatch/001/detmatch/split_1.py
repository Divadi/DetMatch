outputs_dir = 'outputs/detmatch/'

split_folder = 'ssl_splits'
split_num = 1
split_frac = 0.01 
pretrained_split_frac = {0.01: '001', 0.02: '002', 0.20: '020'}[split_frac]

work_dir = outputs_dir + '{}/detmatch/split_{}'.format(pretrained_split_frac, split_num)

pretrained = dict(
    detector_2d="outputs/detmatch/{}/pretrain_frcnn/split_{}/epoch_12.pth".format(pretrained_split_frac, split_num), 
    detector_3d="outputs/detmatch/{}/pretrain_pvrcnn/split_{}/epoch_40.pth".format(pretrained_split_frac, split_num)
)

load_from = None
resume_from = None

batch_size = 4
num_unlabeled_samples = 1

data_root = 'data/kitti/'
train_lab_info_path = data_root + '{}/kitti_infos_train_proj_3d_lab_{}_{}.pkl'.format(split_folder, split_frac, split_num)
train_unlab_info_path = data_root + '{}/kitti_infos_train_unlab_{}_{}.pkl'.format(split_folder, split_frac, split_num)
db_info_path = data_root + '{}/kitti_dbinfos_train_lab_{}_{}.pkl'.format(split_folder, split_frac, split_num)

dataset_type = 'KittiDataset'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
###############################################################################
###############################################################################

voxel_size = [0.05, 0.05, 0.1]

model = dict(
    type='SSL',
    pretrained=pretrained,
    model_cfg=dict(
        type='MMDetector',
        detector_2d=dict(
            type='FasterRCNN',
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN', requires_grad=False),
                norm_eval=True,
                style='caffe',
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='open-mmlab://detectron2/resnet50_caffe')),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5),
            rpn_head=dict(
                type='RPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            roi_head=dict(
                type='StandardRoIHead',
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                bbox_head=dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=3,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=False,
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        loss_weight=1.0,
                        gamma=2.0,
                        alpha=0.5,
                        reduction='mean'),
                    loss_bbox=dict(type='L1Loss', loss_weight=1.0)))),
        detector_3d=dict(
            type='OpenPCDetDetector',
            dataset_fields=dict(
                class_names=['Pedestrian', 'Cyclist', 'Car'],
                point_feature_encoder=dict(num_point_features=4),
                point_cloud_range=[0, -40, -3, 70.4, 40, 1],
                voxel_size=[0.05, 0.05, 0.1],
                depth_downsample_factor=None),
            voxel_layer=dict(
                max_num_points=5,
                point_cloud_range=[0, -40, -3, 70.4, 40, 1],
                voxel_size=[0.05, 0.05, 0.1],
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
                            unmatched_threshold=0.45)
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
                        NMS_POST_MAXSIZE=500))))),
    ssl_cfg=dict(
        labeled=[
            dict( # (3D) Supervised loss
                type='Opd_Supervised_3D',
                ssl_obj_attr='student.detector_3d', batch_dict_key='stu'),
            dict( # (2D) Supervised loss
                type='TwoStageSupervised_2D',
                loss_detach_keys=[], ssl_obj_attr='student.detector_2d', batch_dict_key='stu')
        ],
        unlabeled=[
            dict( # (3D) Teacher: Generate 3D Boxes
                type='Opd_SimpleTest_3D',
                ssl_obj_attr='teacher.detector_3d', batch_dict_key='tea', out_bboxes_key='3d_bboxes_nms'),
            dict( # (3D) Teacher: Remove teacher augs
                type='BboxesTransform_3D',
                reverse=True, img_metas='tea.img_metas',
                in_bboxes_key='tea.3d_bboxes_nms', out_bboxes_key='tea.3d_bboxes_nms_no_aug'),
            dict( # (3D) Teacher: Add student augs
                type='BboxesTransform_3D',
                reverse=False, img_metas='stu.img_metas',
                in_bboxes_key='tea.3d_bboxes_nms_no_aug', out_bboxes_key='tea.3d_bboxes_nms_stu_aug'),
            dict( # (3D) Teacher: Based on 8/22 Jupyter Notebook analysis, 0.5 3D filter before hung.
                type='MaxScoreFilter',
                cls_includes_bg_pred=False,
                score_thr=0.1,
                in_bboxes_key='tea.3d_bboxes_nms_no_aug', out_bboxes_key='tea.3d_bboxes_nms_no_aug_sc_filt'),

            dict( # (2D) Teacher: Get boxes
                type='SimpleTest_2D',
                ssl_obj_attr='teacher.detector_2d', batch_dict_key='tea', out_bboxes_key='2d_bboxes'),
            dict( # (2D) Teacher: NMS
                type='BboxesNMS_2D',
                nms_cfg=dict(nms_pre=-1, score_thr=0.05, max_num=100, iou_thr=0.5),
                cls_includes_bg_pred=True,
                batch_dict_key='tea', in_bboxes_key='2d_bboxes', out_bboxes_key='2d_bboxes_nms'),
            dict( # (2D) Teacher: Remove teacher augs
                type='BboxesTransform_2D',
                reverse=True, img_metas='tea.img_metas',
                in_bboxes_key='tea.2d_bboxes_nms', out_bboxes_key='tea.2d_bboxes_nms_no_aug'),
            dict( # (2D) Teacher: Based on 8/22 Jupyter Notebook analysis, 0.4 2D filter before hung.
                type='MaxScoreFilter',
                cls_includes_bg_pred=True,
                score_thr=0.1,
                in_bboxes_key='tea.2d_bboxes_nms_no_aug', out_bboxes_key='tea.2d_bboxes_nms_no_aug_sc_filt'),

            dict( # (Fusion) Teacher: Hungarian matching
                type='FusionHungarianMatching',
                assigner_cfg=dict(type='ModHungarianAssigner',
                                  cls_cost=dict(type='DoubleSidedFocalLossCost', weight=2.0),
                                  reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                                  iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
                 cost_thr=-1.5, img_metas='stu.img_metas', # the img_metas here is only used for lidar2img and ori_shape
                 # So, it's imperative that these are matched in the "no aug" space.
                 cls_includes_bg_pred_3d=False, cls_includes_bg_pred_2d=True,
                 in_bboxes_3d_key='tea.3d_bboxes_nms_no_aug_sc_filt',
                 in_bboxes_2d_key='tea.2d_bboxes_nms_no_aug_sc_filt',
                 out_bboxes_3d_key='tea.3d_bboxes_nms_no_aug_hung',
                 out_bboxes_2d_key='tea.2d_bboxes_nms_no_aug_hung'),

            dict( # (3D) Teacher: Add student augs
                type='BboxesTransform_3D',
                reverse=False, img_metas='stu.img_metas',
                in_bboxes_key='tea.3d_bboxes_nms_no_aug_hung', out_bboxes_key='tea.3d_bboxes_nms_stu_aug_hung'),

            dict( # (2D) Teacher: Add student augs
                type='BboxesTransform_2D',
                reverse=False, img_metas='stu.img_metas',
                in_bboxes_key='tea.2d_bboxes_nms_no_aug_hung', out_bboxes_key='tea.2d_bboxes_nms_stu_aug_hung'),

            dict( # (3D) Teacher: Detach
                type='DetachBboxes',
                in_bboxes_key='tea.3d_bboxes_nms_stu_aug_hung', out_bboxes_key='tea.3d_bboxes_nms_stu_aug_hung_dtch'),
            dict( # (2D) Teacher: Detach
                type='DetachBboxes',
                in_bboxes_key='tea.2d_bboxes_nms_stu_aug_hung', out_bboxes_key='tea.2d_bboxes_nms_stu_aug_hung_dtch'),

            ### Consumers ######################################################
            ####################################################################
            dict(
                type='Opd_HardPseudoLabel_3D',
                score_thr=0.1, 
                ssl_obj_attr='student.detector_3d', target_bboxes_key='tea.3d_bboxes_nms_stu_aug_hung_dtch',
                target_batch_dict_key='stu', out_bboxes_key='3d_bboxes_nms', no_nms=False),
            dict( # (2D) HardPseudoLabel cls_loss
                type='HardPseudoLabel_2D',
                score_thr=0.1, 
                cls_includes_bg_pred=True, loss_detach_keys=['loss_rpn_bbox', 'loss_bbox'],
                ssl_obj_attr='student.detector_2d', target_bboxes_key='tea.2d_bboxes_nms_stu_aug_hung_dtch',
                target_img_key='stu.img', target_img_metas_key='stu.img_metas',
                name='hard_pseudo_2d', weight=4), # separate weighting for 2D

            ######### 2D GT CONSISTENCY
            
            dict( # (3D) Student: Project to 2D bboxes
                type='Bboxes3DTo2D', # 3d_bboxes_nms -> 3d_bboxes_nms_2d_proj
                img_metas='stu.img_metas', in_bboxes_key='stu.3d_bboxes_nms',
                out_bboxes_key='stu.3d_bboxes_nms_2d_proj'),
            dict( # (3D) Student: 2D NMS the Projected bboxes
                type='BboxesNMS_2D', # 3d_bboxes_nms_2d_proj -> 3d_bboxes_nms_2d_proj_2d_nms
                nms_cfg=dict(nms_pre=-1, score_thr=0.1, max_num=100, iou_thr=0.5),
                cls_includes_bg_pred=False,
                batch_dict_key='stu', in_bboxes_key='3d_bboxes_nms_2d_proj',
                out_bboxes_key='3d_bboxes_nms_2d_proj_2d_nms'),

            dict( # (2D) Teacher: Detach
                type='DetachBboxes',
                in_bboxes_key='tea.2d_bboxes_nms_no_aug_hung', out_bboxes_key='tea.2d_bboxes_nms_no_aug_hung_dtch'),
            dict(
                type='FusionHungarianMatching',
                assigner_cfg=dict(type='ModHungarianAssigner',
                                  cls_cost=dict(type='DoubleSidedFocalLossCost', weight=2.0),
                                  reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                                  iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
                 cost_thr=-1.5, img_metas='stu.img_metas',
                 project_3d_to_2d=False, # still need to match in no aug space beacuse ori_shape is used.
                 cls_includes_bg_pred_3d=False, cls_includes_bg_pred_2d=True,
                 in_bboxes_3d_key='stu.3d_bboxes_nms_2d_proj_2d_nms',
                 in_bboxes_2d_key='tea.2d_bboxes_nms_no_aug_hung_dtch',
                 out_bboxes_3d_key='stu.3d_bboxes_nms_2d_proj_2d_nms_hung',
                 out_bboxes_2d_key='tea.2d_bboxes_nms_no_aug_hung_dtch_hung'),
            dict( # (3D) Student: Add 2D student augs
                type='BboxesTransform_2D',
                reverse=False, img_metas='stu.img_metas',
                in_bboxes_key='stu.3d_bboxes_nms_2d_proj_2d_nms_hung', out_bboxes_key='stu.3d_bboxes_nms_2d_proj_2d_nms_hung_stu_aug'),
            dict( # (2D) Teacher: Add 2D student augs
                type='BboxesTransform_2D',
                reverse=False, img_metas='stu.img_metas',
                in_bboxes_key='tea.2d_bboxes_nms_no_aug_hung_dtch_hung', out_bboxes_key='tea.2d_bboxes_nms_no_aug_hung_dtch_hung_stu_aug'),
            dict(
                type='HungarianConsistency',
                loss_cls_cfg=dict(type='FocalLoss', loss_weight=1.0, reduction='mean'),
                loss_l1_cfg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
                loss_iou_cfg=dict(type='GIoULoss', loss_weight=1.0, reduction='mean'),
                loss_weights_cfg=dict(cls_loss=2, l1_loss=5 * 4, iou_loss=2), 
                # * 4 here, since looking at match_cost and torch.cdist, not normalized by xyxy (not divided by 4)
                cls_includes_bg_pred_in=False, cls_includes_bg_pred_target=True,
                in_bboxes_key='stu.3d_bboxes_nms_2d_proj_2d_nms_hung_stu_aug',
                target_bboxes_key='tea.2d_bboxes_nms_no_aug_hung_dtch_hung_stu_aug',
                target_img_metas_key='stu.img_metas',
                name='2D_to_3D_hung'), # so needs to be in student 2d space now

            ### Visualizations #################################################
            ####################################################################

            dict(
                type='NumPreds',
                bboxes_key='tea.3d_bboxes_nms_stu_aug_hung_dtch', out_name='num_tea_hung'),
            dict(
                type='NumPreds',
                bboxes_key='tea.2d_bboxes_nms_no_aug_hung_dtch_hung', out_name='2D_to_3D_hung'),
            dict(
                type='Vis3D',
                class_names=class_names, batch_dict_key='stu',
                tea_bboxes_key='tea.3d_bboxes_nms_stu_aug_hung_dtch', stu_bboxes_key='stu.3d_bboxes_nms',
                vis_idxs=train_unlab_info_path, vis_idxs_interval=50,
                out_name_prefix='tea'),
        ]),
    # model training and testing settings
    train_cfg=dict(
        ssl=dict(
            set_teacher_eval=True, # needed for Opd i think
            ema_params=dict(
                ema_decay=0.999, true_avg_rampup=True,
                rampup_start_decay=0.99),
            weight_params=dict(weight=1)),
        teacher=None,  # should be unused
        student=dict(
            detector_2d=dict(
                rpn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=False),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False),
                rpn_proposal=dict(
                    nms_pre=2000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0),
                rcnn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False)),
            detector_3d=dict(
                assigner=[
                    dict(  # for Pedestrian
                        type='MaxIoUAssigner',
                        iou_calculator=dict(type='BboxOverlapsNearest3D'),
                        pos_iou_thr=0.35,
                        neg_iou_thr=0.2,
                        min_pos_iou=0.2,
                        ignore_iof_thr=-1),
                    dict(  # for Cyclist
                        type='MaxIoUAssigner',
                        iou_calculator=dict(type='BboxOverlapsNearest3D'),
                        pos_iou_thr=0.35,
                        neg_iou_thr=0.2,
                        min_pos_iou=0.2,
                        ignore_iof_thr=-1),
                    dict(  # for Car
                        type='MaxIoUAssigner',
                        iou_calculator=dict(type='BboxOverlapsNearest3D'),
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.45,
                        min_pos_iou=0.45,
                        ignore_iof_thr=-1),
                ],
                allowed_border=0,
                pos_weight=-1,
                debug=False))),
    test_cfg=dict(
        teacher=dict(
            detector_2d=dict(
                rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0),
                rcnn=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)),
            detector_3d=dict()),
        student=dict(
            detector_2d=dict(
                rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0),
                rcnn=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)),
            detector_3d=dict())))

###############################################################################
###############################################################################

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

input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
###############################################################################

labeled_shared_pipeline = [
    dict(type='LoadImageFromFile'),
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
        with_bbox=True,
        with_label=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict( # must be after object sampling to rescale 2D
        type='Resize',
        img_scale=[(640, 192), (2560, 768)],
        multiscale_mode='range',
        keep_ratio=True),
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
    ### UBteacher Augs ############################################
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
    ###############################################################
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
                                 'img', 'gt_bboxes', 'gt_labels'])
]

labeled_teacher_pipeline = [
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img'])
]

###############################################################################
unlabeled_shared_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='Resize',
        img_scale=[(640, 192), (2560, 768)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
]

unlabeled_student_pipeline = [
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    ### UBteacher Augs ############################################
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
    ###############################################################
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img'])
]

unlabeled_teacher_pipeline = [
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img'])
]

###############################################################################

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

###############################################################################
###############################################################################
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=1,
    train_lab=dict(
        type='TS_SSL_Dataset',
        dataset=dict(
            type='RepeatDataset',
            times=100, # 100 here because the labeled data is 1%, otherwise training slows down a lot re-loading stuff
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=train_lab_info_path,
                split='training',
                pts_prefix='velodyne_reduced',
                pipeline=labeled_shared_pipeline,
                modality=input_modality,
                classes=class_names,
                test_mode=False,
                box_type_3d='LiDAR',
                completely_remove_other_classes=True)),
        student_pipeline=labeled_student_pipeline,
        teacher_pipeline=labeled_teacher_pipeline),
    train_unlab=dict(
        type='TS_SSL_Dataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=train_unlab_info_path,
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=unlabeled_shared_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            filter_empty_gt=False,
            completely_remove_other_classes=True),
        student_pipeline=unlabeled_student_pipeline,
        teacher_pipeline=unlabeled_teacher_pipeline),
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
###############################################################################
lr_3d = 0.001 / 2 * batch_size * (1 + num_unlabeled_samples) * 10
lr_2d = 0.02 / 2 * batch_size * (1 + num_unlabeled_samples)

optimizer = {
    'constructor': 'HybridOptimizerConstructor',
    'student.detector_3d': dict(
        type='AdamW',
        lr=lr_3d,
        betas=(0.95, 0.99),
        weight_decay=0.01,
        step_interval=1),
    'student.detector_2d': dict(
        type='SGD',
        lr=lr_2d,
        momentum=0.9,
        weight_decay=0.0001,
        step_interval=1),
    'teacher': dict( # dummy
        type='SGD',
        lr=1e-9,
        momentum=0.9,
        weight_decay=0.0001,
        step_interval=1)
}

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])
momentum_config = None

runner = dict(type='IterBasedSSLRunner', max_iters=5000)

evaluation = dict(interval=5000)
checkpoint_config = dict(by_epoch=False, interval=5000)

## Special hook
custom_imports = dict(
    imports=['mmdet3d.core.utils.model_iter_epoch',
             'mmdet3d.core.runner.iter_based_ssl_runner'],
    allow_failed_imports=False)
custom_hooks = [dict(type='ModelIterEpochHook'),
                dict(type='WandbVisHook')
                ]

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook'),
           dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='detmatch',
                name='/'.join(work_dir.split('/')[-3:]),
                tags=['kitti']))
                ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = True
workflow = [('train', 1)]
