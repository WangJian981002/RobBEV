# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

voxel_size = [0.05, 0.05, 0.2]
out_size_factor = 8

numC_Trans = 64
multi_adj_frame_id_cfg = (1, 8+1, 1)

img_pre_weight = 'work_dirs/LidarSys-bevdet4d-r50-M-d_0.5-cbgs/epoch_12.pth'
pts_pre_weight = 'checkpoints/transfusion_nusc_voxel_L_v2-0.4-mAP.6610-NDS.7071.pth'

model = dict(
    type='BEVFusion4D_tf',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512, 1024, 2048],
        out_channels=128,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=128,
        out_channels=numC_Trans,
        downsample=8),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        base_channels=32,
        sparse_shape=[41, 2048, 2048],
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((32, 32, 64),
                          (64, 64, 128),
                          (128, 128, 256),
                          (256, 256)),
        encoder_paddings=((0, 0, 1),
                          (0, 0, 1),
                          (0, 0, [0, 1, 1]),
                          (0, 0)),
        block_type='basicblock'),
    fusion_neck=dict(
        type='MDA_TAM',
        spatial_size=[[256,256]],
        single_frame_image_feat_channels=numC_Trans,
        adj_image_frames = len(range(*multi_adj_frame_id_cfg)),
        img_feat_channels=numC_Trans*(1+len(range(*multi_adj_frame_id_cfg))),
        pts_feat_channels=512,
        out_channels=512,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=512,
        out_channels=[128, 256, 256],
        layer_nums=[3, 3, 3],
        layer_strides=[1, 2, 2],
        with_cp=True,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[2048, 2048, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[2048, 2048, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-45, 45),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['points','img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevfusion-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=2,#2*8GPUs
    workers_per_gpu=6,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevfusion-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)



# Optimizer
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(8, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.3,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.3,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=10)

evaluation = dict(interval=1, pipeline=test_pipeline)

'''
load_from = 'work_dirs/RobBEV-r50-M-Spatial0.4-d_0.5-mAP.6946-NDS.7235/epoch_9.pth'
# Optimizer
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    step=[2])
runner = dict(type='EpochBasedRunner', max_epochs=3)

evaluation = dict(interval=1, pipeline=test_pipeline)
'''
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]
