_base_ = [
    '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]
num_stages = 6
num_proposals = 100
model = dict(
    type='SparseRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=20,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=2.0),
                loss_iou=dict(type='GIoULoss', loss_weight=1.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=1.0),
                        dict(type='BBoxL1Cost', weight=2.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=1.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# load pretrained model
load_from = 'myconfigs/voc_mask_sparse/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth'
resume = False

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        eps=1e-8
    )
)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=500,
        end=24000,
        T_max=24000,
        by_epoch=False)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# working dir config
work_dir = './work_dirs/sparse_rcnn_voc_1x'

# log config
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# visualizer config
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)

# data root config
data_root = 'data/VOCdevkit/'

# Modify train pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=[(1333, 480), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(0.8, 0.8), crop_type='relative_range'),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                  'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='VOCDataset',
            data_root=data_root,
            ann_file='VOC2007/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2007/', img='JPEGImages/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VOCDataset',
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/', img='JPEGImages/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=12,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


