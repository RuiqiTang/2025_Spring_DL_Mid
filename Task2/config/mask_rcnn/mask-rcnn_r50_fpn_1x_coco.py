_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Import custom modules
custom_imports = dict(imports=['myconfigs.voc_mask_sparse'])

# 修改 num_classes 为 VOC 数据集的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

# 加载预训练模型
load_from = 'myconfigs/voc_mask_sparse/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
resume = False

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 默认运行时配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# 工作目录配置
work_dir = './work_dirs/mask_rcnn_voc_1x'

# 日志配置
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# 可视化配置
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)

# 数据集配置
data_root = 'data/VOCdevkit/'
