# dataset settings
dataset_type = 'UltrasoundVideoDataset'
data_root = '/home/mh/Zhangcx/EXPERIMENT/ISICDM_frames1'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadVideoOneFramesFromFiles', train_or_not=True, to_float32=True),
    dict(type='LoadAnnotationsFrames'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomBlur', prob=2),
    dict(type='GroupRandomFlip', prob=5),
    dict(type='GroupColorJitter', prob=3),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadVideoOneFramesFromFiles', train_or_not=False, to_float32=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotationsFrames'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler_zhang', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        train = True,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        train=False,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline),
)
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessorThreeFrames',
    size=crop_size,
    mean=[53.39, 53.74, 55.30],
    std=[44.05, 44.12, 44.50],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

type='small'
if type == 'small':
    checkpoint_file = '/home/mh/Zhangcx/pths/WTConvNeXt_small_5_300e_ema.pth'
elif type == 'tiny':
    checkpoint_file = '/home/mh/Zhangcx/pths/WTConvNeXt_tiny_5_300e_ema.pth'
elif type == 'base':
    checkpoint_file = '/home/mh/Zhangcx/pths/WTConvNeXt_base_5_300e_ema.pth'
model = dict(
type='EncoderDecoder_one',
data_preprocessor=data_preprocessor,
backbone=dict(
    type='MWConv',
    layer_fuse=False,
    arch=type,
    frames=3,
    out_indices=[0, 1, 2, 3],
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    gap_before_final_norm=False,
    init_cfg=dict(
        type='Pretrained', checkpoint=checkpoint_file,
        prefix='backbone.')
    ),
decode_head=dict(
    type='HFFDecoder',
    compress_ratio=8,
    conv_kernel=1,
    shared_weights=True,
    initial_fusion=True,
    wavekernel_size=3,
    wt_type='adaptive',

    in_channels=[96, 192, 384, 768],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    num_classes=2,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.5)]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(171, 171)))

# default runtime
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
# load_from = None
load_from = '/home/mh/Zhangcx/ckp/ISICDM_MWNet_Best.pth'
resume = False


# schedule
# optimizer
optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=optimizer,
                     clip_grad=None,
                     paramwise_cfg = dict(
                         custom_keys = {
                             'backbone.downsample_layers': dict(lr_mult=1.0),
                             'backbone.stages.0': dict(lr_mult=0.1),
                             'backbone.stages.1': dict(lr_mult=0.3),
                             'backbone.stages.2': dict(lr_mult=0.7),
                             'backbone.stages.3': dict(lr_mult=1.0),
                             'backbone.temporal': dict(lr_mult=1.0),
                             'backbone.gatedmemory': dict(lr_mult=1.0),
                             'backbone.norm0': dict(lr_mult=1.0),
                             'backbone.norm1': dict(lr_mult=1.0),
                             'backbone.norm2': dict(lr_mult=1.0),
                             'backbone.norm3': dict(lr_mult=1.0),
                             'decode_head': dict(lr_mult=1.0)
                         }
                     )
                     )
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=180000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# training schedule for 40k
train_cfg = dict(type='RefreshIterBasedTransLoop', max_iters=180000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))



