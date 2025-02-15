_base_ = ['./segformer_mit-b0_8xb2-160k_ade20k-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

# dataset settings
dataset_type = 'BaseSegDataset'
data_root = '/data2/dataset/chenhh_dataset/TLS_dataset'
work_dir = "/data2/chenhh/models/TLS_parsing_segformer"
num_classes = 2
batch_size = 8
crop_size=(512,512)

metainfo = dict(    
                classes=('background', 'TLS'),
                palette=[[128, 128, 128], [0, 0, 255]]
                )
# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=num_classes)
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
                 pipeline=train_pipeline,
                 metainfo=metainfo,
                 img_suffix='.png',
                 seg_map_suffix='.png'
                 ),
    batch_size=batch_size,
    num_workers=4,
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        img_suffix='.png',
        seg_map_suffix='.png'
    ),
    batch_size=batch_size,
    num_workers=4
)
test_dataloader = val_dataloader

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=5000)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))