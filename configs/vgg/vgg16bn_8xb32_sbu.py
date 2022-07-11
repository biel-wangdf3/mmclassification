_base_ = [
    # '../_base_/models/vgg16bn.py',
    # '../_base_/datasets/imagenet_bs32_pil_resize.py',
    # '../_base_/schedules/imagenet_bs256.py', 
    '../_base_/default_runtime.py'
]


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VGG', depth=16, norm_cfg=dict(type='BN'), num_classes=2),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 1),
    ))


# dataset settings
dataset_type = 'CustomDataset'              # 'CustomDataset'
# classes = ['roller', 'other']        #

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,   # the BatchSize of each GPU when building the dataloader
    workers_per_gpu=4,    # the number of threads per GPU when building dataloader
    train=dict(
        type=dataset_type,
        data_prefix='/home/wdf/workspace/open-mmlab/mmclassification/base_dataset',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/wdf/workspace/open-mmlab/mmclassification/base_dataset',
        # ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/home/wdf/workspace/open-mmlab/mmclassification/base_dataset',
        # ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='accuracy')  # accuracy  precision
evaluation = dict(interval=1, metric='precision', metric_options={'topk': (1, 1)})

# schedules
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)



