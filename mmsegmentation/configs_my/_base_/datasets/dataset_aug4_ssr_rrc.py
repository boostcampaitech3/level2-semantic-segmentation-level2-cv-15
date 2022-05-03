# dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/input/mmseg/'

# class settings
classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type='RandomShadow', num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ChannelShuffle', always_apply=False, p=1.0),
            dict(type='RandomBrightnessContrast', always_apply = False,  p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.20), brightness_by_max=True),
            dict(type='ToGray', always_apply=False, p=1.0),
            dict(type='HueSaturationValue',always_apply=False, p=1.0, hue_shift_limit=(-20, 20), sat_shift_limit=(-40, 40), val_shift_limit=(-20, 20))
        ],
        p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', always_apply=False, p=1.0, blur_limit=(3, 7)),
            dict(type='GaussNoise', always_apply=False, p=1.0, var_limit=(375, 500.0)),
            dict(type='MotionBlur', always_apply=False, p=1.0, blur_limit=(3, 7))
        ],
        p=0.3),
    dict(
        type='ShiftScaleRotate', always_apply=False, p=1.0, shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None
    ),
    dict(
        type='RandomResizedCrop', always_apply=False, p=1.0, height=384, width=384, scale=(0.3, 1.0), ratio=(0.75, 1.3), interpolation=0
    )

    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Flip', always_apply=False, p=1.0),
    #         dict(type='RandomRotate90', always_apply=False, p=1.0)
    #     ],
    #     p=0.3),
    
]
crop_size = (384,384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(384,384)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap=dict(img='image', gt_semantic_seg='mask'),
        update_pad_shape=True,
        ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],  # for TTA
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        classes=classes,
        # palette=palette,
        type=dataset_type,
        img_dir=data_root + "images",
        ann_dir=data_root + "annotations",
        split = data_root + "train_2_test.txt",
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        # palette=palette,
        type=dataset_type,
        img_dir=data_root + "images",
        ann_dir=data_root + "annotations",
        split = data_root + "valid_2_test.txt",
        pipeline=valid_pipeline),
    test=dict(
        classes=classes,
        # palette=palette,
        type=dataset_type,
        img_dir=data_root + "test",
        pipeline=test_pipeline))