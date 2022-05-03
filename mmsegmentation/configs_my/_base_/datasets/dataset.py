# dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/input/mmseg/'

# class settings
classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']


# train_pipeline = [  # Training pipeline.
#     dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
#     dict(type='LoadAnnotations'),  # Second pipeline to load annotations for current image.
#     dict(type='Resize',  # Augmentation pipeline that resize the images and their annotations.
#         img_scale=(2048, 1024),  # The largest scale of image.
#         ratio_range=(0.5, 2.0)), # The augmented scale range as ratio.
#     dict(type='RandomCrop',  # Augmentation pipeline that randomly crop a patch from current image.
#         crop_size=(512, 1024),  # The crop size of patch.
#         cat_max_ratio=0.75),  # The max area ratio that could be occupied by single category.
#     dict(
#         type='RandomFlip',  # Augmentation pipeline that flip the images and their annotations
#         flip_ratio=0.5),  # The ratio or probability to flip
#     dict(type='PhotoMetricDistortion'),  # Augmentation pipeline that distort current image with several photo metric methods.
#     dict(
#         type='Normalize',  # Augmentation pipeline that normalize the input images
#         mean=[123.675, 116.28, 103.53],  # These keys are the same of img_norm_cfg since the
#         std=[58.395, 57.12, 57.375],  # keys of img_norm_cfg are used here as arguments
#         to_rgb=True),
#     dict(type='Pad',  # Augmentation pipeline that pad the image to specified size.
#         size=(512, 1024),  # The output size of padding.
#         pad_val=0,  # The padding value for image.
#         seg_pad_val=255),  # The padding value of 'gt_semantic_seg'.
#     dict(type='DefaultFormatBundle'),  # Default format bundle to gather data in the pipeline
#     dict(type='Collect',  # Pipeline that decides which keys in the data should be passed to the segmentor
#         keys=['img', 'gt_semantic_seg'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
#     dict(
#         type='MultiScaleFlipAug',  # An encapsulation that encapsulates the test time augmentations
#         img_scale=(2048, 1024),  # Decides the largest scale for testing, used for the Resize pipeline
#         flip=False,  # Whether to flip images during testing
#         transforms=[
#             dict(type='Resize',  # Use resize augmentation
#                  keep_ratio=True),  # Whether to keep the ratio between height and width, the img_scale set here will be suppressed by the img_scale set above.
#             dict(type='RandomFlip'),  # Thought RandomFlip is added in pipeline, it is not used when flip=False
#             dict(
#                 type='Normalize',  # Normalization config, the values are from img_norm_cfg
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='ImageToTensor', # Convert image to tensor
#                 keys=['img']),
#             dict(type='Collect', # Collect pipeline that collect necessary keys for testing.
#                 keys=['img'])
#         ])
# ]
# data = dict(
#     samples_per_gpu=2,  # Batch size of a single GPU
#     workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
#     train=dict(  # Train dataset config
#         type='CityscapesDataset',  # Type of dataset, refer to mmseg/datasets/ for details.
#         data_root='data/cityscapes/',  # The root of dataset.
#         img_dir='leftImg8bit/train',  # The image directory of dataset.
#         ann_dir='gtFine/train',  # The annotation directory of dataset.
#         pipeline=[  # pipeline, this is passed by the train_pipeline created before.
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations'),
#             dict(
#                 type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
#             dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
#             dict(type='RandomFlip', flip_ratio=0.5),
#             dict(type='PhotoMetricDistortion'),
#             dict(
#                 type='Normalize',
#                 mean=[123.675, 116.28, 103.53],
#                 std=[58.395, 57.12, 57.375],
#                 to_rgb=True),
#             dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img', 'gt_semantic_seg'])
#         ]),
#     val=dict(  # Validation dataset config
#         type='CityscapesDataset',
#         data_root='data/cityscapes/',
#         img_dir='leftImg8bit/val',
#         ann_dir='gtFine/val',
#         pipeline=[  # Pipeline is passed by test_pipeline created before
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(2048, 1024),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='RandomFlip'),
#                     dict(
#                         type='Normalize',
#                         mean=[123.675, 116.28, 103.53],
#                         std=[58.395, 57.12, 57.375],
#                         to_rgb=True),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img'])
#                 ])
#         ]),
#     test=dict(
#         type='CityscapesDataset',
#         data_root='data/cityscapes/',
#         img_dir='leftImg8bit/val',
#         ann_dir='gtFine/val',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(2048, 1024),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='RandomFlip'),
#                     dict(
#                         type='Normalize',
#                         mean=[123.675, 116.28, 103.53],
#                         std=[58.395, 57.12, 57.375],
#                         to_rgb=True),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img'])
#                 ])
#         ]))




# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (384,384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(384,384), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
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