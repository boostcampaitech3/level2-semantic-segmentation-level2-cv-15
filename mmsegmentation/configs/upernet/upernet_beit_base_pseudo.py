_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/dataset_aug2_randomcrop.py',  
    '../_base_/default_runtime.py', '../_base_/schedules/adamw_cosanealing_0.0002.py'
]


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook'),
        # '''
        # # Wandb Logger 
        
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='segmentation',
                entity='boostcampaitech3',
                name='33_upernet_beit_pseudo'
            ))
        # '''
    ])
runner = dict(type='EpochBasedRunner', max_epochs=100)

data = dict(
    samples_per_gpu=16)