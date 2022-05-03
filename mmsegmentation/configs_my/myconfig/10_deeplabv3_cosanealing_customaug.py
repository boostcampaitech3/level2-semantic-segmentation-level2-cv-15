_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/dataset_aug1.py',  
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
                name='10_deeplabv3plus_adamw_cosanealing_0.0002_aug1'
            ))
        # '''
    ])
runner = dict(type='EpochBasedRunner', max_epochs=40)