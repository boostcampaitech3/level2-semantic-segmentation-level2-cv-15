_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/dataset.py',  
    '../_base_/default_runtime.py', '../_base_/schedules/adamw_cosrestart.py'
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
                name='deeplabv3plus_adamw_cosrestart'
            ))
        # '''
    ])
runner = dict(type='EpochBasedRunner', max_epochs=40)