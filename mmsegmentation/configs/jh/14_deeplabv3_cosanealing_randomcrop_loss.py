_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/dataset_aug2_randomcrop.py',  
    '../_base_/default_runtime.py', '../_base_/schedules/adamw_cosanealing_0.0002.py'
]

model = dict(
    decode_head = dict(
        loss_decode = dict(type='FocalLoss', use_sigmoid = True, loss_weight=1.0)),
)

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
                name='14_deeplabv3plus_adamw_cosanealing_0.0002_rc_loss'
            ))
        # '''
    ])
runner = dict(type='EpochBasedRunner', max_epochs=40)