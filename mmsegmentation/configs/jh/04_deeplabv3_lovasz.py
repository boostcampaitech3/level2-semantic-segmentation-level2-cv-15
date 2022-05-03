_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/dataset.py',  
    '../_base_/default_runtime.py', '../_base_/schedules/scheduler_adamw_poly2.py'
]

model = dict(
    decode_head = dict(
        loss_decode = dict(_delete_=True,per_image = False,type='LovaszLoss', loss_weight=1.0, reduction='none')),
    auxiliary_head = dict(
        loss_decode = dict(_delete_=True,per_image = False,type='LovaszLoss', loss_weight=0.4,reduction='none'))
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
                name='deeplabv3plus_adamw_0.0001_lovasz'
            ))
        # '''
    ])
runner = dict(type='EpochBasedRunner', max_epochs=40)