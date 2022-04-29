_base_ = [
    './models/fcn_hr18.py', './datasets/dataset.py',
    './default_runtime.py', './schedules/scheduler_adamw.py'
]
# model = dict(decode_head=dict(num_classes=7))