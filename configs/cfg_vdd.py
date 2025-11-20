_base_ = './base_config.py'
import os

# model settings
model = dict(
    name_path='./configs/cls_vdd.txt',
    prob_thd=0.3,
)

# dataset settings
dataset_type = 'VDDDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=os.path.join(_base_.test_data_root, 'VDD/test/src'),
            seg_map_path=os.path.join(_base_.test_data_root, 'VDD/test/gt')),
        pipeline=test_pipeline))