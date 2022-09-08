# dataset settings
dataset_type = 'ADE20KDataset'
data_root = 'dataset'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
crop_size = (352, 352)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(640, 480), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']), 
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),  
        #img_ratios=[0.75, 1.0, 1.25],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),   
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=4, 
    workers_per_gpu=4, 
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TrainDataset/images',
        ann_dir='TrainDataset/masks',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TestDataset/Kvasir/images',
        ann_dir='TestDataset/Kvasir/masks',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='TestDataset/gen_data_test/images',
        # ann_dir='TestDataset/gen_data_test/masks', 
        img_dir='TestDataset/Kvasir/images',
        ann_dir='TestDataset/Kvasir/masks',
        # img_dir='TestDataset/CVC-ClinicDB/images', 
        # ann_dir='TestDataset/CVC-ClinicDB/masks',
        # img_dir='TestDataset/CVC-ColonDB/images',
        # ann_dir='TestDataset/CVC-ColonDB/masks', 
        # img_dir='TestDataset/ETIS-LaribPolypDB/images',
        # ann_dir='TestDataset/ETIS-LaribPolypDB/masks',
        # img_dir='TestDataset/CVC-300/images',  
        # ann_dir='TestDataset/CVC-300/masks',
        pipeline=test_pipeline))