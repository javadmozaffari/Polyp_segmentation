# yapf:disable
log_config = dict(
    interval=362,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from =  'iter_160000.pth'    
resume_from = None 
workflow = [('train', 1)]
cudnn_benchmark = True

