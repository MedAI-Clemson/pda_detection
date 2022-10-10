# config file used for model pretraining
config_pretrain = dict(
    mode_filter =  ['color', 'color_compare'],
    view_filter = ['pdaView', 'pdaRelatedView'],
    val_frac = 0.1, # fraction of data to user for validation
    bs_train = 64,  # batch size for training
    bs_val = 500,  # batch size for validation
    num_workers = 20,  # number of parallel data loading workers
    res = 224, # pixel size along height and width
    device = 'cuda:0',
    model = 'resnet50d',
    num_epochs=4,
    lr = 0.001,
    lr_gamma = 0.92,
    dropout = 0.3,
    weight_decay = 0.001,
    pretrained=True,
    unfreeze_after_n=1,
    lr_unfrozen = 0.00003,
    in_paths = dict(
        train = '/zfs/wficai/pda/model_data/train.csv',
        val = '/zfs/wficai/pda/model_data/val.csv'
    ),
    transforms = dict(
        train = 'train',
        test = 'test'
    )
)

# config file used for model pretraining
config_video =cfg_video = dict(
    mode_filter =  ['color', 'color_compare'],
    view_filter = ['pdaView', 'pdaRelatedView'],
    val_frac = 0.1, # fraction of data to user for validation
    bs_train = 6, 
    bs_val = 10,  
    bs_test = 10, 
    num_workers = 30,  # number of parallel data loading workers
    device = 'cuda:1',
    num_epochs=30,
    lr = 0.001,
    lr_unfrozen = 0.0001,
    lr_gamma = 0.92,
    dropout = 0.3,
    weight_decay = 0.001,
    time_downsample_factor = 7,
    pretrained=True,
    unfreeze_after_n=4,
    in_paths = dict(
        train = '/zfs/wficai/pda/model_data/train.csv',
        val = '/zfs/wficai/pda/model_data/val.csv'
    ),
    video_transforms = dict(
        train = 'train',
        test = 'test'
    )
)

        
        