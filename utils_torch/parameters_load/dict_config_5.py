from Configdict import Config

model = dict(
    type = 'cfenet',
    input_size = 300,
    backbone = 'vgg',
    resume_net = True,
    pretrained = 'weights/vgg16_reducedfc.pth',
    CFENET_CONFIGS = {
        'maps': 6,
        'lat_cfes': 2,
        'channels': [512, 1024, 512, 256, 256, 256],
        'ratios': [6, 6, 6, 6, 4, 4],
    },
    backbone_out_channels = (512, 1024, 1024),
    rgb_means = (104, 117, 123),
    p = 0.6,
    num_classes = dict(
        VOC = 21,
        COCO = 81, # for VOC and COCO
        ),
    save_eposhs = 10,
    weights_save = 'weights/'
    )

train_cfg = dict(
    cuda = True,
    warmup = 5,
    per_batch_size = 8,
    init_lr = 0.002,
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        VOC = [90, 120, 140, 160],
        COCO = [150, 200, 250, 300],
        ),
    print_epochs = 10,
    num_workers= 8,
    )


if __name__ == '__main__':

    cfgfile = 'dict_config_5.py'

    cfg = Config.fromfile(cfgfile)
    print(cfg)
    print(cfg.model.type)