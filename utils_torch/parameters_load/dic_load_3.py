from easydict import EasyDict as edict
__C = edict()
cfg = __C
__C.DATA_SETS_TYPE = 'kitti'
__C.DATA_DIR = '/media/DataCenter/deeplearning/Kitti/object'


if __name__=='__main__':
    print(cfg.DATA_SETS_TYPE)