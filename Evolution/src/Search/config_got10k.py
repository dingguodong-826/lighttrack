import os
import yaml
from easydict import EasyDict as edict

config = edict()

# ------config for general parameters------
config.GPUS = '0,1,2,3,4,5,6,7'
config.WORKERS = 16
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'logs'
config.CHECKPOINT_DIR = 'snapshot'

config.OCEAN = edict()
config.OCEAN.TRAIN = edict()
config.OCEAN.TEST = edict()
config.OCEAN.REID = edict()
config.OCEAN.TUNE = edict()
config.OCEAN.DATASET = edict()
config.OCEAN.DATASET.GOT10K = edict()

# train
config.OCEAN.TRAIN.TEMPLATE_SIZE = 128
config.OCEAN.TRAIN.SEARCH_SIZE = 256
config.OCEAN.TRAIN.STRIDE = 16
config.OCEAN.TRAIN.BATCH = 32
config.OCEAN.TRAIN.WHICH_USE = ['GOT10K']

# augmentation
config.OCEAN.DATASET.SHIFT = 4
config.OCEAN.DATASET.SCALE = 0.05
config.OCEAN.DATASET.COLOR = 1
config.OCEAN.DATASET.FLIP = 0
config.OCEAN.DATASET.BLUR = 0
config.OCEAN.DATASET.GRAY = 0
config.OCEAN.DATASET.MIXUP = 0
config.OCEAN.DATASET.CUTOUT = 0
config.OCEAN.DATASET.CHANNEL6 = 0
config.OCEAN.DATASET.LABELSMOOTH = 0
config.OCEAN.DATASET.ROTATION = 0
config.OCEAN.DATASET.SHIFTs = 64
config.OCEAN.DATASET.SCALEs = 0.18

# got10k
config.OCEAN.DATASET.GOT10K.PATH = 'data/got10k/crop511'
config.OCEAN.DATASET.GOT10K.ANNOTATION = 'data/got10k/all.json'
config.OCEAN.DATASET.GOT10K.RANGE = 100
config.OCEAN.DATASET.GOT10K.USE = 200000



def _update_dict(k, v, model_name):
    if k in ['TRAIN', 'TEST', 'TUNE','REID']:
        for vk, vv in v.items():
            config[model_name][k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'COCO', 'DET', 'YTB', 'LASOT']:
                config[model_name][k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    try:
                        config[model_name][k][vk][vvk] = vvv
                    except:
                        config[model_name][k][vk] = edict()
                        config[model_name][k][vk][vvk] = vvv

    else:
        config[k] = v   # gpu et.


def update_config(config_file):
    """
    ADD new keys to config
    """
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        model_name = list(exp_config.keys())[0]
        if model_name not in ['OCEAN', 'SIAMRPN']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k=OCEAN or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
