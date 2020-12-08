import _init_paths
import os
import shutil
import time
import math
import pprint
import argparse
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.utils import build_lr_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from utils.utils import create_logger, print_speed, load_pretrain, restore_from, save_model
from core.config import config, update_config
# from core.function import ocean_train
'''2020.08.10'''
from lib.dataset.ocean_normalize import OceanDataset
'''2020.09.07'''
import lib.models.models_supernet as models
from core.supernet_function_tracking import supernet_train
'''2020.10.05'''
import glob
from lib.core.supernet_function_tracking import supernet_retrain
eps = 1e-5

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='finetune the supernet')
    # general
    parser.add_argument('--cfg', type=str, default='experiments/train/Ocean.yaml', help='yaml configure file name')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')
    parser.add_argument('--CHECKPOINT_DIR',type=str,default='')
    parser.add_argument('--OUTPUT_DIR',type=str,default='')
    parser.add_argument('--path_name', type=str, required=True)
    parser.add_argument('--pretrain_dir', type=str, required=True)
    args = parser.parse_args()

    return args


def reset_config(config, args):
    """
    set gpus and workers
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.CHECKPOINT_DIR != '':
        config.CHECKPOINT_DIR = args.CHECKPOINT_DIR
    if args.OUTPUT_DIR != '':
        config.OUTPUT_DIR = args.OUTPUT_DIR


def check_trainable(model, logger):
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params


def get_optimizer(cfg, trainable_params):
    """
    get optimizer
    """

    optimizer = torch.optim.SGD(trainable_params, cfg.OCEAN.TRAIN.LR,
                    momentum=cfg.OCEAN.TRAIN.MOMENTUM,
                    weight_decay=cfg.OCEAN.TRAIN.WEIGHT_DECAY)

    return optimizer

def build_opt_lr(cfg, model, current_epoch=0):
    # fix all backbone first
    for param in model.features.parameters():
        param.requires_grad = False
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    if current_epoch >= cfg.OCEAN.TRAIN.UNFIX_EPOCH:
        if len(cfg.OCEAN.TRAIN.TRAINABLE_LAYER) > 0:  # specific trainable layers
            for layer in cfg.OCEAN.TRAIN.TRAINABLE_LAYER:
                for param in getattr(model.features, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.features, layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
        else:    # train all backbone layers
            for param in model.features.parameters():
                param.requires_grad = True
            for m in model.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.features.parameters():
            param.requires_grad = False
        for m in model.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.features.parameters()),
                          'lr': cfg.OCEAN.TRAIN.LAYERS_LR * cfg.OCEAN.TRAIN.BASE_LR}]
    ########## neck ##########
    try:
        trainable_params += [{'params': model.neck.parameters(),
                                  'lr': cfg.OCEAN.TRAIN.BASE_LR}]
    except:
        pass
    ########## connect model ##########
    trainable_params += [{'params': model.feature_fusor.parameters(),
                          'lr': cfg.OCEAN.TRAIN.BASE_LR}]
    try:
        trainable_params += [{'params': model.supernet_head.parameters(),
                              'lr': cfg.OCEAN.TRAIN.BASE_LR}]
    except:
        trainable_params += [{'params': model.head.parameters(),
                              'lr': cfg.OCEAN.TRAIN.BASE_LR}]
    # print trainable parameter (first check)
    print('==========first check trainable==========')
    for param in trainable_params:
        print(param)

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.OCEAN.TRAIN.MOMENTUM,
                                weight_decay=cfg.OCEAN.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, cfg, epochs=cfg.OCEAN.TRAIN.END_EPOCH)
    lr_scheduler.step(cfg.OCEAN.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def lr_decay(cfg, optimizer):
    if cfg.OCEAN.TRAIN.LR_POLICY == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.8685)
    elif cfg.OCEAN.TRAIN.LR_POLICY == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif cfg.OCEAN.TRAIN.LR_POLICY == 'Reduce':
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
    elif cfg.OCEAN.TRAIN.LR_POLICY == 'log':
        scheduler = np.logspace(math.log10(cfg.OCEAN.TRAIN.LR), math.log10(cfg.OCEAN.TRAIN.LR_END), cfg.OCEAN.TRAIN.END_EPOCH)
    else:
        raise ValueError('unsupported learing rate scheduler')

    return scheduler



def main():
    # [*] args, loggers and tensorboard
    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'OCEAN', 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    if not os.path.exists('./pretrain'):
        os.makedirs('./pretrain')
    model = models.__dict__[config.OCEAN.TRAIN.MODEL](args.path_name, search_size=config.OCEAN.TRAIN.SEARCH_SIZE,
                                                      stride=config.OCEAN.TRAIN.STRIDE).cuda()  # build model
    '''2020.08.13 SET MODEL.TRAIN() HERE'''
    model.train()
    # get optimizer
    if not config.OCEAN.TRAIN.START_EPOCH == config.OCEAN.TRAIN.UNFIX_EPOCH:
        optimizer, lr_scheduler = build_opt_lr(config, model, config.OCEAN.TRAIN.START_EPOCH)
    else:
        optimizer, lr_scheduler = build_opt_lr(config, model, 0)  # resume wrong (last line)

    '''2020.10.05 Load pretrained supernet backbone'''
    logger.info(args.pretrain_dir)
    # logger.info(os.listdir(args.pretrain_dir))
    try:
        backbone_ckpt_path = os.path.join(args.pretrain_dir,'model_best.pth.tar')
        logger.info('########### Loading pretrained backbone from %s ###########' % backbone_ckpt_path)
        logger.info(torch.load(backbone_ckpt_path).keys())
        '''2020.10.20 Use the EMA model'''
        # state_dict = torch.load(backbone_ckpt_path)['state_dict']
        state_dict = torch.load(backbone_ckpt_path)['state_dict_ema']
        model.features.load_state_dict(state_dict, strict=False)
    except:
        logger.info('NOT LOADING PRETRAINED BACKBONE')

    # check trainable again
    print('==========double check trainable==========')
    trainable_params = check_trainable(model, logger)           # print trainable params info

    # parallel
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))

    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    logger.info(lr_scheduler)
    logger.info('model prepare done')

    # [*] train

    for epoch in range(config.OCEAN.TRAIN.START_EPOCH, config.OCEAN.TRAIN.END_EPOCH):
        if epoch == config.OCEAN.TRAIN.EARLY_STOP_EPOCH:
            break
        # build dataloader, benefit to tracking
        train_set = OceanDataset(config)
        train_loader = DataLoader(train_set, batch_size=config.OCEAN.TRAIN.BATCH * gpu_num,
                                  num_workers=config.WORKERS, pin_memory=True, drop_last=True)

        # check if it's time to train backbone
        if epoch == config.OCEAN.TRAIN.UNFIX_EPOCH:
            logger.info('training backbone')
            optimizer, lr_scheduler = build_opt_lr(config, model.module, epoch)
            print('==========double check trainable==========')
            check_trainable(model, logger)  # print trainable params info

        lr_scheduler.step(epoch)
        curLR = lr_scheduler.get_cur_lr()

        model, writer_dict = supernet_retrain(train_loader, model, optimizer, epoch + 1,
                                              curLR, config, writer_dict, logger, device=device)

        # save model
        save_model(model, epoch, optimizer, config.OCEAN.TRAIN.MODEL, config, isbest=False)

    writer_dict['writer'].close()

import time
import os
# import numpy as np

def get_FileModifyTime(filePath):
    # filePath = unicode(filePath,'utf8')
    t = os.path.getmtime(filePath)
    # return TimeStampToTime(t)
    return t

if __name__ == '__main__':
    main()




