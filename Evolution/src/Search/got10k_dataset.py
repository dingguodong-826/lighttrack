import torch
from torch.utils.data import DataLoader
from .config_got10k import config
from lib.dataset.ocean_normalize import OceanDataset


def get_train_loader():
    gpu_num = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build dataloader, benefit to tracking
    train_set = OceanDataset(config)
    train_loader = DataLoader(train_set, batch_size=config.OCEAN.TRAIN.BATCH * gpu_num,
                              num_workers=config.WORKERS, pin_memory=True, drop_last=True)
    return train_loader




