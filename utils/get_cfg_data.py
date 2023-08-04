import importlib
import numpy as np


def get_cfg_dataloader(dataset):
    if dataset == 'LongKou':
        config = importlib.import_module('.config_LongKou', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccLongKouLoader
    elif dataset == 'HongHu':
        config = importlib.import_module('.config_HongHu', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccHongHuLoader
    elif dataset == 'HanChuan':
        config = importlib.import_module('.config_HanChuan', package='configs').config
        DataLoader = importlib.import_module('.dataloader', package='data').NewOccHanChuanLoader
    else:
        raise NotImplemented

    return config, DataLoader