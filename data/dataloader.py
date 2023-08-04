#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/29 18:18
# @Author : Hw-Zhao
# @Site : 
# @File : dataloader.py
# @Software: PyCharm
from torch.utils.data.dataloader import DataLoader
from data.data_base.base_minibatchsampler import MinibatchSampler
from data.dataset_HongHu import NewOccHongHuDataset
from data.dataset_LongKou import NewOccLongKouDataset
from data.dataset_HanChuan import NewOccHanChuanDataset


class NewOccHongHuLoader(DataLoader):
    def __init__(self, config):
        self.config = config

        dataset = NewOccHongHuDataset(config=self.config)
        sampler = MinibatchSampler(dataset)
        super(NewOccHongHuLoader, self).__init__(dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 sampler=sampler,
                                                 batch_sampler=None,
                                                 num_workers=self.config['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 timeout=0,
                                                 worker_init_fn=None)


class NewOccLongKouLoader(DataLoader):
    def __init__(self, config):
        self.config = config

        dataset = NewOccLongKouDataset(config=self.config)
        sampler = MinibatchSampler(dataset)
        super(NewOccLongKouLoader, self).__init__(dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  sampler=sampler,
                                                  batch_sampler=None,
                                                  num_workers=self.config['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  timeout=0,
                                                  worker_init_fn=None)


class NewOccHanChuanLoader(DataLoader):
    def __init__(self, config):
        self.config = config

        dataset = NewOccHanChuanDataset(config=self.config)
        sampler = MinibatchSampler(dataset)
        super(NewOccHanChuanLoader, self).__init__(dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   sampler=sampler,
                                                   batch_sampler=None,
                                                   num_workers=self.config['num_workers'],
                                                   pin_memory=True,
                                                   drop_last=False,
                                                   timeout=0,
                                                   worker_init_fn=None)
