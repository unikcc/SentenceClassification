#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Bobo Li
@contact: 932974672@qq.com
@file: util.py
@time: 2020/12/12 14:46
"""

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data):
        self.input_ids, self.labels = zip(*data)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        res = {
            'input_ids': self.input_ids[item],
            'input_labels': self.labels[item]
        }
        return res


class MyDatasetLoader:
    def __init__(self, args, dataset, mode='train'):
        self.args = args
        self.data = dataset[mode]

    def collate_fn(self, batch_data):
        input_ids = [w['input_ids'] for w in batch_data]
        input_labels = [w['input_labels'] for w in batch_data]
        max_length = max(map(len, input_ids))
        # max_length = min(100, max_length)
        input_ids = [w + [0 for _ in range(max_length - len(w))] for w in input_ids]
        res = {
            'input_ids': torch.tensor(input_ids),
            'input_labels': torch.tensor(input_labels)
        }
        return res

    def get_data(self):
        return DataLoader(MyDataset(self.data), shuffle=False, batch_size=self.args['batch_size'], collate_fn=self.collate_fn)


class Metric:
    def __init__(self):
        self.losses, self.predict_labels, self.input_labels = [], [], []
        self.scores = {'train': [], 'valid': [], 'test': []}
        self.best_valid, self.best_test, self.best_iter = 0, 0, 0

    def add_item(self, loss : torch.Tensor, predict_label : torch.Tensor, input_label : torch.Tensor):
        self.losses.append(loss.item())
        self.predict_labels += predict_label.argmax(1).tolist()
        self.input_labels += input_label.tolist()

    def get_score(self, mode='train'):
        loss_score = np.mean(self.losses)
        acc = accuracy_score(self.predict_labels, self.input_labels)
        return loss_score, acc

    def get_final(self, epoch, mode='train'):
        acc = accuracy_score(self.predict_labels, self.input_labels)
        self.scores[mode].append(acc)
        if mode == 'valid' and acc > self.best_valid:
            self.best_valid = acc
            self.best_iter = epoch
        if mode == 'test' and acc > self.best_test:
            self.best_test = acc

    def clear(self):
        self.losses, self.predict_labels, self.input_labels = [], [], []
