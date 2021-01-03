#!/usr/bin/env python
import os
import pickle as pkl
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


class MyDataset(Dataset):
    def __init__(self, data):
        self.string_ids, self.input_ids, self.input_segments, \
        self.input_masks, self.input_labels = data

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.input_ids[index]),
            "input_masks": torch.tensor(self.input_masks[index]),
            "input_segments": torch.tensor(self.input_segments[index]),
            "input_labels": torch.tensor(self.input_labels[index])
        }

    def __len__(self):
        return len(self.string_ids)


class MyDataLoader:
    def __init__(self, args, mode='train'):
        self.args = args

        path = os.path.join(args.data_dir, '{}.pkl'.format(mode))
        self.data = pkl.load(open(path, 'rb'))

    def getdata(self, kfold=False):
        if kfold:
            kf = KFold(n_splits=5)
            res = []
            for train_index, test_index in kf.split(self.data[0], self.data[-1]):
                train_data = [[p[w] for w in train_index] for p in self.data]
                test_data = [[p[w] for w in test_index] for p in self.data]
                train_loader = DataLoader(MyDataset(train_data), shuffle=True, batch_size=self.args.batch_size)
                test_loader = DataLoader(MyDataset(test_data), shuffle=True, batch_size=self.args.batch_size)
                res.append((train_loader, test_loader))
            return res

        return DataLoader(MyDataset(self.data), shuffle=True, batch_size=self.args.batch_size)


