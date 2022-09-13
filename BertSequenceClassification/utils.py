#!/usr/bin/env python
import os
import pickle as pkl
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.string_ids, self.input_ids, self.input_segments, \
        # self.input_masks, self.input_labels = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyDataLoader:
    def __init__(self, args, mode='train'):
        self.args = args
        path = os.path.join(args.data_dir, 'data.pkl')
        self.data = pkl.load(open(path, 'rb'))
    
    def collate(self, batch_data):
        string_ids, input_ids, input_masks, input_segments, input_labels = list(zip(*batch_data))
        max_len = min(len(input_ids), self.args.max_seq_length)
        input_ids = [(w + [0] * max_len)[:max_len] for w in input_ids]
        input_masks = [(w + [0] * max_len)[:max_len] for w in input_masks]
        input_segments = [(w + [0] * max_len)[:max_len] for w in input_segments]

        res = {
            'input_ids': torch.tensor(input_ids).to(self.args.device),
            'input_masks': torch.tensor(input_masks).to(self.args.device),
            'input_segments': torch.tensor(input_segments).to(self.args.device),
            'input_labels': torch.tensor(input_labels).to(self.args.device),
        }

        return res

    def getdata(self):
        modes = 'train valid test'
        res = []
        for i, mode in enumerate(modes.split()):
            cur_data = self.data[i]
            r = DataLoader(MyDataset(cur_data), shuffle=True, batch_size=self.args.batch_size, collate_fn=self.collate)
            res.append(r)
        return res