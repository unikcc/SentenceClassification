#!/usr/bin/env python

import tqdm
import json
import os
import pandas as pd
import pickle as pkl
from transformers import AutoTokenizer
import yaml
from attrdict import AttrDict


class Preprocessor:
    """
    my class for preprocess data
    """
    def __init__(self):
        basename = os.path.basename(os.getcwd())
        config = AttrDict(yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        # config = json.load(open('config.json', 'r'))

        # for cfg in config:
            # self.__setattr__(cfg, config[cfg])

        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)

        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path, use_fast=False)
        self.config = config

    def read_file(self, filename):
        # a = pd.read_csv(filename)
        a = open(filename, 'r', encoding='utf-8').read().splitlines()
        text = [w[1:].strip() for w in a]
        ids = ['{}_{}'.format(filename, i) for i in range(len(text))]
        labels = [int(w[0]) for w in a]
        # if 'label' in a.columns:
            # labels = list(a['label'])
        return ids, text, labels

    def transform2indices(self, data, mode='train'):
        # string_ids, input_ids, input_masks, input_segments, input_labels = [], [], [], [], []
        res = []
        for idx, (string_id, line, label) in enumerate(zip(*data)):
            if idx % 1000 == 0:
                print(idx)
            tokens = self.tokenizer.tokenize(line)
            tokens = tokens[:self.config.max_seq_length - 2]
            tokens = [self.config.CLS] + tokens + [self.config.SEP]

            input_id = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(tokens)
            input_segment = [0] * len(tokens)

            # padding = [0] * (self.config.max_seq_length - len(tokens))

            # input_id += padding
            # input_mask += padding
            # input_segment += padding

            # assert len(input_id) == self.config.max_seq_length
            # assert len(input_mask)== self.config.max_seq_length
            # assert len(input_segment) == self.config.max_seq_length
            res.append([string_id, input_id, input_mask, input_segment, label])

            # string_ids.append(string_id)
            # input_ids.append(input_id)
            # input_segments.append(input_segment)
            # input_masks.append(input_mask)
            # input_labels.append(label)
        
        # res = (string_ids, input_ids, input_segments, input_masks, input_labels)
        return res

        print("total num", len(input_ids))
        path = os.path.join(self.config.data_dir, '{}.pkl'.format(mode))
        pkl.dump((string_ids, input_ids, input_segments, input_masks, input_labels), open(path, 'wb'))

    def manage(self):
        modes = ['train', 'valid', 'test']
        res = []
        for mode  in modes:
            print("Start preprocess {}".format(mode))

            filename = os.path.join(self.config.source_dir, 'stsa.binary.{}'.format('dev' if mode == 'valid' else mode))
            data = self.read_file(filename)
            r = self.transform2indices(data, mode.split('_')[0])

            res.append(r)
            print("End preprocess {}".format(mode))
        path = os.path.join(self.config.data_dir, 'data.pkl')
        pkl.dump(res, open(path, 'wb'))

if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.manage()
