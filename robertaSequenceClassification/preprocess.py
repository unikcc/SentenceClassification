#!/usr/bin/env python

import tqdm
import json
import os
import pandas as pd
import pickle as pkl
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer
import yaml


class Preprocessor:
    """
    my class for preprocess data
    """
    def __init__(self):
        basename = os.path.basename(os.getcwd())
        config = yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        # config = json.load(open('config.json', 'r'))

        for cfg in config:
            self.__setattr__(cfg, config[cfg])

        self.data_dir = self.data_dir.format(basename)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        print(self.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        # self.tokenizer = AutoTokenizer.from_pretrained("../../../opt/embeddings/bert-base-cased", use_fast=False)
        # self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_path, vocab_file=self.bert_path + self.vocab_file, merges_file=self.bert_path  + self.merges_file)
        # self.tokenizer = RobertaTokenizer(vocab_file=self.bert_path + '/' + self.vocab_file, merges_file=self.bert_path + '/' + self.merges_file)
        # self.tokenizer = RobertaTokenizer.from

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
        string_ids, input_ids, input_masks, input_segments, input_labels = [], [], [], [], []
        for idx, (string_id, line, label) in enumerate(zip(*data)):
            if idx % 1000 == 0:
                print(idx)
            tokens = self.tokenizer.tokenize(line)
            tokens = tokens[:self.max_seq_length - 2]
            tokens = [self.CLS] + tokens + [self.SEP]

            input_id = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(tokens)
            input_segment = [0] * len(tokens)

            padding = [1] * (self.max_seq_length - len(tokens))
            input_id += padding

            padding = [0] * (self.max_seq_length - len(tokens))
            input_mask += padding
            input_segment += padding

            assert len(input_id) == self.max_seq_length
            assert len(input_mask)== self.max_seq_length
            assert len(input_segment) == self.max_seq_length

            string_ids.append(string_id)
            input_ids.append(input_id)
            input_segments.append(input_segment)
            input_masks.append(input_mask)
            input_labels.append(label)

        print("total num", len(input_ids))
        path = os.path.join(self.data_dir, '{}.pkl'.format(mode))
        pkl.dump((string_ids, input_ids, input_segments, input_masks, input_labels), open(path, 'wb'))

    def execute(self, mode='train'):
        filename = os.path.join(self.source_dir, 'stsa.binary.{}'.format('dev' if mode == 'valid' else mode))
        data = self.read_file(filename)
        self.transform2indices(data, mode.split('_')[0])

    def manage(self):
        modes = ['train', 'valid', 'test']
        for mode  in modes:
            print("Start preprocess {}".format(mode))
            self.execute(mode)
            print("End preprocess {}".format(mode))

if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.manage()
