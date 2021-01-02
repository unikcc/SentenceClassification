#!/usr/bin/env python  
#_*_ coding:utf-8 _*_  

""" 
@author: libobo
@file: preprocess.py 
@time: 20/12/11 0:01
"""
import yaml
import argparse
import os
from alphabet import Alphabet
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import pickle as pkl
from tqdm import tqdm


class Template:
    def __init__(self, args):
        self.config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
        if args.dataset not in self.config['data_list']:
            raise KeyError("No such dataset named {}.".format(args.dataset))
        self.config['dataset'] = args.dataset
        self.datatype = 'binary'
        if self.config['dataset'] in self.config['datatype']['train_test']:
            self.datatype = 'train_test'
        self.alphabet = Alphabet('word')
        self.set_seed()

    def set_seed(self):
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
    
    def clean_str_sst(self, string):
        """
        Tokenization/string cleaning for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        if self.config['dataset'].startswith('SST'):
            return self.clean_str_sst(string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def read_split_file(self, mode):
        filelist = self.config['data_list'][self.config['dataset']]
        try:
            filename = os.path.join(self.config['dirname'], self.config['dataset'], filelist[mode])
        except:
            return None
        a = open(filename, 'r', encoding='utf-8')
        res = []
        for line in a:
            label, text = int(line[0]), self.clean_str(line[1:]).split()
            res.append((text, label))
        return res
    
    def read_binary_file(self):
        filelist = self.config['data_list'][self.config['dataset']]
        modes = ['pos', 'neg']
        labels = {'pos': 1, 'neg': 0}
        res = []
        for mode in modes:
            filename = os.path.join(self.config['dirname'], self.config['dataset'], filelist[mode])
            # print(filename)
            a = open(filename, 'r', encoding='latin1').read().splitlines()
            for line in a:
                line = self.clean_str(line)
                res.append((line.split(), labels[mode]))
        random.shuffle(res)
        return res
        
        # X, y = zip(*res)
        # train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=self.config['valid_rate'])

    def normalize_word(self, word):
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    def execute(self, data_list):
        res_list = {}
        for key, data in data_list.items():
            cur_res = []
            for line, label in data:
                res_line = []
                for word in line:
                    word = self.normalize_word(word)
                    res_line.append(self.alphabet.get_index(word))
                cur_res.append((res_line, label))
            # self.alphabet.close()
            res_list[key] = cur_res
        return res_list

    def load_pretrain_emb(self, embedding_path, skip_first_row, separator):
        embedd_dim = -1
        embedd_dict = dict()
        if os.path.exists(embedding_path[0]):
            embedding_path = embedding_path[0]
        else:
            embedding_path = embedding_path[1]
        with open(embedding_path, 'r', encoding='utf-8') as file:
            i = 0
            j = 0
            for line in tqdm(file, total=3e6):
                if i == 0:
                    i = i + 1
                    if skip_first_row:
                        _ = line.strip()
                        continue
                j = j + 1
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(separator)
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    if embedd_dim + 1 == len(tokens):
                        embedd = np.empty([1, embedd_dim])
                        embedd[:] = tokens[1:]
                        embedd_dict[tokens[0]] = embedd
                    else:
                        continue
        return embedd_dict, embedd_dim, embedding_path

    def norm2one(self, vec):
        root_sum_square = np.sqrt(np.sum(np.square(vec)))
        return vec / root_sum_square

    def build_pretrain_embedding(self, embedding_path, alphabet, skip_first_row=True, separator=" ", embedd_dim=300,
                                 norm=True):
        embedd_dict = dict()
        if embedding_path != None:
            embedd_dict, embedd_dim, embedding_path = self.load_pretrain_emb(embedding_path, skip_first_row, separator)
        scale = np.sqrt(3.0 / embedd_dim)
        pretrain_emb = np.empty([alphabet.size(), embedd_dim])
        perfect_match = 0
        case_match = 0
        not_match = 0
        for alph, index in alphabet.iteritems():
            if alph in embedd_dict:
                if norm:
                    pretrain_emb[index, :] = self.norm2one(embedd_dict[alph])
                else:
                    pretrain_emb[index, :] = embedd_dict[alph]
                perfect_match += 1
            elif alph.lower() in embedd_dict:
                if norm:
                    pretrain_emb[index, :] = self.norm2one(embedd_dict[alph.lower()])
                else:
                    pretrain_emb[index, :] = embedd_dict[alph.lower()]
                case_match += 1
            else:
                pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
                not_match += 1
        pretrained_size = len(embedd_dict)
        print("Embedding: %s\n     pretrain num:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
            embedding_path, pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet.size()))
        pretrain_emb = np.float32(pretrain_emb)
        self.alphabet.pretrained_emb = pretrain_emb
        return pretrain_emb, embedd_dim

    def run_read_file(self):
        data_list = []
        if self.datatype == 'train_test':
            modes = ['train', 'valid', 'test']
            data_list = list(map(self.read_split_file, modes))
            if data_list[1] is None:
                X, y = zip(*data_list[0])
                train_x, valid_x, train_y, valid_y = train_test_split(X, y,
                                    test_size=self.config['valid_rate'])
                data_list[0] = list(zip(train_x, train_y))
                data_list[1] = list(zip(valid_x, valid_y))
            data_list = {
                'train': data_list[0],
                'valid': data_list[1],
                'test': data_list[2]
                }
        elif self.datatype == 'binary':
            datalist = self.read_binary_file()
            X, y = zip(*datalist)
            kf = StratifiedKFold(n_splits=self.config['kfold'], shuffle=True)
            data_list = []
            for train_index, test_index in kf.split(X, y):
                train_x = [X[w] for w in train_index]
                train_y = [y[w] for w in train_index]
                test_x = [X[w] for w in test_index]
                test_y = [y[w] for w in test_index]

                temp = {'train': list(zip(train_x, train_y)), 'test': list(zip(test_x, test_y))}
                temp['valid'] = temp['test']
                data_list.append(temp)
        return data_list

    def forward(self):
        data_list = self.run_read_file()
        if isinstance(data_list, list):
            processed_list = list(map(self.execute, data_list))
        else:
            processed_list = self.execute(data_list)
        pretrained_emb, emb_dim = self.build_pretrain_embedding(self.config['embedding_path'], self.alphabet, norm=True)
        pkl.dump((processed_list, self.alphabet, pretrained_emb, emb_dim), open(self.config['res_path'].format(self.config['dataset']), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SST-2')
    args = parser.parse_args()
    template = Template(args)
    template.forward()