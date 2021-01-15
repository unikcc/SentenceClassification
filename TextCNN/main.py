#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Unik
@contact: github.com/unikcc 
@file: main.py
@time: 2020/12/12 14:35
"""

import yaml
import pickle as pkl
import argparse
import torch.nn as nn
from util import MyDatasetLoader, Metric
from model import TextCNN
import torch
from tqdm import tqdm
from torch.optim import optimizer, Adam, Adadelta
import sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import os



class Template:
    def __init__(self, args):
        self.config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
        self.config['dataset'] = args.dataset
        self.device = torch.device('cuda:{}'.format(self.config['cuda_index']) if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(self.config['best_model_path']):
            os.makedirs(self.config['best_model_path'])
        self.set_seed()

    def set_seed(self):
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True

    def train(self, epoch, dataset, mode='train'):
        criterion = nn.CrossEntropyLoss()
        dataiter = tqdm(dataset, total=len(dataset), file=sys.stdout) if mode == 'train' else dataset
        self.metircs.clear()
        for index, data in enumerate(dataiter):
            self.model.zero_grad()
            input_ids = data['input_ids'].to(self.device)
            input_labels = data['input_labels'].to(self.device)
            predict_labels = self.model(input_ids)
            if mode == 'train':
                loss = criterion(predict_labels, input_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda x: x.requires_grad, self.model.linear.parameters()), self.config['max_grad_norm'])
                self.optimizer.step()
                # self.model.zero_grad()
            else:
                with torch.no_grad():
                    loss = criterion(predict_labels, input_labels)
            self.metircs.add_item(loss, predict_labels, input_labels)
            description = "Epoch: {}, loss: {:.4f}, accuracy: {:4f}".format(epoch, *self.metircs.get_score())
            if mode == 'train':
                dataiter.set_description(description)
        if mode != 'train':
            # description = "{} score: {}, loss: {:.4f}, accuracy: {:4f}".format(mode.title(), epoch, *self.metircs.get_score())
            loss, acc = self.metircs.get_score()
            self.metircs.get_final(epoch, mode)
            if self.metircs.best_iter == epoch:
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            }, os.path.join(self.config['best_model_path'], 'best.pt'))
            description = "{} score: {}, loss: {:.4f}, accuracy: {:4f}, best socre: {:.4f}".format(mode.title(), epoch, loss, acc, self.metircs.best_valid if mode == 'valid' else self.metircs.best_test)
            print(description)

    def evaluate(self, epoch, dataset, mode='test'):
        criterion = nn.CrossEntropyLoss()
        dataiter = dataset
        self.metircs.clear()
        path = os.path.join(self.config['best_model_path'], 'best.pt')
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
        self.model.eval()
        for index, data in enumerate(dataiter):
            input_ids = data['input_ids'].to(self.device)
            input_labels = data['input_labels'].to(self.device)
            predict_labels = self.model(input_ids)
            with torch.no_grad():
                loss = criterion(predict_labels, input_labels)
            self.metircs.add_item(loss, predict_labels, input_labels)
            # description = "Epoch: {}, loss: {:.4f}, accuracy: {:4f}".format(epoch, *self.metircs.get_score())
            # description = "{} score: {}, loss: {:.4f}, accuracy: {:4f}".format(mode.title(), epoch, *self.metircs.get_score())
        loss, acc = self.metircs.get_score()
        self.metircs.get_final(epoch, 'test')
        description = "{} score: {}, loss: {:.4f}, accuracy: {:4f}, best socre: {:.4f}".format(mode.title(), epoch,loss, acc, self.metircs.best_test)
        print(description)

    def forward(self, train_data, valid_data, test_data):
        for epoch in range(self.config['epoch_size']):
            self.model.train()
            self.train(epoch, train_data, 'train')
            self.model.eval()
            self.train(epoch, valid_data, 'valid')
            self.train(epoch, test_data, 'test')
            if epoch - self.metircs.best_iter > self.config['patience']:
                break
        self.evaluate(epoch, test_data, 'test')
        loss, acc = self.metircs.get_score()
        print("Final test: {:.4f}".format(acc))
        return acc, self.metircs.best_test

        # self.train(epoch, test_data, 'test')


    def main(self):
        processed_list, alphabet, _, emb_dim = pkl.load(open(self.config['res_path'].format(self.config['dataset']), 'rb'))
        if isinstance(processed_list, dict):
            processed_list = [processed_list]
        scores = []
        for data_list in processed_list:
            train_data = MyDatasetLoader(self.config, data_list, 'train').get_data()
            valid_data = MyDatasetLoader(self.config, data_list, 'valid').get_data()
            test_data = MyDatasetLoader(self.config, data_list, 'test').get_data()

            self.model = TextCNN(self.config, alphabet, emb_dim, self.device).to(self.device)
            for w in self.model.parameters():
                print(w.shape, w.requires_grad)
            self.optimizer = Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.config['lr'], weight_decay=float(self.config['l2']), eps=float(self.config['esp']))
            self.metircs = Metric()
            for name, im in self.model.named_parameters():
                print(im.shape, name, im.requires_grad)
            score = self.forward(train_data, valid_data, test_data)
            scores.append(score)
        print('| valid best | global best|')
        print('| --- | --- |')
        for w in scores:
            print("| {:.4f} | {:.4f} |".format(w[0], w[1]))
        if len(scores) > 1:
            print("valid Avg\tglobal Avg")
            print("| {:.4f} | {:.4f} |".format(np.mean([w[0] for w in scores]), np.mean([w[1] for w in scores])))
        # self.optimizer = Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=float(self.config['l2']), eps=float(self.config['esp']))
        
        # self.optimizer = Adagrad(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.config['learning_rate'], weight_decay=1e-5)
        # self.optimizer = Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.config['lr'], weight_decay=float(self.config['l2']), eps=float(self.config['esp']))
        # self.optimizer = Adadelta(filter(lambda x: x.requires_grad, self.model.parameters()), rho=0.95, eps=1e-5, weight_decay=float(self.config['l2']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MR')
    args = parser.parse_args()
    template = Template(args)
    template.main()
