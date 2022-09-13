#!/usr/bin/env python

import os
import sys
import json
from tqdm import tqdm
import yaml
from attrdict import AttrDict
import torch
from transformers import get_linear_schedule_with_warmup, AdamW, BertConfig
import torch.nn as nn
from sklearn.metrics import f1_score
from utils import MyDataLoader
from model import myClassification

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

class Pipeline:
    def __init__(self):
        config = AttrDict(yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        config.device = device

        if not os.path.exists(config.target_dir):
            os.makedirs(config.target_dir)
        self.config = config

    def train_iter(self):
        #print("size of", self.trainLoader.__len__())
        self.model.train()
        train_data = tqdm(self.trainLoader, total=self.trainLoader.__len__(), file=sys.stdout)
        count, correct_count = 0, 0
        losses = 0
        true_labels, predict_labels = [], []
        for data in train_data:
            input_ids = data['input_ids'].to(device)
            input_masks = data['input_masks'].to(device)
            input_segments = data['input_segments'].to(device)
            input_labels = data['input_labels'].to(device)
            out = self.model(input_ids, input_segments, input_masks)
            loss = self.criterion(out, input_labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

            losses += loss.item()
            true_labels += input_labels.tolist()
            predict_labels += out.argmax(1).tolist()
            correct_count += (out.argmax(1) == input_labels).sum().item()
            count += len(input_ids)
            description = "Epoch {}, loss: {:.4f}, acc: {:.4f}".format(self.global_epcoh, losses / count, correct_count / count)
            train_data.set_description(description)


    def evaluate_iter(self):
        self.model.eval()
        dataloader = self.validLoader
        dataloader = self.testLoader
        dataiter = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)
        # dataiter = tqdm(self.te, total=self.validLoader.__len__(), file=sys.stdout)
        res = []
        correct_count, count = 0, 0
        losses = 0.0
        true_labels, predict_labels = [], []
        for data in dataiter:
            input_ids = data['input_ids'].to(device)
            input_masks = data['input_masks'].to(device)
            input_segments = data['input_segments'].to(device)
            input_labels = data['input_labels'].to(device)
            with torch.no_grad():
                out = self.model(input_ids, input_segments, input_masks)
            loss = self.criterion(out, input_labels)
            losses += loss.item()
            res += out.argmax(1).tolist()
            correct_count += (out.argmax(1) == input_labels).sum().item()
            count += len(input_ids)
            true_labels += input_labels.tolist()
            predict_labels += out.argmax(1).tolist()
            f1 = f1_score(true_labels, predict_labels)
            description = "Valid epoch {}, loss: {:.4f}, acc: {:.4f}, f1:{:.4f}".format(self.global_epcoh, losses / count, correct_count / count, f1)
            dataiter.set_description(description)
        f1 = f1_score(true_labels, predict_labels)
        print("Valid f1: {:.4f}".format(f1))
        return f1, losses / count

    def forward(self):
        best_score, best_iter = 0, 0
        for epoch in range(self.config.epoch_size):
            self.global_epcoh = epoch
            self.train_iter()
            score, loss = self.evaluate_iter()
            if score > best_score:
                best_score = score
                best_iter = epoch
                torch.save({'epoch': epoch,
                            'model': self.model.cpu().state_dict(),
                            'best_score': best_score},
                           os.path.join(self.config.target_dir, "best_{}.pth.tar".format(0)))
                self.model.to(device)

            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break

    def main(self):
        config = BertConfig.from_pretrained(self.config.bert_path, num_labels=2)
        self.trainLoader, self.validLoader, self.testLoader = MyDataLoader(self.config).getdata()

        self.model = myClassification.from_pretrained(self.config.bert_path, config=config).to(device)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-8},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 1e-8}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                        lr=float(self.config.learning_rate),
                        eps=float(self.config.adam_epsilon), weight_decay=1e-6)
        # self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=self.warmup_steps,
                                    #  t_total=self.epoch_size * self.trainLoader.__len__())
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps,
                                        num_training_steps=self.config.epoch_size * self.trainLoader.__len__())


        self.criterion = nn.CrossEntropyLoss()
        self.forward()



if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.main()
