#!/usr/bin/env python

import os
import sys
import json
from tqdm import tqdm
import torch
from transformers import get_linear_schedule_with_warmup, AdamW, BertConfig, RobertaTokenizer, RobertaConfig
import torch.nn as nn
from sklearn.metrics import f1_score
import yaml

try:
    from scripts.utils import MyDataLoader
    from scripts.model import myClassification
except:
    from utils import MyDataLoader
    from model import myClassification
    from roberta_model import myClassification as myRobertaModel


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class Pipeline:
    def __init__(self):
        basename = os.path.basename(os.getcwd())
        config = yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)

        # config = json.load(open('config.json', 'r'))
        for cfg in config:
            self.__setattr__(cfg, config[cfg])

        self.data_dir = self.data_dir.format(basename)
        self.target_dir = self.target_dir.format(basename)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

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
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

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
        dataiter = tqdm(self.validLoader, total=self.validLoader.__len__(), file=sys.stdout)
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
        for epoch in range(self.epoch_size):
            self.global_epcoh = epoch
            self.train_iter()
            score, loss = self.evaluate_iter()
            if score > best_score:
                best_score = score
                best_iter = epoch
                torch.save({'epoch': epoch,
                            'model': self.model.cpu().state_dict(),
                            'best_score': best_score},
                           os.path.join(self.target_dir, "best_{}.pth.tar".format(0)))
                self.model.to(device)

            elif epoch - best_iter > self.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.patience))
                break

    def main(self):
        #config = PretrainedConfig.from_pretrained(self.bert_path, num_labels=2)
        # print(type(config))
        # config = BertConfig.from_json_file(os.path.join(self.bert_path, 'config.json'))
        # config = BertConfig.from_pretrained(self.bert_path, num_labels=2)
        config = RobertaConfig.from_json_file(self.bert_path + '/' + 'config.json')

        # alldata = MyDataLoader(self, mode='train').getdata(kfold=False)
        self.trainLoader = MyDataLoader(self, mode='train').getdata()
        self.validLoader = MyDataLoader(self, mode='valid').getdata()
        self.testLoader = MyDataLoader(self, mode='test').getdata()

        self.tokenizer = RobertaTokenizer(vocab_file=self.bert_path + '/' + self.vocab_file, merges_file=self.bert_path + '/' + self.merges_file)

        if True:
        # for data in alldata:
            # self.trainLoader, self.validLoader = data
            self.model = myRobertaModel(self.bert_path, config=config).to(device)
            # self.model.bert.resize_token_embeddings(len(self.tokenizer))
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-8},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 1e-8}
            ]

            self.optimizer = AdamW(optimizer_grouped_parameters,
                          lr=float(self.learning_rate))
                        #   eps=float(self.adam_epsilon), weight_decay=1e-6)
            # self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=self.warmup_steps,
                                        #  t_total=self.epoch_size * self.trainLoader.__len__())
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps,
                                         num_training_steps=self.epoch_size * self.trainLoader.__len__())


            self.criterion = nn.CrossEntropyLoss()
            self.forward()



if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.main()
