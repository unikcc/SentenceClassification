#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Bobo Li
@contact: 932974672@qq.com
@file: model.py
@time: 2020/12/12 13:51
"""

import torch
import torch.nn as nn
import numpy as np
from alphabet import Alphabet


class TextCNN(nn.Module):
    def __init__(self, config, alphabet : Alphabet, emb_dim, device):
        super(TextCNN, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(alphabet.size(), emb_dim)
        # self.embeddings.weight.requires_grad = False
        if config['train_mode'] == 'static':
            self.embeddings = self.embeddings.from_pretrained(torch.from_numpy(alphabet.pretrained_emb))
        elif config['train_mode'] == 'fine-tuned':
            self.embeddings.weight.data.copy_(torch.from_numpy(alphabet.pretrained_emb))

        filters = config['filters']
        self.cnn = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, config['output_channels'], [w, emb_dim]),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)) for w in filters])

        self.linear = nn.Linear(config['output_channels'] * len(filters), 2, bias=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.scale = np.sqrt(3.0 / emb_dim)
        self.apply(self._init_esim_weights)

    def _init_esim_weights(self, module):
        """
        Initialise the weights of the ESIM model.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight) # random 
            nn.init.constant_(module.bias.data, 0.0) # 0.0-> 85.45, 0.1-> 85.28
        elif isinstance(module, nn.Conv2d):
            nn.init.uniform_(module.weight.data, -0.1, 0.1) # 81.71
            nn.init.constant_(module.bias.data, 0.0) # 无所谓
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_uniform_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0.0)
            nn.init.constant_(module.bias_hh_l0.data, 0.0)
            hidden_size = module.bias_hh_l0.data.shape[0] // 4
            module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

            if (module.bidirectional):
                nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
                nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
                module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
        if isinstance(module, nn.Embedding) and self.config['train_mode'] == 'random':
            nn.init.uniform_(module.weight, -0.1, 0.1) # 81.71

    def forward(self, input):
        # input: (batch_size, sentence_length)
        # input: (batch_size, sentence_length, emb_dim)
        input = self.embeddings(input).unsqueeze(1)
        # batch_size, output_channel, 1
        cnn = [conv(input) for conv in self.cnn]
        output = torch.cat(cnn, 1).squeeze(2).squeeze(2)
        output = self.dropout(output)
        output = self.linear(output)
        return output

