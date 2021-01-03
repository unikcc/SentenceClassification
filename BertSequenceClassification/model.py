import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel, BertConfig


class myClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifiers = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask):
        outputs = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        pooledoutput = self.dropout(outputs[1])
        logits = self.classifiers(pooledoutput)
        return logits
