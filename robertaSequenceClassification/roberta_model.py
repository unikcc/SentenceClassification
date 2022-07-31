import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel, BertConfig, RobertaModel, RobertaPreTrainedModel


# class myClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.classifiers = nn.Linear(config.hidden_size, config.num_labels)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.init_weights()

#     def forward(self, input_ids, segment_ids, input_mask):
#         outputs = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
#         pooledoutput = self.dropout(outputs[1])
#         logits = self.classifiers(pooledoutput)
#         return logits


class myClassification(nn.Module):
    def __init__(self, path, config):
        super(myClassification, self).__init__()
        self.bert = RobertaModel.from_pretrained(path)
        self.classifiers = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask):
        x = input_ids.tolist()
        # outputs = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        outputs = self.bert(input_ids, attention_mask=input_mask)
        pooledoutput = outputs[0][:, 0]
        # pooledoutput = self.dropout(outputs[1])
        logits = self.classifiers(pooledoutput)
        return logits