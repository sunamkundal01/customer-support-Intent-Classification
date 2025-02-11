## bert_classifier.py
import torch.nn as nn
from transformers import BertModel
from utils.config import Config

class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        return self.linear(self.dropout(pooled_output))
