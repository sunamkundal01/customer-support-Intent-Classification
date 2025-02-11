
## dataset.py
import json
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
from utils.config import Config

tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

class IntentDataset(Dataset):
    def __init__(self, df, label_map):
        self.labels = [label_map[label] for label in df['label']]
        self.texts = [
            tokenizer(text, padding='max_length', max_length=Config.MAX_LENGTH, truncation=True, return_tensors='pt')
            for text in df['text']
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], np.array(self.labels[idx])