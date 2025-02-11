import pandas as pd
from utils.dataset import load_data
from inference.predict import load_model, predict
from transformers import BertTokenizer

data = load_data('data/customer_intent.json')
intent_to_label = {intent: idx for idx, intent in enumerate(data['label'].unique())}