import torch
from transformers import BertTokenizer
from models.bert_classifier import BertClassifier
from utils.config import Config

def load_model(label_map):
    model = BertClassifier(num_classes=len(label_map))
    model.load_state_dict(torch.load('intent_model.pth', map_location=Config.DEVICE))
    model.to(Config.DEVICE).eval()
    return model

def predict(text, model, tokenizer, label_map):
    inputs = tokenizer(text, padding='max_length', max_length=Config.MAX_LENGTH, truncation=True, return_tensors='pt')
    mask, input_id = inputs['attention_mask'].to(Config.DEVICE), inputs['input_ids'].squeeze(1).to(Config.DEVICE)
    with torch.no_grad():
        output = model(input_id, mask)
    return list(label_map.keys())[output.argmax(dim=1).item()]