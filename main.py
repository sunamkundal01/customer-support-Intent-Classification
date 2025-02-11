import pandas as pd
from utils.dataset import load_data
from inference.predict import load_model, predict
from transformers import BertTokenizer

data = load_data('data/customer_intent.json')
intent_to_label = {intent: idx for idx, intent in enumerate(data['label'].unique())}
label_map = {v: k for k, v in intent_to_label.items()}

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = load_model(intent_to_label)

while True:
    text = input("Enter text ('exit' for exit): ")
    if text.lower() == "exit":
        print("Exiting the program.")
        break
    predicted_label = predict(text, model, tokenizer, label_map)
    print("Predicted Intent:", label_map[predicted_label])