from flask import Flask, request, jsonify, render_template
import pandas as pd
from utils.dataset import load_data
from inference.predict import load_model, predict
from transformers import BertTokenizer

app = Flask(__name__)

data = load_data('data/customer_intent.json')
intent_to_label = {intent: idx for idx, intent in enumerate(data['label'].unique())}
label_map = {v: k for k, v in intent_to_label.items()}

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = load_model(intent_to_label)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_intent():
    text = request.form['text']
    if text.lower() == "exit":
        return jsonify({'intent': 'Exiting the program.'})
    predicted_label = predict(text, model, tokenizer, label_map)
    return jsonify({'intent': label_map[predicted_label]})

# if __name__ == '__main__':
#     app.run(debug=True)