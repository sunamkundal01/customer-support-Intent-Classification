## train.py
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.bert_classifier import BertClassifier
from utils.dataset import load_data, IntentDataset
from utils.config import Config

def train(model, train_data, val_data, label_map):
    train_dataset, val_dataset = IntentDataset(train_data, label_map), IntentDataset(val_data, label_map)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    device = Config.DEVICE
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=Config.LR)
    model.to(device)
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss, total_acc = 0, 0
        for texts, labels in tqdm(train_loader):
            labels = labels.to(device)
            mask = texts['attention_mask'].to(device)
            input_id = texts['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            output = model(input_id, mask)
            loss = criterion(output, labels.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (output.argmax(dim=1) == labels).sum().item()
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_data):.3f}, Accuracy={total_acc/len(train_data):.3f}")
    torch.save(model.state_dict(), 'intent_model.pth')