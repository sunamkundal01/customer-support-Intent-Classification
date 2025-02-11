## config.py
import torch
class Config:
    EPOCHS = 7
    BATCH_SIZE = 32
    LR = 1e-5
    MAX_LENGTH = 128
    MODEL_NAME = 'bert-base-cased'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = None  # Set dynamically
