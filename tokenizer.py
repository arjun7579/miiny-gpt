import string
import json
from config import train_data_path

class CharTokenizer:
    def __init__(self, data_path=train_data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        allowed = set(string.printable)
        text = ''.join([c for c in text if c in allowed])

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])