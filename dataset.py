import torch
from torch.utils.data import Dataset
from tokenizer import CharTokenizer
import config

class CharDataset(Dataset):
    def __init__(self, split='train'):
        assert split in ('train', 'val')
        self.tokenizer = CharTokenizer()
        config.vocab_size = self.tokenizer.vocab_size

        path = config.train_data_path if split == 'train' else config.val_data_path
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - config.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + config.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y