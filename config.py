import torch

# Training
batch_size = 32
learning_rate = 3e-4
max_iters = 2000
eval_interval = 100
eval_iters = 50

# Model
block_size = 64
n_layers = 4
n_heads = 4
n_embd = 128
vocab_size = None  # Will be set after tokenizer loads

# File Paths
train_data_path = "data/train.txt"
val_data_path = "data/val.txt"
checkpoint_path = "minigpt_model.pt"

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
