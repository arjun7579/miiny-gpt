## === FILE: train.py ===

import torch
from torch.utils.data import DataLoader
from model import MiniGPT
from dataset import CharDataset
import config
from tokenizer import CharTokenizer

def main():
    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    train_dataset = CharDataset(split='train')
    val_dataset = CharDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = MiniGPT().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for step in range(config.max_iters):
        model.train()
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

        if step % config.eval_interval == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(config.device), y.to(config.device)
                    _, loss = model(x, y)
                    losses.append(loss.item())
                    if len(losses) >= config.eval_iters:
                        break
            print(f"Step {step}: val loss {sum(losses)/len(losses):.4f}")

    torch.save(model.state_dict(), config.checkpoint_path)
    print("Model saved.")

if __name__ == "__main__":
    main()
