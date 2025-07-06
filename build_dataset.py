# build_dataset.py

from datasets import load_dataset
import os
import random

os.makedirs("data", exist_ok=True)

def format_example(example):
    task = f"# Task: {example['text']}\n"
    code = example['code'].strip()
    tests = '\n'.join(example['test_list'])
    return f"{task}{code}\n{tests}\n\n"

def build_mbpp_dataset():
    dataset = load_dataset("mbpp")

    all_samples = []
    for split in ["train", "validation", "test"]:
        for ex in dataset[split]:
            formatted = format_example(ex)
            all_samples.append(formatted)

    random.shuffle(all_samples)

    # 90% train, 10% val
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    with open("data/train.txt", "w", encoding="utf-8") as f:
        f.writelines(train_samples)

    with open("data/val.txt", "w", encoding="utf-8") as f:
        f.writelines(val_samples)

    print(f"✅ Wrote {len(train_samples)} train and {len(val_samples)} val samples.")

if __name__ == "__main__":
    build_mbpp_dataset()
