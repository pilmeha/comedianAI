import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from v2tokenizer import train_tokenizer
from v2model import JokeModel
from v2dataset_utils import JokeDataset
from tokenizers import Tokenizer
import json
from v2collate import collate_batch
import os
from tqdm import tqdm

MAX_LENGTH = 128
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Подгружаем токенизатор
tokenizer = Tokenizer.from_file("datasets\\v3translated_filtered_more1_jokes.json")

# Загружаем и токенизируем
with open("datasets\\translated_filtered_more1_jokes.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

samples = []
for line in lines:
    if "<|context|>" in line and "<|joke|>" in line:
        samples.append(tokenizer.encode(line))

# Датасет и DataLoader
train_dataset = JokeDataset(samples, max_length=MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Модель
model = JokeModel(vocab_size=tokenizer.get_vocab_size()).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for input_ids, target_ids in train_loader:
        input_ids, target_ids = input_ids.to(DEVICE), target_ids.to(DEVICE)
        logits = model(input_ids)

        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        # отображаем текущий лосс в прогрессбаре
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch}: loss = {total_loss / len(train_loader):.4f}")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/v3translated_filtered_more1_jokes{epoch}.pt")
