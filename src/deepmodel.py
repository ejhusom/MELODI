#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep learning for LLMEC.

Author:
    Erik Johannes Husom

Created:
    2024-03-11

"""
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence

from config import config

# Load dataset
data = pd.read_csv(config.MAIN_DATASET_PATH)

# Split the dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(data['prompt'], data['energy_consumption_llm'], test_size=0.2, random_state=42)

# Tokenization
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Numericalize, pad, and create tensors
def process_texts(data_iter):
    processed_text = []
    for text in data_iter:
        processed_text.append(torch.tensor(vocab(tokenizer(text)), dtype=torch.long))
    return pad_sequence(processed_text, batch_first=True, padding_value=vocab['<pad>'])

train_data = process_texts(train_texts)
test_data = process_texts(test_texts)

train_labels = torch.tensor(train_labels.values, dtype=torch.float)
test_labels = torch.tensor(test_labels.values, dtype=torch.float)

from torch.utils.data import Dataset, DataLoader

class EnergyConsumptionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Create Dataset instances
train_dataset = EnergyConsumptionDataset(train_data, train_labels)
test_dataset = EnergyConsumptionDataset(test_data, test_labels)

# Create DataLoader instances
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch.nn as nn

class EnergyConsumptionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EnergyConsumptionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, text):
        embedded = self.embedding(text)
        _, hidden = self.gru(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

# Initialize model
vocab_size = len(vocab)
embedding_dim = 100
model = EnergyConsumptionModel(vocab_size, embedding_dim)


from torch.optim import Adam

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters())

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
total_loss = 0
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        total_loss += loss.item()

print(f'Test Loss: {total_loss / len(test_loader)}')


