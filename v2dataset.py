import torch
from torch.utils.data import Dataset, DataLoader
from v2tokenizer import tokenizer

class JokeDataset(Dataset):
    def __init__(self, contexts, jokes, tokenizer, max_len=64):
        self.contexts = contexts
        self.jokes = jokes
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        joke = self.jokes[idx]

        input_tokens = self.tokenizer.encode(f"{context}")
        target_tokens = self.tokenizer.encode(f"{joke}")

        input_ids = [self.tokenizer.word2idx[SOS_TOKEN]] + input_tokens + [self.tokenizer.word2idx[EOS_TOKEN]]
        target_ids = input_tokens + [self.tokenizer.word2idx[EOS_TOKEN]] + self.tokenizer.encode(joke)

        return torch.tensor(input_ids[:self.max_len]), torch.tensor(target_ids[:self.max_len])

def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=tokenizer.word2idx[PAD_TOKEN])
    target_batch = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=tokenizer.word2idx[PAD_TOKEN])
    return input_batch, target_batch

from sklearn.model_selection import train_test_split

# Делим на train / val
train_ctx, val_ctx, train_jokes, val_jokes = train_test_split(contexts, jokes, test_size=0.1, random_state=42)

train_dataset = JokeDataset(train_ctx, train_jokes, tokenizer)
val_dataset = JokeDataset(val_ctx, val_jokes, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
