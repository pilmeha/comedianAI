import torch
from torch.utils.data import Dataset

class JokeDataset(Dataset):
    def __init__(self, tokenized_samples, max_length=64):
        self.samples = tokenized_samples
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = sample.ids[:self.max_length]
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        return input_tensor, input_tensor  # (input, target) одинаковы для LM

# import re
# from collections import Counter
# import torch

# class JokeTokenizer:
#     def __init__(self, min_freq=2):
#         self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
#         self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
#         self.vocab_size = 4
#         self.min_freq = min_freq

#     def tokenize(self, text):
#         return re.findall(r"\b\w+\b", text.lower())

#     def build_vocab(self, texts):
#         counter = Counter()
#         for text in texts:
#             counter.update(self.tokenize(text))
#         for word, freq in counter.items():
#             if freq >= self.min_freq:
#                 self.word2idx[word] = self.vocab_size
#                 self.idx2word[self.vocab_size] = word
#                 self.vocab_size += 1

#     def encode(self, text):
#         tokens = self.tokenize(text)
#         return [self.word2idx.get(w, self.word2idx['<unk>']) for w in tokens]

#     def decode(self, ids):
#         return ' '.join(self.idx2word.get(i, '<unk>') for i in ids)

# def load_joke_dataset(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     pairs = []
#     for line in lines:
#         if '<|context|>' in line and '<|joke|>' in line:
#             ctx, joke = line.split('<|joke|>')
#             ctx = ctx.replace('<|context|>', '').strip()
#             joke = joke.strip()
#             pairs.append((ctx, joke))

#     return pairs

# def collate_batch(pairs, tokenizer, device='cpu', max_len=40):
#     batch_contexts = []
#     batch_jokes = []

#     for ctx, joke in pairs:
#         ctx_ids = tokenizer.encode(ctx)[:max_len]
#         joke_ids = tokenizer.encode(joke)[:max_len]
#         joke_ids = [tokenizer.word2idx['<sos>']] + joke_ids + [tokenizer.word2idx['<eos>']]

#         batch_contexts.append(ctx_ids)
#         batch_jokes.append(joke_ids)

#     ctx_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in batch_contexts], batch_first=True, padding_value=0)
#     joke_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in batch_jokes], batch_first=True, padding_value=0)

#     return ctx_padded.to(device), joke_padded.to(device)
