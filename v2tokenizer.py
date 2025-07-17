from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

def train_tokenizer(joke_file, save_path="datasets\\v2joke-tokenizer.json"):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(special_tokens=["<|pad|>", "<|context|>", "<|joke|>", "<|endoftext|>"])

    tokenizer.train([joke_file], trainer)
    tokenizer.save(save_path)
    print(f"✅ Tokenizer saved to {save_path}")

if __name__ == "__main__":
    train_tokenizer("datasets\\translated_filtered_more1_jokes.txt")

# import re
# from collections import Counter
# import torch

# # Спец. токены
# PAD_TOKEN = "<PAD>"
# SOS_TOKEN = "<SOS>"
# EOS_TOKEN = "<EOS>"
# UNK_TOKEN = "<UNK>"

# SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# class SimpleTokenizer:
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = {}
#         self.vocab = set()

#     def build_vocab(self, texts, min_freq=2):
#         counter = Counter()
#         for text in texts:
#             tokens = self.tokenize(text)
#             counter.update(tokens)

#         vocab_words = [word for word, freq in counter.items() if freq >= min_freq]
#         vocab = SPECIAL_TOKENS + vocab_words

#         self.word2idx = {word: i for i, word in enumerate(vocab)}
#         self.idx2word = {i: word for word, i in self.word2idx.items()}
#         self.vocab = set(vocab)

#     def tokenize(self, text):
#         # Простой токенизатор: разбивает по словам и знакам препинания
#         return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

#     def encode(self, text):
#         tokens = self.tokenize(text)
#         return [self.word2idx.get(token, self.word2idx[UNK_TOKEN]) for token in tokens]

#     def decode(self, indices):
#         return " ".join([self.idx2word.get(idx, UNK_TOKEN) for idx in indices])

#     def vocab_size(self):
#         return len(self.word2idx)

# def load_context_joke_pairs(filepath):
#     contexts = []
#     jokes = []
#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             if "<|context|>" in line and "<|joke|>" in line:
#                 context_part, joke_part = line.split("<|joke|>")
#                 context = context_part.replace("<|context|>", "").strip()
#                 joke = joke_part.strip()
#                 contexts.append(context)
#                 jokes.append(joke)
#     return contexts, jokes

# contexts, jokes = load_context_joke_pairs("datasets\\translated_filtered_more1_jokes.txt")

# tokenizer = SimpleTokenizer()
# tokenizer.build_vocab(contexts + jokes, min_freq=2)

# print("Пример токенизации:")
# print("Оригинал:", jokes[0])
# print("Токены:", tokenizer.tokenize(jokes[0]))
# print("Индексы:", tokenizer.encode(jokes[0]))