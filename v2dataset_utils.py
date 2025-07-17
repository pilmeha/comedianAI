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
        # Смещаем на 1 токен: всё кроме последнего — input, всё кроме первого — target
        input_tensor = input_tensor
        target_tensor = input_tensor.clone()
        return input_tensor, target_tensor

