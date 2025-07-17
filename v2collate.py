import torch
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    input_ids, target_ids = zip(*batch)
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)

    return input_ids, target_ids
