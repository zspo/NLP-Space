# -*- coding: utf-8 -*-

import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_data import sentence_pairs, vocab2id, Tokenizer
from utils import subsequent_mask

tok = Tokenizer(vocab2id=vocab2id)

class TrainDatasets(Dataset):
    def __init__(self) -> None:
        self.sentence_pairs = sentence_pairs
        self.train_data = []
        for src, tgt in self.sentence_pairs:
            self.train_data.append([tok.encode(src), tok.encode(tgt)])

    def __len__(self):
        return len(self.sentence_pairs)
    
    def __getitem__(self, i):
        return self.train_data[i]


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
        tgt_mask.data
    )
    return tgt_mask

def data_collator(batch, max_length=20):
    src = []
    tgt = []
    s_max_len = max([len(i[0]) for i in batch])
    t_max_len = max([len(i[1]) for i in batch])
    for s, t in batch:
        s = s + [tok.padding_token_id] * (s_max_len - len(s))
        src.append(s)

        t = t + [tok.padding_token_id] * (t_max_len - len(t))
        tgt.append(t)
    
    src = torch.LongTensor(src)
    src_mask = (src != tok.padding_token_id).unsqueeze(-2)
    tgt = torch.LongTensor(tgt)
    tgt = tgt[:, :-1]
    label = tgt[:, 1:]
    tgt_mask = make_std_mask(tgt, tok.padding_token_id)

    return {
        "src": src,
        "tgt": tgt,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "label": label
    }


if __name__ == "__main__":
    train_dataset = TrainDatasets()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=data_collator,
        batch_size=2
    )

    for batch in train_dataloader:
        print(batch)
        break