# -*- coding: utf-8 -*-

import copy
import functools
import os
import sys
import random
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler



class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero((target.data == self.padding_idx).int())
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class TrainerArgs:
    epochs: int = 10
    learning_rate: float = 1e-4
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    evaluation_steps: int = 1000
    logging_steps: int = 1
    save_steps: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:

    def __init__(
        self,
        model,
        tokenizer,
        args: TrainerArgs,
        data_collator = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        loss_fn: nn.Module = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if loss_fn is None:
            self.loss_fn = LabelSmoothingLoss(tokenizer.vocab_size, tokenizer.padding_token_id)
        else:
            self.loss_fn = loss_fn
        self.optimizer, self.lr_scheduler = optimizers

    def get_data_loader(self, dataset, batch_size, shuffle=True) -> DataLoader:
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "sampler": RandomSampler(dataset) if shuffle else SequentialSampler(dataset),
        }
        return DataLoader(dataset, **dataloader_params)

    def compute_loss(self, output, tgt):
        if self.loss_fn is not None:
            return self.loss_fn(output, tgt)
        else:
            return nn.CrossEntropyLoss()(output, tgt)
    
    def training_step(self, model, batch):
        model.train()
        output = model.forward(**batch)
        output = output.reshape(-1, output.size(-1))
        labels = batch["label"].reshape(-1)
        loss = self.compute_loss(output, labels)
        print(loss)

        del batch
        torch.cuda.empty_cache()

        loss.backward()

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def evaluation_loop(self, model, dataloader):
        model.eval()

        total_loss = 0
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                output = model.forward(**batch)
                loss = self.compute_loss(output, batch["label"])
                total_loss += loss.item()

                del batch
                torch.cuda.empty_cache()

        return total_loss / len(dataloader)

    def train(self):

        start_time = time.time()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        total_steps = 0
        self.model.zero_grad()

        self.train_dataloader = self.get_data_loader(self.train_dataset, self.args.train_batch_size)
        if self.eval_dataset is not None:
            self.eval_dataloader = self.get_data_loader(self.eval_dataset, self.args.eval_batch_size, shuffle=False)

        for epoch in range(self.args.epochs):
            for batch in self.train_dataloader:
                tr_loss_step = self.training_step(self.model, batch)
                tr_loss += tr_loss_step

                if (total_steps + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                self.model.zero_grad()

                total_steps += 1

                if total_steps % self.args.logging_steps == 0:
                    train_loss = tr_loss.item() / total_steps
                    print(f"Epoch {epoch} | Steps {total_steps} | Loss: {train_loss} | Time: {time.time() - start_time}")

                if self.eval_dataset and total_steps % self.args.evaluation_steps == 0:
                    eval_loss = self.evaluation_loop(self.model, self.eval_dataloader)
                    print(f"Validation Loss: {eval_loss} | Time: {time.time() - start_time}")
