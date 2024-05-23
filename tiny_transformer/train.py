# -*- coding: utf-8 -*-

import torch
from torch.optim.lr_scheduler import LambdaLR

from configuration import ModelConfig
from model import Transformer

from trainer import Trainer, TrainerArgs

from dataset import tok, TrainDatasets, data_collator

config = ModelConfig()
config.vocab_size = tok.vocab_size
print(config.vocab_size)
model = Transformer(config=config)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
)
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
lr_scheduler = LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step: rate(
        step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
    ),
)

args = TrainerArgs()
trainer = Trainer(
    model=model,
    tokenizer=tok,
    args=args,
    data_collator=data_collator,
    train_dataset=TrainDatasets(),
    optimizers=(optimizer, lr_scheduler))
trainer.train()