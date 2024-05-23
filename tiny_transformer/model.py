import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
import numpy as np
import pandas as pd

from configuration import ModelConfig


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


"""
六大组件：
    单词嵌入、位置编码、多头注意力、前馈网络、层归一化、残差连接、
三个中间件：
    编码器、解码器、生成器
Transformer = 编码器 + 解码器 + 生成器
"""


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model) -> None:
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(hidden_size))
        self.b_2 = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores =torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = config.hidden_size
        self.h = config.num_attention_heads

        assert self.d_model % self.h == 0
        self.d_k = self.d_model // self.h
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attn = None
        self.attention = ScaleDotProductAttention()

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = nn.Linear(self.d_model, self.d_model)(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = nn.Linear(self.d_model, self.d_model)(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = nn.Linear(self.d_model, self.d_model)(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        del query, key, value

        return nn.Linear(self.d_model, self.d_model)(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class ResidualConnection(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(ResidualConnection(config), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(Encoder, self).__init__()
        self.encoder_layer = EncoderLayer(config)
        self.layers = clones(self.encoder_layer, config.encoder_layer_nums)
        self.norm = LayerNorm(config.hidden_size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.src_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(ResidualConnection(config), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(Decoder, self).__init__()
        self.decoder_layer = DecoderLayer(config)
        self.layers = clones(self.decoder_layer, config.decoder_layer_nums)
        self.norm = LayerNorm(config.hidden_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.src_embed = nn.Sequential(
            Embedding(config.vocab_size, config.hidden_size),
            PositionalEncoding(config.hidden_size, max_len=config.max_position_embeddings)
        )
        self.tgt_embed = nn.Sequential(
            Embedding(config.vocab_size, config.hidden_size), 
            PositionalEncoding(config.hidden_size, max_len=config.max_position_embeddings)
        )
        self.generator = Generator(config)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, label=None):
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
