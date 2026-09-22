"""A minimal decoder-only Transformer, written from scratch for transparency.

The model is deliberately small so that every component is inspectable for
mechanistic interpretability (Project 1, step 4). Two layers is the minimal
depth at which an induction circuit can form: a *previous-token head* in
layer 0 writes each token's predecessor into the residual stream, and an
*induction head* in layer 1 uses the query to match that information and copy
the associated label (Olsson et al., 2022; Singh et al., 2024).

Set ``attention_only=True`` in the config to drop the MLP blocks and study a
cleaner attention-only circuit.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))
        self._last_attn: Optional[torch.Tensor] = None  # [B, H, T, T] when stored

    def forward(
        self,
        x: torch.Tensor,
        store_attn: bool = False,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)     # [B, H, T, T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if store_attn:
            self._last_attn = att.detach()
        att = self.dropout(att)

        y = att @ v                                                   # [B, H, T, d]
        if head_mask is not None:
            # head_mask: [n_heads] with 0 to ablate a head, 1 to keep it.
            y = y * head_mask.view(1, self.n_heads, 1, 1).to(y.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(d_model, d_mlp)
        self.proj = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_only = cfg.attention_only
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.dropout)
        if not cfg.attention_only:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = MLP(cfg.d_model, cfg.d_mlp, cfg.dropout)

    def forward(self, x, store_attn=False, head_mask=None):
        x = x + self.attn(self.ln1(x), store_attn=store_attn, head_mask=head_mask)
        if not self.attention_only:
            x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        store_attn: bool = False,
        head_masks: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """idx: [B, T] token ids -> logits [B, T, vocab_size].

        head_masks: optional {layer_index: [n_heads] tensor} to ablate heads,
        used by the interpretability tools (src/interpret.py).
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for i, block in enumerate(self.blocks):
            hm = None if head_masks is None else head_masks.get(i)
            x = block(x, store_attn=store_attn, head_mask=hm)
        x = self.ln_f(x)
        return self.head(x)

    def get_attention(self) -> List[Optional[torch.Tensor]]:
        """Return the most recent attention patterns per layer ([B, H, T, T]).

        Call ``model(idx, store_attn=True)`` first.
        """
        return [b.attn._last_attn for b in self.blocks]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
