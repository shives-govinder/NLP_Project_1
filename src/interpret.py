"""Mechanistic interpretability tools for the ICL model (Project 1, step 4).

Provides the building blocks to (a) read attention patterns, (b) score how
"induction-like" each head is, and (c) ablate heads to test which ones the
model actually relies on. These are starting points - the brief rewards going
further (e.g. following Conmy et al., 2023 for automated circuit discovery).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .config import Config
from .data import make_icl_batch
from .metrics import query_accuracy


@torch.no_grad()
def attention_patterns(model, seq: torch.Tensor) -> List[torch.Tensor]:
    """Return per-layer attention tensors [B, H, T, T] for a batch of sequences."""
    model.eval()
    model(seq, store_attn=True)
    return model.get_attention()


@torch.no_grad()
def induction_score(model, cfg: Config, n_batches: int = 10) -> torch.Tensor:
    """Heuristic induction score per (layer, head).

    For each sequence the query token (final position) repeats an earlier
    symbol; the *label* for that symbol sits one position after that earlier
    symbol. An induction head should attend, from the query position, to that
    label position. We measure the average attention mass a head places there.

    Returns a tensor [n_layers, n_heads] of scores in [0, 1].
    """
    model.eval()
    L, H = cfg.n_layers, cfg.n_heads
    scores = torch.zeros(L, H)
    seen = 0

    for _ in range(n_batches):
        seq, _ = make_icl_batch(
            cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels, device=cfg.device
        )
        patterns = attention_patterns(model, seq)
        B, T = seq.shape
        query_pos = T - 1
        query_syms = seq[:, query_pos]

        # For each row find where the query symbol first appears among exemplar
        # symbol positions (0, 2, 4, ...); the label is at that position + 1.
        sym_positions = torch.arange(0, T - 1, 2, device=seq.device)  # exemplar symbol slots
        for l in range(L):
            att = patterns[l]  # [B, H, T, T]
            if att is None:
                continue
            for b in range(B):
                # locate the matching exemplar symbol
                match = (seq[b, sym_positions] == query_syms[b]).nonzero(as_tuple=True)[0]
                if len(match) == 0:
                    continue
                label_pos = int(sym_positions[match[0]].item()) + 1
                scores[l] += att[b, :, query_pos, label_pos].cpu()
            seen += B

    return scores / max(seen, 1)


@torch.no_grad()
def ablate_head_accuracy(model, cfg: Config, layer: int, head: int, n_batches: int = 20) -> float:
    """Accuracy with a single head zeroed - large drops flag a load-bearing head."""
    model.eval()
    mask = torch.ones(cfg.n_heads)
    mask[head] = 0.0
    head_masks: Dict[int, torch.Tensor] = {layer: mask.to(cfg.device)}

    tot, n = 0.0, 0
    for _ in range(n_batches):
        seq, tgt = make_icl_batch(
            cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels, device=cfg.device
        )
        logits = model(seq, head_masks=head_masks)
        tot += float(query_accuracy(logits, tgt).item())
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def ablation_grid(model, cfg: Config) -> torch.Tensor:
    """Accuracy after ablating each (layer, head) individually. [n_layers, n_heads]."""
    grid = torch.zeros(cfg.n_layers, cfg.n_heads)
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            grid[l, h] = ablate_head_accuracy(model, cfg, l, h)
    return grid


def plot_attention(seq_row: torch.Tensor, attn_row: torch.Tensor, n_symbols: int, path: str) -> None:
    """Save a heatmap of one head's attention for one sequence.

    seq_row: [T] token ids; attn_row: [T, T] attention for one (batch, head).
    Requires matplotlib.
    """
    import matplotlib.pyplot as plt
    from .data import decode_tokens

    labels = decode_tokens(seq_row, n_symbols).split(" ")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_row.cpu().numpy(), aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("attended-to (key)")
    ax.set_ylabel("query position")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
