"""Synthetic data for the in-context-learning (ICL) symbol task.

This reproduces the abstract task from Singh et al. (2024): the model sees a
sequence of (symbol, label) exemplar pairs whose pairing is random but
consistent *within* a sequence, followed by a query symbol that appeared
earlier. To answer, the model must (1) find the earlier occurrence of the
query symbol, (2) read the label one step to its right, and (3) copy it out.
That three-step "algorithm" is exactly what an induction circuit implements.

Token layout (single shared embedding table):
    symbol tokens : ids [0, n_symbols)
    label  tokens : ids [n_symbols, n_symbols + n_labels)

A sequence of ``n_pairs`` exemplars has length ``2 * n_pairs + 1``:
    s0 l0 s1 l1 ... s_{k-1} l_{k-1}  query
The supervised target is the label token for the query, read at the final
position.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def make_icl_batch(
    batch_size: int,
    n_pairs: int,
    n_symbols: int,
    n_labels: int,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
    label_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of ICL symbol-task sequences.

    Args:
        batch_size: number of sequences.
        n_pairs: exemplar pairs per sequence (k).
        n_symbols, n_labels: sizes of the symbol / label universes.
        device: where to place the returned tensors.
        generator: optional torch.Generator (on CPU) for reproducibility.
        label_probs: optional [n_labels] probability vector. When given,
            labels are drawn from this distribution instead of uniformly.
            This is the hook the model-collapse pipeline uses to inject a
            skewed (previous-generation) label distribution.

    Returns:
        seq: LongTensor [batch_size, 2*n_pairs + 1] of token ids.
        target: LongTensor [batch_size] target *label token* for the query.
    """
    B, K = batch_size, n_pairs

    # Distinct symbols per row: argsort of uniform noise gives a per-row permutation.
    noise = torch.rand(B, n_symbols, generator=generator)
    symbols = noise.argsort(dim=1)[:, :K]  # [B, K] distinct symbol ids in [0, n_symbols)

    # Labels for each exemplar (with replacement).
    if label_probs is None:
        labels = torch.randint(0, n_labels, (B, K), generator=generator)
    else:
        label_probs = label_probs.to(dtype=torch.float)
        labels = torch.multinomial(
            label_probs, B * K, replacement=True, generator=generator
        ).view(B, K)

    # Query: pick one exemplar position per row; the query symbol repeats it.
    q_idx = torch.randint(0, K, (B,), generator=generator)
    rows = torch.arange(B)
    query_symbol = symbols[rows, q_idx]         # [B]
    target_label = labels[rows, q_idx]          # [B] in [0, n_labels)

    # Assemble the interleaved sequence.
    seq = torch.empty(B, 2 * K + 1, dtype=torch.long)
    seq[:, 0 : 2 * K : 2] = symbols
    seq[:, 1 : 2 * K : 2] = labels + n_symbols  # shift labels into label-token range
    seq[:, -1] = query_symbol

    target_token = target_label + n_symbols     # supervised as a token id
    return seq.to(device), target_token.to(device)


def make_fixed_set(
    n_batches: int,
    batch_size: int,
    n_pairs: int,
    n_symbols: int,
    n_labels: int,
    seed: int,
    device: str = "cpu",
    label_probs: Optional[torch.Tensor] = None,
):
    """Build a *frozen* list of batches with a fixed seed.

    Because the data is generated on the fly, we obtain disjoint
    train / validation / test splits by using different, fixed seeds for the
    val and test sets (training uses fresh random draws each step). This
    satisfies the brief's train-val-test requirement; document this design
    choice in your write-up and adjust if you prefer materialised splits.
    """
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        batches.append(
            make_icl_batch(
                batch_size, n_pairs, n_symbols, n_labels,
                device=device, generator=g, label_probs=label_probs,
            )
        )
    return batches


def decode_tokens(seq: torch.Tensor, n_symbols: int) -> str:
    """Human-readable rendering of one sequence, for debugging/plots."""
    parts = []
    for tok in seq.tolist():
        if tok < n_symbols:
            parts.append(f"S{tok}")
        else:
            parts.append(f"L{tok - n_symbols}")
    return " ".join(parts)
