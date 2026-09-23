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
    n_unique: Optional[int] = None,
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
        n_unique: optional number of *distinct* symbols per sequence. When
            smaller than n_pairs, symbols repeat within the context (always
            with the same label), so a single sequence contains several
            induction opportunities. Combine with ``dense_targets`` for a much
            stronger training signal. None = every symbol appears once.

    Returns:
        seq: LongTensor [batch_size, 2*n_pairs + 1] of token ids.
        target: LongTensor [batch_size] target *label token* for the query.
    """
    B, K = batch_size, n_pairs

    def sample_labels(rows: int, cols: int) -> torch.Tensor:
        if label_probs is None:
            return torch.randint(0, n_labels, (rows, cols), generator=generator)
        probs = label_probs.to(device="cpu", dtype=torch.float)
        return torch.multinomial(
            probs, rows * cols, replacement=True, generator=generator
        ).view(rows, cols)

    # Distinct symbols per row: argsort of uniform noise gives a per-row permutation.
    noise = torch.rand(B, n_symbols, generator=generator)
    if n_unique is None or n_unique >= K:
        symbols = noise.argsort(dim=1)[:, :K]   # [B, K] each symbol appears once
        labels = sample_labels(B, K)
    else:
        uniq = noise.argsort(dim=1)[:, :n_unique]            # [B, U] distinct symbols
        uniq_labels = sample_labels(B, n_unique)             # one fixed label per symbol
        slot = torch.randint(0, n_unique, (B, K), generator=generator)
        symbols = uniq.gather(1, slot)                        # [B, K] with repeats
        labels = uniq_labels.gather(1, slot)                  # consistent pairing

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


def dense_targets(seq: torch.Tensor, target_token: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Per-position targets that supervise *every* induction opportunity.

    At each exemplar symbol position p the next token is that symbol's label.
    It is only predictable if the same symbol already appeared earlier in the
    sequence (then the model can copy the earlier label - induction). Those
    positions get the label as target; all others are ignored. The final
    (query) position always gets ``target_token``.

    Returns a LongTensor [B, T] with ``ignore_index`` where there is no target.
    """
    B, T = seq.shape
    tgt = torch.full((B, T), ignore_index, dtype=torch.long, device=seq.device)
    sym_pos = torch.arange(0, T - 1, 2, device=seq.device)    # exemplar symbol positions
    syms = seq[:, sym_pos]                                     # [B, K]
    K = syms.shape[1]
    same = syms.unsqueeze(2) == syms.unsqueeze(1)              # same[b, i, j]
    earlier = torch.tril(torch.ones(K, K), diagonal=-1).bool().to(seq.device)  # j < i
    seen_before = (same & earlier).any(dim=2)                  # [B, K]
    next_labels = seq[:, sym_pos + 1]
    tgt[:, sym_pos] = torch.where(seen_before, next_labels, torch.full_like(next_labels, ignore_index))
    tgt[:, -1] = target_token
    return tgt


def make_fixed_set(
    n_batches: int,
    batch_size: int,
    n_pairs: int,
    n_symbols: int,
    n_labels: int,
    seed: int,
    device: str = "cpu",
    label_probs: Optional[torch.Tensor] = None,
    n_unique: Optional[int] = None,
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
                n_unique=n_unique,
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


# ---------------------------------------------------------------------------
# Extended variant: context + query, then three more tokens
#     s0 l0 ... s_{k-1} l_{k-1}  q  | l_q  s_next  l_next
# The model must (1) give the query's label, (2) choose a next symbol, and
# (3) give that symbol's label. In real data the next symbol is a uniformly
# random context slot's symbol (so it always appears in the context and its
# label can be copied by induction). Step (2) has no single correct answer,
# which is what lets the model's own preferences feed back under recursion.
# ---------------------------------------------------------------------------

def make_extended_batch(
    batch_size: int,
    n_pairs: int,
    n_symbols: int,
    n_labels: int,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
    n_unique: Optional[int] = None,
) -> torch.Tensor:
    """Real extended sequences, LongTensor [batch_size, 2*n_pairs + 4]."""
    seq, target = make_icl_batch(
        batch_size, n_pairs, n_symbols, n_labels,
        device="cpu", generator=generator, n_unique=n_unique,
    )
    B = seq.shape[0]
    rows = torch.arange(B)
    j = torch.randint(0, n_pairs, (B,), generator=generator)   # random context slot
    next_sym = seq[rows, 2 * j]
    next_lab = seq[rows, 2 * j + 1]
    full = torch.cat([seq, target[:, None], next_sym[:, None], next_lab[:, None]], dim=1)
    return full.to(device)


def extended_targets(
    seq: torch.Tensor, n_pairs: int, context_targets: bool = True, ignore_index: int = -100
) -> torch.Tensor:
    """Next-token targets for an extended sequence [B, 2*n_pairs + 4].

    The query position predicts l_q, the l_q position predicts s_next, and the
    s_next position predicts l_next (all read from the sequence itself, so under
    recursion they are whatever the previous generation generated). With
    ``context_targets`` the context also gets the dense induction targets.
    """
    B, T = seq.shape
    q = 2 * n_pairs                                   # query position
    tgt = torch.full((B, T), ignore_index, dtype=torch.long, device=seq.device)
    if context_targets:
        tgt[:, : q + 1] = dense_targets(seq[:, : q + 1], seq[:, q + 1], ignore_index)
    else:
        tgt[:, q] = seq[:, q + 1]
    tgt[:, q + 1] = seq[:, q + 2]
    tgt[:, q + 2] = seq[:, q + 3]
    return tgt
