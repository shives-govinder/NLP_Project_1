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

    Reference points: uniform attention over the whole sequence gives about
    1/T; spreading attention evenly over the label positions (the counting
    shortcut) gives about 1/n_pairs per occurrence; a true induction head
    approaches 1.

    Returns a tensor [n_layers, n_heads] of scores in [0, 1].
    """
    model.eval()
    L, H = cfg.n_layers, cfg.n_heads
    scores = torch.zeros(L, H)
    seen = 0

    for _ in range(n_batches):
        seq, _ = make_icl_batch(
            cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
            device=cfg.device, n_unique=cfg.n_unique,
        )
        patterns = attention_patterns(model, seq)
        B, T = seq.shape
        query_pos = T - 1
        query_syms = seq[:, query_pos]

        # Exemplar symbols sit at positions 0, 2, 4, ...; each symbol's label is
        # one position to its right. With repeats (n_unique < n_pairs) the query
        # symbol can occur several times, so we sum attention over *all* of its
        # label positions.
        sym_positions = torch.arange(0, T - 1, 2, device=seq.device)
        for l in range(L):
            att = patterns[l]  # [B, H, T, T]
            if att is None:
                continue
            for b in range(B):
                match = seq[b, sym_positions] == query_syms[b]
                label_pos = sym_positions[match] + 1
                if len(label_pos) == 0:
                    continue
                scores[l] += att[b, :, query_pos, label_pos].sum(dim=-1).cpu()
        seen += B  # once per batch (not per layer)

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
            cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
            device=cfg.device, n_unique=cfg.n_unique,
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


@torch.no_grad()
def previous_token_score(model, cfg: Config, n_batches: int = 10) -> torch.Tensor:
    """How much each head attends from every label position to the token right
    before it (that label's own symbol). A previous-token head scores near 1.

    Returns a tensor [n_layers, n_heads]. Only layer-0 scores are meaningful
    for the induction circuit, but all layers are returned for completeness.
    """
    model.eval()
    scores = torch.zeros(cfg.n_layers, cfg.n_heads)
    for _ in range(n_batches):
        seq, _ = make_icl_batch(
            cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
            device=cfg.device, n_unique=cfg.n_unique,
        )
        patterns = attention_patterns(model, seq)
        T = seq.shape[1]
        label_pos = torch.arange(1, T - 1, 2, device=seq.device)   # exemplar label slots
        for l, att in enumerate(patterns):
            # att[:, :, p, p-1] for every label position p -> [B, H, K]
            scores[l] += att[:, :, label_pos, label_pos - 1].mean(dim=(0, 2)).cpu()
    return scores / n_batches


@torch.no_grad()
def masked_accuracy(model, cfg: Config, head_masks, batches) -> float:
    """Query accuracy on fixed batches with the given heads ablated (None = intact)."""
    model.eval()
    tot = 0.0
    for seq, tgt in batches:
        tot += float(query_accuracy(model(seq, head_masks=head_masks), tgt).item())
    return tot / max(len(batches), 1)


@torch.no_grad()
def circuit_summary(model, cfg: Config, batches) -> dict:
    """Everything needed to compare the induction circuit across models.

    Head *indices* are arbitrary per training run (a different seed can put the
    previous-token head in any slot), so comparisons across generations should
    use the summary numbers (best score, accuracy when the key head or the whole
    last layer is removed), not specific head ids.
    """
    L, H, dev = cfg.n_layers, cfg.n_heads, cfg.device
    last = L - 1
    ind = induction_score(model, cfg)
    prev = previous_token_score(model, cfg)

    def mask_for(layer_heads):
        masks = {}
        for layer, heads in layer_heads.items():
            m = torch.ones(H, device=dev)
            m[heads] = 0.0
            masks[layer] = m
        return masks

    grid = torch.zeros(L, H)
    for l in range(L):
        for h in range(H):
            grid[l, h] = masked_accuracy(model, cfg, mask_for({l: [h]}), batches)

    prev_head = int(prev[0].argmax())
    ind_head = int(ind[last].argmax())
    base_acc = masked_accuracy(model, cfg, None, batches)
    return {
        "test_acc": base_acc,
        "induction_score": ind.tolist(),
        "prev_token_score": prev.tolist(),
        "ablation_grid": grid.tolist(),
        "prev_head": prev_head,
        "prev_head_score": float(prev[0, prev_head]),
        "induction_head": ind_head,
        "induction_head_score": float(ind[last, ind_head]),
        "acc_without_prev_head": float(grid[0, prev_head]),
        "acc_without_last_layer": masked_accuracy(model, cfg, mask_for({last: list(range(H))}), batches),
        "largest_single_head_drop": float(base_acc - grid.min()),
    }
