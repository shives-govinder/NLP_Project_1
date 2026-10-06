"""Causal interventions on the 2-layer model: resample ablation and path patching.

Zero ablation (src/interpret.py) pushes activations off-distribution and can
exaggerate a head's importance. Following Conmy et al. (2023) and Chan et al.
(2022), here a head's output is instead *resampled*: replaced by that head's
output on a different sequence from the same distribution (the batch rolled by
one row). Every sequence shares one layout, so position t always holds the
same kind of token.

Path patching changes what one later layer *reads* from an earlier head, via
only its queries, keys or values. This tests the specific composition of the
induction circuit (Singh et al., 2024): the previous-token head in layer 0
should matter to the induction head in layer 1 through its keys
(K-composition), not its queries or values.

``run`` re-implements Transformer.forward from the model's own weights so that
these interventions are explicit; ``check_run_matches`` confirms it gives the
same logits as the model.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .config import Config


def _heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, T, C = x.shape
    return x.view(B, T, n_heads, C // n_heads).transpose(1, 2)        # [B, H, T, d]


def _attention(attn, xq, xk, xv):
    """One attention layer with separate (layer-normed) inputs for Q, K and V.
    Returns per-head outputs z [B, H, T, d] and patterns [B, H, T, T]."""
    C, H = xq.shape[-1], attn.n_heads
    W = attn.qkv.weight
    q = _heads(xq @ W[:C].T, H)
    k = _heads(xk @ W[C:2 * C].T, H)
    v = _heads(xv @ W[2 * C:].T, H)
    T = xq.shape[1]
    att = (q @ k.transpose(-2, -1)) / math.sqrt(C // H)
    att = att.masked_fill(attn.mask[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
    return att @ v, att


def _head_contributions(attn, z: torch.Tensor) -> torch.Tensor:
    """Each head's write to the residual stream, [B, H, T, C] (proj has no bias)."""
    _, H, _, d = z.shape
    Wp = attn.proj.weight.view(H * d, H, d)                            # out[c] = sum_{h,j} Wp[c,h,j] z[h,j]
    return torch.einsum("bhtd,chd->bhtc", z, Wp)


@torch.no_grad()
def run(
    model,
    idx: torch.Tensor,
    z_patch: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    path_delta: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
):
    """Forward pass with interventions. Returns (logits, cache).

    z_patch     {(layer, head): z [B, T, d]} replaces that head's output, so
                everything downstream sees the replacement.
    path_delta  {(layer, "q"|"k"|"v"): [B, T, C]} is added to the residual
                stream only where ``layer`` reads its queries / keys / values.
    cache       per layer: "z" head outputs, "att" patterns, "contrib" each
                head's residual write [B, H, T, C].
    """
    z_patch, path_delta = z_patch or {}, path_delta or {}
    model.eval()
    T = idx.shape[1]
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T, device=idx.device))[None]
    cache = {"z": [], "att": [], "contrib": []}
    for l, blk in enumerate(model.blocks):
        xs = {p: blk.ln1(x + path_delta[(l, p)]) if (l, p) in path_delta else None for p in "qkv"}
        base = blk.ln1(x)
        z, att = _attention(blk.attn, *(xs[p] if xs[p] is not None else base for p in "qkv"))
        for (pl, h), zz in z_patch.items():
            if pl == l:
                z[:, h] = zz
        contrib = _head_contributions(blk.attn, z)
        cache["z"].append(z)
        cache["att"].append(att)
        cache["contrib"].append(contrib)
        x = x + contrib.sum(dim=1)
        if not blk.attention_only:
            x = x + blk.mlp(blk.ln2(x))
    return model.head(model.ln_f(x)), cache


@torch.no_grad()
def check_run_matches(model, idx: torch.Tensor) -> float:
    """Largest absolute difference between ``run`` and the model's own forward."""
    return float((run(model, idx)[0] - model(idx)).abs().max())


def corrupt(seq: torch.Tensor) -> torch.Tensor:
    """Resampling source: every row paired with a different sequence."""
    return seq.roll(1, dims=0)


def _score(logits: torch.Tensor, pos: int, target: torch.Tensor) -> Tuple[float, float]:
    """(accuracy, mean log-probability of the target) at one position."""
    lp = logits[:, pos].float().log_softmax(-1)
    return (float((logits[:, pos].argmax(-1) == target).float().mean()),
            float(lp.gather(1, target[:, None]).mean()))


@torch.no_grad()
def query_circuit_tests(model, cfg: Config, batches) -> dict:
    """Resample ablation and path patching for the query-label prediction.

    Uses batches of (seq, target) for base sequences, scored at the query.
    Returns accuracies and mean log-probs (summed over batches, then averaged):

    intact                       no intervention
    resample[l][h], zero[l][h]   one head resampled / zeroed
    resample_layer[l]            every head of layer l resampled at once
    path[h][p]                   layer-0 head h resampled only where layer 1
                                 reads its p in {"q","k","v"} (path patching)
    """
    L, H = cfg.n_layers, cfg.n_heads
    acc = {"intact": 0.0, "resample": torch.zeros(L, H), "zero": torch.zeros(L, H),
           "resample_layer": torch.zeros(L), "path": torch.zeros(H, 3)}
    lp = {k: (v.clone() if torch.is_tensor(v) else 0.0) for k, v in acc.items()}

    for seq, tgt in batches:
        pos = seq.shape[1] - 1
        _, clean = run(model, seq)
        _, bad = run(model, corrupt(seq))
        a, p = _score(run(model, seq)[0], pos, tgt)
        acc["intact"] += a
        lp["intact"] += p
        for l in range(L):
            for h in range(H):
                for kind, zz in (("resample", bad["z"][l][:, h]), ("zero", torch.zeros_like(bad["z"][l][:, h]))):
                    a, p = _score(run(model, seq, z_patch={(l, h): zz})[0], pos, tgt)
                    acc[kind][l, h] += a
                    lp[kind][l, h] += p
            a, p = _score(run(model, seq, z_patch={(l, h): bad["z"][l][:, h] for h in range(H)})[0], pos, tgt)
            acc["resample_layer"][l] += a
            lp["resample_layer"][l] += p
        if L > 1:
            for h in range(H):
                delta = bad["contrib"][0][:, h] - clean["contrib"][0][:, h]
                for j, path in enumerate("qkv"):
                    a, p = _score(run(model, seq, path_delta={(1, path): delta})[0], pos, tgt)
                    acc["path"][h, j] += a
                    lp["path"][h, j] += p

    n = max(len(batches), 1)
    out = {}
    for name, d in (("acc", acc), ("logprob", lp)):
        out[name] = {k: (v / n).tolist() if torch.is_tensor(v) else v / n for k, v in d.items()}
    return out


def choice_metrics(logits: torch.Tensor, seq: torch.Tensor, cfg: Config) -> dict:
    """The model's next-symbol distribution on real extended sequences [B, 2K+4]
    (restricted to symbols, as when generating), against the true one."""
    K, S = cfg.n_pairs, cfg.n_symbols
    q = 2 * K
    ctx = seq[:, 0:q:2]
    p_true = torch.zeros(seq.shape[0], S, device=seq.device).scatter_add_(
        1, ctx, torch.full(ctx.shape, 1.0 / K, device=seq.device))
    lps = logits[:, q + 1, :S].float().log_softmax(-1)
    ps = lps.exp()
    query = seq[:, q][:, None]
    return {
        "kl": float((p_true * (p_true.clamp_min(1e-12).log() - lps)).sum(1).mean()),
        "mass_out": float((ps * (p_true == 0)).sum(1).mean()),
        "query_prob": float(ps.gather(1, query).mean()),
        "query_true": float(p_true.gather(1, query).mean()),
    }


@torch.no_grad()
def choice_mechanism(model, cfg: Config, data: torch.Tensor, batch: int = 1000) -> dict:
    """Which heads produce the free next-symbol choice (extended models).

    At the position that predicts the next symbol (the query label, 2K+1):

    attn[l][h]      share of attention on: the query token, context symbols,
                    context labels, and the position itself
    resample[l][h]  choice metrics (see choice_metrics) with that head resampled
    intact          choice metrics with no intervention
    """
    K = cfg.n_pairs
    q = 2 * K
    p = q + 1
    L, H = cfg.n_layers, cfg.n_heads
    sym_pos = torch.arange(0, q, 2)
    lab_pos = torch.arange(1, q, 2)
    attn = torch.zeros(L, H, 4)
    res = [[[] for _ in range(H)] for _ in range(L)]
    intact = []
    n = 0
    for i in range(0, data.shape[0], batch):
        seq = data[i:i + batch].to(cfg.device)
        logits, clean = run(model, seq)
        _, bad = run(model, corrupt(seq))
        intact.append(choice_metrics(logits, seq, cfg))
        for l in range(L):
            a = clean["att"][l][:, :, p].cpu()                     # [B, H, T]
            attn[l, :, 0] += a[:, :, q].sum(0)
            attn[l, :, 1] += a[:, :, sym_pos].sum((0, 2))
            attn[l, :, 2] += a[:, :, lab_pos].sum((0, 2))
            attn[l, :, 3] += a[:, :, p].sum(0)
            for h in range(H):
                pl, _ = run(model, seq, z_patch={(l, h): bad["z"][l][:, h]})
                res[l][h].append(choice_metrics(pl, seq, cfg))
        n += seq.shape[0]

    def mean(ms):
        return {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}

    return {
        "attn_labels": ["query", "context symbols", "context labels", "self"],
        "attn": (attn / n).tolist(),
        "intact": mean(intact),
        "resample": [[mean(res[l][h]) for h in range(H)] for l in range(L)],
    }
