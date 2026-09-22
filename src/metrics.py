"""Metrics for the ICL symbol task and for tracking model collapse.

The task is graded on the *query position* (the final token), where the model
must emit the correct label. Beyond accuracy/loss/perplexity we also track the
entropy of the model's output distribution, which is the most direct signature
of collapse: as a model over-represents common patterns, its output
distribution narrows and its entropy falls (Shumailov et al., 2024).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def query_loss(logits: torch.Tensor, target_token: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at the final (query) position. logits: [B, T, V]."""
    return F.cross_entropy(logits[:, -1, :], target_token)


def query_accuracy(logits: torch.Tensor, target_token: torch.Tensor) -> torch.Tensor:
    """Fraction of queries answered with the correct label token."""
    preds = logits[:, -1, :].argmax(dim=-1)
    return (preds == target_token).float().mean()


def perplexity_from_loss(loss: torch.Tensor) -> float:
    return float(torch.exp(loss.detach()).item())


def label_predictions(logits: torch.Tensor, n_symbols: int, n_labels: int) -> torch.Tensor:
    """Predicted *label indices* in [0, n_labels) at the query position.

    Predictions that fall outside the label range are dropped (they are wrong
    anyway) so the returned tensor may be shorter than the batch.
    """
    preds = logits[:, -1, :].argmax(dim=-1) - n_symbols
    return preds[(preds >= 0) & (preds < n_labels)]


def distribution_entropy(counts: torch.Tensor) -> float:
    """Shannon entropy (nats) of a count/probability vector."""
    p = counts.float()
    total = p.sum().clamp(min=1.0)
    p = p / total
    p = p[p > 0]
    return float(-(p * p.log()).sum().item())


def label_entropy(preds: torch.Tensor, n_labels: int) -> float:
    """Entropy of an empirical label distribution given predicted label indices."""
    counts = torch.bincount(preds, minlength=n_labels)
    return distribution_entropy(counts)
