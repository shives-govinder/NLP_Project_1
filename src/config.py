"""Central configuration for the induction-head / model-collapse experiments.

Everything downstream (data, model, training, collapse) reads its
hyper-parameters from a single ``Config`` object so that experiments are
reproducible and easy to sweep. Load defaults from ``config.yaml`` or
override fields programmatically.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # ---- data / task -------------------------------------------------
    n_symbols: int = 32       # size of the "symbol" universe (query tokens)
    n_labels: int = 8         # size of the "value/number" universe (targets)
    n_pairs: int = 8          # number of (symbol, label) exemplars per sequence
    n_unique: Optional[int] = None  # distinct symbols per sequence (< n_pairs => repeats)
    dense_loss: bool = False  # supervise every induction opportunity, not just the query

    # ---- model -------------------------------------------------------
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2         # 2 is the minimal depth for an induction circuit
    d_mlp: int = 256
    attention_only: bool = False   # set True to remove MLP blocks (cleaner circuit)
    dropout: float = 0.0
    tie_weights: bool = True

    # ---- training ----------------------------------------------------
    steps: int = 3000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 0
    device: str = "cpu"       # overridden to "cuda" automatically in train.py

    # ---- model collapse ----------------------------------------------
    n_generations: int = 5
    variant: str = "base"     # "base" or "extended" (see src/collapse.py)
    collapse_batches: int = 40      # batches used to estimate each generation's output distribution
    collapse_temperature: float = 1.0   # sampling temperature for generated data (<= 0 means argmax)
    dataset_size: int = 200_000         # sequences per generation in the extended variant

    # ---- derived -----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Symbols occupy ids [0, n_symbols); labels occupy [n_symbols, vocab_size)."""
        return self.n_symbols + self.n_labels

    @property
    def max_seq_len(self) -> int:
        """Interleaved sequence: s0 l0 s1 l1 ... s_{k-1} l_{k-1} query.

        The extended variant appends three tokens (query label, next symbol,
        next label), so its models need three more positions.
        """
        return 2 * self.n_pairs + 1 + (3 if self.variant == "extended" else 0)

    # ---- (de)serialisation ------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)

    def describe(self) -> str:
        return (
            f"Config(vocab={self.vocab_size}, seq_len={self.max_seq_len}, "
            f"d_model={self.d_model}, heads={self.n_heads}, layers={self.n_layers}, "
            f"attention_only={self.attention_only})"
        )
