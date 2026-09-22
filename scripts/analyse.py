"""Run the interpretability tools on a saved model.

    python scripts/analyse.py results/base.pt

Prints per-head induction scores and ablation accuracies, and saves an
attention heatmap for every (layer, head) on one example sequence to results/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import decode_tokens, make_icl_batch
from src.interpret import ablation_grid, attention_patterns, induction_score, plot_attention
from src.train import TEST_SEED, auto_device, evaluate, fixed_split, load_checkpoint

path = sys.argv[1] if len(sys.argv) > 1 else "results/base.pt"
device = auto_device(sys.argv[2] if len(sys.argv) > 2 else None)
model, cfg, _ = load_checkpoint(path, device=device)
print(f"loaded {path} | {cfg.describe()} | n_unique={cfg.n_unique} | device={device}")

test = evaluate(model, cfg, fixed_split(cfg, TEST_SEED))
print(f"test acc {test['acc']:.3f} (baseline accuracy is the no-induction ceiling for this setting)")

torch.set_printoptions(precision=3, sci_mode=False)
print("\ninduction score [layer x head] (near 1 = induction head):")
print(induction_score(model, cfg))
print("\naccuracy with each head ablated [layer x head] (big drop = head matters):")
print(ablation_grid(model, cfg))

seq, _ = make_icl_batch(1, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
                        device=device, generator=torch.Generator().manual_seed(0),
                        n_unique=cfg.n_unique)
patterns = attention_patterns(model, seq)
print("\nexample sequence:", decode_tokens(seq[0], cfg.n_symbols))
stem = os.path.splitext(os.path.basename(path))[0]
os.makedirs("results", exist_ok=True)
for l, att in enumerate(patterns):
    for h in range(cfg.n_heads):
        out = f"results/{stem}_attn_L{l}H{h}.png"
        plot_attention(seq[0], att[0, h], cfg.n_symbols, out)
print(f"saved attention heatmaps to results/{stem}_attn_L*H*.png")
