"""Quick end-to-end check (~1 min on CPU): trains a tiny model and runs the
interpretability tools. Run from the repo root:  python scripts/smoke_test.py"""
import dataclasses, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import Config
from src.train import train_model, save_checkpoint, load_checkpoint
from src.interpret import induction_score, ablation_grid
from src.collapse import run_collapse

cfg = Config(steps=600, eval_every=200, eval_batches=5, batch_size=128,
             n_generations=2, collapse_batches=5, n_unique=4, dense_loss=True)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
print(cfg.describe(), "| device", cfg.device)

model, hist = train_model(cfg)
print("final val acc:", round(hist[-1]["val_acc"], 3), "| chance:", round(1 / cfg.n_labels, 3), "| no-induction ceiling (n_unique=4): 0.523")
save_checkpoint(model, cfg, "results/smoke.pt", hist)
model, cfg, _ = load_checkpoint("results/smoke.pt", device=cfg.device)
print("induction score [layer x head]:\n", induction_score(model, cfg, n_batches=2))
print("accuracy with each head ablated:\n", ablation_grid(model, cfg))

cfg.steps = 300
cfg.dataset_size = 2000
history, real_ref = run_collapse(cfg)                       # recursive base variant
assert real_ref["query_label_correct"] == 1.0, "real query labels must match the context"
assert "real_eval" in history[-1]
run_collapse(dataclasses.replace(cfg, variant="base_reweight"))   # original protocol

# extended variant: tiny dataset, short training - only checks the code path runs
ext = Config(steps=300, eval_every=150, eval_batches=3, batch_size=128, n_generations=2,
             n_unique=4, dense_loss=True, variant="extended", dataset_size=2000, real_frac=0.1)
ext.device = cfg.device
history, real_ref = run_collapse(ext, save_dir="results/smoke_extended")
assert real_ref["next_symbol_in_context"] == 1.0, "real next symbols must come from the context"
assert history[-1]["extended_stats"]["next_symbol_in_context"] >= 0.0
assert abs(real_ref["next_symbol_is_query"] - 0.344) < 0.03
print("\nSMOKE TEST PASSED")
