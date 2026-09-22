"""Quick end-to-end check (~1 min on CPU): trains a tiny model and runs the
interpretability tools. Run from the repo root:  python scripts/smoke_test.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import Config
from src.train import train_model
from src.interpret import induction_score, ablation_grid
from src.collapse import run_collapse

cfg = Config(steps=600, eval_every=200, eval_batches=5, batch_size=128,
             n_generations=2, collapse_batches=5)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
print(cfg.describe(), "| device", cfg.device)

model, hist = train_model(cfg)
print("final val acc:", round(hist[-1]["val_acc"], 3), "| chance:", round(1 / cfg.n_labels, 3))
print("induction score [layer x head]:\n", induction_score(model, cfg, n_batches=2))
print("accuracy with each head ablated:\n", ablation_grid(model, cfg))

cfg.steps = 300
run_collapse(cfg)
print("\nSMOKE TEST PASSED")
