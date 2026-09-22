# Project 1 — Interpretability of Model Collapse

COMS4054A Natural Language Processing, Wits 2026. Due **27 Oct 2026, 17:00**.

We reproduce the in-context-learning (ICL) symbol task of Singh et al. (2024), place it inside a model-collapse pipeline (Shumailov et al., 2024), and use mechanistic interpretability to study how collapse affects induction heads.

## The task in one line

A sequence of `(symbol, label)` pairs with a random but consistent pairing, then a repeated query symbol: `S3 L1 S7 L4 S0 L2 ... S7 → L4`. To solve it, the model has to find the earlier `S7` and copy the label that comes after it. That is what an induction circuit does: a previous-token head in layer 0 feeds an induction head in layer 1.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU helps with the multi-generation runs, but everything also works on CPU. The code picks CUDA automatically when it's available.

## How to run

All commands run from the repo root.

| What | Command |
|---|---|
| Sanity check (about 1 minute; run this first) | `python scripts/smoke_test.py` |
| Train the base model (recommended setting) and save it | `python -m src.train --steps 20000 --lr 0.001 --n_unique 4 --dense_loss --device mps --save results/base.pt` |
| Same, attention-only | add `--attention_only` and save to `results/base_attn.pt` |
| Original single-query setting (plateaus; keep as a documented negative result) | `python -m src.train --steps 20000 --lr 0.001 --device mps` |
| Interpretability on a saved model | `python scripts/analyse.py results/base.pt` |
| Model-collapse run, base variant (saves every generation) | `python -m src.collapse --variant base --n_generations 5 --steps 25000 --n_unique 4 --dense_loss --lr 0.001 --device mps --save_dir results/collapse_base` |
| Circuit across generations (table, JSON, 2 figures) | `python scripts/analyse_generations.py results/collapse_base mps` |
| Log to Weights & Biases | add `--wandb` to `src.train` (install `wandb` first) |

`--device mps` uses the GPU on Apple Silicon Macs; use `cuda` on an NVIDIA machine, or leave it out to use the CPU.

### Reading the numbers: the no-induction ceiling

A model without an induction circuit can still beat chance by guessing labels that appear often in the context. Treat accuracy as evidence of induction only once it's clearly above this ceiling (computed by simulation):

| Setting | Chance | No-induction ceiling (acc / loss) |
|---|---|---|
| `n_pairs 8`, every symbol once (default) | 0.125 | 0.325 / 1.556 |
| `n_pairs 8`, `n_unique 4` | 0.125 | 0.523 / 0.972 |
| `n_pairs 8`, `n_unique 4`, `n_labels 32` | 0.031 | 0.464 / 1.114 |

With the default setting and a loss only at the query, the model stayed at the 0.325 ceiling for 20,000 steps. It learned label counting, not induction. `--n_unique 4 --dense_loss` lets symbols repeat and supervises every repeat, so each sequence gives about 5.6 induction targets instead of 1.

## Repository layout

```
config.yaml          default hyper-parameters
src/config.py        Config dataclass (loads config.yaml)
src/data.py          synthetic ICL data generator plus fixed val/test splits
src/model.py         2-layer transformer from scratch, with attention capture and head ablation
src/metrics.py       accuracy, loss, perplexity, output entropy (the collapse signal)
src/train.py         training and evaluation for a single generation
src/collapse.py      multi-generation collapse loop (base works; extended is a TODO)
src/interpret.py     attention maps, induction scores, per-head ablation
scripts/smoke_test.py  end-to-end check
scripts/analyse.py   induction scores, ablations and attention heatmaps for a saved model
scripts/analyse_generations.py  circuit metrics and figures across collapse generations
results/             outputs (git-ignored)
```

**Data splits:** the data is generated on the fly. Training draws fresh samples, and validation and test use fixed, different seeds (`VAL_SEED`, `TEST_SEED` in `src/train.py`). Explain this in the write-up, because the brief requires a train, validation and test split.

## Roadmap (maps to the brief's steps)

- [ ] **1. Reproduce the base task.** Get `smoke_test.py` passing, then do a full `src.train` run. Save the loss curve and look for the phase change.
- [ ] **1b. Tune hyper-parameters.** The brief deducts marks if you don't tune at all. Sweep `lr`, `d_model` and `steps` on validation, and keep test for the end.
- [ ] **2a. Base collapse.** Run `src.collapse --variant base` and plot test accuracy and `next_label_entropy` against generation. Expect little collapse, and use this as the control.
- [ ] **2b. Extended collapse.** Implement `sample_extended_distribution` in `src/collapse.py`. The model also predicts the next symbol and its value, and those preferences feed back into the data. Designing this protocol is part of the graded science.
- [ ] **3. Metrics over training.** Track train and validation loss and accuracy, final test accuracy, perplexity and output entropy across generations.
- [ ] **4. Interpretability.** Use `induction_score` and `ablation_grid` from `src/interpret.py` to find the previous-token and induction heads, then see how they degrade across generations. For more depth, see Conmy et al. (2023) and Chen et al. (2026).
- [ ] **5. Extended abstract.** Two pages on the Moodle template, plus the NeurIPS ethics checklist and the Faculty AI ethics statement.

## References

- Singh et al. (2024). *What needs to go right for an induction head?* ICML. Code: https://github.com/aadityasingh/icl-dynamics
- Olsson et al. (2022). *In-context learning and induction heads.* arXiv:2209.11895
- Shumailov et al. (2024). *AI models collapse when trained on recursively generated data.* Nature 631.
- Conmy et al. (2023). *Towards automated circuit discovery for mechanistic interpretability.* NeurIPS.
- Chen et al. (2026). *Mechanistic data attribution.* arXiv:2601.21996
