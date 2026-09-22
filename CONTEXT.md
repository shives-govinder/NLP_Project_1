# Project context and handover

This file is the running record of where Project 1 stands: what has been built and found, the decisions behind it, and what's left to do. Read it before picking up work, and update it when something changes. The README covers setup and commands; this file covers everything else.

*Last updated: 22 Sep 2026.*

## Deadlines and deliverables

- **Submission: 27 Oct 2026, 17:00.** Project 2 is due the same day.
- **Preliminary contribution statement: about 13 Oct.** The brief says "two weeks before submission"; confirm the exact date on Moodle.
- **What to submit:** the full code, `requirements.txt`, a **`README.txt`** (we have `README.md`, so run `cp README.md README.txt` before submitting), and a **two-page extended abstract as a PDF** using the Moodle template. The NeurIPS 2024 ethics checklist and the Faculty AI ethics statement go after the references. **Without the NeurIPS checklist the project isn't marked.**
- **Marking:** only the write-up is graded, using this rubric: structure 10%, background 10%, **method 30%, results 30%, discussion 20%**. The brief says marks go to the quality of the science and interpretation, not model accuracy, and that honest limitations and well-explained failed experiments earn credit. The code only matters as the source of our results.

## The research question

The brief asks how **model collapse** affects **in-context learning and induction heads**. We reproduce the symbol→label in-context learning task from Singh et al. (2024). Then we train models over several generations, each generation learning from data produced by the one before, as in Shumailov et al. (2024). At each generation we check what happens to the induction circuit. The brief requires two variants:

- **Base:** the model predicts only the label for the query.
- **Extended:** the model also predicts a next symbol and that symbol's label.

## Setup we settled on

- **Task:** 32 symbols, 8 labels, 8 (symbol, label) pairs per sequence, then a query symbol, so sequences are 17 tokens long.
- **Model:** a 2-layer transformer written from scratch, with 4 heads per layer, `d_model` 64 and an MLP of size 256. It has 103,232 parameters.
- **Training:** learning rate 1e-3, batch size 256, run on the Mac's GPU (`--device mps`).
- **Training setup that works:** `--n_unique 4 --dense_loss`.
  - `--n_unique 4`: each sequence uses only 4 distinct symbols across its 8 pairs, so symbols repeat, always with the same label.
  - `--dense_loss`: the model is graded on every repeated symbol's label, not just the final query. That gives about 5.6 copying targets per sequence instead of 1.
- **Data splits:** training data is generated fresh at every step. Validation and test sets are fixed and generated with their own seeds (`VAL_SEED`, `TEST_SEED` in `src/train.py`). Explain this in the method section.

## Findings so far (with the numbers)

### 1. The original setup never learns induction (a negative result worth reporting)

When the loss is taken only at the final query, the model sat at **test accuracy 0.326 for 20,000 steps**. We simulated the best any model can do *without* induction, by guessing labels in proportion to how often they appear in the context. That ceiling is **0.325 accuracy and 1.556 loss**, and the model's results match it almost exactly. So the model learned to count labels rather than to copy. Chance is 0.125.

### 2. With the dense setup, induction forms suddenly

With `--n_unique 4 --dense_loss`, counting alone can reach **0.523 accuracy and 0.972 loss**. The seed-0 model sat on that plateau until about step 17,500. It then jumped to 0.94 in roughly 1,000 steps and finished at **test accuracy 1.000**. That sudden jump is the phase change described by Olsson et al. The model is saved in `results/base.pt`.

### 3. The circuit in the seed-0 model

These tests use zero ablation, meaning the head's output is set to zero.

- **Layer 0, head 0 is the previous-token head.** Each label position attends to the symbol just before it. Removing this head drops accuracy from **1.00 to 0.25**.
- **Every layer-1 head looks like an induction head** by attention score (0.95–0.99), but none of them can do the job alone:
  - keeping only L1H1 gives 0.61;
  - removing L1H1 gives 0.61;
  - removing the whole of layer 1 gives **0.107**, below chance.

  So the induction step is spread across several heads.
- **Removing L0H0 and L0H3 together gives 0.254**, the same as removing L0H0 alone. L0H3's role is still unclear.

### 4. The base collapse variant does not collapse (the control)

We ran 5 generations with seeds 0–4, 25,000 steps each (`results/collapse_base/`):

| gen | seed | phase-change step | test acc | label entropy | #prev-token heads | #induction heads | acc, all prev-token heads removed | acc, all induction heads removed |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 18,750 | 1.000 | 2.0793 | 1 | 4 | 0.269 | 0.116 |
| 1 | 1 | 22,000 | 0.999 | 2.0789 | 2 | 4 | 0.283 | 0.147 |
| 2 | 2 | 6,500 | 1.000 | 2.0790 | 1 | 2 | 0.211 | 0.140 |
| 3 | 3 | 13,500 | 1.000 | 2.0792 | 2 | 4 | 0.253 | 0.156 |
| 4 | 4 | 16,000 | 1.000 | 2.0788 | 1 | 4 | 0.262 | 0.129 |

The phase-change step is the first logged step where validation accuracy reaches at least 0.9. Maximum entropy for 8 labels is 2.0794. A head counts as a previous-token or induction head if its score is above 0.8.

What the table shows:

- **No collapse.** Accuracy and label entropy stay at their maximum in every generation.
  - *Why:* the induction circuit copies whatever label appears in the context and has no preference for particular labels. Each generation's label distribution is also measured on fresh uniform contexts, so every generation is pulled back towards the real data.
- **The same circuit forms in every seed, but its redundancy varies.**
  - Every seed has one or two previous-token heads feeding two to four induction heads.
  - Removing **all** heads of either type always breaks the model: 0.21–0.28 without the previous-token heads, and about chance without the induction heads.
  - Removing only the *strongest* previous-token head understates the damage when there are two (seeds 1 and 3). That's why we report the "all heads removed" numbers.
- **When the phase change happens depends heavily on the seed**, anywhere from 6,500 to 22,000 steps. The data distribution was the same in all five runs, so this is seed noise alone. It's the baseline spread the extended variant must be compared against.

## Bugs found and fixed (mention in the method where relevant)

- `induction_score` used to halve every score. Numbers from before the fix (about 0.06) are wrong; the corrected value is about 0.12.
- The first collapse run gave every generation the same seed, so generations 2–4 were identical runs. Now generation *g* uses seed + *g*. In that first run the phase change moved from about 17,750 to about 2,500 steps just because of a different data order: another sign of how much the seed matters.

## Things to keep in mind

- **Head numbers aren't comparable across generations or seeds.** The previous-token head might be H0 in one run and H3 in the next. Compare the summary numbers (head counts, best scores, "all removed" accuracy), not specific head IDs.
- **A generation that never reaches the phase change looks like collapse but isn't.** Check that every generation clearly passes the 0.523 counting ceiling before interpreting its circuit.
  - 25,000 steps was only just enough (seed 1 broke through at 22,000). Use about 35,000 for the extended variant.
- **Zero ablation can exaggerate a head's importance.** Mean ablation (replacing a head's output with its average) is the more standard method (Conmy et al., 2023). It isn't implemented yet; mention it as a limitation or add it.
- **`results/` is git-ignored.** The trained models, JSON files and figures exist only on the machine that produced them, which is currently Shives' Mac. Share figures separately, or re-run the scripts to reproduce them.

## Decisions still open

1. **Extended variant design** (not built yet; `sample_extended_distribution` in `src/collapse.py` stops with an error until it is). The proposed defaults:
   - After the query, the model generates the query's label, a **new symbol**, and that symbol's label, by **sampling at temperature 1**.
   - The new symbol has no correct answer, so it exposes the model's own preferences. Any preference then compounds over generations.
   - Each generation trains **only** on the previous generation's generated sequences (the pure recursive setting).
   - Run **35,000 steps** per generation.
   - **What to measure:** symbol and label entropy, test accuracy on real data, and the circuit (using `analyse_generations.py`).
   - **Code needed:** `train_model` has to accept a fixed set of generated sequences instead of making fresh data each step.
2. **Fully recursive base variant** (optional, not built). Measure each generation on contexts drawn from the *previous* generation's label distribution, instead of fresh uniform contexts. A perfect copier would then pass sampling noise on from generation to generation, and the entropy should drift downward. That would test whether collapse can happen with no model errors at all.
3. **Mean ablation** (optional): adds rigour to the circuit claims.

## Next steps, in order

1. Decide the extended-variant design, then build it (the biggest remaining piece of work).
2. Run the extended variant for 5 generations with `--save_dir results/collapse_extended`, then run `analyse_generations.py` on it.
3. If there's time, add the fully recursive base variant and/or mean ablation.
4. Hyper-parameter evidence: the 5-seed base run already shows the spread across seeds. Add a small sweep of learning rate or `d_model` on validation, because the brief deducts marks for no tuning.
5. Final figures, then write the two-page abstract.
6. Before submitting: `README.txt`, a check of `requirements.txt`, the NeurIPS checklist, and the Faculty AI ethics statement.

**Suggested split:** one person builds the extended variant; one runs experiments and makes the figures (seeds, sweep, mean ablation); one drafts the background and method sections of the abstract now, since the base results are final. Project 2 hasn't started yet, so the work will need to be divided across both projects.

## Commands you'll use most

```bash
# base model (about 20k steps until the phase change), then its circuit
python -m src.train --steps 20000 --lr 0.001 --n_unique 4 --dense_loss --device mps --save results/base.pt
python scripts/analyse.py results/base.pt mps

# collapse run that saves every generation, then the analysis and figures across generations
python -m src.collapse --variant base --n_generations 5 --steps 25000 --n_unique 4 --dense_loss --lr 0.001 --device mps --save_dir results/collapse_base
python scripts/analyse_generations.py results/collapse_base mps
```

`analyse_generations.py` writes `circuit_summary.json`, `circuit_across_generations.png` and `ablation_by_generation.png` into the run folder.

## References

Singh et al. (2024), *What needs to go right for an induction head?*, ICML · Olsson et al. (2022), *In-context learning and induction heads* · Shumailov et al. (2024), *AI models collapse when trained on recursively generated data*, Nature · Conmy et al. (2023), *Towards automated circuit discovery*, NeurIPS · Chen et al. (2026), *Mechanistic data attribution*.
