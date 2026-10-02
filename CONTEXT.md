# Project context and handover

This file is the running record of where Project 1 stands: what has been built and found, the decisions behind it, and what's left to do. Read it before picking up work, and update it when something changes. The README covers setup and commands; this file covers everything else.

*Last updated: 2 Oct 2026 (15-generation extended run).*

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

### 5. Extended variant, 5 generations × 35,000 steps (`results/collapse_extended/`)

- Test accuracy on real data is 1.000 in every generation, and the generated query labels are always correct.
- **Errors in the model's own choices build up.** The share of generated next symbols that don't appear in the context rises steadily: 0.8% → 1.8% → 2.7% → 3.6% → 4.4%, about 0.9 points per generation.
- **The circuit is present in every generation** (previous-token score 0.94–0.97, induction score ≥ 0.99). But these models depend less on the identified heads than the base models do, **already in generation 0**:

  | | Base runs (5 seeds) | Extended, gen 0 → 4 |
  |---|---|---|
  | Accuracy, all previous-token heads removed | 0.21–0.28 | 0.39, 0.42, 0.29, 0.41, 0.70 |
  | Accuracy, all induction heads removed | 0.12–0.16 | 0.22, 0.20, 0.39, 0.40, 0.43 |

  Because generation 0 trained only on real data, this difference comes from the extended setup (more training targets, and/or far more training after the circuit forms), **not** from collapse. The rise across generations is suggestive only: it's a single chain, and seed noise is large.
- **The phase change arrives at 750–1,500 steps**, versus 6,500–22,000 in the base runs. The training loss levels off around 0.15 rather than 0, because choosing the next symbol is genuinely random. That floor is expected and is a useful sanity check.

### 6. Extended variant, 15 generations × 8,000 steps (`results/collapse_extended_long/`), the main collapse result

What each generation wrote for the next one (real data: in context 1.000, query share 0.343, symbol-ID entropy 3.466):

| gen | next symbol not in context | query share among in-context choices | within-context choice entropy | symbol-ID entropy | next-label correct |
|---|---|---|---|---|---|
| 0 | 1.9% | 0.351 | 1.318 | 3.459 | 1.000 |
| 2 | 6.1% | 0.362 | 1.308 | 3.442 | 0.998 |
| 4 | 9.2% | 0.373 | 1.308 | 3.420 | 0.995 |
| 6 | 13.4% | 0.380 | 1.305 | 3.406 | 0.993 |
| 8 | 17.2% | 0.383 | 1.298 | 3.402 | 0.985 |
| 10 | 20.1% | 0.394 | 1.283 | 3.395 | 0.994 |
| 12 | 23.6% | 0.404 | 1.282 | 3.396 | 0.991 |
| 14 | **26.4%** | **0.398** | 1.296 | **3.372** | 0.982 |

Columns 3 and 4 are derived from the logged statistics. The query share is "next symbol is the query" divided by "next symbol in context". The within-context entropy removes the "not in context" bucket: H_in = (H(choice) − h(p_out)) / (1 − p_out), where p_out is the out-of-context share and h(p_out) its binary entropy. Test accuracy is 1.000 in every generation, query-label accuracy ≥ 0.999, and label entropy 2.0794 → 2.0781.

Three collapse signatures, all from the part of the output the model *chooses* rather than copies:

1. **Errors accumulate in a straight line.** The out-of-context share grows by **1.8 points per generation** (a linear fit leaves a largest error of 0.7 points, and the quadratic term is negligible). So each generation adds a roughly constant amount of new error on top of the error it inherits; the errors aren't compounding.
   - The rate is **double** the 35,000-step run's 0.9 points per generation, and generation 0 already has 1.9% versus 0.8%. Less training per generation means a less accurate copy, which means faster drift. The training budget sets how fast collapse happens.
2. **The most common choice gets more common.** Among choices that *are* in the context, the query symbol's share rises from 0.343 (real) to about 0.40, and the within-context entropy falls from 1.320 to about 1.28–1.30. This is the "favour the mode" behaviour the brief predicted. The query symbol is the one the model has just attended to, which makes it the easy default.
   - Note: the raw "which context symbol" entropy rises and then levels off (1.39 → 1.53). That rise is caused by the growing out-of-context bucket and hides this narrowing; report the within-context version.
3. **Symbol IDs drift away from uniform.** Symbol-ID entropy falls steadily from 3.466 to 3.372.

**The copying parts are untouched.** Query labels, test accuracy on real data, and label entropy barely move. The induction circuit does its job in every generation. Collapse shows up only where the model has freedom to choose, which is the main point for the discussion section.

Also: the training loss floor rises across generations (about 0.16 in gen 0 → about 0.30 in gen 14). That's expected, because the generated data becomes harder to predict as invented symbols build up.

**Not yet done for this run:** the circuit analysis (`python scripts/analyse_generations.py results/collapse_extended_long mps`). That's the step that links these output trends to the induction heads.

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

1. **Extended variant: built and run** (results in findings 5 and 6). How it works:
   - **Sequence:** the context and query are followed by three tokens: the query's label, a **next symbol**, and that symbol's label (20 tokens in total).
   - **Real data:** the next symbol is a randomly chosen symbol from the context, and its label is the one it has in the context.
   - **Generation 0** trains on a fixed set of 200,000 real sequences.
   - **Every later generation** trains only on 200,000 sequences whose three extra tokens were **sampled at temperature 1 by the previous generation**. The context and query stay real.
   - **Format rule:** each generated token is restricted to the right type (label, symbol, label), so every sequence stays well-formed.
   - **Steps:** 35,000 per generation.
   - **What we record for each generation's output (`extended_stats`, compared against the same statistics on real data):**
     - whether the query label is correct;
     - whether the next symbol appears in the context, and whether it's the query symbol;
     - whether the next symbol's label is correct;
     - the entropy of the chosen symbol IDs;
     - the entropy of *which* context symbol is chosen;
     - the entropy of the labels.
   - `analyse_generations.py` also plots these in `extended_generated_data.png`.
   - **Options:** `--temperature` (0 means always pick the most likely token) and `--dataset_size`.
   - **Design decisions to justify in the method section:**
     - the contexts stay real, so collapse can only enter through the model's own choices;
     - there's a fixed, finite dataset per generation, because finite samples are where Shumailov-style drift comes from;
     - generation 0 also uses a fixed dataset, so every generation has the same amount of data.
2. **Fully recursive base variant** (optional, not built). Measure each generation on contexts drawn from the *previous* generation's label distribution, instead of fresh uniform contexts. A perfect copier would then pass sampling noise on from generation to generation, and the entropy should drift downward. That would test whether collapse can happen with no model errors at all.
3. **Mean ablation** (optional): adds rigour to the circuit claims.

## Next steps, in order

1. **Circuit analysis on the long run:** `python scripts/analyse_generations.py results/collapse_extended_long mps`. Does reliance on the identified heads keep changing over 15 generations?
2. **A second chain with a different seed**, to show the drift isn't a one-chain accident: `python -m src.collapse --variant extended --n_generations 10 --steps 8000 --seed 100 --n_unique 4 --dense_loss --lr 0.001 --device mps --save_dir results/collapse_extended_s100`
3. Optional: the same setup with `--steps 35000` for more generations, to confirm that training budget sets the drift rate (0.9 vs 1.8 points per generation).
4. Hyper-parameter evidence: a small learning-rate or `d_model` sweep on validation (the brief deducts marks for no tuning). Mean ablation if there's time.
5. Final figures, then write the two-page abstract. The background, method and base results can be written now.
6. Before submitting: `README.txt`, a check of `requirements.txt`, the NeurIPS checklist, the Faculty AI ethics statement, and the contribution statement (about 13 Oct).

**Suggested split:** one person on the remaining Project 1 runs and figures; one drafting the Project 1 abstract; one starting Project 2, which hasn't begun yet.

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
