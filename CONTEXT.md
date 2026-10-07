# Project context and handover

This file is the running record of where Project 1 stands: what has been built and found, the decisions behind it, and what's left to do. Read it before picking up work, and update it when something changes. The README covers setup and commands; this file covers everything else.

*Last updated: 7 Oct 2026 (v2 experiments: recursive base variant, real-data evaluation, causal circuit tests, data interventions). The v2 findings below are the ones to write up; findings 1–8 are the earlier (v1) record.*

## Deadlines and deliverables

- **Submission: 27 Oct 2026, 17:00.** Project 2 is due the same day.
- **Preliminary contribution statement: about 13 Oct.** The brief says "two weeks before submission"; confirm the exact date on Moodle. Groups are locked after this.
- **Marking:** only the write-up is graded, using this rubric: structure 10%, background 10%, **method 30%, results 30%, discussion 20%**. The brief says marks go to the quality of the science and interpretation, not model accuracy, and that honest limitations and well-explained failed experiments earn credit. The code only matters as the source of our results.

### Submission checklist (updated 5 Oct)

**Code**

- [x] Full code implementation (this repo)
- [ ] `requirements.txt`: final check (currently torch, numpy, pyyaml, matplotlib, tqdm)
- [ ] **`README.txt`** (the brief asks for `.txt`): how to install and reproduce every result, with the final commands
- [ ] Final test: fresh clone → install → `python scripts/smoke_test.py` passes

**Extended abstract (PDF)**

- [ ] Uploaded to Overleaf: `_NLP_2026__Extended_Abstract_Template.zip` → New Project → Upload Project (as instructed on Moodle). Write in `ccn_style.tex`, add references to `ccn_style.bib`, replace the placeholder `images/image.jpeg`.
- [ ] Report figures (about 3), built from `results/`, uploaded to `images/`
- [ ] Title, authors (all 3), abstract (about 20 lines) and 3 keywords
- [ ] Introduction with background, methods, results, discussion: **strictly 2 pages** including figures and tables
- [ ] Figure and table captions that tell the reader what to look at; every figure referenced in the text
- [ ] References (outside the page limit)
- [ ] Contribution statement (outside the page limit; may change from the preliminary one, but the authors can't)
- [ ] Check the template's two blank `\phantom` pages after `\maketitle`: keep them or remove them as intended
- [ ] Exported as PDF, page limit confirmed

**Appended after the references, before any supplementary material**

- [ ] **NeurIPS 2024 checklist: `.tex` NOT yet available on Moodle.** Without it the project isn't marked. Keep checking Moodle, and ask the lecturer or tutor if it hasn't appeared by about 13 Oct. Fallback (only if the lecturer allows): the public checklist from the official NeurIPS 2024 style files.
- [ ] **Wits Science Faculty AI Ethics Form:** Shives has it and will add it manually.

**Admin**

- [ ] Preliminary contribution statement submitted (about 13 Oct)
- [ ] All three group members agree on the final contribution statement
- [ ] Submitted before **27 Oct, 17:00**: code + `requirements.txt` + `README.txt` + abstract PDF

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

## v2 findings (7 Oct): what to write up

All runs are on the `experiments/collapse-v2` branch: seed 0 unless stated, lr 1e-3, d_model 64, `--n_unique 4 --dense_loss`, 200,000 sequences per generation. Extended chains: 8 generations × 8,000 steps. Base chain: 6 generations × 25,000 steps. Trained on the Wits cluster (`stampede`, CPU); outputs in `results/v2/` (git-ignored; copy from the cluster or from Koven's laptop). Figures: `results/v2/figures/fig_collapse.png` and `fig_interventions.png`, made by `python scripts/paper_figures.py`.

**Two changes to the method since v1:**

- **The base variant is now recursive.** Each generation trains on real contexts plus query labels *sampled* (temperature 1) by the previous model, so its mistakes become training targets. The v1 protocol (now `--variant base_reweight`) only re-weighted how context labels were sampled and always trained on the true labels, so it could not collapse by construction: a perfect copier under it gives label entropy 2.0791 (95% range 2.0786–2.0793), exactly the v1 base chain's 2.0788–2.0793. **Do not present v1 finding 4 as a collapse result.**
- **Every generation is scored on 10,000 held-out *real* sequences** (`real_eval` in `collapse.json`), the analogue of Shumailov et al.'s perplexity on the original data. For the free next-symbol choice the true distribution is known exactly (each context symbol with probability slots/8), so KL(true ‖ model) is computed exactly, without sampling. It splits into two parts (`src/patching.py: choice_profile`):
  - **leakage** = −log(1 − probability on symbols *not* in the context);
  - **misallocation** = KL to the model renormalised over the context's symbols (probability spread wrongly *among* them).

### A. Base variant (recursive): no collapse

Query-label errors in the generated data stay at about 1 in 20,000 and shrink (0.99995 → 0.99999 correct); real-data query loss falls (7e-5 → 6e-6 nats); label entropy stays at the maximum (2.0794). The phase change came at 4,000–7,750 steps in every generation. **Interpretation:** when the answer is a deterministic function of a real context, the rare sampled errors are unpredictable noise that the next model cannot learn, so induction corrects errors instead of copying them. This is now a genuine negative result, not a design artefact.

### B. Extended variant, temperature 1: collapse lives only in the free choice

| | seed 0 (`ext_t1`) | seed 100 (`ext_t1_s100`) | replicates? |
|---|---|---|---|
| Probability on out-of-context symbols, gen 0 → 7 | 2.2% → 14.8% | 1.8% → 15.0% | **yes** (≈1.8 points/gen, as in v1 finding 6) |
| Total KL on real data | 0.099 → 0.558 | 0.099 → 0.550 | **yes** |
| of which misallocation within the context | 0.076 → 0.398 | 0.080 → 0.387 | **yes**: most of the drift |
| Query-symbol preference (model / true, within context) | 1.03 → 1.11 | 1.01 → 1.07 | **yes** |
| Frequency slope (1 = copies slot frequencies exactly) | 1.01 → 1.27 | 0.98 → 1.04 | **no**: strong in seed 0 only |
| Single-slot symbols' choice probability (true 0.125) | 0.118 → 0.085 | 0.126 → 0.118 | **no** |
| Query-label accuracy on real data | ≥ 0.999 | ≥ 0.999 | yes |
| Next-label loss on real data | 0.0016 → 0.0129 | 0.0002 → 0.0048 | yes: small rise, accuracy stays ≥ 0.998 |

- **Most of the drift is misallocation among context symbols (≈70% of the KL), not leakage.** It rises in both seeds. The v1 "within-context entropy" was pooled over all sequences and could not see this per-sequence error; that is why it did not replicate (v1 finding 7).
- **Rich-get-richer frequency sharpening is seed-dependent at T = 1.** Report it as present in one chain, robust only at T = 0.7 (C).
- **Copying stays intact but not perfect:** query and next-label argmax accuracy on real data stays ≥ 0.998, while the loss on the next label rises ~10× (still < 0.02 nats).

### C. Temperature decides which kind of collapse (`ext_t07`, one seed)

At T = 0.7 the distribution *narrows* instead of spreading: out-of-context probability *falls* 2.2% → 0.5%, misallocation explodes 0.076 → 2.56 nats, the frequency slope reaches 1.92, single-slot symbols nearly vanish (0.118 → 0.017, true 0.125), and the query preference reaches 1.28. Training loss falls each generation (0.171 → 0.036) because the data gets more predictable. This is Shumailov et al.'s "tails disappear"; T = 1 instead shows error accumulation into impossible outputs.

### D. Keeping 10% real data (`ext_real10`, one seed)

Total KL at generation 7 is 0.296 vs 0.558 without real data. Misallocation levels off at ≈ 0.17 from generation 5, but leakage keeps rising (11.8% out-of-context probability at generation 7). So real data stops one failure but not the other.

### E. Deletion test on generation 7's data (Chen et al. 2026 style; `scripts/formation.py`, 3 seeds each)

14.5% of `ext_t1`'s generation-7 rows contradict their context (next symbol not in the context, or a wrong label). Fresh models trained on the data:

| Training data | leakage | misallocation | frequency slope |
|---|---|---|---|
| real data | 0.018–0.023 | 0.071–0.076 | 0.98–1.01 |
| generation-7 data | 0.164–0.172 | 0.40–0.42 | 1.28–1.31 |
| **erroneous rows deleted** | **0.016–0.018** | 0.42–0.47 | 1.26–1.31 |
| same number of random rows deleted | 0.167–0.174 | 0.40–0.43 | 1.25–1.31 |

**A clean dissociation:** leakage is caused entirely by the detectably wrong rows (deleting them restores the real-data level; deleting random rows does nothing). Misallocation and sharpening survive the deletion untouched, because they live in rows that look correct. A correctness filter cannot fix that half of collapse.

### F. Induction formation (3 seeds each; phase change = first eval with validation accuracy ≥ 0.9, resolution 250 steps)

- Collapsed data does not change *when* induction forms: generation-7 data 1,000–1,250 steps vs real data 750–1,750.
- **The next-symbol target is what makes induction form fast.** Query-label target only: 6,250–7,750 steps; adding the next-symbol target: 750–1,000; adding the next-label target as well: 750–1,750. No overlap. This explains why the extended variant always formed induction ~7× sooner than the base variant. (A Chen-style "catalyst", but from the training objective rather than data content.)

### G. The circuit, with causal tests (Conmy et al. 2023 style; `scripts/mechanism.py`, all 38 models)

Interventions use **resample ablation** (a head's output replaced by its output on another sequence) instead of zero ablation, and **path patching** (changing only what layer 1 reads as queries, keys or values).

- **The circuit is essential in every generation of every chain:** resampling all of layer 1 gives 0.117–0.132 accuracy (chance 0.125); resampling all of layer 0 gives 0.28–0.41.
- **It is K-composition, as in Singh et al., but relayed by the layer-0 MLP.** In the 33/38 models where one layer-0 head matters (resampled accuracy ≤ 0.88), the path into layer 1's *keys*, including the route through the layer-0 MLP, carries a median **99%** of that head's effect (range 93–107%). The direct key edge alone carries a median **10%**. Queries carry nothing; values at most 12%. Singh et al.'s figure shows an attention-only path; here the MLP sits in the middle.
- **Zero ablation understates induction heads here.** The most important single layer-1 head: median accuracy 0.68 when resampled vs 0.98 when zeroed. The v1 claim "induction is spread across several heads" (finding 3) rests on zero ablation; under resample ablation several models have one head that alone drops accuracy to 0.12–0.14. Report both and say why they differ (zeroing removes a head's vote; resampling makes it vote for a wrong label).
- **Collapse does not damage the circuit.** Resampling layer 1 gives chance in every generation of every chain. The only trend that appears in both T = 1 chains is a weak *fall* in accuracy when all of layer 0 is resampled (r = −0.55 and −0.66 with generation, 8 models each; not significant at that size), i.e. layer 0 becomes, if anything, slightly *more* essential. That matches v1's weak "more essential" trend (finding 7); report it as weak, not as a result.

### H. Limitations to state in the write-up

- One seed for T = 0.7 and for the 10%-real chain; two seeds for T = 1. Generation is confounded with seed (generation g uses seed + g).
- 8 generations only; the base chain has 6.
- Phase-change steps have 250-step resolution and three seeds per condition.
- The deletion test was done once, on generation 7 of one chain; the "erroneous" filter is defined by consistency with the context.
- Resample ablation uses one corruption (each sequence paired with its neighbour in the batch).
- The task is an adaptation of Singh et al.'s (discrete symbols, 8 pairs, repeated symbols, dense loss), not a reproduction (they use image exemplars and 2 pairs).
- No hyper-parameter tuning for the base variant (the v1 sweep tuned the extended variant only).

## Earlier findings (v1 code; see the v2 section for what supersedes them)

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

**Superseded (v2 finding A):** this protocol, now `--variant base_reweight`, cannot collapse by construction, because its training targets are always the true labels. The entropy column below is sampling noise. The circuit columns are still valid observations about the trained models.

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

  Because generation 0 trained only on real data, this difference comes from the extended setup (more training targets, and/or far more training after the circuit forms), **not** from collapse. The rise across generations does not replicate in the 15-generation run (finding 6), so treat it as seed noise.
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
2. **The query symbol is increasingly favoured** (replicated in finding 7). The narrowing of the overall within-context distribution described below does *not* replicate. Among choices that *are* in the context, the query symbol's share rises from 0.343 (real) to about 0.40, and the within-context entropy falls from 1.320 to about 1.28–1.30. This is the "favour the mode" behaviour the brief predicted. The query symbol is the one the model has just attended to, which makes it the easy default.
   - Note: the raw "which context symbol" entropy rises and then levels off (1.39 → 1.53). That rise is caused by the growing out-of-context bucket and hides this narrowing; report the within-context version.
3. **Symbol IDs drift away from uniform.** Symbol-ID entropy falls steadily from 3.466 to 3.372.

**The copying parts are untouched.** Query labels, test accuracy on real data, and label entropy barely move. The induction circuit does its job in every generation. Collapse shows up only where the model has freedom to choose, which is the main point for the discussion section.

Also: the training loss floor rises across generations (about 0.16 in gen 0 → about 0.30 in gen 14). That's expected, because the generated data becomes harder to predict as invented symbols build up.

**Circuit across the 15 generations** (`circuit_summary.json`, `circuit_across_generations.png`, `ablation_by_generation.png` in the run folder)

| | gens 0–4 (mean) | gens 10–14 (mean) | trend over 15 gens |
|---|---|---|---|
| Previous-token head score | 0.954 | 0.986 | rises, r = +0.74 |
| Accuracy, whole last layer removed | 0.231 | 0.168 | falls, r = −0.54 |
| Accuracy, all induction heads removed | 0.337 | 0.187 | falls, r = −0.78 (outliers removed) |
| Accuracy, all previous-token heads removed | 0.515 | 0.406 | falls, r = −0.69 (outliers removed) |

Test accuracy is 1.000 and the induction score is ≥ 0.985 in every generation.

- **The circuit doesn't weaken as the outputs collapse.** (The "sharper and more essential" part below only weakly replicates; see finding 7.) In this chain it also became sharper and more essential: The previous-token head attends more precisely, and removing the circuit hurts more in later generations. The model relies on the induction mechanism *more* as its generated data drifts further from real data. So collapse in this setting damages the model's choices, not its copying machinery.
- **Outliers are artefacts of the 0.8 threshold.** In gens 4, 9 and 12 only one head of a type passed the threshold, so "removing all of them" left untouched heads that were really doing the same job: 0.907, 0.818 and 0.698. The whole-last-layer column doesn't depend on any threshold, and it shows the same downward trend. Report it as the robust measure, and mention the threshold issue as a limitation.
- **This revises finding 5.** The rise in "all induction heads removed" across the 5-generation 35,000-step chain (0.22 → 0.43) does **not** replicate over 15 generations. Treat it as seed noise.
- **Generation 0 here is still less dependent on the identified heads than the base models** (0.408 / 0.334, against base 0.21–0.28 / 0.12–0.16). Yet it had about 7,000 training steps after the phase change, inside the base runs' range of about 3,000–18,500. So "longer training after the phase change" doesn't explain the difference. The extended task itself (extra targets, the free symbol choice) leads to a more spread-out mechanism.
- **Caveats:** this is one chain, and generation is confounded with seed. Each trend correlation is across only 15 models (r of about 0.5–0.8). The second chain (`results/collapse_extended_s100`) is what turns these trends into a claim.

### 7. Replication: second extended chain, seed 100, 10 generations × 8,000 steps (`results/collapse_extended_s100/`)

Same setup as finding 6 with an independent seed (the generation-0 model and all data sampling are different). Gens 0–9 of both chains:

| | Seed-0 chain | Seed-100 chain | Replicates? |
|---|---|---|---|
| Next symbol not in context, gen 0 → 9 | 1.9% → 18.4% | 1.8% → 16.4% | **yes** |
| Drift rate (linear fit) | 1.85 pts/gen (largest error 0.4) | 1.54 pts/gen (largest error 0.5) | **yes**; the chains never differ by more than 2.9 points |
| Query share among in-context choices | 0.343 → 0.386 | 0.343 → 0.377 | **yes** |
| Symbol-ID entropy | 3.466 → 3.386 | 3.466 → 3.376 | **yes** |
| Next-label correct, gen 9 | 0.990 | 0.975 | yes (slowly falls) |
| Within-context choice entropy | 1.320 → 1.289 (falls) | 1.321 → 1.343 (rises) | **no** |
| Test accuracy on real data | 1.000 | 1.000 | yes |

**Conclusions:**

- **Robust:** straight-line error accumulation (about 1.5–1.9 points per generation); a growing preference for the query symbol among in-context choices; symbol IDs drifting from uniform; copying staying intact.
- **Not robust:** the narrowing of the within-context choice distribution. Don't claim it. The query symbol gains share in both chains, but in seed 100 the other in-context choices become *more* even at the same time, so overall within-context entropy goes up. The robust claim is "the model increasingly favours the query symbol", not "its choices narrow".
- **Circuit across this chain** (`circuit_summary.json` in the run folder). The circuit is present and essential in every generation: previous-token score 0.94–0.99, induction score ≥ 0.985, test accuracy 1.000. The finding-6 claim that the circuit becomes *more* essential only partly replicates:

  | Measure | Seed-0 chain | Seed-100 chain | Replicates? |
  |---|---|---|---|
  | Accuracy, whole last layer removed (threshold-free) | 0.231 → 0.168 over 15 gens (r = −0.54); flat over gens 0–9 | 0.237 → 0.180 over 10 gens (r = −0.75) | weakly: a small decline (about 0.06) in both, but on different timescales |
  | Accuracy, all induction heads removed | falls (r = −0.78, outliers removed) | 0.17–0.22 and flat wherever all 4 heads are counted; gens 0 and 2–4 inflated because only 2 heads passed the threshold | no clear evidence beyond threshold artefacts |
  | Accuracy, all previous-token heads removed | falls (r = −0.69, outliers removed) | no trend (r = −0.08) | **no** |
  | Previous-token head score | rises (r = +0.74) | weak rise (r = +0.33) | weak |

- **Claim for the report:** across both chains, the induction circuit is **preserved, and stays essential**, while the model's free choices collapse. There is at most weak evidence that reliance on the circuit *increases*. Don't claim "the circuit gets stronger"; present the last-layer decline as a small, consistent trend worth noting. The threshold-based "all heads removed" numbers are unreliable when few heads pass the 0.8 threshold; list this as a limitation.

### 8. Hyper-parameter sweep (`scripts/sweep.py`, `results/sweep/sweep.json`)

Extended variant, generation 0 only (real data), 8,000 steps, seed 0, the same training data for every configuration. Every number is measured on **validation data** (the test set is untouched). "Not in context" is the share of a fresh 20,000-sequence sample, written by the trained model, whose next symbol isn't in the context: the starting error behind collapse drift.

| lr | d_model | params | phase change | val acc | val loss | not in context |
|---|---|---|---|---|---|---|
| 3e-4 | 32 | 27k | 3,500 | 1.000 | 0.0018 | 19.76% |
| 3e-4 | 64 | 103k | 1,250 | 1.000 | 0.0002 | 2.88% |
| 3e-4 | 128 | 403k | 750 | 1.000 | **0.0001** | **1.22%** |
| 1e-3 | 32 | 27k | 1,750 | 1.000 | 0.0003 | 5.22% |
| **1e-3** | **64** | **103k** | **1,000** | **1.000** | **0.0002** | **2.24%** |
| 1e-3 | 128 | 403k | 1,750 | 0.993 | 0.0208 | 2.00% |
| 3e-3 | 32 | 27k | 2,000 | 0.999 | 0.0036 | 4.94% |
| 3e-3 | 64 | 103k | 6,250 | 1.000 | 0.0004 | 3.13% |
| 3e-3 | 128 | 403k | 4,500 | 0.980 | 0.0596 | 2.36% |

- **Every configuration learns induction** (validation accuracy ≥ 0.98), so the circuit doesn't depend on fine-tuned settings.
- **Our setting (lr 1e-3, d_model 64) is justified.** It ties for second-best validation loss and is the best of the 64-wide models on every measure. The only better configuration (3e-4, 128) has 4× the parameters and trains about 2× slower. We keep 1e-3/64 for all results; in the report, mention 3e-4/128 as the validation-best alternative.
- **lr 3e-3 is too high:** the phase change comes later, and at width 128 training is unstable (validation accuracy 0.98, loss 0.06).
- **Capacity sets the starting error.** The not-in-context share falls with width (32 → 64 → 128 gives roughly 5–20% → 2–3% → 1–2%). Together with the training-budget effect (35,000 steps gave 0.8% in gen 0 and 0.9 points per generation of drift; 8,000 steps gave 1.9% and 1.5–1.9 points), this suggests the **collapse rate is set by how accurately each generation fits the choice distribution**. Not tested directly; an optional 128-wide chain would test it.
- **Caveats:** one seed per configuration, and the base runs showed the phase-change step varying about 3× between seeds. So don't read much into phase-change differences under 2×; validation loss and the not-in-context share are the more reliable comparisons.

## Bugs found and fixed (mention in the method where relevant)

- `induction_score` used to halve every score. Numbers from before the fix (about 0.06) are wrong; the corrected value is about 0.12.
- The first collapse run gave every generation the same seed, so generations 2–4 were identical runs. Now generation *g* uses seed + *g*. In that first run the phase change moved from about 17,750 to about 2,500 steps just because of a different data order: another sign of how much the seed matters.

## Things to keep in mind

- **Head numbers aren't comparable across generations or seeds.** The previous-token head might be H0 in one run and H3 in the next. Compare the summary numbers (head counts, best scores, "all removed" accuracy), not specific head IDs.
- **A generation that never reaches the phase change looks like collapse but isn't.** Check that every generation clearly passes the 0.523 counting ceiling before interpreting its circuit.
  - 25,000 steps was only just enough (seed 1 broke through at 22,000). Use about 35,000 for the extended variant.
- **Zero ablation and resample ablation can disagree** (v2 finding G). Resample ablation (`src/patching.py`) is the one Conmy et al. (2023) prefer; use it for circuit claims.
- **`results/` is git-ignored.** v1 results are on Shives' Mac; v2 results (`results/v2/`) are on the Wits cluster (`~/nlp1/results/v2`) and Koven's laptop. Share figures separately, or re-run the scripts to reproduce them.
- **Absolute next-symbol probabilities are diluted by leakage.** Compare choices within the context (`choice_profile`'s `query_bias`, `kl_in`), not raw `next_symbol_query_prob`.

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
2. ~~Fully recursive base variant~~ (done in v2 as `--variant base`, finding A).
3. ~~Mean ablation~~ (resample ablation done instead, v2 finding G).

## Next steps, in order

1. ~~v2 experiments, figures~~ (done, see the v2 findings).
2. Optional, if there's time, to strengthen the weakest claims: a second seed for the T = 0.7 and 10%-real chains (`--seed 100`), and the deletion test on another generation. Each chain takes a few hours on the cluster.
3. Write the two-page abstract from the v2 findings. Suggested story: collapse leaves the induction circuit intact and only corrupts the model's free choices (B, G); temperature decides spread vs narrowing (C); a correctness filter fixes leakage but not misallocation (E); the next-symbol objective catalyses induction (F).
4. Before submitting: `README.txt`, a check of `requirements.txt`, the NeurIPS checklist, the Faculty AI ethics statement, and the contribution statement (about 13 Oct).

The earlier v1 next-steps (seed-100 chain, sweep, long run) are done; see findings 6–8.

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
