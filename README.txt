Project 1 - Interpretability of Model Collapse
COMS4054A Natural Language Processing, Wits 2026

We train a 2-layer transformer on the in-context-learning symbol task of
Singh et al. (2024), put it in a model-collapse pipeline (Shumailov et al.,
2024) in a base and an extended variant, and test what happens to the
induction circuit across generations.

All commands run from the repository root. Python 3.10+ is needed.


1. INSTALL
----------

    python -m venv .venv
    .venv\Scripts\activate            (Linux / macOS: source .venv/bin/activate)
    pip install -r requirements.txt

Everything runs on a CPU. A CUDA GPU is used automatically when present; on an
Apple Silicon Mac add --device mps. On a CPU, --threads 4 to 8 is usually
faster than using every core.


2. CHECK THE INSTALL (about 1 minute)
-------------------------------------

    python scripts/smoke_test.py

Trains tiny models through every pipeline (single model, base, base_reweight
and extended collapse) and must end with "SMOKE TEST PASSED".


3. REPRODUCE THE RESULTS IN THE ABSTRACT
----------------------------------------

The JSON summaries, logs and figures of every run are already in results/v2/,
so step 3f alone redraws the figures without retraining. Model checkpoints
(gen*.pt) and training data (data_gen*.pt) are included for the first five
chains (base, ext_t1, ext_t1_s100, ext_t07, ext_real10); the other chains keep
only their JSON summaries, and steps 3a to 3e recreate their checkpoints.

Shared settings: 32 symbols, 8 labels, 8 pairs per sequence, 4 distinct
symbols per sequence (--n_unique 4), loss on every repeated symbol
(--dense_loss), learning rate 1e-3, d_model 64, 200,000 sequences per
generation. Validation and test sets are fixed and drawn with their own seeds
(src/train.py: VAL_SEED, TEST_SEED); every generation is also scored on 10,000
held-out real sequences.

3a. Collapse chains (one SLURM array job runs all five chains and 3b):

    sbatch scripts/cluster/collapse_v2.sbatch        (set --partition first)

    or one at a time:

    python -m src.collapse --variant base     --n_generations 6 --steps 25000 --seed 0   --n_unique 4 --dense_loss --lr 0.001 --save_dir results/v2/base
    python -m src.collapse --variant extended --n_generations 8 --steps 8000  --seed 0   --n_unique 4 --dense_loss --lr 0.001 --save_dir results/v2/ext_t1
    python -m src.collapse --variant extended --n_generations 8 --steps 8000  --seed 100 --n_unique 4 --dense_loss --lr 0.001 --save_dir results/v2/ext_t1_s100
    python -m src.collapse --variant extended --n_generations 8 --steps 8000  --seed 0   --n_unique 4 --dense_loss --lr 0.001 --temperature 0.7 --save_dir results/v2/ext_t07
    python -m src.collapse --variant extended --n_generations 8 --steps 8000  --seed 0   --n_unique 4 --dense_loss --lr 0.001 --real_frac 0.1  --save_dir results/v2/ext_real10

    Each writes gen{g}.pt (model), data_gen{g}.pt (its training data) and
    collapse.json (training curves, test metrics, real-data evaluation and
    statistics of the generated data) into its folder. Each chain takes a few
    hours on 8 CPU cores.

3a'. Additional chains (25,000-step budget ablation, second seeds for
    T = 0.7 and 10% real data, generated contexts, accumulated data):

    On the Wits mscluster, from the repo root, with a venv at .venv
    (python3 -m venv .venv; pip install torch --index-url
    https://download.pytorch.org/whl/cpu; pip install -r requirements.txt):

    sbatch --array=1-4 scripts/cluster/revision.sbatch
                                                      (batch: ext_genctx, ext_accum,
                                                       ext_t07_s100, ext_real10_s100
                                                       and their circuit tests)
    sbatch --array=0 --partition=stampede --cpus-per-task=8 --time=3-00:00:00 scripts/cluster/revision.sbatch
                                                      (ext_long; about 3 hours)
    sbatch --dependency=afterok:<job ids> scripts/cluster/revision_analysis.sbatch
                                                      (choice profiles -> revision_profiles.json,
                                                       choice_circuit.json for these chains)

    Three more seeds (200, 300, 400) for each main condition, and every pooled
    analysis rerun once over all chains on the same 10,000-sequence set:

    sbatch scripts/cluster/seeds.sbatch
    sbatch --dependency=afterok:<job> scripts/cluster/seeds_analysis.sbatch

    Circuit tests of the first ten chains rerun on the same 10,000-sequence
    evaluation set, then the choice-circuit tests, attention scores and
    leakage fit:

    sbatch scripts/cluster/consistency.sbatch
    sbatch --dependency=afterok:<job> scripts/cluster/consistency_analysis.sbatch

    --gen_context makes the model write the whole sequence (loss on every
    token); collapse.json then also holds exact context KLs (real_eval) and
    context statistics (generated_stats). --accumulate trains each generation
    on the real data plus every earlier generation's data.

3b. Causal circuit tests (resample ablation, path patching) for every generation:

    python scripts/mechanism.py results/v2/<run>        -> results/v2/<run>/mechanism.json

3c. Data interventions on generation 7 of ext_t1 (3 seeds each):

    python scripts/formation.py --data real                           --seeds 0 1 2 --out results/v2/formation/real.json
    python scripts/formation.py --data results/v2/ext_t1/data_gen7.pt --seeds 0 1 2 --out results/v2/formation/gen7.json
    python scripts/formation.py --data results/v2/ext_t1/data_gen7.pt --seeds 0 1 2 --delete errors --out results/v2/formation/gen7_del_errors.json
    python scripts/formation.py --data results/v2/ext_t1/data_gen7.pt --seeds 0 1 2 --delete random --out results/v2/formation/gen7_del_random.json
    python scripts/formation.py --data real --seeds 0 1 2 --targets query        --out results/v2/formation/targets_query.json
    python scripts/formation.py --data real --seeds 0 1 2 --targets query+symbol --out results/v2/formation/targets_qsym.json

3d. Hyper-parameter sweeps (validation set only; the test set is never used):

    python scripts/sweep.py
    python scripts/sweep.py --variant base --lrs 0.0003 0.001 0.003 --d_models 64 --steps 25000 --out results/sweep/sweep_base.json

3d'. Restoring-force fit of leakage and misallocation (supplementary Table tab:fit; reads only collapse.json):

    python scripts/leakage_fit.py

3d''. Null model of pure temperature sampling (supplementary Table tab:null),
    the same interventions scored on the next-symbol choice (Table tab:choice),
    and previous-token / induction scores of every model:

    python scripts/sampling_null.py results/v2/ext_t07   -> results/v2/ext_t07/sampling_null.json
    python scripts/choice_circuit.py                      -> results/v2/choice_circuit.json
    python scripts/attention_scores.py                    -> results/v2/attention_scores.json

3e. Embedding geometry and attention maps across generations (needs the checkpoints):

    python scripts/embeddings.py results/v2/ext_t1 --compare results/v2/ext_t07 results/v2/base

    Final train accuracy of every generation, recomputed from the saved models
    and their training data (the v2 runs predate train-accuracy logging):

    python scripts/train_accuracy.py results/v2/base results/v2/ext_t1 results/v2/ext_t1_s100 results/v2/ext_t07 results/v2/ext_real10

3f. Figures for the abstract:

    python scripts/paper_figures.py      -> results/v2/figures/fig_main.png, fig_collapse.png,
                                            fig_interventions.png, fig_training.png
    (fig_embeddings.png comes from scripts/embeddings.py, step 3e)

    The paper reads its figures from paper/images/; copy the new ones there:

    cp results/v2/figures/*.png paper/images/

    fig_collapse.png needs results/v2/choice_profiles.json (in the repository);
    if it is deleted, the script recomputes it from the gen*.pt checkpoints.


4. OTHER ENTRY POINTS
---------------------

    python -m src.train --steps 20000 --lr 0.001 --n_unique 4 --dense_loss --save results/base.pt
        one model on real data; prints train / validation loss, accuracy and
        perplexity during training and the final test accuracy
    python scripts/analyse.py results/base.pt
        induction scores, head ablations and attention heatmaps for one model
    python scripts/analyse_generations.py results/v2/ext_t1
        circuit summary across the generations of a run
    python -m src.collapse ... --variant base_reweight
        the original (v1) non-recursive base protocol, kept for comparison


5. LAYOUT
---------

    config.yaml        default hyper-parameters
    src/               data generator, model, training, collapse loop, metrics,
                       interpretability (interpret.py) and causal tests (patching.py)
    scripts/           experiments, analysis and figures (see each file's docstring)
    results/v2/        JSON summaries, logs and figures of the reported runs
    paper/             LaTeX source of the extended abstract (ccn_style.tex), the
                       NeurIPS checklist (neurips_checklist.tex) and the signed
                       AI ethics declaration (ai_declaration.pdf)


6. BUILDING THE EXTENDED ABSTRACT
---------------------------------

    cd paper
    pdflatex ccn_style && bibtex ccn_style && pdflatex ccn_style && pdflatex ccn_style

or upload the paper/ folder to Overleaf (pdfLaTeX). The checklist and the AI
ethics declaration are pulled in automatically after the references.
