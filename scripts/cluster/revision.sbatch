#!/bin/bash
# Additional chains, one array task per chain (SLURM).
#
#   sbatch --array=1-4 scripts/cluster/revision.sbatch                         # 8,000-step chains (batch)
#   sbatch --array=0 --partition=stampede --cpus-per-task=8 --time=3-00:00:00 \
#          scripts/cluster/revision.sbatch                                     # ext_long (may exceed 1 day)
#
#   0 ext_long        T = 1, 25,000 steps per generation: training-budget ablation
#   1 ext_genctx      the model also writes the contexts and queries (full loss)
#   2 ext_accum       real and synthetic data accumulate (Gerstgrasser et al.)
#   3 ext_t07_s100    second seed for T = 0.7
#   4 ext_real10_s100 second seed for 10% real data
#
# Run from the repo root on the Wits mscluster. Uses the venv at .venv (system
# Python + CPU torch; envtest built it). Each chain needs ~6 cores and a few GB,
# so it runs on the small batch nodes (1-day limit); only ext_long goes to stampede.
#
#SBATCH --job-name=revision
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=results/v2/logs/%x_%a.out

source .venv/bin/activate

set -euo pipefail
mkdir -p results/v2/logs
export PYTHONUNBUFFERED=1
COMMON="--variant extended --n_generations 8 --n_unique 4 --dense_loss --lr 0.001"

case "$SLURM_ARRAY_TASK_ID" in
  0) NAME=ext_long;        ARGS="--steps 25000 --seed 0" ;;
  1) NAME=ext_genctx;      ARGS="--steps 8000 --seed 0 --gen_context" ;;
  2) NAME=ext_accum;       ARGS="--steps 8000 --seed 0 --accumulate" ;;
  3) NAME=ext_t07_s100;    ARGS="--steps 8000 --seed 100 --temperature 0.7" ;;
  4) NAME=ext_real10_s100; ARGS="--steps 8000 --seed 100 --real_frac 0.1" ;;
  *) echo "unknown task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

# Uses the GPU automatically if the node has one; --threads only applies on CPU.
python -m src.collapse $COMMON $ARGS --threads "$SLURM_CPUS_PER_TASK" --save_dir "results/v2/$NAME"
python scripts/mechanism.py "results/v2/$NAME" --threads "$SLURM_CPUS_PER_TASK"
