#!/bin/bash
# Collapse chains for the v2 experiments, one array task per chain (SLURM).
#
#   sbatch scripts/cluster/collapse_v2.sbatch            # all chains
#   sbatch --array=0,1 scripts/cluster/collapse_v2.sbatch  # only some
#
# Run from the repo root after `pip install -r requirements.txt` in your env.
# Set --partition (and the env activation below) for your cluster.
#
#SBATCH --job-name=collapse_v2
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --array=0-4
#SBATCH --output=results/v2/logs/%x_%a.out

# source ~/venvs/nlp/bin/activate      # or: conda activate nlp

set -euo pipefail
mkdir -p results/v2/logs
export PYTHONUNBUFFERED=1
COMMON="--n_unique 4 --dense_loss --lr 0.001"

case "$SLURM_ARRAY_TASK_ID" in
  0) NAME=ext_t1;     ARGS="--variant extended --n_generations 8 --steps 8000 --seed 0" ;;
  1) NAME=base;       ARGS="--variant base --n_generations 6 --steps 25000 --seed 0" ;;
  2) NAME=ext_real10; ARGS="--variant extended --n_generations 8 --steps 8000 --seed 0 --real_frac 0.1" ;;
  3) NAME=ext_t07;    ARGS="--variant extended --n_generations 8 --steps 8000 --seed 0 --temperature 0.7" ;;
  4) NAME=ext_t1_s100; ARGS="--variant extended --n_generations 8 --steps 8000 --seed 100" ;;
  *) echo "unknown task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

# Uses the GPU automatically if the node has one; --threads only applies on CPU.
python -m src.collapse $ARGS $COMMON --threads "$SLURM_CPUS_PER_TASK" --save_dir "results/v2/$NAME"
python scripts/mechanism.py "results/v2/$NAME" --threads "$SLURM_CPUS_PER_TASK"
