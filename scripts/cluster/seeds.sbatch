#!/bin/bash
# Three more seeds (200, 300, 400) for each main condition, so every main
# condition has 5 independent chains (seeds 0, 100, 200, 300, 400). Base seeds
# are spaced by 100 because generation g uses seed + g.
#
#   sbatch scripts/cluster/seeds.sbatch
#   sbatch --dependency=afterok:<job> scripts/cluster/seeds_analysis.sbatch
#
#SBATCH --job-name=seeds
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8
#SBATCH --output=results/v2/logs/%x_%a.out

source .venv/bin/activate

set -euo pipefail
export PYTHONUNBUFFERED=1
COMMON="--variant extended --n_generations 8 --n_unique 4 --dense_loss --lr 0.001 --steps 8000"
SEEDS=(200 300 400)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
case $((SLURM_ARRAY_TASK_ID / 3)) in
  0) NAME=ext_t1_s$SEED;     ARGS="" ;;
  1) NAME=ext_t07_s$SEED;    ARGS="--temperature 0.7" ;;
  2) NAME=ext_real10_s$SEED; ARGS="--real_frac 0.1" ;;
esac

python -m src.collapse $COMMON $ARGS --seed $SEED --threads "$SLURM_CPUS_PER_TASK" --save_dir "results/v2/$NAME"
python scripts/mechanism.py "results/v2/$NAME" --threads "$SLURM_CPUS_PER_TASK" > /dev/null
