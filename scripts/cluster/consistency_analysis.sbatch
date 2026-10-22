#!/bin/bash
# After consistency.sbatch: choice-circuit tests, attention scores and the
# leakage fit (in nats), all on the same 10,000-sequence evaluation set.
#
#SBATCH --job-name=consistency_analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/v2/logs/%x.out

source .venv/bin/activate

set -euo pipefail
export PYTHONUNBUFFERED=1
python scripts/choice_circuit.py results/v2 --threads "$SLURM_CPUS_PER_TASK"
python scripts/attention_scores.py results/v2
python scripts/leakage_fit.py results/v2
