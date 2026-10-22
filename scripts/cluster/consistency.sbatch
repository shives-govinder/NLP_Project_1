#!/bin/bash
# Circuit tests of every chain on the one 10,000-sequence evaluation set (SLURM).
#
#   sbatch scripts/cluster/consistency.sbatch            # one array task per chain
#   sbatch --dependency=afterok:<job> scripts/cluster/consistency_analysis.sbatch
#
#SBATCH --job-name=consistency
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=results/v2/logs/%x_%a.out

source .venv/bin/activate

set -euo pipefail
export PYTHONUNBUFFERED=1
CHAINS=(base ext_t1 ext_t1_s100 ext_real10 ext_t07 ext_long ext_t07_s100 ext_real10_s100 ext_genctx ext_accum)
python scripts/mechanism.py "results/v2/${CHAINS[$SLURM_ARRAY_TASK_ID]}" --threads "$SLURM_CPUS_PER_TASK" > /dev/null
echo "done ${CHAINS[$SLURM_ARRAY_TASK_ID]}"
