#!/bin/bash
# After seeds.sbatch: choice profiles of the new chains (seed_profiles.json),
# then every pooled analysis once over all chains, and the sampling null of the
# new T = 0.7 seeds. All on the same 10,000-sequence evaluation set.
#
#SBATCH --job-name=seeds_analysis
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

python - <<'PY'
import glob, json, re
from src.collapse import REAL_EVAL_SEED, REAL_EVAL_SIZE, real_dataset
from src.patching import choice_profile
from src.train import load_checkpoint, set_threads

set_threads(6)
out, data = {}, None
for run in [f"ext_{c}_s{s}" for c in ("t1", "t07", "real10") for s in (200, 300, 400)]:
    out[run] = []
    for p in sorted(glob.glob(f"results/v2/{run}/gen*.pt"), key=lambda p: int(re.search(r"gen(\d+)\.pt$", p).group(1))):
        model, cfg, _ = load_checkpoint(p, device="cpu")
        if data is None:
            data = real_dataset(cfg, REAL_EVAL_SEED, REAL_EVAL_SIZE)
        cp = choice_profile(model, cfg, data)
        cp["gen"] = int(re.search(r"gen(\d+)\.pt$", p).group(1))
        out[run].append(cp)
with open("results/v2/seed_profiles.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved results/v2/seed_profiles.json")
PY

python scripts/choice_circuit.py results/v2 --threads "$SLURM_CPUS_PER_TASK" > /dev/null
python scripts/attention_scores.py results/v2 > /dev/null
python scripts/leakage_fit.py results/v2 > /dev/null
for s in 200 300 400; do python scripts/sampling_null.py "results/v2/ext_t07_s$s"; done
echo "analysis done"
