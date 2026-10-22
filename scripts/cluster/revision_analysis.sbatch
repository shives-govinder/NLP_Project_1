#!/bin/bash
# Analysis of the revision chains, after revision.sbatch has finished:
#
#   sbatch --dependency=afterok:<chains job>:<ext_long job> scripts/cluster/revision_analysis.sbatch
#
# Writes results/v2/revision_profiles.json (choice_profile of every generation,
# as in choice_profiles.json) and results/v2/choice_circuit.json (only the chains
# present on the cluster; merge it with the local one).
#
#SBATCH --job-name=revision_analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=results/v2/logs/%x.out

source .venv/bin/activate

set -euo pipefail
export PYTHONUNBUFFERED=1

python - <<'EOF'
import glob, json, re, torch
from src.collapse import REAL_EVAL_SEED, REAL_EVAL_SIZE, real_dataset
from src.patching import choice_profile
from src.train import load_checkpoint, set_threads

set_threads(6)
out, data = {}, None
for run in ["ext_long", "ext_genctx", "ext_accum", "ext_t07_s100", "ext_real10_s100"]:
    out[run] = []
    for p in sorted(glob.glob(f"results/v2/{run}/gen*.pt"), key=lambda p: int(re.search(r"gen(\d+)\.pt$", p).group(1))):
        model, cfg, _ = load_checkpoint(p, device="cpu")
        if data is None:
            data = real_dataset(cfg, REAL_EVAL_SEED, REAL_EVAL_SIZE)
        cp = choice_profile(model, cfg, data)
        cp["gen"] = int(re.search(r"gen(\d+)\.pt$", p).group(1))
        out[run].append(cp)
        print(run, cp["gen"], {k: round(v, 3) for k, v in cp.items() if isinstance(v, float)})
with open("results/v2/revision_profiles.json", "w") as f:
    json.dump(out, f, indent=1)
EOF

python scripts/choice_circuit.py results/v2 --threads 6
