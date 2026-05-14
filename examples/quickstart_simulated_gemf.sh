#!/usr/bin/env bash
set -euo pipefail

# Quickstart example:
# 1. Simulate a small multimodal dataset in GEMF object format.
# 2. Fit a fixed-rank GEMF model.
# 3. Print a short summary of the fitted model.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

OUTDIR="results/quickstart_simulated_gemf"

echo "Creating synthetic multimodal data..."
python3 scripts/simulate_gemf_energy_data.py \
  --outdir "$OUTDIR/data" \
  --scenario shared_specific \
  --n 60 \
  --K 20 \
  --q 4 \
  --F 16 \
  --C 2 \
  --R 3 \
  --seed 1

echo "Fitting fixed-rank GEMF..."
python3 scripts/fit_fixed_rank_gemf.py \
  --data "$OUTDIR/data/simulated_multiview_data.npz" \
  --outdir "$OUTDIR/fit_R3_L111" \
  --R 3 \
  --Ls 1 \
  --Lf 1 \
  --Le 1 \
  --w-s 1.0 \
  --w-f 1.0 \
  --w-e 1.0 \
  --lambda-s 0.01 \
  --lambda-f 0.01 \
  --lambda-e 0.01 \
  --zeta-e 0.01 \
  --specific-ridge 1e-6 \
  --score-ridge 0.1 \
  --max-iter 50 \
  --tol 1e-5 \
  --seed 1 \
  --quiet

echo "Reading fit summary..."
python3 - <<'PY'
import json
from pathlib import Path

summary_path = Path("results/quickstart_simulated_gemf/fit_R3_L111/fit_summary.json")
with open(summary_path, "r") as f:
    s = json.load(f)

print("\nGEMF quickstart completed.")
print(f"n subjects:        {s.get('n')}")
print(f"K modes:           {s.get('K')}")
print(f"sMRI features q:   {s.get('q')}")
print(f"EEG frequencies F: {s.get('F')}")
print(f"EEG conditions C:  {s.get('C')}")
print(f"iterations run:    {s.get('iterations_run')}")
print(f"final loss:        {s.get('final_loss_total'):.3f}")
PY

echo ""
echo "Output written to: $OUTDIR"
