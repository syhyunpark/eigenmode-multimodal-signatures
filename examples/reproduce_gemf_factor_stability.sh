#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

bundle="${1:-examples/factor_stability}"
output="${2:-results/factor_stability}"

python3 -u -m scripts.run_gemf_factor_stability --bundle "$bundle" --outdir "$output"
python3 -m scripts.summarize_gemf_factor_stability --bundle "$bundle" --outdir "$output"
