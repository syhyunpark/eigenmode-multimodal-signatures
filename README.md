# eigenmode-multimodal-signatures

Code for cortical eigenmode-coordinate representations and geometric eigenmode multiview factorization (GEMF) for multimodal structural MRI, resting-state fMRI, and EEG data.

This repository supports the manuscript:

**Cortical eigenmode coordinates provide compact subject-level signatures across structural MRI, resting-state fMRI, and EEG**

## Overview

A central goal of this project is to represent multiple neuroimaging modalities in a common cortical Laplace--Beltrami eigenmode coordinate system. In this framework:

- structural MRI is represented as cortical eigenmode coefficients of morphometric fields,
- resting-state fMRI is represented as covariance among eigenmode time-series coefficients,
- EEG is represented as mode-by-frequency-by-condition summaries using a forward-projected cortical eigenmode dictionary.

The repository includes code for:

- fixed-rank GEMF model fitting,
- simulation studies for shared and modality-specific recovery,
- fixed-dimension benchmarking of subject-level representations,
- reconstruction-based rank selection using the one-standard-error rule,
- visualization of GEMF shared factors.

## Repository structure

    eigenmode-multimodal-signatures/
      gemf/          Core GEMF package code
      scripts/       Analysis, simulation, and visualization scripts
      examples/      Quickstart examples
      docs/          Input-format and reproducibility notes

## Installation

Create a Python environment and install dependencies:

    git clone https://github.com/syhyunpark/eigenmode-multimodal-signatures.git
    cd eigenmode-multimodal-signatures

    python3 -m venv .venv
    source .venv/bin/activate

    pip install -r requirements.txt
    export PYTHONPATH=$PWD:$PYTHONPATH

## Quick import test

After installation, run:

    python3 - <<'PY'
    from gemf.data import load_multiview_npz
    from gemf.fixed_rank_model import FixedRankGMHF, FixedRankGMHFConfig
    from gemf.score_inference import infer_shared_scores, infer_full_scores
    from gemf.workflows import fit_or_load_gemf, infer_consistent_scores
    print("GEMF package import test passed.")
    PY

### Data

This repository does not redistribute MPI--LEMON data or derived subject-level feature tables. 
The main analysis scripts assume processed feature files as inputs, including:

- a GEMF `.npz` file containing sMRI, rs-fMRI, and EEG harmonic objects,
- CSV files containing harmonic PCA feature blocks,
- CSV files containing conventional atlas/sensor-based comparison features,
- a metadata CSV containing subject identifiers, age, sex, and outcome variables.

See `docs/INPUT_FORMAT.md` for details.



### GEMF fitting

Fit a fixed-rank GEMF model to a processed multimodal `.npz` file:

    python3 scripts/fit_fixed_rank_gemf.py \
      --data /path/to/multimodal_harmonic_objects.npz \
      --outdir results/gemf_fit \
      --R 5 \
      --Ls 1 --Lf 1 --Le 1

### Simulation analyses

Run recovery simulations:

    python3 scripts/run_gemf_recovery_simulations.py \
      --outdir results/gemf_sim_recovery \
      --n-list 50,100,200 \
      --seed-start 1 \
      --seed-end 100 \
      --quiet

    python3 scripts/summarize_gemf_recovery_simulations.py \
      --results_dir results/gemf_sim_recovery

Run rank-selection simulations:

    python3 scripts/run_gemf_rank_selection_simulations.py \
      --data_root results/gemf_sim_recovery \
      --outdir results/gemf_sim_rank_selection \
      --n-list 50,100,200 \
      --seed-start 1 \
      --seed-end 100 \
      --R_grid 1,2,3,4,5,6,7,8,9,10,11,12 \
      --quiet

    python3 scripts/summarize_gemf_rank_selection_simulations.py \
      --results_dir results/gemf_sim_rank_selection

### MPI--LEMON analyses

The following scripts reproduce the main processed-feature analyses when supplied with local processed input files:

- `scripts/outercv_fixed_budget_benchmark.py`
- `scripts/outercv_pca_rankselect_1se.py`
- `scripts/outercv_gemf_rankselect_1se.py`

Use `--help` to see the required arguments:

    python3 scripts/outercv_fixed_budget_benchmark.py --help
    python3 scripts/outercv_pca_rankselect_1se.py --help
    python3 scripts/outercv_gemf_rankselect_1se.py --help


The following scripts generate manuscript tables and figures from analysis outputs:

- `scripts/plot_partA_fixed_dimension_benchmark_clean.py` 
- `scripts/fit_and_export_gemf_panels.py`
- `scripts/plot_gemf_panels.py`
- `scripts/rank_gemf_factors_by_age.py`
- `scripts/make_gemf_main_montage.py`


## Quickstart example

A small synthetic example is included. It simulates multimodal data, fits a fixed-rank GEMF model, and prints a short fit summary.

    ./examples/quickstart_simulated_gemf.sh

Outputs are written to:

    results/quickstart_simulated_gemf/


## Citation

If you use this repository, please cite the associated manuscript.
