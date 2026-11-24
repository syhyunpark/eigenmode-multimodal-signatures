# eigenmode-multimodal-signatures

Code for **Cortical eigenmode coordinates provide compact subject-level signatures across structural MRI, resting-state fMRI, and EEG**.

This project represents structural MRI (sMRI), resting-state fMRI (rs-fMRI), and EEG in a common cortical eigenmode coordinate system and conducts geometric eigenmode multiview factorization (GEMF), which estimates shared and modality-specific participant scores while retaining the matrix and tensor structure of the measurements. 

## Usage

The [small simulated example](#small-example) illustrates how to fit GEMF
and obtain subject scores and modality loadings.

For the MPI–LEMON application, the following instructions are provided:

- [Factor stability analysis (Supplementary Table S3)](#1-factor-stability-supplementary-table-s3)
- [Age and LPS-1 cognitive outcome prediction analyses](#2-age-and-lps-1-prediction)
- [Cortical maps and factor profiles (Figure 2)](docs/REPRODUCING_PAPER_RESULTS.md#4-cortical-maps-and-factor-profiles-figure-2)

The processed imaging data required for the factor stability analysis
are included in this repository. Input requirements and detailed
instructions for the other analyses are described in
[Reproducing paper results](docs/REPRODUCING_PAPER_RESULTS.md).


## Installation

Download or clone the repository. If using Git:

```bash
git clone https://github.com/syhyunpark/eigenmode-multimodal-signatures.git
cd eigenmode-multimodal-signatures
```

From the repository folder, create a Python environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the commands below from this folder, with the environment activated.

## MPI–LEMON analyses

The paper uses 188 participants with sMRI, rs-fMRI, and eyes-open and eyes-closed EEG. The repository includes the processed imaging data needed for the analyses. Repeating the prediction analyses requires additional files, as described below. Raw MRI and EEG recordings are not included.

### 1. Factor stability: Supplementary Table S3

This analysis can be run immediately after installation.

```bash
bash examples/reproduce_gemf_factor_stability.sh \
  examples/factor_stability \
  results/factor_stability_reproduction
```

The command fits GEMF with five shared factors and one modality-specific
factor per modality, as used in the paper.

To fit GEMF with a different number of factors, use
`scripts/fit_fixed_rank_gemf.py` directly. For example, to fit a model
with three shared factors and one modality-specific factor per modality
using the included MPI–LEMON data (50/50/20 modes):

```bash
python3 -m scripts.fit_fixed_rank_gemf \
  --data examples/factor_stability/inputs.npz \
  --R 3 \
  --Ls 1 --Lf 1 --Le 1 \
  --outdir results/lemon_R3_L111
```

Here, the model ranks are specified by:

- `--R`: number of shared factors.
- `--Ls`: number of sMRI-specific factors.
- `--Lf`: number of rs-fMRI-specific factors.
- `--Le`: number of EEG-specific factors.

Set `--Ls 0 --Lf 0 --Le 0` to fit a shared-only model.
The fitted scores and loadings are saved in `fit_results.npz`,
and a summary of the fit is saved in `fit_summary.json`.

To view all available fitting options, run:

```bash
python3 -m scripts.fit_fixed_rank_gemf --help
```

- **Panel A:** 25 refits on participant subsets, each containing approximately 80% of the cohort.
- **Panel B:** a comparison of 20/20/20 and 50/50/20 retained modes for sMRI/rs-fMRI/EEG, using all 188 participants.

It also refits the original 20/20/20 model. Factor matching does not use age or cognition.

Results are written to `results/factor_stability_reproduction/`. Open **`table_s3.csv`** for both panels. The file `reproduction_check.json` reports whether the results match the saved table values at three decimal places. Use a different output-folder name when running the analysis again.

See [Reproducing factor stability](docs/REPRODUCING_FACTOR_STABILITY.md) for the comparison method and other output files.

### 2. Age and LPS-1 cognitive outcome prediction

The prediction analyses compare PCA, GEMF, and conventional
atlas/sensor-based representations for predicting age and the
LPS-1 cognitive score, with adjustment for sex.

The following scripts implement the prediction analyses:

- `scripts/outercv_fixed_budget_benchmark.py`: prediction across
  different numbers of retained subject scores (Figure 1).
- `scripts/outercv_pca_rankselect_1se.py`: PCA with
  reconstruction-based rank selection.
- `scripts/outercv_gemf_rankselect_1se.py`: GEMF with
  reconstruction-based rank selection.

These scripts require processed imaging features and participant-level
outcome data beyond those included in this repository.
The required input formats and analysis commands are described in
[Input format](docs/INPUT_FORMAT.md) and
[Reproducing paper results](docs/REPRODUCING_PAPER_RESULTS.md).

## Small example

To try GEMF without any imaging data:

```bash
bash examples/quickstart_simulated_gemf.sh
```

This creates simulated data, fits GEMF, and prints a fit summary. Results are saved in `results/quickstart_simulated_gemf/`.

## Code and documentation

| Location | Contents |
| --- | --- |
| `gemf/fixed_rank_model.py` | GEMF fitting algorithm |
| `gemf/score_inference.py` | Participant-score estimation from fitted loadings |
| `gemf/data.py` and `gemf/workflows.py` | Data loading and fitting helpers |
| `scripts/` | Prediction, simulation, and plotting programs |
| `examples/` | Runnable examples and factor-stability data |
| [Input format](docs/INPUT_FORMAT.md) | Required arrays, CSV columns, and participant alignment |
| [Reproducing paper results](docs/REPRODUCING_PAPER_RESULTS.md) | Commands for the paper's analyses |

## Citation

Please cite the associated GEMF manuscript when using this code. Analyses using MPI–LEMON should also cite Babayan et al. (2019), *A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults*, Scientific Data, 6, 180308. [Dataset paper](https://doi.org/10.1038/sdata.2018.308).
