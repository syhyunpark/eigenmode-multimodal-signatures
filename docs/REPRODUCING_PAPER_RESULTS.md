# Reproducing paper results

Run the following commands from the repository folder after installation.

For Supplementary Table S3, see
[Factor stability analysis](REPRODUCING_FACTOR_STABILITY.md).
The required data are included in the repository.

## 1. Prediction inputs

The prediction analyses require processed imaging features and outcome
data that are not included in the repository.
See [Input format](INPUT_FORMAT.md) for the required files.

```bash
data_dir="/path/to/processed_lemon_data"
metadata="/path/to/metadata.csv"
```

The analyses use five-fold cross-validation repeated five times,
with seed 0. GEMF uses one modality-specific factor per modality
and the default penalty settings specified in the scripts.

Use a new output folder when changing the data or model settings.

## 2. Fixed-dimension prediction and Figure 1

```bash
python3 -m scripts.outercv_fixed_budget_benchmark \
  --data "$data_dir/lemon_real_main188_K20.npz" \
  --metadata_csv "$metadata" \
  --subject_col subject_id \
  --harm_smri_csv "$data_dir/baseline_blocks/harmonic_smri_block.csv" \
  --harm_fmri_csv "$data_dir/baseline_blocks/harmonic_fmri_block.csv" \
  --harm_eeg_csv "$data_dir/baseline_blocks/harmonic_eeg_block.csv" \
  --harm_concat_csv "$data_dir/baseline_blocks/harmonic_concat_block.csv" \
  --conv_multimodal_csv "$data_dir/conventional_multimodal_main_188.csv" \
  --outcome_specs "age_mid|male;LPS_1|male" \
  --rank_grid 1,2,3,4,5,6,7,8,9,10,11,12 \
  --Ls 1 --Lf 1 --Le 1 \
  --outer_splits 5 --n_repeats 5 --seed 0 \
  --outdir results/fixed_budget_benchmark_age_LPS
```

To generate Figure 1:

```bash
python3 -m scripts.plot_partA_fixed_dimension_benchmark_clean \
  --summary_csv results/fixed_budget_benchmark_age_LPS/outercv_summary.csv \
  --outdir results/figures_main \
  --prefix figure_partA_fixed_dimension_benchmark_main \
  --figure_type main
```

Use `--figure_type supp` for the supplementary figure containing all
methods. For the wider rank comparison, use
`--rank_grid 1,2,3,4,5,6,8,10,12,16,20` with a separate output folder.

## 3. Reconstruction-based rank selection

Rank is selected using reconstruction error in five inner folds
and the one-standard-error rule.

### PCA

```bash
python3 -m scripts.outercv_pca_rankselect_1se \
  --metadata_csv "$metadata" \
  --subject_col subject_id \
  --harm_smri_csv "$data_dir/baseline_blocks/harmonic_smri_block.csv" \
  --harm_fmri_csv "$data_dir/baseline_blocks/harmonic_fmri_block.csv" \
  --harm_eeg_csv "$data_dir/baseline_blocks/harmonic_eeg_block.csv" \
  --harm_concat_csv "$data_dir/baseline_blocks/harmonic_concat_block.csv" \
  --conv_multimodal_csv "$data_dir/conventional_multimodal_main_188.csv" \
  --outcome_specs "age_mid|male;LPS_1|male" \
  --rank_grid 1,2,3,4,5,6,7,8,9,10,11,12 \
  --outer_splits 5 --inner_splits 5 --n_repeats 5 \
  --one_se_tol 1.0 --seed 0 \
  --outdir results/pca_rankselect_1se_age_LPS
```

### GEMF

```bash
python3 -m scripts.outercv_gemf_rankselect_1se \
  --data "$data_dir/lemon_real_main188_K20.npz" \
  --metadata_csv "$metadata" \
  --subject_col subject_id \
  --outcome_specs "age_mid|male;LPS_1|male" \
  --R_grid 1,2,3,4,5,6,7,8,9,10,11,12 \
  --Ls 1 --Lf 1 --Le 1 \
  --outer_splits 5 --inner_splits 5 --n_repeats 5 \
  --one_se_tol 1.0 --seed 0 \
  --outdir results/gemf_rankselect_1se_age_LPS
```

### Rank-selection table

Combine the PCA and GEMF summaries and generate the table:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("results")
folders = [base / "pca_rankselect_1se_age_LPS",
           base / "gemf_rankselect_1se_age_LPS"]

for filename in ["outercv_summary.csv", "selected_rank_summary.csv"]:
    combined = pd.concat([pd.read_csv(p / filename) for p in folders],
                         ignore_index=True)
    combined.to_csv(base / ("combined_" + filename), index=False)
PY

python3 -m scripts.make_partB_rankselect_table \
  --summary_csv results/combined_outercv_summary.csv \
  --rank_csv results/combined_selected_rank_summary.csv \
  --out_tex results/tables_main/table_partB_rank_selection.tex
```

## 4. Cortical maps and factor profiles: Figure 2

These commands require the original 20-mode GEMF input, cortical
eigenbasis, and corresponding `fsaverage5` surface files.

```bash
basis="/path/to/fsaverage5_phi_K200_pial_A.npz"
surfaces="/path/to/fs_subjects/fsaverage5"

python3 -m scripts.fit_and_export_gemf_panels \
  --data "$data_dir/lemon_real_main188_K20.npz" \
  --basis_npz "$basis" \
  --outdir results/gemf_final_R5_L111_panels \
  --R 5 --Ls 1 --Lf 1 --Le 1 \
  --prefer_condition EO --seed 0

python3 -m scripts.plot_gemf_panels \
  --panel_npz results/gemf_final_R5_L111_panels/panel_objects.npz \
  --outdir results/gemf_final_R5_L111_panels/figures \
  --fsaverage_dir "$surfaces" \
  --factors 1,2,3,4,5 --smri_measures_to_show 1,2,3,4 \
  --views lat,med --surface smoothwm --smoothing 10 --qlim 95 \
  --brain_title_fontsize 50 --brain_title_pad 24

python3 -m scripts.make_gemf_main_montage \
  --figures_dir results/gemf_final_R5_L111_panels/figures \
  --panel_npz results/gemf_final_R5_L111_panels/panel_objects.npz \
  --outdir results/gemf_final_R5_L111_panels/figures_main5 \
  --factor 5 --smri_measure_index 1
```

Change `--factor` and the output folder to visualize other factors.
The score-versus-age plots use the `age` field in the original input.

## 5. Simulated data

```bash
bash examples/quickstart_simulated_gemf.sh
```

This example generates simulated data and fits GEMF.
Results are saved in `results/quickstart_simulated_gemf/`.

The full simulation studies additionally require
`evaluate_gemf_simulation_fit.py` and
`select_gemf_rank_by_reconstruction.py`, which are not included
in the current repository.