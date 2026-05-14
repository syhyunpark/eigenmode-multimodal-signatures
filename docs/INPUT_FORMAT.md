# Input format

The main analysis scripts assume processed subject-level feature files. This repository does not include MPI--LEMON data.

## GEMF `.npz` input

The GEMF scripts expect a compressed NumPy `.npz` file with the following fields:

- `subject_ids`: subject identifiers, length `n`
- `S_tilde`: standardized sMRI harmonic coefficient matrices, shape `(n, K, q)`
- `Sigma_tilde`: log-Euclidean rs-fMRI harmonic covariance objects, shape `(n, K, K)`
- `E`: EEG mode-frequency-condition tensors, shape `(n, K, F, C)`
- `K`: number of harmonic modes
- `q`: number of sMRI morphometric features
- `F`: number of EEG frequency bins
- `C`: number of EEG conditions

Optional fields include (and other covariates):

- `age`
- `male`
- `lambda_vals`
- `smri_measures`
- `eeg_freq_centers`
- `condition_names`

## CSV feature inputs for PCA benchmarks

The PCA benchmark scripts assume subject-aligned CSV files for:

- harmonic sMRI features
- harmonic rs-fMRI features
- harmonic EEG features
- concatenated harmonic multimodal features
- conventional atlas/sensor-based multimodal features
- metadata containing subject ID, age, sex, and outcome variables

Each feature CSV should contain one subject identifier column and feature columns. Feature standardization, PCA fitting, and prediction model fitting are performed within cross-validation folds.
