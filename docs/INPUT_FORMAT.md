# Input format

The analysis scripts use processed participant-level imaging features
rather than raw MRI or EEG recordings. The data required for the
factor-stability analysis are provided in `examples/factor_stability/`.

## 1. GEMF input (.npz)

The following arrays are required:

| Field | Shape | Description |
| --- | --- | --- |
| `S_tilde` | `(n, K_s, q)` | sMRI coefficient matrices |
| `Sigma_tilde` | `(n, K_f, K_f)` | rs-fMRI log-covariance matrices |
| `E` | `(n, K_e, F, C)` | EEG mode–frequency–condition tensors |
| `lambda_vals` | `(K_max,)` or longer | Spatial penalty weights |

Here, `n` is the number of participants; `K_s`, `K_f`, and `K_e` are
the numbers of cortical modes; `q` is the number of morphometric
features; `F` and `C` are the numbers of EEG frequency bins and
conditions; and `K_max = max(K_s, K_f, K_e)`.

The empirical analyses use mode indices (1, 2, ..., K_max) as spatial
penalty weights. Include `subject_ids` to identify participants.

### MPI–LEMON data

| Analysis | sMRI | rs-fMRI | EEG |
| --- | --- | --- | --- |
| Original GEMF fit | `(188, 20, 4)` | `(188, 20, 20)` | `(188, 20, 20, 2)` |
| Increased MRI mode count | `(188, 50, 4)` | `(188, 50, 50)` | `(188, 20, 20, 2)` |

The four sMRI features are cortical thickness, sulcal depth, mean
curvature, and surface area. The EEG coefficients represent log-power
topographies under eyes-open (EO) and eyes-closed (EC) conditions.

The 20-mode MRI inputs are obtained by retaining the first 20 modes
of the supplied 50-mode arrays, after the matrix logarithm for rs-fMRI.

## 2. Feature CSVs for prediction

Each file contains a `subject_id` column and numeric feature columns.

| File | Features |
| --- | --- |
| `harmonic_smri_block.csv` | 50 modes × 4 features |
| `harmonic_fmri_block.csv` | Upper triangle of the 50 × 50 log-covariance matrix, including the diagonal |
| `harmonic_eeg_block.csv` | 20 modes × 20 frequencies × 2 conditions |
| `harmonic_concat_block.csv` | Concatenation of the three harmonic feature blocks |
| `conventional_multimodal_main_188.csv` | Regional MRI and sensor-based EEG features |

The GEMF input and `harmonic_concat_block.csv` must have the same
participants in the same row order.

## 3. Participant metadata

The prediction analyses require a separate CSV with the following columns:

| Column | Description |
| --- | --- |
| `subject_id` | Participant identifier |
| `age_mid` | Midpoint of the five-year age interval |
| `male` | Binary sex indicator (0 or 1) |
| `LPS_1` | LPS-1 cognitive score |

## 4. Cortical maps

The cortical maps require the cortical eigenbasis and corresponding
`fsaverage5` surface files. The basis must have the same mode ordering
and signs as those used to construct the imaging coefficients.

For prediction and visualization commands, see
[Reproducing paper results](REPRODUCING_PAPER_RESULTS.md).