# MPI–LEMON data for factor stability analysis

The file `inputs.npz` contains processed sMRI, rs-fMRI, and EEG
features for 188 MPI–LEMON participants.

| Array | Shape | Description |
|---|---|---|
| `S_tilde` | (188, 50, 4) | sMRI coefficient matrices |
| `Sigma_tilde` | (188, 50, 50) | rs-fMRI log-covariance matrices |
| `E` | (188, 20, 20, 2) | EEG mode–frequency–condition tensors |
| `lambda_vals` | (50,) | Spatial penalty weights (1, ..., 50) |

The four sMRI features are cortical thickness, sulcal depth, mean
curvature, and surface area, in that order. EEG features represent
log-power topographies across 20 frequency bins spanning 1–45 Hz
under eyes-open (EO) and eyes-closed (EC) conditions.
Frequency coordinates are stored in the input file.

The cortical modes follow the original `fsaverage5` eigenbasis ordering.
The 20/20/20 input is obtained by retaining the first 20 MRI modes,
after the matrix logarithm for rs-fMRI. No additional feature or
modality scaling is applied.

The `subject_ids` array contains the original pseudonymous MPI–LEMON
participant identifiers, normalized to six digits. Age, sex, and
cognitive variables are not included.

## Additional files

- `reference.npz`: original Figure 2 shared scores and loadings.
- `splits.csv`: participant indices and ordering for the 25 refits
  (zero-based).
- `settings.json`: model fitting and score inference settings.
- `expected/`: reference results for checking reproduction.
- `sha256.json`: file checksums.

See [Factor stability analysis](../../docs/REPRODUCING_FACTOR_STABILITY.md)
for instructions.

## Data source

These features were derived from the publicly available MPI–LEMON dataset:

- [MPI–LEMON data (FCP–INDI)](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html)
- [Original MRI data (GWDG)](https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/MRI_MPILMBB_LEMON/)

The source data are distributed under PDDL and CC0 terms, respectively.

Please cite the GEMF paper and Babayan et al. (2019), *Scientific Data*,
6, 180308. [doi:10.1038/sdata.2018.308](https://doi.org/10.1038/sdata.2018.308).