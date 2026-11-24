# Reproducing GEMF factor stability

This analysis reproduces Supplementary Table S3 using processed MRI and EEG measurements from 188 MPI–LEMON participants. All required data are included in `examples/factor_stability/`. No raw recordings, demographic data, cortical surfaces, or additional downloads are needed.

## Run the analysis

After [installation](../README.md#installation), run this command from the repository folder:

```bash
bash examples/reproduce_gemf_factor_stability.sh \
  examples/factor_stability \
  results/factor_stability_reproduction
```

Choose an output folder that does not already exist. For another run, change the final path, for example to `results/factor_stability_repeat`.

## What the command does

All fits use five shared factors and one modality-specific factor for each of sMRI, rs-fMRI, and EEG.

| Analysis | Participants | Retained modes: sMRI / rs-fMRI / EEG |
| --- | --- | --- |
| Original model (reference check) | All 188 | 20 / 20 / 20 |
| Panel A: participant subsampling | 25 saved subsets, each with 150 or 151 participants | 20 / 20 / 20 |
| Panel B: more MRI modes | All 188 | Compare 20 / 20 / 20 with 50 / 50 / 20 |

The reference check refits the original model and compares its factors with the saved Figure 2 factors. Panel A uses the overlapping training subsets from five-fold splitting repeated five times. Panel B retains more MRI modes while keeping the EEG representation unchanged.

The script uses the saved processed values and fitting settings. The settings are in `settings.json`.


## Results

The output folder contains the following files:

- `table_s3.csv`: both panels of Supplementary Table S3.
- `figure2_stability_summary.csv`: summary across participant subsets.
- `figure2_stability_by_fold.csv`: results for individual subsets.
- `resolution_matches.csv`: comparison of the 20/20/20 and 50/50/20 fits.
- `control_matches.csv`: comparison with the original full-cohort fit.
- `reproduction_check.json`: verification against the original results.

The fitted models are also saved in the output folder.

## Factor comparison

Factors are matched by maximizing the sum of absolute correlations
between shared participant scores, followed by sign alignment.
Scores are compared using Pearson correlation, and loadings are
compared using cosine similarity.

For participant subsampling, comparisons use participants common to
both fits. For the 20/20/20 versus 50/50/20 comparison, loading
similarities are computed using the first 20 MRI modes and the full
EEG loading tensors.


## Included files and data source

| File in `examples/factor_stability/` | Contents |
| --- | --- |
| `inputs.npz` | Processed imaging arrays and participant IDs |
| `reference.npz` | Original shared scores and modality loadings |
| `splits.csv` | Participant membership and order for each subset |
| `settings.json` | Model and comparison settings |
| `expected/` | Saved results used to check reproduction |

Array definitions and source-data details are in [DATA_DESCRIPTION.md](../examples/factor_stability/DATA_DESCRIPTION.md). The files contain processed measurements derived for this study, not an official MPI–LEMON release. Age, sex, and cognitive scores are excluded.

Please cite the GEMF manuscript and Babayan et al. (2019), *A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults*, Scientific Data, 6, 180308. [Dataset paper](https://doi.org/10.1038/sdata.2018.308) · [MPI–LEMON data](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html).
