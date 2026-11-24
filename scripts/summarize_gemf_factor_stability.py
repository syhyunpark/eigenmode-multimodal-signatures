#!/usr/bin/env python3
"""Match refitted GEMF factors and reproduce both panels of Supplementary Table S3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.run_gemf_factor_stability import read_splits, sha256

LOADINGS = ("A_shared", "M_shared", "a_shared", "b_shared", "c_shared")
METRICS = ("score_r", "smri_cosine", "fmri_cosine", "eeg_cosine")


def read_fit(path):
    with np.load(path, allow_pickle=False) as saved:
        fields = ("subject_ids", "shared_scores", *LOADINGS, "smri_measures", "condition_names", "eeg_freq_centers")
        arrays = {key: saved[key] for key in fields}
    ids = arrays["subject_ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate participant IDs in {path}")
    arrays["subject_ids"] = ids
    return arrays


def cosine(x, y):
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator > 0 else np.nan


def eeg_loading(loadings, factor):
    return np.einsum("k,f,c->kfc", *(loadings[key][factor] for key in LOADINGS[2:]))


def compare_factors(reference, candidate, reference_scores, candidate_scores):
    """Production comparison: joint |r| assignment, then one sign per matched factor."""
    x, y = np.asarray(reference_scores, dtype=float), np.asarray(candidate_scores, dtype=float)
    if x.ndim != 2 or y.shape != x.shape or x.shape[0] < 3:
        raise ValueError("Expected aligned subject-by-factor score matrices of equal shape")
    rank = x.shape[1]
    for key in LOADINGS:
        if reference[key].shape != candidate[key].shape or reference[key].shape[0] != rank:
            raise ValueError(f"Incompatible dimensions for {key}")
        if not np.isfinite(reference[key]).all() or not np.isfinite(candidate[key]).all():
            raise ValueError(f"Nonfinite loadings in {key}")
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    denominator = np.outer(np.linalg.norm(x, axis=0), np.linalg.norm(y, axis=0))
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("Cannot match a constant or nonfinite score column")
    correlations = np.clip((x.T @ y) / denominator, -1, 1)
    reference_idx, matched_idx = linear_sum_assignment(-np.abs(correlations))
    rows = []
    for r, k in zip(reference_idx, matched_idx):
        sign = 1 if correlations[r, k] >= 0 else -1
        rows.append(dict(
            reference_factor=int(r + 1), matched_factor=int(k + 1), alignment_sign=sign,
            score_r=float(abs(correlations[r, k])),
            smri_cosine=sign * cosine(reference["A_shared"][r], candidate["A_shared"][k]),
            fmri_cosine=sign * cosine(reference["M_shared"][r], candidate["M_shared"][k]),
            eeg_cosine=sign * cosine(eeg_loading(reference, r), eeg_loading(candidate, k)),
        ))
    return rows


def orient_full_cohort(candidate, condition):
    """Reproduce the production panel-export convention without any outcome fallback."""
    names = list(candidate["condition_names"].astype(str))
    if names.count(condition) != 1:
        raise ValueError(f"Expected one {condition} condition for panel orientation")
    coefficients = candidate["c_shared"][:, names.index(condition)]
    if not np.isfinite(coefficients).all() or np.any(np.abs(coefficients) <= 1e-12):
        raise ValueError("EO condition cannot orient every factor; do not substitute an outcome-based rule")
    signs = np.where(coefficients >= 0, 1, -1)
    oriented = dict(candidate)
    oriented["shared_scores"] = candidate["shared_scores"] * signs
    for key in LOADINGS:
        oriented[key] = candidate[key] * signs.reshape((-1,) + (1,) * (candidate[key].ndim - 1))
    # Three EEG vector sign flips give one sign flip of their outer product.
    return oriented, signs


def compare_fit(reference, candidate, common_mri=False, full_cohort=False):
    reference_index = {sid: row for row, sid in enumerate(reference["subject_ids"])}
    ids = candidate["subject_ids"]
    if any(sid not in reference_index for sid in ids):
        raise ValueError("Candidate contains participants absent from the reference")
    if full_cohort and set(ids) != set(reference["subject_ids"]):
        raise ValueError("The full-cohort comparison must use the same participants")
    for key in ("smri_measures", "condition_names", "eeg_freq_centers"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Loading axis metadata differs: {key}")
    candidate = dict(candidate)
    if common_mri:
        ks, kf = reference["A_shared"].shape[1], reference["M_shared"].shape[1]
        candidate["A_shared"] = candidate["A_shared"][:, :ks, :]
        candidate["M_shared"] = candidate["M_shared"][:, :kf, :kf]
    index = [reference_index[sid] for sid in ids]
    return pd.DataFrame(compare_factors(reference, candidate, reference["shared_scores"][index], candidate["shared_scores"]))


def participant_summary(details):
    rows = []
    for factor, block in details.groupby("reference_factor"):
        row = dict(reference_factor=int(factor), n_fits=len(block))
        for metric in METRICS:
            values = block[metric].dropna()
            row[f"{metric}_n"] = len(values)
            for quantile, label in ((.25, "q25"), (.5, "median"), (.75, "q75")):
                row[f"{metric}_{label}"] = values.quantile(quantile)
        rows.append(row)
    return pd.DataFrame(rows)


def check_expected(actual, expected_path, keys):
    """Report raw numeric differences; check the manuscript's three-decimal precision."""
    expected = pd.read_csv(expected_path)
    a, e = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    if not a.index.equals(e.index) or not set(e.columns).issubset(a.columns):
        raise ValueError(f"Expected-output structure differs: {expected_path.name}")
    details = {}
    passed = True
    for column in e:
        av, ev = a[column].to_numpy(dtype=float), e[column].to_numpy(dtype=float)
        valid = np.isfinite(av).all() and np.isfinite(ev).all()
        same = valid and all(f"{x:.3f}" == f"{y:.3f}" for x, y in zip(av, ev))
        details[column] = {"max_abs_difference": float(np.max(np.abs(av - ev))) if valid else None,
                           "matches_at_three_decimals": bool(same)}
        passed = passed and same
    return {"matches_at_three_decimals": bool(passed), "columns": details}


def summarize(bundle, outdir):
    report = json.loads((outdir / "run_report.json").read_text())
    if not report["complete"]:
        raise ValueError("The fit run has not completed")
    for name, digest in report["input_sha256"].items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Bundle changed after fitting: {name}")
    settings = report["settings"]
    reference = read_fit(bundle / "reference.npz")
    with np.load(bundle / "inputs.npz", allow_pickle=False) as source:
        ids = source["subject_ids"].astype(str)
    if set(ids) != set(reference["subject_ids"]):
        raise ValueError("Reference and input cohorts differ")
    groups = read_splits(bundle / "splits.csv", len(ids))
    full = {}
    orientation = {}
    for name in ("control", "increased_modes"):
        candidate, signs = orient_full_cohort(read_fit(outdir / name / "fit_results.npz"), settings["prefer_condition"])
        full[name] = compare_fit(reference, candidate, common_mri=(name == "increased_modes"), full_cohort=True)
        orientation[name] = signs.tolist()
    full["control"].to_csv(outdir / "control_matches.csv", index=False)
    full["increased_modes"].to_csv(outdir / "resolution_matches.csv", index=False)
    rows = []
    for (repeat, fold), indices in groups.items():
        candidate = read_fit(outdir / f"participant_refits/rep{repeat:02d}_fold{fold:02d}/fit_results.npz")
        if not np.array_equal(candidate["subject_ids"], ids[indices]):
            raise ValueError(f"Participant order differs from the saved split: {repeat}, {fold}")
        matches = compare_fit(reference, candidate)
        for row in matches.to_dict("records"):
            rows.append(dict(repeat=repeat, fold=fold, n_train=len(indices),
                             scale_s=1., scale_f=1., scale_e=1., **row))
    details = pd.DataFrame(rows)
    summary = participant_summary(details)
    details.to_csv(outdir / "figure2_stability_by_fold.csv", index=False)
    summary.to_csv(outdir / "figure2_stability_summary.csv", index=False)
    table = []
    for row in summary.to_dict("records"):
        for metric in METRICS:
            table.append(dict(panel="A", reference_factor=row["reference_factor"], metric=metric,
                              value=row[f"{metric}_median"], q25=row[f"{metric}_q25"], q75=row[f"{metric}_q75"]))
    for row in full["increased_modes"].to_dict("records"):
        for metric in METRICS:
            table.append(dict(panel="B", reference_factor=row["reference_factor"], metric=metric,
                              value=row[metric], q25=np.nan, q75=np.nan))
    pd.DataFrame(table).to_csv(outdir / "table_s3.csv", index=False)
    checks = {"full_cohort_export_signs": orientation,
              "control_all_metrics_equal_one_at_three_decimals": all(
                  f"{value:.3f}" == "1.000" for value in full["control"][list(METRICS)].to_numpy().ravel()),
              "expected_output_checks": {}}
    expected_files = [
        ("figure2_stability_summary.csv", summary, ["reference_factor"], True),
        ("resolution_matches.csv", full["increased_modes"], ["reference_factor"], True),
        ("figure2_stability_by_fold.csv", details, ["repeat", "fold", "reference_factor"], False),
        ("control_matches.csv", full["control"], ["reference_factor"], False),
    ]
    for name, actual, keys, required in expected_files:
        path = bundle / "expected" / name
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"Required completed-analysis output: {path}")
            continue
        checks["expected_output_checks"][name] = check_expected(actual, path, keys)
    checks["passed"] = bool(checks["control_all_metrics_equal_one_at_three_decimals"] and all(
        item["matches_at_three_decimals"] for item in checks["expected_output_checks"].values()))
    (outdir / "reproduction_check.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nIncreased MRI modes:\n" + full["increased_modes"].to_string(index=False))
    print(f"\nSaved table_s3.csv and reproduction_check.json in {outdir}")
    if not checks["passed"]:
        raise SystemExit("Reproduction differs at reported precision; inspect reproduction_check.json. No results were adjusted.")
    print("All five factors and the 20/20/20 control match at the reported precision.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("examples/factor_stability"))
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.bundle, args.outdir)
