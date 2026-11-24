#!/usr/bin/env python3
"""Refit the two GEMF factor-stability analyses from the released input bundle."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from gemf.data import MultiViewData
from gemf.fixed_rank_model import FixedRankGEMF, FixedRankGEMFConfig
from gemf.score_inference import infer_shared_scores

AXES = ("smri_measures", "eeg_freq_centers", "eeg_freq_bands", "condition_names")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bundle(path):
    with np.load(path / "inputs.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    required = {"subject_ids", "S_tilde", "Sigma_tilde", "E", "lambda_vals", *AXES}
    if not required.issubset(arrays):
        raise ValueError(f"Missing input fields: {sorted(required - arrays.keys())}")
    expected = ((188, 50, 4), (188, 50, 50), (188, 20, 20, 2))
    if tuple(arrays[k].shape for k in ("S_tilde", "Sigma_tilde", "E")) != expected:
        raise ValueError("Expected the released 188-participant 50/50/20 input arrays")
    for key in ("S_tilde", "Sigma_tilde", "E"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite input: {key}")
    if not np.array_equal(arrays["lambda_vals"], np.arange(1, 51)):
        raise ValueError("The completed analysis used spatial weights 1,...,50")
    ids = arrays["subject_ids"].astype(str)
    if ids.shape != (188,) or len(set(ids)) != len(ids):
        raise ValueError("Input participant IDs must be unique and follow the released row order")
    arrays["subject_ids"] = ids
    settings = json.loads((path / "settings.json").read_text())
    if settings["schema_version"] != 1:
        raise ValueError("Unsupported bundle schema")
    config_keys = {field.name for field in fields(FixedRankGEMFConfig)}
    for name in ("control", "participant_refits", "increased_modes"):
        cfg = settings["model_configs"][name]
        if set(cfg) != config_keys:
            raise ValueError(f"{name}: a complete archived model configuration is required")
        if [cfg[k] for k in ("R", "Ls", "Lf", "Le")] != [5, 1, 1, 1]:
            raise ValueError(f"{name}: expected R=5 and one specific factor per modality")
        if cfg["score_ridge"] != settings["score_ridge"]:
            raise ValueError(f"{name}: inconsistent shared-score ridge settings")
    return arrays, settings


def read_splits(path, n):
    """Read saved memberships and their order; never regenerate random folds."""
    table = pd.read_csv(path)
    columns = ["repeat", "fold", "train_order", "row_index"]
    if not set(columns).issubset(table):
        raise ValueError(f"Split manifest must contain {columns}")
    if table[columns].isna().any().any() or not np.equal(
            table[columns], table[columns].astype(int)).all().all():
        raise ValueError("Split manifest entries must be integers")
    table[columns] = table[columns].astype(int)
    expected = {(repeat, fold) for repeat in range(1, 6) for fold in range(1, 6)}
    groups = {}
    for key, rows in table.groupby(["repeat", "fold"], sort=True):
        rows = rows.sort_values("train_order")
        indices = rows["row_index"].to_numpy()
        if not np.array_equal(rows["train_order"], np.arange(len(rows))):
            raise ValueError(f"Split {key}: train_order must be zero-based and consecutive")
        if len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError(f"Split {key}: duplicate or out-of-range participant rows")
        if len(indices) not in {n - n // 5, n - (n + 4) // 5}:
            raise ValueError(f"Split {key}: expected approximately 80% of participants")
        groups[key] = indices
    if set(groups) != expected:
        raise ValueError("Expected exactly five saved folds in each of five repeats")
    for repeat in range(1, 6):
        counts = np.bincount(np.concatenate([groups[repeat, fold] for fold in range(1, 6)]), minlength=n)
        if not np.array_equal(counts, np.full(n, 4)):
            raise ValueError(f"Repeat {repeat}: each participant must appear in four training subsets")
    return groups


def make_data(arrays, indices, mri_modes):
    # The matrix logarithm has already been applied. Restrict its stored entries.
    data = MultiViewData(
        S_tilde=np.asarray(arrays["S_tilde"][indices, :mri_modes], dtype=float),
        Sigma_tilde=np.asarray(arrays["Sigma_tilde"][indices, :mri_modes, :mri_modes], dtype=float),
        E=np.asarray(arrays["E"][indices], dtype=float),
        lambda_vals=np.asarray(arrays["lambda_vals"][:max(mri_modes, arrays["E"].shape[1])], dtype=float),
        subject_ids=arrays["subject_ids"][indices],
    )
    data.validate()
    return data


def fit_once(data, config, score_ridge, axes, destination):
    start = time.perf_counter()
    model = FixedRankGEMF(FixedRankGEMFConfig(**config)).fit(data)
    saved = model.save_dict(data)
    scores = infer_shared_scores(saved, data, config["w_s"], config["w_f"], config["w_e"], ridge=score_ridge)
    # Preserve the raw score/loading orientation, including the fitted Z separately.
    saved.pop("config")
    saved.update(shared_scores=scores, config_json=json.dumps(config, sort_keys=True),
                 score_definition="shared-only inference", score_ridge=score_ridge,
                 **axes)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "fit_results.npz", **saved)
    return {"seconds": time.perf_counter() - start, "iterations": len(saved["history_loss_total"])}


def run(bundle, outdir):
    arrays, settings = read_bundle(bundle)
    groups = read_splits(bundle / "splits.csv", len(arrays["subject_ids"]))
    if not (bundle / "reference.npz").is_file():
        raise FileNotFoundError(bundle / "reference.npz")
    outdir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    axes = {key: arrays[key] for key in AXES}
    inputs = {name: sha256(bundle / name) for name in ("inputs.npz", "reference.npz", "splits.csv", "settings.json")}
    report = {"settings": settings, "input_sha256": inputs,
              "python": platform.python_version(), "platform": platform.platform(),
              "packages": {key: version(key) for key in ("numpy", "scipy", "pandas", "scikit-learn")},
              "fits": {}, "complete": False}
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "scripts/summarize_gemf_factor_stability.py",
               *(root / "gemf").glob("*.py")]
    report["source_sha256"] = {str(path.relative_to(root)): sha256(path) for path in sources}
    (outdir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    all_rows = np.arange(len(arrays["subject_ids"]))
    for name, modes in (("control", 20), ("increased_modes", 50)):
        print(f"Fitting {name}: {modes}/{modes}/20 modes", flush=True)
        report["fits"][name] = fit_once(make_data(arrays, all_rows, modes), settings["model_configs"][name],
                                         settings["score_ridge"], axes, outdir / name)
    for (repeat, fold), rows in groups.items():
        name = f"participant_refits/rep{repeat:02d}_fold{fold:02d}"
        print(f"Fitting repeat {repeat}, fold {fold}: {len(rows)} participants", flush=True)
        report["fits"][name] = fit_once(make_data(arrays, rows, 20), settings["model_configs"]["participant_refits"],
                                         settings["score_ridge"], axes, outdir / name)
    report.update(seconds=time.perf_counter() - start, complete=True)
    (outdir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed 27 fits in {report['seconds']:.1f} seconds. Results: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("examples/factor_stability"))
    parser.add_argument("--outdir", type=Path, required=True, help="New output directory; existing paths are not reused")
    args = parser.parse_args()
    run(args.bundle, args.outdir)
