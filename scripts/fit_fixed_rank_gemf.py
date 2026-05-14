#!/usr/bin/env python3
# script: fit_fixed_rank_gemf.py

import argparse
import json
from pathlib import Path

import numpy as np

from gemf.workflows import fit_gemf_model, augment_fit_with_consistent_scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the fixed-rank GEMF model to a standardized .npz dataset."
    )

    parser.add_argument("--data", required=True, help="Path to .npz data file.")
    parser.add_argument("--outdir", required=True, help="Output directory.")

    parser.add_argument("--R", type=int, default=3, help="Shared rank.")
    parser.add_argument("--Ls", type=int, default=0, help="sMRI-specific rank.")
    parser.add_argument("--Lf", type=int, default=0, help="fMRI-specific rank.")
    parser.add_argument("--Le", type=int, default=0, help="EEG-specific rank.")

    parser.add_argument("--w-s", type=float, default=1.0, dest="w_s")
    parser.add_argument("--w-f", type=float, default=1.0, dest="w_f")
    parser.add_argument("--w-e", type=float, default=1.0, dest="w_e")

    parser.add_argument("--lambda-s", type=float, default=1e-2, dest="lambda_s")
    parser.add_argument("--lambda-f", type=float, default=1e-2, dest="lambda_f")
    parser.add_argument("--lambda-e", type=float, default=1e-2, dest="lambda_e")
    parser.add_argument("--zeta-e", type=float, default=1e-2, dest="zeta_e")

    parser.add_argument("--specific-ridge", type=float, default=1e-6, dest="specific_ridge")
    parser.add_argument("--score-ridge", type=float, default=1e-1, dest="score_ridge")

    parser.add_argument("--max-iter", type=int, default=100, dest="max_iter")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def in_sample_r2(y, X):
    """Return the in-sample R^2 from a least-squares regression of y on X."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def make_summary(data, fit, model, args):
    history = model.history_

    summary = {
        "n": data.n,
        "K": data.K,
        "q": data.q,
        "F": data.F,
        "C": data.C,
        "final_loss_total": float(history["loss_total"][-1]),
        "final_loss_s": float(history["loss_s"][-1]),
        "final_loss_f": float(history["loss_f"][-1]),
        "final_loss_e": float(history["loss_e"][-1]),
        "final_penalty": float(history["penalty"][-1]),
        "iterations_run": len(history["loss_total"]),
        "config": vars(args),
    }

    if getattr(data, "age", None) is not None:
        age = np.asarray(data.age, dtype=float)

        score_sets = {
            "age_regression_r2_in_sample": fit["Z"],
            "age_regression_r2_in_sample_Zhat": fit["Zhat_train"],
            "age_regression_r2_in_sample_score": fit["score_train"],
        }

        for name, scores in score_sets.items():
            summary[name] = in_sample_r2(age, scores)

    return summary


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fit, data, model = fit_gemf_model(
        data_source=args.data,
        R=args.R,
        Ls=args.Ls,
        Lf=args.Lf,
        Le=args.Le,
        w_s=args.w_s,
        w_f=args.w_f,
        w_e=args.w_e,
        lambda_s=args.lambda_s,
        lambda_f=args.lambda_f,
        lambda_e=args.lambda_e,
        zeta_e=args.zeta_e,
        specific_ridge=args.specific_ridge,
        score_ridge=args.score_ridge,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        quiet=args.quiet,
    )

    fit = augment_fit_with_consistent_scores(
        fit=fit,
        data_source=data,
        w_s=args.w_s,
        w_f=args.w_f,
        w_e=args.w_e,
        score_ridge=args.score_ridge,
    )

    result_file = outdir / "fit_results.npz"
    summary_file = outdir / "fit_summary.json"

    np.savez_compressed(result_file, **fit)

    summary = make_summary(data, fit, model, args)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved fit results to: {result_file}")
    print(f"Saved summary to:     {summary_file}")


if __name__ == "__main__":
    main()