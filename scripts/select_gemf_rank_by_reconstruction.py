#!/usr/bin/env python3
# script: select_gemf_rank_by_reconstruction.py

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from gemf.data import load_multiview_npz
from gemf.fixed_rank_model import FixedRankGMHFConfig as FixedRankGEMFConfig
from gemf.fixed_rank_model import FixedRankGMHF as FixedRankGEMF
from gemf.score_inference import infer_full_scores


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select GEMF shared rank by cross-validated normalized reconstruction error."
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--R_grid", default="1,2,3,4,5,6,7,8,9,10,11,12")

    ap.add_argument("--Ls", type=int, default=1)
    ap.add_argument("--Lf", type=int, default=1)
    ap.add_argument("--Le", type=int, default=1)

    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--w_s", type=float, default=1.0)
    ap.add_argument("--w_f", type=float, default=1.0)
    ap.add_argument("--w_e", type=float, default=1.0)

    ap.add_argument("--lambda_s", type=float, default=0.01)
    ap.add_argument("--lambda_f", type=float, default=0.01)
    ap.add_argument("--lambda_e", type=float, default=0.01)
    ap.add_argument("--zeta_e", type=float, default=0.01)

    ap.add_argument("--specific_ridge", type=float, default=1e-6)
    ap.add_argument("--score_ridge", type=float, default=0.1)

    ap.add_argument("--max_iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--quiet", action="store_true")

    return ap.parse_args()


def parse_grid(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    vals = sorted(set(vals))
    if any(v <= 0 for v in vals):
        raise ValueError("All candidate ranks must be positive.")
    return vals


def make_folds(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    sizes = np.full(k, n // k, dtype=int)
    sizes[: n % k] += 1

    folds = []
    start = 0
    for size in sizes:
        folds.append(perm[start : start + size])
        start += size

    out = []
    for j in range(k):
        test = np.sort(folds[j])
        train = np.sort(np.concatenate([folds[m] for m in range(k) if m != j]))
        out.append((train, test))
    return out


def fit_model(train_data, args, R: int, fold_id: int):
    config = FixedRankGEMFConfig(
        R=R,
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
        random_state=args.seed + 1000 * R + fold_id,
        verbose=not args.quiet,
    )

    model = FixedRankGEMF(config=config)
    model.fit(train_data)
    return model.save_dict(data=train_data)


def unpack_full_scores(score: np.ndarray, R: int, Ls: int, Lf: int, Le: int):
    z = score[:, :R]
    idx = R
    us = score[:, idx : idx + Ls]
    idx += Ls
    uf = score[:, idx : idx + Lf]
    idx += Lf
    ue = score[:, idx : idx + Le]
    return z, us, uf, ue


def reconstruct_from_score(fit: Dict[str, np.ndarray], score: np.ndarray, R: int, Ls: int, Lf: int, Le: int):
    z, us, uf, ue = unpack_full_scores(score, R, Ls, Lf, Le)

    S_hat = np.einsum("nr,rkq->nkq", z, fit["A_shared"])
    F_hat = np.einsum("nr,rkm->nkm", z, fit["M_shared"])
    E_hat = np.einsum("nr,rk,rf,rc->nkfc", z, fit["a_shared"], fit["b_shared"], fit["c_shared"])

    if Ls > 0 and "A_specific_s" in fit and fit["A_specific_s"].size > 0:
        S_hat += np.einsum("nl,lkq->nkq", us, fit["A_specific_s"])

    if Lf > 0 and "M_specific_f" in fit and fit["M_specific_f"].size > 0:
        F_hat += np.einsum("nl,lkm->nkm", uf, fit["M_specific_f"])

    if Le > 0 and "a_specific_e" in fit and fit["a_specific_e"].size > 0:
        E_hat += np.einsum(
            "nl,lk,lf,lc->nkfc",
            ue,
            fit["a_specific_e"],
            fit["b_specific_e"],
            fit["c_specific_e"],
        )

    return S_hat, F_hat, E_hat


def safe_nre(Y: np.ndarray, Yhat: np.ndarray) -> float:
    denom = float(np.sum(Y ** 2))
    if denom <= 1e-12:
        return np.nan
    return float(np.sum((Y - Yhat) ** 2)) / denom


def modality_balanced_nre(test_data, fit: Dict[str, np.ndarray], args, R: int) -> Dict[str, float]:
    score = infer_full_scores(
        fit=fit,
        data_like=test_data,
        w_s=args.w_s,
        w_f=args.w_f,
        w_e=args.w_e,
        ridge=args.score_ridge,
    )

    S_hat, F_hat, E_hat = reconstruct_from_score(
        fit=fit,
        score=score,
        R=R,
        Ls=args.Ls,
        Lf=args.Lf,
        Le=args.Le,
    )

    nre_s = safe_nre(test_data.S_tilde, S_hat)
    nre_f = safe_nre(test_data.Sigma_tilde, F_hat)
    nre_e = safe_nre(test_data.E, E_hat)

    # Equal-modality normalized reconstruction error.
    total = np.nanmean([nre_s, nre_f, nre_e])

    return {
        "nre_total": float(total),
        "nre_s": float(nre_s),
        "nre_f": float(nre_f),
        "nre_e": float(nre_e),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_multiview_npz(args.data)
    R_grid = parse_grid(args.R_grid)
    folds = make_folds(data.n, args.n_folds, args.seed)

    fold_rows = []

    for R in R_grid:
        if not args.quiet:
            print(f"\n===== R={R} =====")

        for fold_id, (train_idx, test_idx) in enumerate(folds):
            train_data = data.subset(train_idx)
            test_data = data.subset(test_idx)

            fit = fit_model(train_data, args, R=R, fold_id=fold_id)
            scores = modality_balanced_nre(test_data, fit, args, R=R)

            row = {
                "R": R,
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
            row.update(scores)
            fold_rows.append(row)

    with open(outdir / "cv_fold_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["R", "fold", "n_train", "n_test", "nre_total", "nre_s", "nre_f", "nre_e"],
        )
        writer.writeheader()
        writer.writerows(fold_rows)

    summary_rows = []
    for R in R_grid:
        rows = [r for r in fold_rows if r["R"] == R]
        vals = np.array([r["nre_total"] for r in rows], dtype=float)

        row = {
            "R": R,
            "mean_nre": float(np.mean(vals)),
            "sd_nre": float(np.std(vals, ddof=1)),
            "se_nre": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
            "mean_nre_s": float(np.mean([r["nre_s"] for r in rows])),
            "mean_nre_f": float(np.mean([r["nre_f"] for r in rows])),
            "mean_nre_e": float(np.mean([r["nre_e"] for r in rows])),
        }
        summary_rows.append(row)

    with open(outdir / "cv_rank_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "R",
                "mean_nre",
                "sd_nre",
                "se_nre",
                "mean_nre_s",
                "mean_nre_f",
                "mean_nre_e",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    means = np.array([r["mean_nre"] for r in summary_rows], dtype=float)
    ses = np.array([r["se_nre"] for r in summary_rows], dtype=float)
    Rs = np.array([r["R"] for r in summary_rows], dtype=int)

    best_idx = int(np.argmin(means))
    best_R = int(Rs[best_idx])
    best_mean = float(means[best_idx])
    best_se = float(ses[best_idx])
    one_se_threshold = best_mean + best_se
    selected_R = int(np.min(Rs[means <= one_se_threshold]))

    selection = {
        "R_grid": R_grid,
        "best_R_by_mean": best_R,
        "best_mean_nre": best_mean,
        "best_se_nre": best_se,
        "one_se_threshold": one_se_threshold,
        "selected_R_one_se": selected_R,
        "selection_rule": "smallest R with mean modality-balanced NRE within one SE of the minimum",
    }

    with open(outdir / "selected_rank_summary.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    print("Wrote:", outdir / "cv_fold_results.csv")
    print("Wrote:", outdir / "cv_rank_summary.csv")
    print("Wrote:", outdir / "selected_rank_summary.json")
    print("Best R:", best_R)
    print("Selected R by one-SE:", selected_R)


if __name__ == "__main__":
    main()
