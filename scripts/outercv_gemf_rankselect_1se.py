#!/usr/bin/env python3
# script: outercv_gemf_rankselect_1se.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from gemf.workflows import (
    load_npz_dict,
    subset_npz_dict,
    save_npz_dict,
    scale_views_by_train,
    fit_or_load_gemf,
    infer_consistent_scores,
    normalized_reconstruction_error,
)


RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def norm_sid(x):
    sid = str(x).strip().replace("sub-", "")
    return sid.zfill(6) if sid.isdigit() else sid


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_outcome_specs(spec):
    """
    Parse outcome specifications of the form:

        age_mid|male;LPS_1|male,site
    """
    specs = []

    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue

        if "|" not in item:
            raise ValueError(f"Bad outcome spec: {item}; expected outcome|cov1,cov2")

        outcome, covars = item.split("|", 1)
        covars = [c.strip() for c in covars.split(",") if c.strip()]
        specs.append((outcome.strip(), covars))

    return specs


def round_numeric(df, ndigits=3):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(ndigits)
    return out


def load_metadata(path, subject_col, subject_ids):
    meta = pd.read_csv(path)

    if subject_col not in meta.columns:
        raise RuntimeError(f"{path} missing subject column {subject_col}")

    meta[subject_col] = meta[subject_col].map(norm_sid)

    return (
        meta.drop_duplicates(subset=[subject_col])
        .set_index(subject_col)
        .loc[subject_ids]
        .reset_index()
        .rename(columns={subject_col: "subject_id"})
    )


def preprocess_covars(Ctr, Cte):
    if Ctr.shape[1] == 0:
        return Ctr.copy(), Cte.copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    Ctr = imputer.fit_transform(Ctr)
    Cte = imputer.transform(Cte)

    return scaler.fit_transform(Ctr), scaler.transform(Cte)


def preprocess_features(Xtr, Xte):
    if Xtr.shape[1] == 0:
        return Xtr.copy(), Xte.copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)

    return scaler.fit_transform(Xtr), scaler.transform(Xte)


def fit_predict_base(ytr, Ctr, Cte):
    Ctr, Cte = preprocess_covars(Ctr, Cte)

    if Ctr.shape[1] == 0:
        return np.repeat(np.mean(ytr), Cte.shape[0])

    return LinearRegression().fit(Ctr, ytr).predict(Cte)


def fit_predict_ridge_features(Xtr, Xte, ytr, Ctr, Cte):
    Xtr, Xte = preprocess_features(Xtr, Xte)
    Ctr, Cte = preprocess_covars(Ctr, Cte)

    train_design = np.column_stack([Ctr, Xtr])
    test_design = np.column_stack([Cte, Xte])

    return RidgeCV(alphas=RIDGE_ALPHAS).fit(train_design, ytr).predict(test_design)


def regression_metrics(y_true, pred, base_pred):
    r2_full = r2_score(y_true, pred)
    r2_base = r2_score(y_true, base_pred)

    return {
        "reg_r2_full": r2_full,
        "reg_r2_base": r2_base,
        "reg_delta_r2": r2_full - r2_base,
        "reg_rmse_full": np.sqrt(mean_squared_error(y_true, pred)),
        "reg_corr_full": np.corrcoef(y_true, pred)[0, 1],
    }


def add_fold_result(rows, outcome, rep, fold, family, method, selected_rank, complexity, y_true, pred, base_pred):
    row = {
        "outcome": outcome,
        "repeat": rep,
        "outer_fold": fold,
        "family": family,
        "method": method,
        "selected_rank": selected_rank,
        "complexity": complexity,
    }
    row.update(regression_metrics(y_true, pred, base_pred))
    rows.append(row)


def add_predictions(rows, outcome, rep, fold, subject_ids, family, method, selected_rank, complexity, y_true, pred, base_pred):
    for j, sid in enumerate(subject_ids):
        rows.append(
            {
                "outcome": outcome,
                "repeat": rep,
                "outer_fold": fold,
                "subject_id": sid,
                "family": family,
                "method": method,
                "selected_rank": selected_rank,
                "complexity": complexity,
                "y_true": y_true[j],
                "pred": pred[j],
                "pred_base": base_pred[j],
            }
        )


def inner_cv_gemf_loss(
    train_dict,
    R_grid,
    args,
    seed,
    workdir,
):
    rows = []
    n = train_dict["S_tilde"].shape[0]
    inner_cv = KFold(n_splits=args.inner_splits, shuffle=True, random_state=seed)

    for inner_fold, (tr, va) in enumerate(inner_cv.split(np.arange(n)), start=1):
        train0 = subset_npz_dict(train_dict, tr)
        valid0 = subset_npz_dict(train_dict, va)

        train_scaled, valid_scaled, _ = scale_views_by_train(
            train0,
            valid0,
            mode="entry_rms",
        )

        train_npz = workdir / f"inner_fold{inner_fold:02d}" / "train_data.npz"
        save_npz_dict(train_npz, train_scaled)

        for R in R_grid:
            fit_dir = workdir / f"inner_fold{inner_fold:02d}" / f"R{R}_Ls{args.Ls}_Lf{args.Lf}_Le{args.Le}"

            fit, _ = fit_or_load_gemf(
                data_source=train_npz,
                fit_dir=fit_dir,
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
                seed=seed + inner_fold + R,
                resume=True,
                quiet=args.quiet,
            )

            nre = normalized_reconstruction_error(
                fit=fit,
                data_source=valid_scaled,
                w_s=args.w_s,
                w_f=args.w_f,
                w_e=args.w_e,
                score_ridge=args.score_ridge,
            )

            rows.append(
                {
                    "inner_fold": inner_fold,
                    "R": R,
                    "nre": nre,
                }
            )

    return pd.DataFrame(rows)


def select_rank_1se(inner_df, one_se_tol=1.0):
    rank_summary = (
        inner_df.groupby("R", as_index=False)
        .agg(
            mean_nre=("nre", "mean"),
            sd_nre=("nre", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            n=("nre", "size"),
        )
        .sort_values("R")
        .reset_index(drop=True)
    )

    rank_summary["se_nre"] = rank_summary["sd_nre"] / np.sqrt(
        rank_summary["n"].clip(lower=1)
    )

    best_idx = rank_summary["mean_nre"].idxmin()
    best_rank = int(rank_summary.loc[best_idx, "R"])
    best_mean = float(rank_summary.loc[best_idx, "mean_nre"])
    best_se = float(rank_summary.loc[best_idx, "se_nre"])

    threshold = best_mean + one_se_tol * best_se
    selected_rank = int(rank_summary.loc[rank_summary["mean_nre"] <= threshold, "R"].min())

    return {
        "best_rank": best_rank,
        "best_mean_nre": best_mean,
        "best_se_nre": best_se,
        "one_se_threshold": threshold,
        "selected_rank": selected_rank,
    }


def make_outer_summary(fold_df):
    summary = (
        fold_df.groupby(["outcome", "family", "method"], as_index=False)
        .agg(
            n_folds=("reg_delta_r2", "size"),
            mean_delta_r2=("reg_delta_r2", "mean"),
            sd_delta_r2=(
                "reg_delta_r2",
                lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            ),
            median_delta_r2=("reg_delta_r2", "median"),
            mean_r2_full=("reg_r2_full", "mean"),
            mean_rmse=("reg_rmse_full", "mean"),
            mean_corr=("reg_corr_full", "mean"),
        )
    )

    summary["se_delta_r2"] = summary["sd_delta_r2"] / np.sqrt(
        summary["n_folds"].clip(lower=1)
    )

    return summary


def make_selected_rank_summary(sel_df):
    return (
        sel_df.groupby(["outcome", "family"], as_index=False)
        .agg(
            median_selected_rank=("selected_rank", "median"),
            mean_selected_rank=("selected_rank", "mean"),
            mode_selected_rank=("selected_rank", lambda x: pd.Series(x).mode().iloc[0]),
            min_selected_rank=("selected_rank", "min"),
            max_selected_rank=("selected_rank", "max"),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Outer-CV GEMF rank selection by inner-CV reconstruction + 1-SE."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--subject_col", default="subject_id")
    parser.add_argument(
        "--outcome_specs",
        required=True,
        help='Example: "age_mid|male;LPS_1|male"',
    )

    parser.add_argument("--R_grid", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--Ls", type=int, default=1)
    parser.add_argument("--Lf", type=int, default=1)
    parser.add_argument("--Le", type=int, default=1)

    parser.add_argument("--w_s", type=float, default=1.0)
    parser.add_argument("--w_f", type=float, default=1.0)
    parser.add_argument("--w_e", type=float, default=1.0)

    parser.add_argument("--lambda_s", type=float, default=1e-2)
    parser.add_argument("--lambda_f", type=float, default=1e-2)
    parser.add_argument("--lambda_e", type=float, default=1e-2)
    parser.add_argument("--zeta_e", type=float, default=1e-2)

    parser.add_argument("--specific_ridge", type=float, default=1e-6)
    parser.add_argument("--score_ridge", type=float, default=1e-1)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-5)

    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=4)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--one_se_tol", type=float, default=1.0)

    parser.add_argument("--outdir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    fold_root = outdir / "folds"

    outdir.mkdir(parents=True, exist_ok=True)
    fold_root.mkdir(parents=True, exist_ok=True)

    R_grid = parse_int_list(args.R_grid)
    outcome_specs = parse_outcome_specs(args.outcome_specs)

    data = load_npz_dict(args.data)
    subject_ids = np.array([norm_sid(x) for x in data["subject_ids"]], dtype="U6")
    meta = load_metadata(args.metadata_csv, args.subject_col, subject_ids)

    inner_rows = []
    sel_rows = []
    fold_rows = []
    pred_rows = []

    for outcome, covars in outcome_specs:
        needed = [outcome] + covars
        missing = [c for c in needed if c not in meta.columns]

        if missing:
            raise RuntimeError(f"Missing metadata columns for outcome {outcome}: {missing}")

        y_all = pd.to_numeric(meta[outcome], errors="coerce").to_numpy(dtype=float)

        if covars:
            C_all = np.column_stack(
                [
                    pd.to_numeric(meta[c], errors="coerce").to_numpy(dtype=float)
                    for c in covars
                ]
            )
        else:
            C_all = np.zeros((len(meta), 0), dtype=float)

        ok = np.isfinite(y_all) & np.isfinite(C_all).all(axis=1)
        idx_ok = np.where(ok)[0]

        sid_ok = subject_ids[idx_ok]
        y = y_all[idx_ok]
        C = C_all[idx_ok]
        data_ok = subset_npz_dict(data, idx_ok)

        for rep0 in range(args.n_repeats):
            rep = rep0 + 1
            outer_seed = args.seed + rep0

            outer_cv = KFold(
                n_splits=args.outer_splits,
                shuffle=True,
                random_state=outer_seed,
            )

            for outer_fold, (tr, te) in enumerate(outer_cv.split(y), start=1):
                y_tr, y_te = y[tr], y[te]
                C_tr, C_te = C[tr], C[te]
                test_subjects = sid_ok[te]

                pred_base = fit_predict_base(y_tr, C_tr, C_te)

                outer_train = subset_npz_dict(data_ok, tr)
                outer_test = subset_npz_dict(data_ok, te)

                fold_dir = fold_root / outcome / f"rep{rep:02d}_fold{outer_fold:02d}"
                inner_dir = fold_dir / "inner"

                inner_df = inner_cv_gemf_loss(
                    outer_train,
                    R_grid,
                    args,
                    seed=outer_seed + 1000 + outer_fold,
                    workdir=inner_dir,
                )

                inner_df["outcome"] = outcome
                inner_df["repeat"] = rep
                inner_df["outer_fold"] = outer_fold
                inner_df["family"] = "gemf"
                inner_rows.append(inner_df)

                selected = select_rank_1se(inner_df, one_se_tol=args.one_se_tol)
                selected_rank = int(selected["selected_rank"])

                sel_rows.append(
                    {
                        "outcome": outcome,
                        "repeat": rep,
                        "outer_fold": outer_fold,
                        "family": "gemf",
                        **selected,
                    }
                )

                train_scaled, test_scaled, _ = scale_views_by_train(
                    outer_train,
                    outer_test,
                    mode="entry_rms",
                )

                train_npz = fold_dir / "outer_train_scaled.npz"
                save_npz_dict(train_npz, train_scaled)

                fit_dir = fold_dir / (
                    f"outer_R{selected_rank}_Ls{args.Ls}_Lf{args.Lf}_Le{args.Le}"
                )

                fit, _ = fit_or_load_gemf(
                    data_source=train_npz,
                    fit_dir=fit_dir,
                    R=selected_rank,
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
                    seed=outer_seed,
                    resume=True,
                    quiet=args.quiet,
                )

                train_scores = infer_consistent_scores(
                    fit=fit,
                    data_source=train_scaled,
                    w_s=args.w_s,
                    w_f=args.w_f,
                    w_e=args.w_e,
                    score_ridge=args.score_ridge,
                )
                test_scores = infer_consistent_scores(
                    fit=fit,
                    data_source=test_scaled,
                    w_s=args.w_s,
                    w_f=args.w_f,
                    w_e=args.w_e,
                    score_ridge=args.score_ridge,
                )

                # GEMF shared factors only.
                pred = fit_predict_ridge_features(
                    train_scores["Zhat"],
                    test_scores["Zhat"],
                    y_tr,
                    C_tr,
                    C_te,
                )

                add_fold_result(
                    fold_rows,
                    outcome,
                    rep,
                    outer_fold,
                    "gemf_shared",
                    "gemf_shared_sel1se",
                    selected_rank,
                    selected_rank,
                    y_te,
                    pred,
                    pred_base,
                )
                add_predictions(
                    pred_rows,
                    outcome,
                    rep,
                    outer_fold,
                    test_subjects,
                    "gemf_shared",
                    "gemf_shared_sel1se",
                    selected_rank,
                    selected_rank,
                    y_te,
                    pred,
                    pred_base,
                )

                # GEMF shared plus modality-specific scores.
                full_dim = train_scores["score"].shape[1]

                pred = fit_predict_ridge_features(
                    train_scores["score"],
                    test_scores["score"],
                    y_tr,
                    C_tr,
                    C_te,
                )

                add_fold_result(
                    fold_rows,
                    outcome,
                    rep,
                    outer_fold,
                    "gemf_full",
                    "gemf_full_sel1se",
                    selected_rank,
                    full_dim,
                    y_te,
                    pred,
                    pred_base,
                )
                add_predictions(
                    pred_rows,
                    outcome,
                    rep,
                    outer_fold,
                    test_subjects,
                    "gemf_full",
                    "gemf_full_sel1se",
                    selected_rank,
                    full_dim,
                    y_te,
                    pred,
                    pred_base,
                )

    inner_df = round_numeric(pd.concat(inner_rows, ignore_index=True), 3)
    pred_df = round_numeric(pd.DataFrame(pred_rows), 3)
    fold_df = round_numeric(pd.DataFrame(fold_rows), 3)
    sel_df = round_numeric(pd.DataFrame(sel_rows), 3)

    summary = round_numeric(make_outer_summary(fold_df), 3)
    sel_summary = round_numeric(make_selected_rank_summary(sel_df), 3)

    inner_df.to_csv(outdir / "innercv_loss_curves.csv", index=False)
    pred_df.to_csv(outdir / "outercv_predictions.csv", index=False)
    fold_df.to_csv(outdir / "outercv_fold_metrics.csv", index=False)
    sel_df.to_csv(outdir / "innercv_rank_selection.csv", index=False)
    summary.to_csv(outdir / "outercv_summary.csv", index=False)
    sel_summary.to_csv(outdir / "selected_rank_summary.csv", index=False)

    for name in [
        "innercv_loss_curves.csv",
        "outercv_predictions.csv",
        "outercv_fold_metrics.csv",
        "innercv_rank_selection.csv",
        "outercv_summary.csv",
        "selected_rank_summary.csv",
    ]:
        print("Wrote:", outdir / name)


if __name__ == "__main__":
    main()