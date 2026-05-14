#!/usr/bin/env python3
# script: outercv_fixed_budget_benchmark.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
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


def load_feature_block(path, anchor_ids):
    df = pd.read_csv(path)

    if "subject_id" not in df.columns:
        raise RuntimeError(f"{path} is missing subject_id column.")

    df["subject_id"] = df["subject_id"].map(norm_sid)
    return df.set_index("subject_id").loc[anchor_ids].reset_index()


def load_metadata(path, subject_col, anchor_ids):
    meta = pd.read_csv(path)

    if subject_col not in meta.columns:
        raise RuntimeError(f"{path} missing subject column {subject_col}")

    meta[subject_col] = meta[subject_col].map(norm_sid)

    return (
        meta.drop_duplicates(subset=[subject_col])
        .set_index(subject_col)
        .loc[anchor_ids]
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


def fit_predict_pca_features(Xtr, Xte, ytr, Ctr, Cte, rank):
    Xtr, Xte = preprocess_features(Xtr, Xte)
    Ctr, Cte = preprocess_covars(Ctr, Cte)

    k = min(rank, Xtr.shape[0] - 1, Xtr.shape[1])

    pca = PCA(n_components=k, svd_solver="full", random_state=0)
    Xtr_pc = pca.fit_transform(Xtr)
    Xte_pc = pca.transform(Xte)

    train_design = np.column_stack([Ctr, Xtr_pc])
    test_design = np.column_stack([Cte, Xte_pc])

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


def add_fold_result(rows, outcome, rep, fold, family, method, rank, complexity, y_true, pred, base_pred):
    row = {
        "outcome": outcome,
        "repeat": rep,
        "outer_fold": fold,
        "family": family,
        "method": method,
        "budget_rank": rank,
        "complexity": complexity,
    }
    row.update(regression_metrics(y_true, pred, base_pred))
    rows.append(row)


def add_predictions(rows, outcome, rep, fold, subject_ids, family, method, rank, complexity, y_true, pred, base_pred):
    for j, sid in enumerate(subject_ids):
        rows.append(
            {
                "outcome": outcome,
                "repeat": rep,
                "outer_fold": fold,
                "subject_id": sid,
                "family": family,
                "method": method,
                "budget_rank": rank,
                "complexity": complexity,
                "y_true": y_true[j],
                "pred": pred[j],
                "pred_base": base_pred[j],
            }
        )


def make_summary(fold_df):
    summary = (
        fold_df.groupby(
            ["outcome", "family", "method", "budget_rank", "complexity"],
            as_index=False,
        )
        .agg(
            n_folds=("reg_delta_r2", "size"),
            mean_delta_r2=("reg_delta_r2", "mean"),
            sd_delta_r2=(
                "reg_delta_r2",
                lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            ),
            median_delta_r2=("reg_delta_r2", "median"),
            q25_delta_r2=("reg_delta_r2", lambda x: float(np.quantile(x, 0.25))),
            q75_delta_r2=("reg_delta_r2", lambda x: float(np.quantile(x, 0.75))),
            mean_r2_full=("reg_r2_full", "mean"),
            mean_rmse=("reg_rmse_full", "mean"),
            mean_corr=("reg_corr_full", "mean"),
        )
    )

    summary["iqr_delta_r2"] = summary["q75_delta_r2"] - summary["q25_delta_r2"]
    summary["se_delta_r2"] = summary["sd_delta_r2"] / np.sqrt(
        summary["n_folds"].clip(lower=1)
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Outer-CV fixed-budget benchmark across PCA families, conv_ridge, and GEMF."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--subject_col", default="subject_id")

    parser.add_argument("--harm_smri_csv", required=True)
    parser.add_argument("--harm_fmri_csv", required=True)
    parser.add_argument("--harm_eeg_csv", required=True)
    parser.add_argument("--harm_concat_csv", required=True)
    parser.add_argument("--conv_multimodal_csv", required=True)

    parser.add_argument(
        "--outcome_specs",
        required=True,
        help='Example: "age_mid|male;LPS_1|male"',
    )
    parser.add_argument("--rank_grid", default="1,2,3,4,5,6,8,10,12,16,20")

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
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

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

    rank_grid = parse_int_list(args.rank_grid)
    outcome_specs = parse_outcome_specs(args.outcome_specs)

    data = load_npz_dict(args.data)
    subject_ids = np.array([norm_sid(x) for x in data["subject_ids"]], dtype="U6")

    harm_concat_raw = pd.read_csv(args.harm_concat_csv)
    harm_concat_raw["subject_id"] = harm_concat_raw["subject_id"].map(norm_sid)
    anchor_ids = harm_concat_raw["subject_id"].to_numpy()

    meta = load_metadata(args.metadata_csv, args.subject_col, anchor_ids)

    feature_blocks = {
        "harm_smri": load_feature_block(args.harm_smri_csv, anchor_ids),
        "harm_fmri": load_feature_block(args.harm_fmri_csv, anchor_ids),
        "harm_eeg": load_feature_block(args.harm_eeg_csv, anchor_ids),
        "lb_pca": load_feature_block(args.harm_concat_csv, anchor_ids),
        "conv_pca": load_feature_block(args.conv_multimodal_csv, anchor_ids),
    }

    X_blocks = {
        name: df.drop(columns=["subject_id"]).to_numpy(dtype=float)
        for name, df in feature_blocks.items()
    }

    pred_rows = []
    fold_rows = []

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

        sid_ok = anchor_ids[idx_ok]
        y = y_all[idx_ok]
        C = C_all[idx_ok]

        data_ok = subset_npz_dict(data, idx_ok)
        X_ok = {name: X[idx_ok] for name, X in X_blocks.items()}

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

                pred_base = fit_predict_base(y_tr, C_tr, C_te)
                test_subjects = sid_ok[te]

                # Conventional multimodal ridge on the original feature block.
                pred = fit_predict_ridge_features(
                    X_ok["conv_pca"][tr],
                    X_ok["conv_pca"][te],
                    y_tr,
                    C_tr,
                    C_te,
                )

                add_fold_result(
                    fold_rows,
                    outcome,
                    rep,
                    outer_fold,
                    "conv_ridge",
                    "conv_ridge",
                    np.nan,
                    np.nan,
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
                    "conv_ridge",
                    "conv_ridge",
                    np.nan,
                    np.nan,
                    y_te,
                    pred,
                    pred_base,
                )

                for r in rank_grid:
                    # PCA benchmarks at the same nominal feature budget.
                    for family, X in X_ok.items():
                        pred = fit_predict_pca_features(
                            X[tr],
                            X[te],
                            y_tr,
                            C_tr,
                            C_te,
                            rank=r,
                        )

                        method = f"{family}_r{r:02d}"

                        add_fold_result(
                            fold_rows,
                            outcome,
                            rep,
                            outer_fold,
                            family,
                            method,
                            r,
                            r,
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
                            family,
                            method,
                            r,
                            r,
                            y_te,
                            pred,
                            pred_base,
                        )

                    fold_dir = fold_root / outcome / f"rep{rep:02d}_fold{outer_fold:02d}"

                    outer_train = subset_npz_dict(data_ok, tr)
                    outer_test = subset_npz_dict(data_ok, te)

                    outer_train_scaled, outer_test_scaled, _ = scale_views_by_train(
                        outer_train,
                        outer_test,
                        mode="entry_rms",
                    )

                    train_npz = fold_dir / "outer_train_scaled.npz"
                    save_npz_dict(train_npz, outer_train_scaled)

                    fit_dir = fold_dir / f"outer_R{r}_Ls{args.Ls}_Lf{args.Lf}_Le{args.Le}"

                    fit, _ = fit_or_load_gemf(
                        data_source=train_npz,
                        fit_dir=fit_dir,
                        R=r,
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
                        seed=outer_seed + r,
                        resume=True,
                        quiet=args.quiet,
                    )

                    train_scores = infer_consistent_scores(
                        fit=fit,
                        data_source=outer_train_scaled,
                        w_s=args.w_s,
                        w_f=args.w_f,
                        w_e=args.w_e,
                        score_ridge=args.score_ridge,
                    )
                    test_scores = infer_consistent_scores(
                        fit=fit,
                        data_source=outer_test_scaled,
                        w_s=args.w_s,
                        w_f=args.w_f,
                        w_e=args.w_e,
                        score_ridge=args.score_ridge,
                    )

                    # GEMF shared scores only.
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
                        f"gemf_shared_R{r}",
                        r,
                        r,
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
                        f"gemf_shared_R{r}",
                        r,
                        r,
                        y_te,
                        pred,
                        pred_base,
                    )

                    # GEMF shared + modality-specific scores.
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
                        f"gemf_full_R{r}",
                        r,
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
                        f"gemf_full_R{r}",
                        r,
                        full_dim,
                        y_te,
                        pred,
                        pred_base,
                    )

    pred_df = round_numeric(pd.DataFrame(pred_rows), 3)
    fold_df = round_numeric(pd.DataFrame(fold_rows), 3)
    summary = round_numeric(make_summary(fold_df), 3)

    pred_df.to_csv(outdir / "outercv_predictions.csv", index=False)
    fold_df.to_csv(outdir / "outercv_fold_metrics.csv", index=False)
    summary.to_csv(outdir / "outercv_summary.csv", index=False)

    print("Wrote:", outdir / "outercv_predictions.csv")
    print("Wrote:", outdir / "outercv_fold_metrics.csv")
    print("Wrote:", outdir / "outercv_summary.csv")


if __name__ == "__main__":
    main()