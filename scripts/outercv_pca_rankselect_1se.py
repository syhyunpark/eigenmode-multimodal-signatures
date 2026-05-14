#!/usr/bin/env python3
# script: outercv_pca_rankselect_1se.py

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
    if k < 1:
        raise RuntimeError(f"Effective PCA rank < 1 for requested rank={rank}")

    pca = PCA(n_components=k, svd_solver="full", random_state=0)
    Xtr_pc = pca.fit_transform(Xtr)
    Xte_pc = pca.transform(Xte)

    train_design = np.column_stack([Ctr, Xtr_pc])
    test_design = np.column_stack([Cte, Xte_pc])

    return RidgeCV(alphas=RIDGE_ALPHAS).fit(train_design, ytr).predict(test_design)


def inner_cv_recon_loss(X, rank_grid, n_splits, seed):
    rows = []
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for inner_fold, (tr, va) in enumerate(inner_cv.split(X), start=1):
        Xtr, Xva = X[tr], X[va]
        Xtr, Xva = preprocess_features(Xtr, Xva)

        for rank in rank_grid:
            k = min(rank, Xtr.shape[0] - 1, Xtr.shape[1])
            if k < 1:
                continue

            pca = PCA(n_components=k, svd_solver="full", random_state=0)
            Zva = pca.fit(Xtr).transform(Xva)
            Xhat = pca.inverse_transform(Zva)

            denom = float(np.sum(Xva**2))
            nre = float(np.sum((Xva - Xhat) ** 2)) / max(denom, 1e-12)

            rows.append(
                {
                    "inner_fold": inner_fold,
                    "rank": rank,
                    "nre": nre,
                }
            )

    return pd.DataFrame(rows)


def select_rank_1se(inner_df, one_se_tol=1.0):
    rank_summary = (
        inner_df.groupby("rank", as_index=False)
        .agg(
            mean_nre=("nre", "mean"),
            sd_nre=("nre", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            n=("nre", "size"),
        )
        .sort_values("rank")
        .reset_index(drop=True)
    )

    rank_summary["se_nre"] = rank_summary["sd_nre"] / np.sqrt(
        rank_summary["n"].clip(lower=1)
    )

    best_idx = rank_summary["mean_nre"].idxmin()
    best_rank = int(rank_summary.loc[best_idx, "rank"])
    best_mean = float(rank_summary.loc[best_idx, "mean_nre"])
    best_se = float(rank_summary.loc[best_idx, "se_nre"])

    threshold = best_mean + one_se_tol * best_se
    selected_rank = int(rank_summary.loc[rank_summary["mean_nre"] <= threshold, "rank"].min())

    return {
        "best_rank": best_rank,
        "best_mean_nre": best_mean,
        "best_se_nre": best_se,
        "one_se_threshold": threshold,
        "selected_rank": selected_rank,
    }


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


def add_fold_result(rows, outcome, rep, fold, family, method, selected_rank, y_true, pred, base_pred):
    row = {
        "outcome": outcome,
        "repeat": rep,
        "outer_fold": fold,
        "family": family,
        "method": method,
        "selected_rank": selected_rank,
    }
    row.update(regression_metrics(y_true, pred, base_pred))
    rows.append(row)


def add_predictions(rows, outcome, rep, fold, subject_ids, family, method, selected_rank, y_true, pred, base_pred):
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
                "y_true": y_true[j],
                "pred": pred[j],
                "pred_base": base_pred[j],
            }
        )


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
        description="Outer-CV PCA rank selection by inner-CV reconstruction + 1-SE."
    )

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

    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=4)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--one_se_tol", type=float, default=1.0)

    parser.add_argument("--outdir", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rank_grid = parse_int_list(args.rank_grid)
    outcome_specs = parse_outcome_specs(args.outcome_specs)

    # Use the harmonic concatenated file as the subject anchor.
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

        sid_ok = anchor_ids[idx_ok]
        y = y_all[idx_ok]
        C = C_all[idx_ok]
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
                test_subjects = sid_ok[te]

                pred_base = fit_predict_base(y_tr, C_tr, C_te)

                # Conventional multimodal ridge reference.
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
                    y_te,
                    pred,
                    pred_base,
                )

                for family, X in X_ok.items():
                    X_train = X[tr]
                    X_test = X[te]

                    inner_df = inner_cv_recon_loss(
                        X_train,
                        rank_grid,
                        n_splits=args.inner_splits,
                        seed=outer_seed + 1000 + outer_fold,
                    )

                    inner_df["outcome"] = outcome
                    inner_df["repeat"] = rep
                    inner_df["outer_fold"] = outer_fold
                    inner_df["family"] = family
                    inner_rows.append(inner_df)

                    selected = select_rank_1se(inner_df, one_se_tol=args.one_se_tol)
                    selected_rank = int(selected["selected_rank"])

                    sel_rows.append(
                        {
                            "outcome": outcome,
                            "repeat": rep,
                            "outer_fold": outer_fold,
                            "family": family,
                            **selected,
                        }
                    )

                    pred = fit_predict_pca_features(
                        X_train,
                        X_test,
                        y_tr,
                        C_tr,
                        C_te,
                        rank=selected_rank,
                    )

                    method = f"{family}_sel1se"

                    add_fold_result(
                        fold_rows,
                        outcome,
                        rep,
                        outer_fold,
                        family,
                        method,
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
                        family,
                        method,
                        selected_rank,
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