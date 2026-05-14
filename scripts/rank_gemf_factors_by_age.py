#!/usr/bin/env python3
# script: rank_gemf_factors_by_age.py

import argparse
import re

import numpy as np
import pandas as pd


def residualize(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    resid = np.full(len(y), np.nan, dtype=float)

    if ok.sum() < X.shape[1] + 2:
        return resid

    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    resid[ok] = y[ok] - X[ok] @ beta

    return resid


def correlation(x, y):
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 3:
        return np.nan

    x = x[ok] - np.mean(x[ok])
    y = y[ok] - np.mean(y[ok])

    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom <= 1e-12:
        return np.nan

    return float(np.sum(x * y) / denom)


def find_z_columns(df):
    z_cols = [c for c in df.columns if re.fullmatch(r"z\d+", str(c))]
    z_cols = sorted(z_cols, key=lambda c: int(c.replace("z", "")))

    if not z_cols:
        raise ValueError("No z-factor columns found. Expected columns like z1, z2, ...")

    return z_cols


def get_age_column(df, requested_col):
    if requested_col in df.columns:
        return requested_col

    if "age" in df.columns:
        return "age"

    raise ValueError(f"Could not find age column '{requested_col}' or fallback 'age'.")


def rank_factors_by_age(df, age_col, sex_col):
    if sex_col not in df.columns:
        raise ValueError(f"Could not find sex column '{sex_col}'.")

    z_cols = find_z_columns(df)

    age = df[age_col].to_numpy(dtype=float)
    sex = df[sex_col].to_numpy(dtype=float)

    X_sex = np.column_stack([np.ones(len(df)), sex])
    age_resid = residualize(age, X_sex)

    rows = []

    for z_col in z_cols:
        z = df[z_col].to_numpy(dtype=float)
        z_resid = residualize(z, X_sex)

        raw_r = correlation(z, age)
        adjusted_r = correlation(z_resid, age_resid)

        rows.append(
            {
                "factor": z_col,
                "factor_index": int(z_col.replace("z", "")),
                "corr_age_raw": raw_r,
                "corr_age_sex_adjusted": adjusted_r,
                "abs_corr_age_sex_adjusted": (
                    abs(adjusted_r) if np.isfinite(adjusted_r) else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_corr_age_sex_adjusted", ascending=False)
        .round(3)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank GEMF shared factors by sex-adjusted association with age."
    )

    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--age_col", default="age_mid")
    parser.add_argument("--sex_col", default="male")

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.scores_csv)

    age_col = get_age_column(df, args.age_col)
    ranked = rank_factors_by_age(df, age_col, args.sex_col)

    ranked.to_csv(args.out_csv, index=False)

    print(f"Wrote: {args.out_csv}")
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()