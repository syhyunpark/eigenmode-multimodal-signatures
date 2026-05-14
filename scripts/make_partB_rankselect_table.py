#!/usr/bin/env python3
# script: make_partB_rankselect_table.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_MAP = {
    ("lb_pca", "lb_pca_sel1se"): "Multimodal harmonic PCA",
    ("conv_pca", "conv_pca_sel1se"): "Conventional PCA",
    ("gemf_shared", "gemf_shared_sel1se"): "GEMF (shared)",
    ("gemf_full", "gemf_full_sel1se"): "GEMF (full)",
    ("harm_smri", "harm_smri_sel1se"): "sMRI harmonic PCA",
    ("harm_fmri", "harm_fmri_sel1se"): "fMRI harmonic PCA",
    ("harm_eeg", "harm_eeg_sel1se"): "EEG harmonic PCA",
    ("conv_ridge", "conv_ridge"): "Conventional ridge",
}

ORDER = [
    ("lb_pca", "lb_pca_sel1se"),
    ("conv_pca", "conv_pca_sel1se"),
    ("gemf_shared", "gemf_shared_sel1se"),
    ("gemf_full", "gemf_full_sel1se"),
    ("harm_smri", "harm_smri_sel1se"),
    ("harm_fmri", "harm_fmri_sel1se"),
    ("harm_eeg", "harm_eeg_sel1se"),
    ("conv_ridge", "conv_ridge"),
]


def get_rank_string(fam: str, ranks: pd.DataFrame) -> str:
    if fam == "conv_ridge":
        return "--"
    if fam in ["gemf_shared", "gemf_full"]:
        dr = ranks[(ranks["outcome"] == "age_mid") & (ranks["family"] == "gemf")]
    else:
        dr = ranks[(ranks["outcome"] == "age_mid") & (ranks["family"] == fam)]

    if dr.empty:
        return "--"
    return str(int(round(float(dr["median_selected_rank"].iloc[0]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--rank_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    summ = pd.read_csv(args.summary_csv)
    ranks = pd.read_csv(args.rank_csv)

    rows = []
    for fam, method in ORDER:
        label = METHOD_MAP[(fam, method)]

        d_age = summ[
            (summ["outcome"] == "age_mid")
            & (summ["family"] == fam)
            & (summ["method"] == method)
        ]
        d_lps = summ[
            (summ["outcome"] == "LPS_1")
            & (summ["family"] == fam)
            & (summ["method"] == method)
        ]

        if d_age.empty or d_lps.empty:
            raise RuntimeError(f"Missing row for {fam}, {method}")

        rank_str = get_rank_string(fam, ranks)

        rows.append(
            [
                label,
                rank_str,
                float(d_age["mean_delta_r2"].iloc[0]),
                float(d_age["mean_corr"].iloc[0]),
                float(d_age["mean_rmse"].iloc[0]),
                float(d_lps["mean_delta_r2"].iloc[0]),
                float(d_lps["mean_corr"].iloc[0]),
                float(d_lps["mean_rmse"].iloc[0]),
            ]
        )

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{")
    tex.append(
        r"Outcome-independent rank selection using inner cross-validation and the one-standard-error rule. "
        r"Rank was selected from \(r=1,\ldots,12\) using normalized reconstruction error within each outer training set, "
        r"without using the external outcomes. Performance is summarized over 25 outer test folds. "
        r"\(\Delta R^2\) is the increase in cross-validated \(R^2\) relative to a sex-only baseline model; "
        r"Corr. is the prediction--outcome correlation; RMSE is the root mean squared prediction error. "
        r"For GEMF, the selected rank is the shared rank \(R\); GEMF (full) additionally includes one modality-specific score per modality."
    )
    tex.append(r"}")
    tex.append(r"\label{tab:app_rank_selection}")
    tex.append(r"\resizebox{\textwidth}{!}{")
    tex.append(r"\begin{tabular}{lccccccc}")
    tex.append(r"\toprule")
    tex.append(r"Representation &")
    tex.append(r"Median selected rank &")
    tex.append(r"\multicolumn{3}{c}{Outcome: Age} &")
    tex.append(r"\multicolumn{3}{c}{Outcome: LPS-1} \\")
    tex.append(r"\cmidrule(lr){3-5}")
    tex.append(r"\cmidrule(lr){6-8}")
    tex.append(r"&")
    tex.append(r"&")
    tex.append(r"\(\Delta R^2\) & Corr. & RMSE &")
    tex.append(r"\(\Delta R^2\) & Corr. & RMSE \\")
    tex.append(r"\midrule")

    for label, rank, age_dr2, age_corr, age_rmse, lps_dr2, lps_corr, lps_rmse in rows:
        tex.append(
            f"{label} & {rank} & "
            f"{age_dr2:.3f} & {age_corr:.3f} & {age_rmse:.3f} & "
            f"{lps_dr2:.3f} & {lps_corr:.3f} & {lps_rmse:.3f} \\\\"
        )

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"}")
    tex.append(r"\end{table}")

    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(tex), encoding="utf-8")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
