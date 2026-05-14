#!/usr/bin/env python3
# script: summarize_gemf_rank_selection_simulations.py

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "shared_only": "Shared structure only",
    "shared_specific": "Shared and modality-specific structure",
    "low_snr_specific": "Low-SNR modality-specific structure",
}

SCENARIO_ORDER = {
    "shared_only": 1,
    "shared_specific": 2,
    "low_snr_specific": 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize GEMF rank-selection simulation results."
    )
    parser.add_argument("--results_dir", default="results/gemf_sim_rank_selection_energy")
    return parser.parse_args()


def rank_distribution(values):
    counts = Counter(values)
    return ", ".join(f"{k}:{counts[k]}" for k in sorted(counts))


def read_per_seed_results(manifest):
    rows = []

    for item in manifest:
        with open(item["selection_json"], "r", encoding="utf-8") as f:
            selected = json.load(f)

        rows.append(
            {
                "scenario": item["scenario"],
                "n": int(item["n"]),
                "seed": int(item["seed"]),
                "true_R": int(item.get("true_R", 3)),
                "best_R_by_mean": int(selected["best_R_by_mean"]),
                "selected_R_one_se": int(selected["selected_R_one_se"]),
                "best_mean_nre": float(selected["best_mean_nre"]),
                "best_se_nre": float(selected["best_se_nre"]),
                "one_se_threshold": float(selected["one_se_threshold"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_rank_selection(df):
    rows = []

    for (scenario, n), group in df.groupby(["scenario", "n"], sort=False):
        true_R = int(group["true_R"].iloc[0])

        best = group["best_R_by_mean"].astype(int).to_numpy()
        one_se = group["selected_R_one_se"].astype(int).to_numpy()

        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                "n": n,
                "n_seeds": len(group),
                "true_R": true_R,
                "best_R_by_mean_mean": best.mean(),
                "best_R_by_mean_sd": best.std(ddof=1) if len(best) > 1 else 0.0,
                "best_R_by_mean_prop_exact": float(np.mean(best == true_R)),
                "best_R_by_mean_distribution": rank_distribution(best),
                "selected_R_one_se_mean": one_se.mean(),
                "selected_R_one_se_sd": one_se.std(ddof=1) if len(one_se) > 1 else 0.0,
                "selected_R_one_se_prop_exact": float(np.mean(one_se == true_R)),
                "selected_R_one_se_distribution": rank_distribution(one_se),
            }
        )

    return pd.DataFrame(rows)


def make_latex_table(summary):
    summary = summary.copy()
    summary["_scenario_order"] = summary["scenario"].map(SCENARIO_ORDER).fillna(99)
    summary = summary.sort_values(["n", "_scenario_order"])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        (
            r"Supplementary simulation results for cross-validated shared-rank selection. "
            r"Candidate ranks were \(R=1,\ldots,12\), and the true shared rank was \(R=3\). "
            r"Best by CV mean denotes the rank minimizing the mean modality-balanced normalized reconstruction error. "
            r"The one-standard-error rule selects the smallest rank whose mean reconstruction error is within one standard error of the minimum. "
            r"Prop. exact denotes the proportion of replications in which the one-standard-error rule selected the true shared rank."
        ),
        r"}",
        r"\label{tab:supp_sim_rank_selection}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\(n\) & Scenario & True \(R\) & Best by CV mean & One-SE rule & Prop. exact (One-SE) \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{int(row['n'])} & {row['scenario_label']} & {int(row['true_R'])} & "
            f"{row['best_R_by_mean_mean']:.2f} $\\pm$ {row['best_R_by_mean_sd']:.2f} & "
            f"{row['selected_R_one_se_mean']:.2f} $\\pm$ {row['selected_R_one_se_sd']:.2f} & "
            f"{row['selected_R_one_se_prop_exact']:.2f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]

    return "\n".join(lines)


def main():
    args = parse_args()
    root = Path(args.results_dir)

    with open(root / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    per_seed = read_per_seed_results(manifest)
    summary = summarize_rank_selection(per_seed)
    latex_table = make_latex_table(summary)

    per_seed_file = root / "rank_selection_per_seed.csv"
    summary_file = root / "rank_selection_summary.csv"
    table_file = root / "table_supp_sim_rank_selection.tex"

    per_seed.to_csv(per_seed_file, index=False)
    summary.to_csv(summary_file, index=False)
    table_file.write_text(latex_table, encoding="utf-8")

    print("Wrote:", per_seed_file)
    print("Wrote:", summary_file)
    print("Wrote:", table_file)


if __name__ == "__main__":
    main()