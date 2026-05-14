#!/usr/bin/env python3
# script: summarize_gemf_recovery_simulations.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "shared_only": "Shared structure only",
    "shared_specific": "Shared and modality-specific structure",
    "low_snr_specific": "Low-SNR modality-specific structure",
}

SCENARIO_ORDER = [
    "shared_only",
    "shared_specific",
    "low_snr_specific",
]

SPECIFIC_SCENARIOS = [
    "shared_specific",
    "low_snr_specific",
]

MAIN_METRICS = [
    "Z_subspace_cc_mean",
    "smri_shared_r2",
    "fmri_shared_r2",
    "eeg_shared_r2",
]

SUPP_METRICS = [
    "U_s_subspace_cc_mean",
    "U_f_subspace_cc_mean",
    "U_e_subspace_cc_mean",
    "smri_specific_r2",
    "fmri_specific_r2",
    "eeg_specific_r2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize GEMF recovery simulation results."
    )
    parser.add_argument("--results_dir", default="results/gemf_sim_recovery_energy")
    return parser.parse_args()


def mean_sd_text(mean, sd):
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def read_per_seed_results(manifest):
    rows = []

    for item in manifest:
        with open(item["eval_json"], "r", encoding="utf-8") as f:
            row = json.load(f)

        row["scenario"] = item["scenario"]
        row["n"] = item["n"]
        row["seed"] = item["seed"]

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_results(df):
    rows = []

    for (scenario, n), group in df.groupby(["scenario", "n"], sort=False):
        row = {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
            "n": n,
            "n_seeds": group.shape[0],
        }

        for metric in MAIN_METRICS + SUPP_METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_sd"] = vals.std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["scenario", "n"])


def make_shared_recovery_table(summary):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        (
            r"Fixed-rank simulation recovery of shared structure across sample sizes. "
            r"Values are mean \(\pm\) SD across 100 replications. "
            r"Shared-score recovery is the mean canonical correlation between the estimated and true shared score subspaces. "
            r"Shared reconstruction is the energy-based \(R^2\) for the true shared component in each modality."
        ),
        r"}",
        r"\label{tab:sim_shared_recovery}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"Scenario & \(n\) & Shared-score recovery & "
            r"Shared \(R^2\): sMRI & Shared \(R^2\): fMRI & Shared \(R^2\): EEG \\"
        ),
        r"\midrule",
    ]

    for scenario in SCENARIO_ORDER:
        scenario_rows = summary[summary["scenario"] == scenario].sort_values("n")

        first_row = True
        for _, row in scenario_rows.iterrows():
            label = row["scenario_label"] if first_row else ""
            first_row = False

            lines.append(
                f"{label} & {int(row['n'])} & "
                f"{mean_sd_text(row['Z_subspace_cc_mean_mean'], row['Z_subspace_cc_mean_sd'])} & "
                f"{mean_sd_text(row['smri_shared_r2_mean'], row['smri_shared_r2_sd'])} & "
                f"{mean_sd_text(row['fmri_shared_r2_mean'], row['fmri_shared_r2_sd'])} & "
                f"{mean_sd_text(row['eeg_shared_r2_mean'], row['eeg_shared_r2_sd'])} \\\\"
            )

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"

    lines += [
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]

    return "\n".join(lines)


def make_specific_recovery_table(summary):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        (
            r"Supplementary fixed-rank simulation recovery of modality-specific structure across sample sizes. "
            r"Values are mean \(\pm\) SD across 100 replications. "
            r"Specific-score recovery is the mean canonical correlation between estimated and true modality-specific score subspaces. "
            r"Specific reconstruction is the energy-based \(R^2\) for the true modality-specific component. "
            r"Entries are not applicable for the shared-only scenario."
        ),
        r"}",
        r"\label{tab:supp_sim_specific_recovery}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        (
            r"Scenario & \(n\) & "
            r"\multicolumn{3}{c}{Specific-score recovery} & "
            r"\multicolumn{3}{c}{Specific reconstruction \(R^2\)} \\"
        ),
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"& & sMRI & fMRI & EEG & sMRI & fMRI & EEG \\",
        r"\midrule",
    ]

    for scenario in SPECIFIC_SCENARIOS:
        scenario_rows = summary[summary["scenario"] == scenario].sort_values("n")

        first_row = True
        for _, row in scenario_rows.iterrows():
            label = row["scenario_label"] if first_row else ""
            first_row = False

            lines.append(
                f"{label} & {int(row['n'])} & "
                f"{mean_sd_text(row['U_s_subspace_cc_mean_mean'], row['U_s_subspace_cc_mean_sd'])} & "
                f"{mean_sd_text(row['U_f_subspace_cc_mean_mean'], row['U_f_subspace_cc_mean_sd'])} & "
                f"{mean_sd_text(row['U_e_subspace_cc_mean_mean'], row['U_e_subspace_cc_mean_sd'])} & "
                f"{mean_sd_text(row['smri_specific_r2_mean'], row['smri_specific_r2_sd'])} & "
                f"{mean_sd_text(row['fmri_specific_r2_mean'], row['fmri_specific_r2_sd'])} & "
                f"{mean_sd_text(row['eeg_specific_r2_mean'], row['eeg_specific_r2_sd'])} \\\\"
            )

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"

    lines += [
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
    summary = summarize_results(per_seed)

    per_seed_file = root / "simulation_recovery_per_seed.csv"
    summary_file = root / "simulation_recovery_summary.csv"
    shared_table_file = root / "table_sim_shared_recovery.tex"
    specific_table_file = root / "table_supp_sim_specific_recovery.tex"

    per_seed.to_csv(per_seed_file, index=False)
    summary.to_csv(summary_file, index=False)

    shared_table_file.write_text(make_shared_recovery_table(summary), encoding="utf-8")
    specific_table_file.write_text(make_specific_recovery_table(summary), encoding="utf-8")

    print("Wrote:", per_seed_file)
    print("Wrote:", summary_file)
    print("Wrote:", shared_table_file)
    print("Wrote:", specific_table_file)


if __name__ == "__main__":
    main()