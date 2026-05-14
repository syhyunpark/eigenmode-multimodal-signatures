#!/usr/bin/env python3
# script: plot_partA_fixed_dimension_benchmark_clean.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MAIN_METHOD_ORDER = [
    "lb_pca",
    "conv_pca",
    "gemf_shared",
    "gemf_full",
]

# Curve plotting order and intended legend row order.
SUPP_METHOD_ORDER = [
    "lb_pca",
    "conv_pca",
    "gemf_shared",
    "gemf_full",
    "harm_fmri",
    "harm_smri",
    "harm_eeg",
]

METHOD_LABELS = {
    "lb_pca": "Multimodal harmonic-coordinate PCA",
    "conv_pca": "Multimodal conventional PCA",
    "gemf_shared": "GEMF (shared)",
    "gemf_full": "GEMF (full)",
    "harm_fmri": "fMRI harmonic PCA",
    "harm_smri": "sMRI harmonic PCA",
    "harm_eeg": "EEG harmonic PCA",
}

OUTCOME_LABELS = {
    "age_mid": "Outcome: Age",
    "LPS_1": "Outcome: LPS-1 cognitive score",
}

# Reverted to the cleaner previous style:
# all main method curves solid except GEMF full dashed.
LINE_STYLES = {
    "lb_pca": "-",
    "conv_pca": "-",
    "gemf_shared": "-",
    "gemf_full": "--",
    "harm_fmri": "-",
    "harm_smri": "-",
    "harm_eeg": "-",
}

MARKERS = {
    "lb_pca": "o",
    "conv_pca": "s",
    "gemf_shared": "P",
    "gemf_full": "X",
    "harm_fmri": "^",
    "harm_smri": "D",
    "harm_eeg": "v",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create fixed-dimensional benchmark figure.")
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="figure_partA_fixed_dimension_benchmark_clean")
    ap.add_argument(
        "--figure_type",
        choices=["main", "supp"],
        default="main",
        help="main: core curves; supp: full comparison including unimodal harmonic PCA.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def legend_order_for_matplotlib_rows(method_order: list[str], ncol: int) -> list[str]:
    """
    Matplotlib fills multi-row legends column-wise. To visually display rows in
    the desired order, reorder handles before calling fig.legend.

    For 7 items with ncol=4, desired visual rows:
      row 1: [1,2,3,4]
      row 2: [5,6,7]
    Matplotlib needs input order:
      [1,5,2,6,3,7,4]
    """
    n = len(method_order)
    nrow = int(np.ceil(n / ncol))
    rows = []
    for r in range(nrow):
        rows.append(method_order[r * ncol : min((r + 1) * ncol, n)])

    reordered = []
    for c in range(ncol):
        for r in range(nrow):
            if c < len(rows[r]):
                reordered.append(rows[r][c])
    return reordered


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    method_order = MAIN_METHOD_ORDER if args.figure_type == "main" else SUPP_METHOD_ORDER

    outcomes = ["age_mid", "LPS_1"]
    outcomes = [o for o in outcomes if o in set(df["outcome"].unique())]

    # Slightly compact but not overly tight.
    fig_width = 10.2 if len(outcomes) == 2 else 5.3
    fig_height = 4.45 if args.figure_type == "main" else 4.75

    fig, axes = plt.subplots(
        1,
        len(outcomes),
        figsize=(fig_width, fig_height),
        sharex=True,
    )

    if len(outcomes) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, outcomes):
        dfo = df[df["outcome"] == outcome].copy()

        for fam in method_order:
            d = dfo[dfo["family"] == fam].copy()
            if d.empty:
                continue

            d = d.sort_values("budget_rank")
            x = d["budget_rank"].to_numpy(dtype=float)
            y = d["mean_delta_r2"].to_numpy(dtype=float)
            se = d["se_delta_r2"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                linestyle=LINE_STYLES.get(fam, "-"),
                marker=MARKERS.get(fam, "o"),
                linewidth=2.15,
                markersize=5.1,
                label=METHOD_LABELS.get(fam, fam),
            )
            ax.fill_between(x, y - se, y + se, alpha=0.12)

        ax.axhline(0.0, color="gray", linewidth=1.0)
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome), fontsize=14)
        ax.set_xlabel(r"Representation dimension $r$", fontsize=12.5)
        ax.set_ylabel(r"Outer-CV $\Delta R^2$", fontsize=12.5)
        ax.set_xticks(np.arange(1, 13, 1))
        ax.grid(True, alpha=0.25)

    # Collect handles.
    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            handles_by_label[l] = h

    ncol = 4

    if args.figure_type == "supp":
        # Force visual row order:
        # Row 1: multimodal harmonic PCA, conventional PCA, GEMF shared, GEMF full
        # Row 2: fMRI harmonic PCA, sMRI harmonic PCA, EEG harmonic PCA
        legend_methods = legend_order_for_matplotlib_rows(SUPP_METHOD_ORDER, ncol=ncol)
    else:
        legend_methods = MAIN_METHOD_ORDER

    legend_labels = [
        METHOD_LABELS[m] for m in legend_methods
        if METHOD_LABELS[m] in handles_by_label
    ]
    legend_handles = [handles_by_label[l] for l in legend_labels]

    # Slightly more room than the previous attempt, much less than the original.
    if args.figure_type == "main":
        legend_y = 0.018
        bottom_margin = 0.205
    else:
        legend_y = 0.020
        bottom_margin = 0.265

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol,
        frameon=False,
        fontsize=10.4,
        handlelength=2.6,
        columnspacing=1.4,
        labelspacing=0.8,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.875,
        bottom=bottom_margin,
        wspace=0.18,
    )

    out_png = outdir / f"{args.prefix}.png"
    out_pdf = outdir / f"{args.prefix}.pdf"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
