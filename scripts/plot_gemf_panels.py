#!/usr/bin/env python3
# script: plot_gemf_panels.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import mne


VALID_VIEWS = ("cau", "dor", "fro", "lat", "med", "par", "ros", "ven")


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_subject_dir(fsaverage_dir: str | Path) -> Path:
    p = Path(fsaverage_dir).expanduser().resolve()
    candidates = [p, p / "fsaverage5"]
    for c in candidates:
        if (c / "surf").exists():
            return c
    raise FileNotFoundError(
        f"Could not resolve an fsaverage5 subject directory from: {p}\n"
        f"Tried: {candidates}"
    )


def pretty_measure_name(x: str) -> str:
    s = str(x)
    s = s.replace("beta_", "")
    s = s.replace("Beta_", "")
    s = s.replace("_", " ")
    s = s.replace("meancurv", "mean curvature")
    s = s.replace("sulc", "sulcal depth")
    s = s.strip()
    if len(s) == 0:
        return s
    return s[0].upper() + s[1:]


def diverging_clim(v, q=95.0, eps=1e-12):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        a = 1.0
    else:
        a = np.percentile(np.abs(v), q)
        if not np.isfinite(a) or a < eps:
            a = np.max(np.abs(v)) if v.size else 1.0
        if not np.isfinite(a) or a < eps:
            a = 1.0
    mid = 0.5 * a
    return dict(
        kind="value",
        pos_lims=[0.0, mid, a],
        neg_lims=[-a, -mid, 0.0],
    )


def plot_one_hemi_view(
    data_hemi: np.ndarray,
    subject: str,
    subjects_dir: str,
    hemi: str,
    surf_name: str,
    view: str,
    out_png: str,
    title: str = "",
    smoothing: int = 10,
    cmap: str = "RdBu_r",
    qlim: float = 95.0,
    size=(900, 420),
    colorbar: bool = True,
    title_fontsize: int = 26,
    title_pad: int = 20,
):
    n = len(data_hemi)
    if hemi == "lh":
        vertices = [np.arange(n), np.array([], dtype=int)]
        stc_data = np.asarray(data_hemi, float)[:, None]
    else:
        vertices = [np.array([], dtype=int), np.arange(n)]
        stc_data = np.asarray(data_hemi, float)[:, None]

    stc = mne.SourceEstimate(
        data=stc_data,
        vertices=vertices,
        tmin=0.0,
        tstep=1.0,
        subject=subject,
    )

    fig = stc.plot(
        subjects_dir=subjects_dir,
        hemi=hemi,
        surface=surf_name,
        time_viewer=False,
        smoothing_steps=smoothing,
        views=view,
        colormap=cmap,
        clim=diverging_clim(data_hemi, q=qlim),
        colorbar=colorbar,
        backend="matplotlib",
        size=size,
    )

    # Reserve title space even if the title is blank so two panels line up vertically.
    shown_title = title if title else " "
    try:
        fig.axes[0].set_title(shown_title, fontsize=title_fontsize, pad=title_pad)
    except Exception:
        try:
            fig.suptitle(shown_title, fontsize=title_fontsize, y=0.98)
        except Exception:
            pass

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def stitch_horizontal(png_paths: List[str], out_png: str, valign: str = "bottom"):
    imgs = [Image.open(p).convert("RGBA") for p in png_paths if Path(p).exists()]
    if not imgs:
        return

    H = max(im.height for im in imgs)
    W = sum(im.width for im in imgs)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    x = 0
    for im in imgs:
        if valign == "top":
            y = 0
        elif valign == "center":
            y = (H - im.height) // 2
        else:
            y = H - im.height
        canvas.paste(im, (x, y))
        x += im.width

    canvas.save(out_png)


def render_two_views(
    data_hemi: np.ndarray,
    out_png: str,
    *,
    subject: str,
    subjects_dir: str,
    hemi: str,
    surf_name: str,
    views: List[str],
    title: str,
    smoothing: int,
    qlim: float,
    brain_title_fontsize: int,
    brain_title_pad: int,
):
    tmp_pngs = []
    for j, v in enumerate(views):
        p = str(Path(out_png).with_name(f"_tmp_{Path(out_png).stem}_{v}.png"))
        plot_one_hemi_view(
            data_hemi=data_hemi,
            subject=subject,
            subjects_dir=subjects_dir,
            hemi=hemi,
            surf_name=surf_name,
            view=v,
            out_png=p,
            title=title if j == 0 else "",
            smoothing=smoothing,
            qlim=qlim,
            colorbar=True,
            title_fontsize=brain_title_fontsize,
            title_pad=brain_title_pad,
        )
        tmp_pngs.append(p)

    stitch_horizontal(tmp_pngs, out_png, valign="bottom")
    for p in tmp_pngs:
        Path(p).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Plot GEMF Part C figures from panel_objects.npz using MNE matplotlib backend."
    )
    ap.add_argument("--panel_npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--fsaverage_dir",
        required=True,
        help="Path to fsaverage5 subject dir or its parent subjects dir.",
    )
    ap.add_argument("--factors", default="1,2,3,4,5", help="1-based factor indices to plot.")
    ap.add_argument(
        "--smri_measures_to_show",
        default="1,2,3,4",
        help="1-based sMRI measure indices to plot.",
    )
    ap.add_argument("--views", default="lat,med")
    ap.add_argument("--surface", default="smoothwm")
    ap.add_argument("--smoothing", type=int, default=10)
    ap.add_argument("--qlim", type=float, default=95.0)

    # Brain-surface title settings only. These do not affect profile-plot titles.
    ap.add_argument("--brain_title_fontsize", type=int, default=26)
    ap.add_argument("--brain_title_pad", type=int, default=20)

    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.panel_npz, allow_pickle=True)
    panel = {k: z[k] for k in z.files}

    factors = [i - 1 for i in parse_int_list(args.factors)]
    smri_idx = [i - 1 for i in parse_int_list(args.smri_measures_to_show)]
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for v in views:
        if v not in VALID_VIEWS:
            raise ValueError(f"Invalid view '{v}'. Allowed: {VALID_VIEWS}")

    R = int(panel["R"])
    age = np.asarray(panel["age"], dtype=float) if "age" in panel else None
    shared_scores = np.asarray(panel["shared_scores"])
    smri_measures = [pretty_measure_name(str(x)) for x in np.asarray(panel["smri_measures"])]
    eeg_freq_centers = np.asarray(panel["eeg_freq_centers"], dtype=float)
    condition_names = [str(x) for x in np.asarray(panel["condition_names"])]

    subj_dir = resolve_subject_dir(args.fsaverage_dir)
    subjects_dir = str(subj_dir.parent)
    subject = subj_dir.name

    coords_lh, _ = mne.read_surface(str(subj_dir / "surf" / f"lh.{args.surface}"))
    n_lh = coords_lh.shape[0]

    # 1. Shared-factor vs age scatterplots. Age is on the x-axis.
    if age is not None:
        ncols = min(3, R)
        nrows = int(np.ceil(R / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for r in range(R):
            ax = axes[r]
            ax.scatter(age, shared_scores[:, r], s=18, alpha=0.75)
            ok = np.isfinite(shared_scores[:, r]) & np.isfinite(age)

            if np.sum(ok) >= 3:
                b = np.polyfit(age[ok], shared_scores[ok, r], 1)
                xx = np.linspace(age[ok].min(), age[ok].max(), 100)
                ax.plot(xx, b[0] * xx + b[1], linewidth=2)
                c = np.corrcoef(age[ok], shared_scores[ok, r])[0, 1]
                ax.set_title(rf"$z_{{{r+1}}}$ (r = {c:.2f})")
            else:
                ax.set_title(rf"$z_{{{r+1}}}$")

            ax.set_xlabel("Age")
            ax.set_ylabel(rf"$z_{{{r+1}}}$ score")

        for j in range(R, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        fig.savefig(outdir / "gemf_shared_scores_vs_age.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 2. Factor profile plots. These title sizes are unchanged by brain_title_fontsize.
    for r in factors:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

        axes[0].plot(eeg_freq_centers, panel["b_shared"][r], linewidth=2)
        axes[0].axhline(0, linewidth=1)
        axes[0].set_title(rf"$z_{{{r+1}}}$: EEG spectral loading")
        axes[0].set_xlabel("Frequency")
        axes[0].set_ylabel("Loading")

        axes[1].bar(np.arange(len(condition_names)), panel["c_shared"][r])
        axes[1].axhline(0, linewidth=1)
        axes[1].set_xticks(np.arange(len(condition_names)))
        axes[1].set_xticklabels(condition_names)
        axes[1].set_title(rf"$z_{{{r+1}}}$: EEG condition loading")
        axes[1].set_xlabel("Condition")
        axes[1].set_ylabel("Loading")

        axes[2].plot(
            np.arange(1, panel["fmri_eigenmode_loadings"][r].shape[0] + 1),
            panel["fmri_eigenmode_loadings"][r],
            linewidth=2,
        )
        axes[2].axhline(0, linewidth=1)
        axes[2].set_title(rf"$z_{{{r+1}}}$: fMRI eigenmode loading")
        axes[2].set_xlabel("Eigenmode")
        axes[2].set_ylabel("Loading")

        fig.tight_layout()
        fig.savefig(outdir / f"gemf_factor_z{r+1}_profiles.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 3. Cortical maps: LH only, two stitched views, one title.
    for r in factors:
        eeg_map = np.asarray(panel["eeg_cortex_maps"][r])[:n_lh]
        render_two_views(
            eeg_map,
            str(outdir / f"gemf_factor_z{r+1}_eeg_LH.png"),
            subject=subject,
            subjects_dir=subjects_dir,
            hemi="lh",
            surf_name=args.surface,
            views=views,
            title=rf"$z_{{{r+1}}}$: EEG cortical loading",
            smoothing=args.smoothing,
            qlim=args.qlim,
            brain_title_fontsize=args.brain_title_fontsize,
            brain_title_pad=args.brain_title_pad,
        )

        fmri_map = np.asarray(panel["fmri_cortex_maps"][r])[:n_lh]
        render_two_views(
            fmri_map,
            str(outdir / f"gemf_factor_z{r+1}_fmri_LH.png"),
            subject=subject,
            subjects_dir=subjects_dir,
            hemi="lh",
            surf_name=args.surface,
            views=views,
            title=rf"$z_{{{r+1}}}$: fMRI cortical pattern",
            smoothing=args.smoothing,
            qlim=args.qlim,
            brain_title_fontsize=args.brain_title_fontsize,
            brain_title_pad=args.brain_title_pad,
        )

        for j in smri_idx:
            smri_map = np.asarray(panel["smri_cortex_maps"][r, j])[:n_lh]
            mname = smri_measures[j] if j < len(smri_measures) else f"Measure {j+1}"
            render_two_views(
                smri_map,
                str(outdir / f"gemf_factor_z{r+1}_smri_{j+1}_LH.png"),
                subject=subject,
                subjects_dir=subjects_dir,
                hemi="lh",
                surf_name=args.surface,
                views=views,
                title=rf"$z_{{{r+1}}}$: sMRI {mname} loading",
                smoothing=args.smoothing,
                qlim=args.qlim,
                brain_title_fontsize=args.brain_title_fontsize,
                brain_title_pad=args.brain_title_pad,
            )

    print("Wrote figures to:", outdir)


if __name__ == "__main__":
    main()