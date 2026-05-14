#!/usr/bin/env python3
# script: make_gemf_main_montage.py

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def open_image(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing figure file: {path}")
    return Image.open(path).convert("RGBA")


def resize_width(image, width):
    height = int(image.height * width / image.width)
    return image.resize((width, height), Image.LANCZOS)


def get_font(size=46):
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for path in font_paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def add_panel_label(image, label, top_margin=78, left_margin=18, font_size=50):
    """Add a panel label above the image, rather than over the figure itself."""
    width, height = image.size

    canvas = Image.new("RGBA", (width, height + top_margin), "white")
    canvas.paste(image, (0, top_margin))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (left_margin, 12),
        label,
        fill="black",
        font=get_font(font_size),
    )

    return canvas


def stack_horizontally(images, pad=30):
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + pad * (len(images) - 1)

    canvas = Image.new("RGBA", (width, height), "white")

    x = 0
    for im in images:
        y = (height - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + pad

    return canvas


def stack_vertically(images, pad=36):
    width = max(im.width for im in images)
    height = sum(im.height for im in images) + pad * (len(images) - 1)

    canvas = Image.new("RGBA", (width, height), "white")

    y = 0
    for im in images:
        x = (width - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + pad

    return canvas


def clean_condition_names(names):
    clean = []

    for name in names:
        s = str(name).strip()
        key = s.lower()

        if key in {"eo", "eyes-open", "eyes open"}:
            clean.append("EO")
        elif key in {"ec", "eyes-closed", "eyes closed"}:
            clean.append("EC")
        else:
            clean.append(s)

    return clean


def make_eeg_profile_panel(panel_npz, factor, out_png, width_inches=10.5, height_inches=3.5):
    """
    Make a compact EEG profile panel with spectral and condition loadings.

    The fMRI eigenmode-loading line plot is intentionally omitted here.
    """
    z = np.load(panel_npz, allow_pickle=True)
    r = factor - 1

    spectral_loading = np.asarray(z["b_shared"])[r]
    condition_loading = np.asarray(z["c_shared"])[r]

    if "eeg_freq_centers" in z.files:
        freq = np.asarray(z["eeg_freq_centers"], dtype=float)
    else:
        freq = np.arange(1, len(spectral_loading) + 1)

    if "condition_names" in z.files:
        condition_names = clean_condition_names(z["condition_names"])
    else:
        condition_names = [f"C{j + 1}" for j in range(len(condition_loading))]

    fig, axes = plt.subplots(1, 2, figsize=(width_inches, height_inches))

    axes[0].plot(freq, spectral_loading, linewidth=2)
    axes[0].axhline(0, linewidth=1)
    axes[0].set_title(rf"EEG spectral loading for $z_{{{factor}}}$")
    axes[0].set_xlabel("Frequency")
    axes[0].set_ylabel("Loading")

    axes[1].bar(np.arange(len(condition_names)), condition_loading)
    axes[1].axhline(0, linewidth=1)
    axes[1].set_xticks(np.arange(len(condition_names)))
    axes[1].set_xticklabels(condition_names)
    axes[1].set_title(rf"EEG condition loading for $z_{{{factor}}}$")
    axes[1].set_xlabel("Condition")
    axes[1].set_ylabel("Loading")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return open_image(out_png)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a main-paper GEMF representative-factor montage from existing panel figures."
    )

    parser.add_argument("--figures_dir", required=True)
    parser.add_argument("--panel_npz", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--factor",
        type=int,
        required=True,
        help="1-based GEMF shared factor index, e.g., 5.",
    )
    parser.add_argument(
        "--smri_measure_index",
        type=int,
        default=1,
        help="1-based sMRI measure index to display, e.g., 1 if measure 1 is cortical thickness.",
    )
    parser.add_argument("--panel_width", type=int, default=1040)
    parser.add_argument("--top_margin", type=int, default=80)
    parser.add_argument("--pad", type=int, default=30)

    return parser.parse_args()


def main():
    args = parse_args()

    figdir = Path(args.figures_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    panel_npz = Path(args.panel_npz).expanduser().resolve()

    outdir.mkdir(parents=True, exist_ok=True)

    factor = args.factor
    smri_index = args.smri_measure_index

    image_files = {
        "A": figdir / f"gemf_factor_z{factor}_smri_{smri_index}_LH.png",
        "B": figdir / f"gemf_factor_z{factor}_fmri_LH.png",
        "C": figdir / f"gemf_factor_z{factor}_eeg_LH.png",
    }

    panels = []
    for label, path in image_files.items():
        image = open_image(path)
        image = resize_width(image, args.panel_width)
        image = add_panel_label(image, label, top_margin=args.top_margin)
        panels.append(image)

    top_row = stack_horizontally(panels, pad=args.pad)

    profile_tmp = outdir / f"_tmp_gemf_factor_z{factor}_eeg_profiles_only.png"
    profile = make_eeg_profile_panel(panel_npz, factor, profile_tmp)
    profile = resize_width(profile, top_row.width)
    profile = add_panel_label(profile, "D", top_margin=args.top_margin)

    montage = stack_vertically([top_row, profile], pad=42)

    out_png = outdir / f"figure_gemf_representative_factor_z{factor}.png"
    out_pdf = outdir / f"figure_gemf_representative_factor_z{factor}.pdf"

    montage.save(out_png)
    montage.convert("RGB").save(out_pdf)

    profile_tmp.unlink(missing_ok=True)

    print(f"Factor: z{factor}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()