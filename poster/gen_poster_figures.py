"""
Generate poster figures for the mmDar Gaussian radar model poster.
Auburn University color scheme: Navy #03244d, Orange #DD550C.
Run from repo root: python3 poster/gen_poster_figures.py
"""

import shutil
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
POSTER_ASSETS = REPO_ROOT / "poster" / "assets"
SAMPLES_DIR = POSTER_ASSETS / "samples"

TEASER_SRC = REPO_ROOT / "imgs" / "teaser.png"
PLOTS_DIR = REPO_ROOT / "results" / "baseline_5090_adapted" / "plots"

SAMPLE_SRCS = {
    "sample1.png": PLOTS_DIR / "best_117_1000.png",
    "sample2.png": PLOTS_DIR / "best_229_332.png",
    "sample3.png": PLOTS_DIR / "best_238_97.png",
}

# ---------------------------------------------------------------------------
# Auburn color scheme
# ---------------------------------------------------------------------------
NAVY   = "#03244d"
ORANGE = "#DD550C"

# Grouped bar colors: [Chamfer, mod-H] per method
BAR_COLORS = {
    "Published RadarHD":            ("#888888", "#aaaaaa"),
    "Our Baseline\n(U-Net, 17.5M)": ("#888888", "#aaaaaa"),
    "ConvLSTM\n(Temporal)":         ("#cc4444", "#dd7777"),
    "Ours: Gaussian\n(3.1M)":       (ORANGE,    NAVY),
}

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":          18,
    "axes.labelsize":     20,
    "axes.titlesize":     22,
    "xtick.labelsize":    16,
    "ytick.labelsize":    16,
    "legend.fontsize":    16,
    "figure.dpi":         200,
    "savefig.dpi":        200,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
METHODS = [
    "Published RadarHD",
    "Our Baseline\n(U-Net, 17.5M)",
    "ConvLSTM\n(Temporal)",
    "Ours: Gaussian\n(3.1M)",
]

CHAMFER    = [0.360, 0.406, 0.603, 0.280]
MOD_HAUS   = [0.240, 0.296, 0.467, 0.205]


def make_comparison_chart(out_path: Path) -> None:
    """Grouped bar chart: Chamfer + mod-Hausdorff for each method."""
    n = len(METHODS)
    x = np.arange(n)
    width = 0.35

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(14, 7))

        for i, (method, ch, mh) in enumerate(zip(METHODS, CHAMFER, MOD_HAUS)):
            c_ch, c_mh = BAR_COLORS[method]

            b1 = ax.bar(x[i] - width / 2, ch, width, color=c_ch,
                        label="Chamfer" if i == 0 else "_nolegend_",
                        edgecolor="white", linewidth=0.8)
            b2 = ax.bar(x[i] + width / 2, mh, width, color=c_mh,
                        label="Mod-Hausdorff" if i == 0 else "_nolegend_",
                        edgecolor="white", linewidth=0.8)

            # Value labels on top of bars
            for bar, val in [(b1, ch), (b2, mh)]:
                ax.text(
                    bar[0].get_x() + bar[0].get_width() / 2,
                    val + 0.012,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=16, fontweight="bold",
                    color="#222222",
                )

        # Build a clean custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#888888", label="Chamfer"),
            Patch(facecolor="#aaaaaa", label="Mod-Hausdorff"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=16)
        ax.set_ylabel("Distance (m) — lower is better", fontsize=20)
        ax.set_ylim(0, 0.72)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)

        # Highlight the "Ours" column with a subtle background
        ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, alpha=0.07, color=ORANGE, zorder=0)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved: {out_path}")


def copy_asset(src: Path, dst: Path) -> None:
    if not src.exists():
        warnings.warn(f"Source not found, skipping: {src}", stacklevel=2)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Copied: {src.name} → {dst}")


def main() -> None:
    print("Generating poster figures ...")

    POSTER_ASSETS.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Comparison bar chart
    make_comparison_chart(POSTER_ASSETS / "comparison.png")

    # 2. Teaser
    copy_asset(TEASER_SRC, POSTER_ASSETS / "teaser.png")

    # 3. Sample BEV plots
    for dst_name, src_path in SAMPLE_SRCS.items():
        copy_asset(src_path, SAMPLES_DIR / dst_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
