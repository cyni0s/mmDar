"""
Generate all publication figures for IEEE conference paper.
Saves PDFs to reports/initial_report/figures/
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = "/git/mmDar/reports/initial_report/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "pdf.fonttype": 42,   # embed fonts as Type-1 for IEEE
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
})

IEEE_1COL = 3.5   # single-column width in inches
IEEE_2COL = 7.16  # double-column width in inches


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Baseline U-Net Training Curves
# ─────────────────────────────────────────────────────────────────────────────
def parse_baseline_log(path):
    """Parse auto_launch.log up to '=== TEST EVALUATION ===' line."""
    epochs, losses = [], []
    val_epochs, val_cd, val_mh = [], [], []

    with open(path) as f:
        for line in f:
            if "=== TEST EVALUATION" in line:
                break
            # Training loss line: "Ep   N | loss X.XXXX | val_cd ..."
            m = re.match(
                r"Ep\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+val_cd\s+([\d.]+)\s+val_mh\s+([\d.]+)",
                line.strip(),
            )
            if m:
                ep = int(m.group(1))
                epochs.append(ep)
                losses.append(float(m.group(2)))
                val_epochs.append(ep)
                val_cd.append(float(m.group(3)))
                val_mh.append(float(m.group(4)))
                continue

            # Loss-only line: "Ep   N | loss X.XXXX | val skip ..."
            m2 = re.match(
                r"Ep\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+val skip",
                line.strip(),
            )
            if m2:
                epochs.append(int(m2.group(1)))
                losses.append(float(m2.group(2)))

    return epochs, losses, val_epochs, val_cd, val_mh


def fig_baseline_training():
    log = "/git/mmDar/logs/auto_launch.log"
    epochs, losses, val_epochs, val_cd, val_mh = parse_baseline_log(log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_2COL, 2.0))

    # Left: training loss
    ax1.plot(epochs, losses, color="#1f77b4", lw=1.4, label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE+Dice Loss")
    ax1.set_title("(a) Training Loss")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax1.grid(True, lw=0.4, alpha=0.5)

    # Right: validation metrics
    ax2.plot(val_epochs, val_cd, color="#2ca02c", marker="o", ms=3, lw=1.2, label="Val Chamfer")
    ax2.plot(val_epochs, val_mh, color="#d62728", marker="s", ms=3, lw=1.2, label="Val mod-H")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Distance (m)")
    ax2.set_title("(b) Validation Metrics")
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax2.grid(True, lw=0.4, alpha=0.5)

    fig.suptitle("Baseline U-Net Training (batch=12, lr=7e-5, fp32)", fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_baseline_training.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Gaussian Model Training Curves (Exp 1)
# ─────────────────────────────────────────────────────────────────────────────
def parse_gaussian_log(path):
    """Parse EXP 1 block from sweep_output.txt."""
    epochs, losses, nlls = [], [], []
    val_epochs, val_mh = [], []
    in_exp1 = False

    with open(path) as f:
        for line in f:
            if "EXP 1: Relax sigma_r_prior" in line:
                in_exp1 = True
                continue
            if in_exp1 and line.startswith("==="):
                if "TEST EVALUATION" in line:
                    break
            if not in_exp1:
                continue

            # Full epoch line with val
            m = re.match(
                r"Ep\s+(\d+)\s+\|\s+loss\s+([-\d.]+)\s+\|"
                r"\s+nll=([-\d.]+)\s+existence=[-\d.]+\s+coverage=[-\d.]+\s+"
                r"cardinality=[-\d.]+\s+repulsion=[-\d.]+\s+sigma_prior=[-\d.]+\s+"
                r"huber_range=[-\d.]+\s+\|\s+val_mh_traj\s+([\d.]+)",
                line.strip(),
            )
            if m:
                ep = int(m.group(1))
                epochs.append(ep)
                losses.append(float(m.group(2)))
                nlls.append(float(m.group(3)))
                val_epochs.append(ep)
                val_mh.append(float(m.group(4)))
                continue

            # Epoch line without val
            m2 = re.match(
                r"Ep\s+(\d+)\s+\|\s+loss\s+([-\d.]+)\s+\|"
                r"\s+nll=([-\d.]+)\s+existence=[-\d.]+\s+coverage=[-\d.]+\s+"
                r"cardinality=[-\d.]+\s+repulsion=[-\d.]+\s+sigma_prior=[-\d.]+\s+"
                r"huber_range=[-\d.]+\s+\|\s+val skip",
                line.strip(),
            )
            if m2:
                epochs.append(int(m2.group(1)))
                losses.append(float(m2.group(2)))
                nlls.append(float(m2.group(3)))

    return epochs, losses, nlls, val_epochs, val_mh


def fig_gaussian_training():
    log = "/git/mmDar/logs/sweep_output.txt"
    epochs, losses, nlls, val_epochs, val_mh = parse_gaussian_log(log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_2COL, 2.0))

    # Left: loss + NLL component
    ax1.plot(epochs, losses, color="#1f77b4", lw=1.4, label="Total loss")
    ax1.plot(epochs, nlls, color="#ff7f0e", lw=1.1, ls="--", label="NLL component")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("(a) Training Loss Components")
    ax1.legend(loc="lower right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax1.grid(True, lw=0.4, alpha=0.5)

    # Right: validation mod-H
    ax2.plot(val_epochs, val_mh, color="#d62728", marker="o", ms=3, lw=1.2,
             label="Val mod-H (traj median)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mod-Hausdorff (m)")
    ax2.set_title("(b) Validation mod-H")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax2.grid(True, lw=0.4, alpha=0.5)

    fig.suptitle(r"Physics Gaussian Model — Exp 1 ($\sigma_r$=0.3, Huber=0.0)", fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_gaussian_training.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Main Results Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_main_comparison():
    labels = [
        "Honest Baseline\n(UNet, 17.5M)",
        "ConvLSTM\n(27.5M)",
        "Physics Gaussian\n(3.1M, val-sel.)",
        "Published\nRadarHD",
    ]
    chamfer = [0.406, 0.603, 0.318, 0.36]
    mod_h   = [0.296, 0.467, 0.230, 0.24]

    # Fair comparison: all four for Chamfer; for mod-H the Gaussian is val-selected
    # Highlight differences with hatching for unfair comparisons
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(IEEE_2COL, 2.6))

    colors_cd  = ["#4878cf", "#4878cf", "#6acc65", "#d65f5f"]
    colors_mh  = ["#4878cf", "#4878cf", "#6acc65", "#d65f5f"]
    # "honest" bars: solid; "test-selected" / published: hatched
    hatches_cd = ["", "", "", "//"]
    hatches_mh = ["", "", "", "//"]

    bars1 = ax.bar(x - width / 2, chamfer, width, label="Chamfer (m)",
                   color=colors_cd, hatch=[h + "" for h in hatches_cd],
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, mod_h, width, label="mod-H (m)",
                   color=colors_mh, hatch=[h + ".." for h in hatches_mh],
                   edgecolor="white", linewidth=0.5, alpha=0.85)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=6)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Distance (m)")
    ax.set_title("Model Comparison — Test Set (lower is better)", fontsize=9)
    ax.set_ylim(0, 0.75)

    # Legend — manual with color patches
    patch_cd = mpatches.Patch(color="#4878cf", label="Chamfer (m)")
    patch_mh = mpatches.Patch(facecolor="#6acc65", alpha=0.85, label="mod-H (m)")
    hatch_ref = mpatches.Patch(facecolor="white", hatch="//", edgecolor="gray",
                                label="Published (ref.)")
    ax.legend(handles=[patch_cd, patch_mh, hatch_ref], loc="upper left", fontsize=7)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    # Annotate honest vs published note
    ax.text(0.99, 0.97, "All results val-selected except Published RadarHD",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
            style="italic", color="gray")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_main_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Checkpoint Selection Bias
# ─────────────────────────────────────────────────────────────────────────────
def fig_checkpoint_bias():
    epochs      = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    chamfer_med = [0.4451, 0.3720, 0.4651, 0.3837, 0.4546, 0.4419, 0.3946, 0.3778, 0.4531, 0.4045]
    mod_h_med   = [0.2959, 0.2277, 0.2837, 0.2709, 0.2965, 0.3641, 0.2678, 0.2678, 0.2576, 0.2995]

    test_sel_chamfer  = 0.295
    honest_chamfer    = 0.406
    test_sel_mh       = min(mod_h_med)   # 0.2277 at ep20 — approximate "test-selected"
    honest_mh         = 0.296

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_2COL, 2.2))

    # Left: Chamfer
    ax1.plot(epochs, chamfer_med, color="#1f77b4", marker="o", ms=4, lw=1.3,
             label="Test Chamfer (per-epoch)")
    ax1.axhline(test_sel_chamfer, color="#d62728", ls="--", lw=1.1,
                label=f"Test-selected best ({test_sel_chamfer:.3f})")
    ax1.axhline(honest_chamfer, color="#2ca02c", ls="-.", lw=1.1,
                label=f"Honest val-selected ({honest_chamfer:.3f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Chamfer Distance (m)")
    ax1.set_title("(a) Chamfer vs Epoch")
    ax1.legend(loc="upper right", fontsize=6)
    ax1.set_ylim(0.25, 0.55)
    ax1.grid(True, lw=0.4, alpha=0.5)

    # Right: mod-H
    ax2.plot(epochs, mod_h_med, color="#ff7f0e", marker="s", ms=4, lw=1.3,
             label="Test mod-H (per-epoch)")
    ax2.axhline(test_sel_mh, color="#d62728", ls="--", lw=1.1,
                label=f"Test-selected best ({test_sel_mh:.3f})")
    ax2.axhline(honest_mh, color="#2ca02c", ls="-.", lw=1.1,
                label=f"Honest val-selected ({honest_mh:.3f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mod-Hausdorff (m)")
    ax2.set_title("(b) mod-H vs Epoch")
    ax2.legend(loc="upper right", fontsize=6)
    ax2.set_ylim(0.15, 0.45)
    ax2.grid(True, lw=0.4, alpha=0.5)

    fig.suptitle("Checkpoint Selection Bias: Test-Peek vs. Honest Val-Selected",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_checkpoint_bias.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Loss Tuning Ablation
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation():
    configs = [
        "Pre-sweep\n(σ=0.1,H=0.0)",
        "Exp1\n(σ=0.3,H=0.0)",
        "Exp2\n(FPS protos)",
        "Exp3\n(σ=0.3,H=0.05)",
        "Exp4 [BEST]\n(σ=0.3,H=0.1)",
        "Exp5\n(σ=0.3,H=0.2)",
        "Exp6\n(σ=0.5,H=0.05)",
    ]
    mod_h_vals = [0.219, 0.213, 0.324, 0.210, 0.205, 0.225, 0.215]
    is_best = [False, False, False, False, True, False, False]
    is_bad  = [False, False, True,  False, False, False, False]

    colors = []
    for best, bad in zip(is_best, is_bad):
        if best:
            colors.append("#2ca02c")
        elif bad:
            colors.append("#d62728")
        else:
            colors.append("#4878cf")

    x = np.arange(len(configs))

    fig, ax = plt.subplots(figsize=(IEEE_2COL, 2.4))
    bars = ax.bar(x, mod_h_vals, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    # Value labels
    for bar, val, best in zip(bars, mod_h_vals, is_best):
        weight = "bold" if best else "normal"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight=weight)

    # Best annotation arrow
    best_idx = mod_h_vals.index(min(mod_h_vals))
    ax.annotate("Best", xy=(x[best_idx], mod_h_vals[best_idx]),
                xytext=(x[best_idx] + 0.6, mod_h_vals[best_idx] + 0.02),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0),
                fontsize=7, color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=6.5)
    ax.set_ylabel("mod-Hausdorff (m, val-selected)")
    ax.set_title("Loss Tuning Ablation — Physics Gaussian Model", fontsize=9)
    ax.set_ylim(0.17, 0.37)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    # Legend
    patch_best = mpatches.Patch(color="#2ca02c", label="Best config")
    patch_reg  = mpatches.Patch(color="#4878cf", label="Standard runs")
    patch_bad  = mpatches.Patch(color="#d62728", label="Degraded (FPS)")
    ax.legend(handles=[patch_best, patch_reg, patch_bad], loc="upper left", fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_ablation.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Error vs Range Decomposition
# ─────────────────────────────────────────────────────────────────────────────
def fig_error_range():
    bins          = ["0–2m", "2–4m", "4–6m", "6–8m", "8–11m"]
    range_err     = [0.1724, 0.1891, 0.2885, 0.4948, 0.2659]
    angular_err   = [0.1141, 0.2007, 0.2419, 0.2795, 0.2509]
    total_err     = [0.2268, 0.3097, 0.4148, 0.6561, 0.4233]
    n_points      = [14003, 6724, 2784, 870, 311]

    x = np.arange(len(bins))
    width = 0.55

    fig, ax = plt.subplots(figsize=(IEEE_1COL * 1.5, 2.5))

    bar_r = ax.bar(x, range_err, width, label="Range error", color="#4878cf", edgecolor="white")
    bar_a = ax.bar(x, angular_err, width, bottom=range_err, label="Angular error",
                   color="#e8a838", edgecolor="white")

    # Total error line
    ax.plot(x, total_err, color="#d62728", marker="D", ms=4, lw=1.2, zorder=5,
            label="Total error")

    # Annotate N (sample counts)
    for i, (xi, n) in enumerate(zip(x, n_points)):
        ax.text(xi, total_err[i] + 0.02, f"n={n:,}", ha="center", va="bottom",
                fontsize=6, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Range Bin")
    ax.set_ylabel("Mean Nearest-Neighbour Error (m)")
    ax.set_title("Error Decomposition vs. Range\n(Exp 1, Gaussian Model, pred→GT)", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0, 0.82)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_error_range.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures ...")
    fig_baseline_training()
    fig_gaussian_training()
    fig_main_comparison()
    fig_checkpoint_bias()
    fig_ablation()
    fig_error_range()
    print("Done. All PDFs written to:", OUT_DIR)
