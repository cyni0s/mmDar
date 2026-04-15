"""Generate 4-panel comparison: Radar Input | Baseline U-Net | Gaussian (Ours) | Lidar GT.

Runs inside Docker:
  docker compose run --rm mmdar python3 poster/gen_comparison.py

Outputs: poster/assets/comparison_panels.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
TRAJ = 117
FRAME = 1000
CHECKPOINT = "logs/v2_physics_gaussian/best.pt"
PROCESSED_DIR = "data/processed"
RADAR_PNG = f"dataset_5/test/radar/R_{TRAJ}_{FRAME}.png"
LIDAR_PNG = f"dataset_5/test/lidar/L_{TRAJ}_{FRAME}.png"
BASELINE_PLOT = f"results/baseline_5090_adapted/plots/best_{TRAJ}_{FRAME}.png"
OUT = "poster/assets/comparison_panels.png"

NAVY = "#03244d"
ORANGE = "#DD550C"

# ── Load radar & lidar PNGs ──────────────────────────────────────────────────
radar_img = np.array(Image.open(RADAR_PNG).convert("L")).astype(np.float32)
lidar_img = np.array(Image.open(LIDAR_PNG).convert("L")).astype(np.float32)

# Enhance radar: log scale + viridis colormap to make returns visible
radar_enhanced = np.log1p(radar_img)

from matplotlib.patches import Ellipse

# ── Run Gaussian model inference ─────────────────────────────────────────────
from model.physics_frontend import PhysicsGaussianModel
from train.train import GaussianDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = PhysicsGaussianModel(N_az=64, T=41, K=96).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load the specific frame from processed data
radar_pt = torch.load(os.path.join(PROCESSED_DIR, f"radar_{TRAJ}.pt"), map_location="cpu")
lidar_pt = torch.load(os.path.join(PROCESSED_DIR, f"lidar_{TRAJ}.pt"), map_location="cpu")

# Find the frame index
# Frames are stored sequentially for each trajectory
# Frame 1000 offset depends on trajectory start
frame_table_path = os.path.join(PROCESSED_DIR, "frame_table.json")
if os.path.exists(frame_table_path):
    import json
    with open(frame_table_path) as f:
        frame_table = json.load(f)
    # Find the local index for this frame
    traj_key = str(TRAJ)
    if traj_key in frame_table:
        frames = frame_table[traj_key]
        if FRAME in frames:
            local_idx = frames.index(FRAME)
        else:
            local_idx = FRAME  # fallback
    else:
        local_idx = FRAME
else:
    local_idx = FRAME

# Build window of 41 frames centered on target
W = 41
n_frames = radar_pt.shape[0]
center = min(local_idx, n_frames - 1)
start = max(0, center - W // 2)
end = min(n_frames, start + W)
start = max(0, end - W)  # adjust if near end
indices = list(range(start, end))
if len(indices) < W:
    # Pad by repeating last frame
    indices = indices + [indices[-1]] * (W - len(indices))

radar_window = radar_pt[indices]  # (41, 8, 512) complex
radar_batch = radar_window.unsqueeze(0).to(device)  # (1, 41, 8, 512)

with torch.no_grad():
    out = model(radar_batch)  # dict with 'mu_xy', 'mu_r', 'mu_phi', 'sigma_r', 'sigma_perp', 'existence'
    gaussian_pts_list = model.predict_points(radar_batch, threshold=0.3)
    gaussian_pts = gaussian_pts_list[0].cpu().numpy()  # (N, 2) — x, y
    existence_prob = torch.sigmoid(out['existence'][0]).cpu().numpy()  # (K,)
    mu_xy = out['mu_xy'][0].cpu().numpy()  # (K, 2)
    mu_r = out['mu_r'][0].cpu().numpy()  # (K,)
    mu_phi = out['mu_phi'][0].cpu().numpy()  # (K,)
    sigma_r = out['sigma_r'][0].cpu().numpy()  # (K,)
    sigma_perp = out['sigma_perp'][0].cpu().numpy()  # (K,)
    n_above = (existence_prob > 0.3).sum()

# Also get lidar GT points for this frame
lidar_gt = lidar_pt[center].cpu().numpy()  # (M, 3) or (M, 2)

# ── Auto-range from lidar GT ─────────────────────────────────────────────────
if len(lidar_gt) > 0:
    xmin, xmax = lidar_gt[:, 0].min() - 1, lidar_gt[:, 0].max() + 1
    ymin, ymax = lidar_gt[:, 1].min() - 1, lidar_gt[:, 1].max() + 1
else:
    xmin, xmax, ymin, ymax = -12, 12, 0, 12

# ── Create 4-panel figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 6), facecolor="white")

# Panel 1: Radar Input (enhanced with colormap)
ax = axes[0]
ax.imshow(radar_enhanced, cmap="inferno", aspect="auto", vmin=0, vmax=np.percentile(radar_enhanced, 99))
ax.set_title("Radar Input\n(FFT Beamformed)", fontsize=16, fontweight="bold", color=NAVY)
ax.set_xlabel("Range Bin", fontsize=12)
ax.set_ylabel("Azimuth Bin", fontsize=12)
ax.tick_params(labelsize=10)

# Panel 2: Gaussian Ellipses (showing uncertainty before collapsing to points)
ax = axes[1]
ax.set_facecolor("black")
mask = existence_prob > 0.3
for i in range(len(mu_xy)):
    if not mask[i]:
        continue
    x, y = mu_xy[i]
    # Ellipse axes: sigma_perp is perpendicular to radial direction, sigma_r is radial
    # Orientation: angle from mu_phi (radial direction)
    angle_deg = np.degrees(mu_phi[i])
    # 2-sigma ellipse (95% confidence)
    w = 2 * sigma_perp[i] * 2  # width = perpendicular axis (2-sigma)
    h = 2 * sigma_r[i] * 2     # height = radial axis (2-sigma)
    alpha = min(1.0, existence_prob[i])
    e = Ellipse((x, y), width=w, height=h, angle=angle_deg,
                facecolor=ORANGE, edgecolor="white", linewidth=0.5,
                alpha=alpha * 0.6)
    ax.add_patch(e)
    # Small dot at center
    ax.plot(x, y, '.', color="white", markersize=2, alpha=alpha)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_title(f"Gaussian Ellipses\n(2σ confidence, {n_above} active)", fontsize=16, fontweight="bold", color=ORANGE)
ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.tick_params(labelsize=10, colors="#999")
ax.spines[:].set_color("#333")

# Panel 3: Gaussian Points (centers only, after existence filtering)
ax = axes[2]
if len(gaussian_pts) > 0:
    ax.scatter(gaussian_pts[:, 0], gaussian_pts[:, 1], s=8, c=ORANGE, alpha=0.9)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_facecolor("black")
ax.set_title(f"Gaussian Centers\n(3.1M params, {n_above} points)", fontsize=16, fontweight="bold", color=ORANGE)
ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.tick_params(labelsize=10, colors="#999")
ax.spines[:].set_color("#333")

# Panel 4: Lidar Ground Truth — scatter plot
ax = axes[3]
if len(lidar_gt) > 0:
    ax.scatter(lidar_gt[:, 0], lidar_gt[:, 1], s=2, c="#2ca02c", alpha=0.6)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_facecolor("black")
ax.set_title(f"Lidar Ground Truth\n({len(lidar_gt)} points)", fontsize=16, fontweight="bold", color="#2ca02c")
ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.tick_params(labelsize=10, colors="#999")
ax.spines[:].set_color("#333")

plt.tight_layout(pad=1.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
print(f"Gaussian points: {n_above}/96 (threshold=0.3)")
