"""Deep dive into error sources: what's the model getting wrong and why?

Analyzes:
1. Gaussian spread — are predicted Gaussians too wide/narrow?
2. Coverage gaps — which GT regions have no nearby predictions?
3. Hallucinations — where does the model predict points with no GT?
4. Range bias — does the model systematically over/under-estimate range?
5. Angular bias — systematic azimuth errors?
6. Per-trajectory breakdown — are some trajectories much worse?

Usage:
  docker compose run --rm mmdar python3 eval/error_deep_dive.py \
    --checkpoint logs/verify_cleanup/best.pt
"""

import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.physics_frontend import PhysicsGaussianModel
from train.train import AugmentedGaussianDataset, get_split


def analyze_frame(model, radar, lidar, device, threshold=0.3):
    """Analyze one frame in detail."""
    radar = radar.to(device)
    gt_xy = lidar[0, :, :2].to(device)

    # Get raw Gaussian output (not just points)
    out = model(radar)
    mu_xy = out['mu_xy'][0]         # (K, 2)
    sigma_r = out['sigma_r'][0]     # (K,)
    sigma_perp = out['sigma_perp'][0]  # (K,)
    existence = out['existence'][0]  # (K,)

    prob = torch.sigmoid(existence)  # (K,)

    # Filter by threshold
    keep = prob > threshold
    mu_keep = mu_xy[keep]          # (N, 2)
    sigma_r_keep = sigma_r[keep]   # (N,)
    sigma_p_keep = sigma_perp[keep]  # (N,)
    prob_keep = prob[keep]          # (N,)

    if mu_keep.shape[0] < 2:
        return None

    # Convert to polar for analysis
    pred_r = torch.norm(mu_keep, dim=1)
    pred_theta = torch.atan2(mu_keep[:, 1], mu_keep[:, 0]) * 180 / np.pi
    gt_r = torch.norm(gt_xy, dim=1)
    gt_theta = torch.atan2(gt_xy[:, 1], gt_xy[:, 0]) * 180 / np.pi

    # NN matching: pred -> GT
    D = torch.cdist(mu_keep, gt_xy)  # (N, M)
    nn_dist_p2g = D.min(dim=1).values
    nn_idx_p2g = D.argmin(dim=1)

    # NN matching: GT -> pred
    nn_dist_g2p = D.min(dim=0).values
    nn_idx_g2p = D.argmin(dim=0)

    # Range and angle errors (signed) for pred -> GT
    nn_gt = gt_xy[nn_idx_p2g]
    dr = pred_r - torch.norm(nn_gt, dim=1)  # signed range error
    dtheta = pred_theta - torch.atan2(nn_gt[:, 1], nn_gt[:, 0]) * 180 / np.pi

    # Wrap angle difference to [-180, 180]
    dtheta = (dtheta + 180) % 360 - 180

    return {
        'n_pred': int(mu_keep.shape[0]),
        'n_gt': int(gt_xy.shape[0]),
        # Predicted Gaussian stats
        'sigma_range_mean': float(sigma_r_keep.mean()),
        'sigma_xrange_mean': float(sigma_p_keep.mean()),
        'prob_mean': float(prob_keep.mean()),
        # Signed errors (pred -> GT)
        'range_bias': float(dr.mean()),
        'range_bias_std': float(dr.std()),
        'angle_bias_deg': float(dtheta.mean()),
        'angle_bias_std_deg': float(dtheta.std()),
        # NN distances
        'p2g_mean': float(nn_dist_p2g.mean()),
        'p2g_median': float(nn_dist_p2g.median()),
        'g2p_mean': float(nn_dist_g2p.mean()),
        'g2p_median': float(nn_dist_g2p.median()),
        # Coverage: fraction of GT points with a pred within 0.1m, 0.2m, 0.5m
        'coverage_0.1': float((nn_dist_g2p < 0.1).float().mean()),
        'coverage_0.2': float((nn_dist_g2p < 0.2).float().mean()),
        'coverage_0.5': float((nn_dist_g2p < 0.5).float().mean()),
        # Hallucinations: pred points with no GT within 0.5m, 1.0m
        'halluc_0.5': float((nn_dist_p2g > 0.5).float().mean()),
        'halluc_1.0': float((nn_dist_p2g > 1.0).float().mean()),
        # Range-binned errors
        'pred_ranges': pred_r.cpu(),
        'signed_range_err': dr.cpu(),
        'signed_angle_err': dtheta.cpu(),
        'nn_dist_p2g': nn_dist_p2g.cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split", default="mixed")
    parser.add_argument("--window-size", type=int, default=41)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = PhysicsGaussianModel(
        N_az=cfg.get("N_az", 64), T=cfg.get("window_size", args.window_size),
        K=cfg.get("K", 96),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch', '?')}", flush=True)

    _, _, TEST_TRAJS, _ = get_split(args.split)

    # Per-trajectory and aggregate analysis
    traj_results = {}
    all_results = []
    all_ranges = []
    all_range_err = []
    all_angle_err = []
    all_nn_dist = []

    with torch.no_grad():
        for tid in TEST_TRAJS:
            proto_path = os.path.join(args.processed_dir, f"proto_{tid}.pt")
            if not os.path.exists(proto_path):
                continue
            ds = AugmentedGaussianDataset(tid, args.processed_dir,
                                           args.window_size, augment=False)
            loader = DataLoader(ds, batch_size=1, shuffle=False)

            traj_frames = []
            for radar, lidar, protos in loader:
                if len(all_results) >= args.max_frames:
                    break
                r = analyze_frame(model, radar, lidar, device, args.threshold)
                if r is None:
                    continue
                traj_frames.append(r)
                all_results.append(r)
                all_ranges.append(r['pred_ranges'])
                all_range_err.append(r['signed_range_err'])
                all_angle_err.append(r['signed_angle_err'])
                all_nn_dist.append(r['nn_dist_p2g'])

            if traj_frames:
                traj_results[tid] = traj_frames

            if len(all_results) >= args.max_frames:
                break

    n = len(all_results)
    print(f"\nAnalyzed {n} frames across {len(traj_results)} trajectories\n")

    # Aggregate stats
    print("=" * 60)
    print("GAUSSIAN OUTPUT STATS")
    print("=" * 60)
    print(f"  Mean predicted Gaussians/frame: {np.mean([r['n_pred'] for r in all_results]):.1f}")
    print(f"  Mean sigma_range:  {np.mean([r['sigma_range_mean'] for r in all_results]):.4f}m")
    print(f"  Mean sigma_xrange: {np.mean([r['sigma_xrange_mean'] for r in all_results]):.4f}m")
    print(f"  Mean confidence:   {np.mean([r['prob_mean'] for r in all_results]):.3f}")

    print(f"\n{'=' * 60}")
    print("SYSTEMATIC BIASES")
    print("=" * 60)
    print(f"  Range bias (signed):  {np.mean([r['range_bias'] for r in all_results]):+.4f}m "
          f"(std {np.mean([r['range_bias_std'] for r in all_results]):.4f})")
    print(f"  Angle bias (signed):  {np.mean([r['angle_bias_deg'] for r in all_results]):+.2f}° "
          f"(std {np.mean([r['angle_bias_std_deg'] for r in all_results]):.2f}°)")

    print(f"\n{'=' * 60}")
    print("COVERAGE (GT -> pred)")
    print("=" * 60)
    print(f"  GT points within 0.1m of a pred: {np.mean([r['coverage_0.1'] for r in all_results]):.1%}")
    print(f"  GT points within 0.2m of a pred: {np.mean([r['coverage_0.2'] for r in all_results]):.1%}")
    print(f"  GT points within 0.5m of a pred: {np.mean([r['coverage_0.5'] for r in all_results]):.1%}")

    print(f"\n{'=' * 60}")
    print("HALLUCINATIONS (pred -> GT)")
    print("=" * 60)
    print(f"  Pred points >0.5m from any GT: {np.mean([r['halluc_0.5'] for r in all_results]):.1%}")
    print(f"  Pred points >1.0m from any GT: {np.mean([r['halluc_1.0'] for r in all_results]):.1%}")

    # Range-binned signed bias
    ranges = torch.cat(all_ranges)
    range_err = torch.cat(all_range_err)
    angle_err = torch.cat(all_angle_err)
    nn_dist = torch.cat(all_nn_dist)

    print(f"\n{'=' * 60}")
    print("SIGNED BIAS vs RANGE")
    print("=" * 60)
    print(f"  {'Range':8s} {'N':>7s} {'Range bias':>11s} {'Angle bias':>11s} {'NN dist':>8s}")
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 11)]:
        mask = (ranges >= lo) & (ranges < hi)
        if mask.sum() < 10:
            continue
        print(f"  {lo:2d}-{hi:2d}m   {mask.sum():7d} {range_err[mask].mean():+11.4f}m "
              f"{angle_err[mask].mean():+10.2f}° {nn_dist[mask].mean():8.4f}m")

    # Per-trajectory breakdown
    print(f"\n{'=' * 60}")
    print("PER-TRAJECTORY BREAKDOWN")
    print("=" * 60)
    print(f"  {'Traj':>6s} {'Frames':>7s} {'p2g_med':>8s} {'g2p_med':>8s} {'Cov@0.2':>8s} {'Halluc':>7s} {'R bias':>8s}")
    for tid in sorted(traj_results.keys()):
        frames = traj_results[tid]
        print(f"  {tid:6d} {len(frames):7d} "
              f"{np.mean([f['p2g_median'] for f in frames]):8.4f} "
              f"{np.mean([f['g2p_median'] for f in frames]):8.4f} "
              f"{np.mean([f['coverage_0.2'] for f in frames]):8.1%} "
              f"{np.mean([f['halluc_0.5'] for f in frames]):7.1%} "
              f"{np.mean([f['range_bias'] for f in frames]):+8.4f}")


if __name__ == "__main__":
    main()
