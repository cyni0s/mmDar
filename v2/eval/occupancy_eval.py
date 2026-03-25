"""Polar occupancy evaluation: convert occupancy maps to point clouds,
compare against original lidar for Chamfer/mod-H metrics.

Uses LISTA's angular grid: sin_theta[k] = -1 + 2*k/(N_az-1)
"""
import numpy as np
import torch
from v2.eval.eval_adapter import chamfer_distance_np, mod_hausdorff_np

MAX_PENALTY_DIST = 20.0  # penalty for empty predictions


def occupancy_to_points(occ, threshold=0.5, r_max=10.8):
    """Convert polar occupancy grid to XYZ point cloud.

    Args:
        occ: (N_az, N_r) float32 occupancy in [0,1]
        threshold: detection threshold
        r_max: max range meters
    Returns:
        (N_pts, 3) float32 [x, y, z]
    """
    N_az, N_r = occ.shape
    az_bins, r_bins = np.where(occ > threshold)
    if len(az_bins) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    sin_theta = -1.0 + 2.0 * az_bins / (N_az - 1)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    cos_theta = np.sqrt(1.0 - sin_theta**2)
    r = r_bins * r_max / (N_r - 1)
    x = r * cos_theta
    y = r * sin_theta
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def evaluate_occupancy_epoch(model, dataloader, device, threshold=0.5, r_max=10.8):
    """Eval occupancy model against ORIGINAL lidar points.

    Dataloader yields (radar, lidar_pts, occ_label, norm) 4-tuples.
    Predictions compared against lidar_pts, NOT occ_label.
    Empty predictions get MAX_PENALTY_DIST penalty.
    """
    model.eval()
    chamfer_sum = 0.0
    hausdorff_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            radar, lidar_gt, _occ_label, _norm = batch
            radar = radar.to(device)
            logits = model(radar)
            pred_occ = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            gt_pts_batch = lidar_gt.numpy()

            for i in range(pred_occ.shape[0]):
                pred_pts = occupancy_to_points(pred_occ[i], threshold, r_max)
                gt_pts = gt_pts_batch[i]

                # Filter GT to scene volume
                r_gt = np.sqrt(gt_pts[:, 0]**2 + gt_pts[:, 1]**2)
                valid = (gt_pts[:, 0] > 0) & (r_gt <= r_max) & (r_gt > 0.01)
                gt_pts = gt_pts[valid]

                if len(gt_pts) == 0:
                    continue  # no valid GT

                if len(pred_pts) == 0:
                    chamfer_sum += MAX_PENALTY_DIST
                    hausdorff_sum += MAX_PENALTY_DIST
                else:
                    chamfer_sum += chamfer_distance_np(pred_pts, gt_pts)
                    hausdorff_sum += mod_hausdorff_np(pred_pts, gt_pts)
                n_samples += 1

    if n_samples == 0:
        return {"chamfer": float("nan"), "mod_hausdorff": float("nan"), "n_samples": 0}

    return {
        "chamfer": chamfer_sum / n_samples,
        "mod_hausdorff": hausdorff_sum / n_samples,
        "n_samples": n_samples,
    }
