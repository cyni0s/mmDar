# v2/eval/occupancy_to_pc.py
"""Convert polar occupancy predictions to point clouds and evaluate.

Uses eval/eval_pointcloud.py for the polar → Cartesian conversion and metrics.
Threshold → polar_image_to_pointcloud(legacy_cartesian) → Chamfer/mod-H.
"""
import numpy as np
import torch
from eval.eval_pointcloud import (
    polar_image_to_pointcloud, COORD_MODE_LEGACY,
    chamfer_distance, modified_hausdorff,
)


def occupancy_to_pointcloud(occ: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert (256, 512) occupancy probability to (N, 2) point cloud.

    The input is on the (range, angle-uniform) polar grid which matches
    eval_pointcloud.py's legacy_cartesian mode convention.

    Args:
        occ: (256, 512) float, values in [0, 1]
        threshold: binarization threshold

    Returns:
        (N, 2) float64 point cloud in meters (x, y)
    """
    binary = (occ >= threshold).astype(np.uint8) * 255
    return polar_image_to_pointcloud(binary, threshold=1,
                                     coordinate_mode=COORD_MODE_LEGACY)


def evaluate_occupancy_model(model, dataloader, device,
                             threshold: float = 0.5) -> dict:
    """Run occupancy model inference and compute point cloud metrics.

    Args:
        model: U-Net occupancy model, output (B, 1, 256, 512) sigmoid
        dataloader: yields (features, labels) batches
        device: torch device
        threshold: occupancy binarization threshold

    Returns:
        dict with chamfer_mean, mod_h_mean, n_samples
    """
    model.eval()
    chamfer_accum = 0.0
    hausdorff_accum = 0.0
    n_samples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            pred = model(features.to(device))
            pred_np = pred.squeeze(1).cpu().numpy()
            label_np = labels.squeeze(1).cpu().numpy()

            for i in range(pred_np.shape[0]):
                pc_pred = occupancy_to_pointcloud(pred_np[i], threshold)
                pc_label = occupancy_to_pointcloud(label_np[i], 0.5)

                if pc_pred.shape[0] == 0 or pc_label.shape[0] == 0:
                    continue

                chamfer_accum += chamfer_distance(pc_pred, pc_label)
                hausdorff_accum += modified_hausdorff(pc_pred, pc_label)
                n_samples += 1

    if n_samples == 0:
        return {'chamfer_mean': float('nan'), 'mod_h_mean': float('nan'),
                'n_samples': 0}

    return {
        'chamfer_mean': chamfer_accum / n_samples,
        'mod_h_mean': hausdorff_accum / n_samples,
        'n_samples': n_samples,
    }
