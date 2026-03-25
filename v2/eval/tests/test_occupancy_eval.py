import numpy as np
import torch
import pytest


def test_occupancy_to_points_broadside():
    from v2.eval.occupancy_eval import occupancy_to_points
    occ = np.zeros((256, 512), dtype=np.float32)
    r_bin = 236  # ~5.0m
    az_bin = 128  # sin(theta)~0 -> broadside
    occ[az_bin, r_bin] = 1.0
    pts = occupancy_to_points(occ, threshold=0.5, r_max=10.8)
    assert pts.shape[1] == 3
    assert len(pts) == 1
    assert abs(pts[0, 0] - 5.0) < 0.1, f"x={pts[0,0]}, expected ~5.0"
    assert abs(pts[0, 1]) < 0.1, f"y={pts[0,1]}, expected ~0.0"


def test_occupancy_to_points_empty():
    from v2.eval.occupancy_eval import occupancy_to_points
    occ = np.zeros((256, 512), dtype=np.float32)
    pts = occupancy_to_points(occ, threshold=0.5)
    assert len(pts) == 0 or pts.shape == (0, 3)


def test_evaluate_occupancy_epoch_smoke():
    from v2.eval.occupancy_eval import evaluate_occupancy_epoch

    class MockModel(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            return torch.zeros(B, 1, 256, 512)

    model = MockModel()
    radar = torch.randn(2, 8, 512, dtype=torch.complex64)
    lidar = torch.zeros(2, 8192, 3)
    lidar[0, 0] = torch.tensor([5.0, 0.0, 0.0])
    lidar[1, 0] = torch.tensor([5.0, 0.0, 0.0])
    occ = torch.zeros(2, 256, 512)
    norm = torch.ones(2)
    loader = [(radar, lidar, occ, norm)]

    metrics = evaluate_occupancy_epoch(model, loader, torch.device("cpu"))
    assert "chamfer" in metrics
    assert "mod_hausdorff" in metrics
    # Model predicts all-zero logits -> sigmoid(0)=0.5 -> all cells "occupied"
    # at threshold=0.5 (default), so predictions are not empty. Metrics should be finite.
    assert metrics["chamfer"] < 20.0, "Should not be penalty distance"


def test_empty_prediction_gets_penalty():
    """Empty predictions should receive MAX_PENALTY_DIST, not be skipped."""
    from v2.eval.occupancy_eval import evaluate_occupancy_epoch, MAX_PENALTY_DIST

    class NothingModel(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            # Very negative logits -> sigmoid ~ 0 -> no cells above threshold
            return torch.full((B, 1, 256, 512), -100.0)

    model = NothingModel()
    radar = torch.randn(1, 8, 512, dtype=torch.complex64)
    lidar = torch.zeros(1, 8192, 3)
    lidar[0, 0] = torch.tensor([5.0, 0.0, 0.0])  # valid GT point
    occ = torch.zeros(1, 256, 512)
    norm = torch.ones(1)
    loader = [(radar, lidar, occ, norm)]

    metrics = evaluate_occupancy_epoch(model, loader, torch.device("cpu"))
    assert metrics["chamfer"] == MAX_PENALTY_DIST, (
        f"Empty prediction should get penalty {MAX_PENALTY_DIST}, got {metrics['chamfer']}"
    )
