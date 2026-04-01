"""Unit tests for eval/eval_adapter.py.

Tests need only numpy and scipy — no Docker required.

Tests:
    1. chamfer_distance_np on identical point clouds returns 0.0
    2. chamfer_distance_np on known offset points returns expected distance
    3. mod_hausdorff_np on identical point clouds returns 0.0
    4. evaluate_batch returns dict with 'chamfer' and 'mod_hausdorff' keys
    5. Metrics use only XY columns (2D) — adding z offset does NOT change the metric
    6. chamfer_distance_np matches formula: 0.5*mean(nn_A->B) + 0.5*mean(nn_B->A)
"""

import numpy as np
import pytest

from eval.eval_adapter import (
    chamfer_distance_np,
    mod_hausdorff_np,
    evaluate_batch,
)


# ---------------------------------------------------------------------------
# Test 1: chamfer_distance_np on identical clouds returns 0.0
# ---------------------------------------------------------------------------

def test_chamfer_identical():
    """chamfer_distance_np returns 0.0 for identical point clouds."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((100, 3)).astype(np.float32)
    result = chamfer_distance_np(pts, pts)
    assert result == pytest.approx(0.0, abs=1e-6), \
        f"Expected 0.0 for identical clouds, got {result}"


# ---------------------------------------------------------------------------
# Test 2: chamfer_distance_np with known offset
# ---------------------------------------------------------------------------

def test_chamfer_known_offset():
    """chamfer_distance_np returns expected distance for uniformly offset clouds."""
    # Single point shifted by (1, 0, 0) -> nearest-neighbor dist = 1.0 in XY
    # chamfer = 0.5*1.0 + 0.5*1.0 = 1.0
    pred = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    gt   = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = chamfer_distance_np(pred, gt)
    assert result == pytest.approx(1.0, abs=1e-5), \
        f"Expected 1.0 for unit-offset single points, got {result}"


# ---------------------------------------------------------------------------
# Test 3: mod_hausdorff_np on identical clouds returns 0.0
# ---------------------------------------------------------------------------

def test_mod_hausdorff_identical():
    """mod_hausdorff_np returns 0.0 for identical point clouds."""
    rng = np.random.default_rng(43)
    pts = rng.standard_normal((50, 3)).astype(np.float32)
    result = mod_hausdorff_np(pts, pts)
    assert result == pytest.approx(0.0, abs=1e-6), \
        f"Expected 0.0 for identical clouds, got {result}"


# ---------------------------------------------------------------------------
# Test 4: evaluate_batch returns correct keys
# ---------------------------------------------------------------------------

def test_evaluate_batch_keys():
    """evaluate_batch returns dict with 'chamfer' and 'mod_hausdorff' keys."""
    rng = np.random.default_rng(44)
    B, N, M = 3, 64, 64
    pred = rng.standard_normal((B, N, 3)).astype(np.float32)
    gt   = rng.standard_normal((B, M, 3)).astype(np.float32)
    result = evaluate_batch(pred, gt)
    assert "chamfer" in result, "Missing 'chamfer' key in evaluate_batch output"
    assert "mod_hausdorff" in result, "Missing 'mod_hausdorff' key in evaluate_batch output"
    assert np.isfinite(result["chamfer"]), f"Non-finite chamfer: {result['chamfer']}"
    assert np.isfinite(result["mod_hausdorff"]), f"Non-finite mod_hausdorff: {result['mod_hausdorff']}"


# ---------------------------------------------------------------------------
# Test 5: metrics use only XY columns (z offset does not change metrics)
# ---------------------------------------------------------------------------

def test_metrics_xy_only():
    """Adding z offset does NOT change chamfer or mod_hausdorff — 2D XY only."""
    rng = np.random.default_rng(45)
    N = 50
    pred = rng.standard_normal((N, 3)).astype(np.float32)
    gt   = rng.standard_normal((N, 3)).astype(np.float32)

    # Compute baseline
    chamfer_base = chamfer_distance_np(pred, gt)
    mh_base = mod_hausdorff_np(pred, gt)

    # Add large z offset — should not change XY-only metrics
    pred_z = pred.copy()
    pred_z[:, 2] += 100.0  # massive z shift

    chamfer_shifted = chamfer_distance_np(pred_z, gt)
    mh_shifted = mod_hausdorff_np(pred_z, gt)

    assert chamfer_shifted == pytest.approx(chamfer_base, abs=1e-5), \
        f"chamfer changed with z offset: {chamfer_base} -> {chamfer_shifted}"
    assert mh_shifted == pytest.approx(mh_base, abs=1e-5), \
        f"mod_hausdorff changed with z offset: {mh_base} -> {mh_shifted}"


# ---------------------------------------------------------------------------
# Test 6: chamfer_distance_np matches manual formula
# ---------------------------------------------------------------------------

def test_chamfer_formula():
    """chamfer_distance_np matches 0.5*mean(nn_A->B) + 0.5*mean(nn_B->A)."""
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(46)
    pred = rng.standard_normal((30, 3)).astype(np.float32)
    gt   = rng.standard_normal((40, 3)).astype(np.float32)

    # Manual formula using XY columns only
    D = cdist(pred[:, :2], gt[:, :2])
    expected = 0.5 * D.min(axis=1).mean() + 0.5 * D.min(axis=0).mean()

    result = chamfer_distance_np(pred, gt)
    assert result == pytest.approx(float(expected), abs=1e-6), \
        f"chamfer formula mismatch: expected {expected}, got {result}"
