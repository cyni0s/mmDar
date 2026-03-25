# v2/data/tests/test_rasterize.py
import numpy as np
import torch
from v2.data.rasterize import rasterize_to_polar

def test_single_point_broadside():
    """A point at (x=5, y=0, z=0) should land at azimuth=0, range=5m."""
    pts = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    occ = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8)
    assert occ.shape == (256, 512)
    assert occ.dtype == np.float32
    az_bin = 128  # round((0+1)*255/2) = 127.5 -> 128
    r_bin = round(5.0 / 10.8 * 511)  # ~236
    assert occ[az_bin, r_bin] > 0, f"Expected occupied at ({az_bin}, {r_bin})"
    assert occ.sum() > 0 and occ.sum() < 10, "Should have ~1 occupied cell"

def test_point_at_45deg():
    """A point at 45deg azimuth, range=7m."""
    theta = np.radians(45)
    r = 7.0
    pts = np.array([[r * np.cos(theta), r * np.sin(theta), 0.0]], dtype=np.float32)
    occ = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8)
    sin_val = np.sin(theta)
    expected_az = round((sin_val + 1.0) * 255 / 2.0)
    expected_r = round(r / 10.8 * 511)
    assert occ[expected_az, expected_r] > 0

def test_empty_cloud():
    pts = np.zeros((0, 3), dtype=np.float32)
    occ = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8)
    assert occ.sum() == 0

def test_out_of_range_filtered():
    pts = np.array([
        [15.0, 0.0, 0.0],   # beyond r_max=10.8
        [-1.0, 0.0, 0.0],   # behind sensor
        [5.0, 0.0, 0.0],    # valid
    ], dtype=np.float32)
    occ = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8)
    assert occ.sum() > 0 and occ.sum() < 5

def test_gaussian_softening():
    pts = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    occ_hard = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8, sigma=0)
    occ_soft = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8, sigma=1.0)
    assert occ_soft.sum() > occ_hard.sum(), "Soft labels should spread"
    assert occ_soft.max() <= 1.0
