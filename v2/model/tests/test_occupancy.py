import torch
import pytest


def test_channelizer_output_shape():
    from v2.model.occupancy import Channelizer
    ch = Channelizer()
    x = torch.randn(2, 256, 512, dtype=torch.complex64)
    out = ch(x)
    assert out.shape == (2, 3, 256, 512), f"Expected (2,3,256,512), got {out.shape}"
    assert out.dtype == torch.float32


def test_channelizer_preserves_info():
    from v2.model.occupancy import Channelizer
    ch = Channelizer()
    x = torch.randn(2, 256, 512, dtype=torch.complex64)
    out = ch(x)
    assert out.shape == (2, 3, 256, 512)
    assert torch.isfinite(out).all(), "Output should be all finite"
    assert out.mean().abs() < 0.5, f"Mean should be near zero after norm: {out.mean()}"


def test_dilated_res_head_output_shape():
    from v2.model.occupancy import DilatedResHead
    head = DilatedResHead(in_ch=3, mid_ch=32, n_blocks=3)
    x = torch.randn(2, 3, 256, 512)
    out = head(x)
    assert out.shape == (2, 1, 256, 512), f"Expected (2,1,256,512), got {out.shape}"


def test_occupancy_model_end_to_end():
    from v2.model.occupancy import OccupancyModel
    model = OccupancyModel(beamformer="fft")
    x = torch.randn(2, 8, 512, dtype=torch.complex64)
    logits = model(x)
    assert logits.shape == (2, 1, 256, 512)
    assert logits.dtype == torch.float32


def test_occupancy_model_param_count():
    from v2.model.occupancy import OccupancyModel
    model = OccupancyModel(beamformer="fft", mid_ch=32, n_blocks=4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 1_000_000, f"Model too large: {n_params} params (budget: <1M)"
    print(f"Occupancy model params: {n_params:,}")


def test_occupancy_model_lista():
    from v2.model.occupancy import OccupancyModel
    model = OccupancyModel(beamformer="lista", K=3)
    x = torch.randn(2, 8, 512, dtype=torch.complex64)
    logits = model(x)
    assert logits.shape == (2, 1, 256, 512)
