"""Unit tests for ConvLSTMCell and UNet1ConvLSTM in model.py."""
import pytest
import torch
import torch.nn as nn

from train_test_utils.model import ConvLSTMCell, UNet1ConvLSTM


# ---------- ConvLSTMCell tests ----------

def test_convlstm_cell_output_shapes():
    """ConvLSTMCell forward returns h_new, c_new with same shape as inputs."""
    cell = ConvLSTMCell(256, 256, kernel_size=3)
    x = torch.randn(2, 256, 16, 4)
    h = torch.zeros(2, 256, 16, 4)
    c = torch.zeros(2, 256, 16, 4)
    h_new, c_new = cell(x, h, c)
    assert h_new.shape == (2, 256, 16, 4), f"Expected h_new (2,256,16,4), got {h_new.shape}"
    assert c_new.shape == (2, 256, 16, 4), f"Expected c_new (2,256,16,4), got {c_new.shape}"


def test_convlstm_cell_output_dtype_float32():
    """ConvLSTMCell must return h, c in float32 regardless of input dtype."""
    cell = ConvLSTMCell(256, 256, kernel_size=3)
    x = torch.randn(2, 256, 16, 4)
    h = torch.zeros(2, 256, 16, 4)
    c = torch.zeros(2, 256, 16, 4)
    h_new, c_new = cell(x, h, c)
    assert h_new.dtype == torch.float32, f"Expected float32 h_new, got {h_new.dtype}"
    assert c_new.dtype == torch.float32, f"Expected float32 c_new, got {c_new.dtype}"


def test_convlstm_cell_forget_gate_bias_initialized_to_1():
    """Forget gate bias slice [hidden:2*hidden] must be 1.0 after init."""
    hidden = 256
    cell = ConvLSTMCell(hidden, hidden, kernel_size=3)
    bias = cell.conv.bias.data
    forget_slice = bias[hidden:2 * hidden]
    assert torch.allclose(forget_slice, torch.ones_like(forget_slice)), \
        f"Forget gate bias not 1.0: min={forget_slice.min():.4f}, max={forget_slice.max():.4f}"


def test_convlstm_cell_has_per_gate_groupnorm():
    """ConvLSTMCell must have GroupNorm for each gate: norm_i, norm_f, norm_g, norm_o."""
    cell = ConvLSTMCell(256, 256, kernel_size=3)
    assert isinstance(cell.norm_i, nn.GroupNorm), "Missing GroupNorm norm_i"
    assert isinstance(cell.norm_f, nn.GroupNorm), "Missing GroupNorm norm_f"
    assert isinstance(cell.norm_g, nn.GroupNorm), "Missing GroupNorm norm_g"
    assert isinstance(cell.norm_o, nn.GroupNorm), "Missing GroupNorm norm_o"


def test_convlstm_cell_fp32_state_under_bf16_autocast():
    """Under bf16 autocast, ConvLSTMCell must still return h, c in float32."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bf16 autocast test")
    cell = ConvLSTMCell(256, 256, kernel_size=3).cuda()
    x = torch.randn(2, 256, 16, 4, device='cuda')
    h = torch.zeros(2, 256, 16, 4, device='cuda', dtype=torch.float32)
    c = torch.zeros(2, 256, 16, 4, device='cuda', dtype=torch.float32)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        h_new, c_new = cell(x, h, c)
    assert h_new.dtype == torch.float32, f"h_new must be float32 under bf16 autocast, got {h_new.dtype}"
    assert c_new.dtype == torch.float32, f"c_new must be float32 under bf16 autocast, got {c_new.dtype}"


# ---------- UNet1ConvLSTM tests ----------

def test_unet1convlstm_output_shape_full():
    """UNet1ConvLSTM(1,1) with input (2,5,1,256,64) must produce output (2,5,1,256,512)."""
    model = UNet1ConvLSTM(1, 1)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 5, 1, 256, 64)
        out, state = model(x)
    assert out.shape == (2, 5, 1, 256, 512), f"Expected (2,5,1,256,512), got {out.shape}"


def test_unet1convlstm_returns_state_tuple():
    """UNet1ConvLSTM forward must return a (out, state) tuple where state = ((h1,c1),(h2,c2))."""
    model = UNet1ConvLSTM(1, 1)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 1, 256, 64)
        out, state = model(x)
    assert isinstance(state, tuple) and len(state) == 2, "State must be length-2 tuple"
    (h1, c1), (h2, c2) = state
    assert h1.shape == (1, 256, 16, 4), f"Expected h1 (1,256,16,4), got {h1.shape}"
    assert h2.shape == (1, 256, 32, 8), f"Expected h2 (1,256,32,8), got {h2.shape}"


def test_unet1convlstm_single_frame_cold_start():
    """UNet1ConvLSTM with T=1 must work (cold start, zero state)."""
    model = UNet1ConvLSTM(1, 1)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 1, 1, 256, 64)
        out, state = model(x)
    assert out.shape == (2, 1, 1, 256, 512), f"Expected (2,1,1,256,512), got {out.shape}"


def test_unet1convlstm_state_passthrough():
    """State from one forward call can be passed to next (streaming mode)."""
    model = UNet1ConvLSTM(1, 1)
    model.eval()
    with torch.no_grad():
        x1 = torch.randn(2, 3, 1, 256, 64)
        out1, state1 = model(x1)
        # Pass state1 into next call
        x2 = torch.randn(2, 2, 1, 256, 64)
        out2, state2 = model(x2, state=state1)
    assert out2.shape == (2, 2, 1, 256, 512), f"Expected (2,2,1,256,512), got {out2.shape}"


def test_unet1convlstm_uses_groupnorm_not_batchnorm():
    """UNet1ConvLSTM must NOT contain any BatchNorm2d (all GroupNorm)."""
    model = UNet1ConvLSTM(1, 1)
    for name, module in model.named_modules():
        assert not isinstance(module, nn.BatchNorm2d), \
            f"Found BatchNorm2d at {name} — should be GroupNorm"


def test_unet1convlstm_has_projection_layers():
    """UNet1ConvLSTM must have all 4 1x1 projection Conv2d layers."""
    model = UNet1ConvLSTM(1, 1)
    assert hasattr(model, 'proj_in1') and isinstance(model.proj_in1, nn.Conv2d), "Missing proj_in1"
    assert hasattr(model, 'proj_out1') and isinstance(model.proj_out1, nn.Conv2d), "Missing proj_out1"
    assert hasattr(model, 'proj_in2') and isinstance(model.proj_in2, nn.Conv2d), "Missing proj_in2"
    assert hasattr(model, 'proj_out2') and isinstance(model.proj_out2, nn.Conv2d), "Missing proj_out2"
    # Verify shapes: 512->256 and 256->512
    assert model.proj_in1.in_channels == 512 and model.proj_in1.out_channels == 256
    assert model.proj_out1.in_channels == 256 and model.proj_out1.out_channels == 512
    assert model.proj_in2.in_channels == 512 and model.proj_in2.out_channels == 256
    assert model.proj_out2.in_channels == 256 and model.proj_out2.out_channels == 512


def test_unet1convlstm_state_shape_structure():
    """UNet1ConvLSTM state tuple has expected shapes: h1 (B,256,16,4), h2 (B,256,32,8)."""
    B = 3
    model = UNet1ConvLSTM(1, 1)
    model.eval()
    with torch.no_grad():
        x = torch.randn(B, 2, 1, 256, 64)
        out, state = model(x)
    (h1, c1), (h2, c2) = state
    assert h1.shape == (B, 256, 16, 4), f"h1 shape mismatch: {h1.shape}"
    assert c1.shape == (B, 256, 16, 4), f"c1 shape mismatch: {c1.shape}"
    assert h2.shape == (B, 256, 32, 8), f"h2 shape mismatch: {h2.shape}"
    assert c2.shape == (B, 256, 32, 8), f"c2 shape mismatch: {c2.shape}"
