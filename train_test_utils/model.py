# Model for RadarHD

# Adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch.nn as nn
import torch

from train_test_utils.unet_parts import *


class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.up5 = Up_nocat(64, 64, bilinear)
        self.up6 = Up_nocat(64, 64, bilinear)
        self.up7 = Up_nocat(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        conv_out = self.outc(x)
        logits = self.final_sigmoid(conv_out)

        return logits


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell with per-gate GroupNorm and fp32 state.

    Gate order in conv output (chunk 4): i (input), f (forget), g (cell), o (output).
    Forget gate bias initialized to 1.0 (Jozefowicz 2015 — "remember by default").
    State update runs in float32 even under bf16 autocast to avoid numerical drift
    over long sequences (AMP contract).

    Args:
        input_channels: channels of the input feature map x
        hidden_channels: channels of the hidden state h (and c)
        kernel_size: spatial kernel size for gate convolution (default 3, same-padding)
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # same-padding: output spatial == input spatial

        # All 4 gates fused into one conv (input + hidden concatenated along channel)
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # GroupNorm applied per gate after the gate split (NOT on the 4*H concatenation).
        # This preserves forget gate bias semantics and keeps gate statistics independent.
        # 32 groups works for hidden_channels=256 (8 channels/group).
        self.norm_i = nn.GroupNorm(32, hidden_channels)
        self.norm_f = nn.GroupNorm(32, hidden_channels)
        self.norm_g = nn.GroupNorm(32, hidden_channels)
        self.norm_o = nn.GroupNorm(32, hidden_channels)

        # Forget gate bias = 1.0.
        # chunk(4) order: i, f, g, o — forget slice is [hidden_channels : 2*hidden_channels].
        nn.init.constant_(self.conv.bias[hidden_channels:2 * hidden_channels], 1.0)

    def forward(self, x, h, c):
        """Step the cell one timestep forward.

        Args:
            x: (B, input_channels, H, W) — current frame features (any dtype)
            h: (B, hidden_channels, H, W) — previous hidden state (float32)
            c: (B, hidden_channels, H, W) — previous cell state (float32)

        Returns:
            h_new, c_new: both (B, hidden_channels, H, W) in float32
        """
        # Cast h to x.dtype for the gate convolution (runs inside outer autocast context)
        combined = torch.cat([x, h.to(x.dtype)], dim=1)
        gates = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)

        # State update in float32 to prevent bf16 numerical drift over long sequences.
        # autocast(enabled=False) is a no-op on CPU (which has no bf16 autocast).
        device_type = x.device.type if x.device.type != 'cpu' else 'cuda'
        with torch.autocast(device_type='cuda', enabled=False):
            i = torch.sigmoid(self.norm_i(i.float()))
            f = torch.sigmoid(self.norm_f(f.float()))
            g = torch.tanh(self.norm_g(g.float()))
            o = torch.sigmoid(self.norm_o(o.float()))
            c_new = f * c + i * g
            h_new = o * torch.tanh(c_new)

        return h_new, c_new  # float32


class UNet1ConvLSTM(nn.Module):
    """UNet1 with two ConvLSTM cells replacing the static bottleneck/skip representation.

    Architecture:
    - Shared single-frame 2D encoder (n_channels=1, GroupNorm throughout)
    - ConvLSTM cell 1 at bottleneck (after down4, 512ch -> proj 256ch -> LSTM -> proj 512ch)
    - ConvLSTM cell 2 at deepest skip (after down3, 512ch -> proj 256ch -> LSTM -> proj 512ch)
    - Current-frame skip features at down2/down1/inc levels (no temporal processing)
    - Decoder identical to UNet1 (except GroupNorm)
    - Output: (B, T, 1, H_out, W_out) with one prediction per input frame

    Input x_seq: (B, T, 1, 256, 64)
    Output:      (B, T, 1, 256, 512) + state tuple ((h1,c1),(h2,c2))

    State can be passed back in for streaming/stateful inference. Pass state=None
    (default) to zero-initialize (training: always pass None per batch).
    """

    def __init__(self, n_channels=1, n_classes=1, bilinear=True, hidden_channels=256):
        super().__init__()
        self.hidden_channels = hidden_channels
        factor = 2 if bilinear else 1

        # Encoder — shared weights, single frame, GroupNorm throughout
        self.inc   = DoubleConv(n_channels, 64,           norm_type='group')
        self.down1 = Down(64,  128,                       norm_type='group')
        self.down2 = Down(128, 256,                       norm_type='group')
        self.down3 = Down(256, 512,                       norm_type='group')
        self.down4 = Down(512, 1024 // factor,            norm_type='group')  # -> 512ch

        # 1x1 projection layers: 512->256 before ConvLSTM, 256->512 after
        self.proj_in1  = nn.Conv2d(512, hidden_channels, 1)  # bottleneck input
        self.proj_out1 = nn.Conv2d(hidden_channels, 512, 1)  # bottleneck output
        self.proj_in2  = nn.Conv2d(512, hidden_channels, 1)  # deepest skip input
        self.proj_out2 = nn.Conv2d(hidden_channels, 512, 1)  # deepest skip output

        # ConvLSTM cells — both 256ch hidden, 3x3 kernel
        self.convlstm1 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=3)  # bottleneck
        self.convlstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=3)  # deepest skip

        # Decoder — channel math mirrors UNet1 exactly, GroupNorm throughout
        # up1 expects cat([x4_out, up(x5_out)]) = 512+512 = 1024 in, bilinear -> 256 out
        self.up1  = Up(1024, 512 // factor, bilinear, norm_type='group')   # 256
        self.up2  = Up(512,  256 // factor, bilinear, norm_type='group')   # 128
        self.up3  = Up(256,  128 // factor, bilinear, norm_type='group')   # 64
        self.up4  = Up(128,  64,            bilinear, norm_type='group')
        self.up5  = Up_nocat(64, 64, bilinear, norm_type='group')
        self.up6  = Up_nocat(64, 64, bilinear, norm_type='group')
        self.up7  = Up_nocat(64, 64, bilinear, norm_type='group')
        self.outc = OutConv(64, n_classes)
        self.final_sigmoid = nn.Sigmoid()

    def _init_state(self, B, device):
        """Zero-initialize hidden state tensors in float32."""
        z1 = torch.zeros(B, self.hidden_channels, 16, 4,  device=device, dtype=torch.float32)
        z2 = torch.zeros(B, self.hidden_channels, 32, 8,  device=device, dtype=torch.float32)
        return (z1.clone(), z1.clone()), (z2.clone(), z2.clone())

    def forward(self, x_seq, state=None):
        """Forward pass over a sequence of radar frames.

        Args:
            x_seq: (B, T, 1, H, W) radar sequence. Also accepts (B, 1, H, W) single frame.
            state: optional ((h1,c1),(h2,c2)) from a previous call (streaming mode).
                   Pass None (default) to zero-initialize (training mode).

        Returns:
            out:       (B, T, 1, 256, 512) — one lidar prediction per input frame
            state_out: ((h1,c1),(h2,c2)) — final hidden states in float32
        """
        if x_seq.dim() == 4:
            # Single frame (B, 1, H, W) -> add time dim -> (B, 1, 1, H, W)
            x_seq = x_seq.unsqueeze(1)

        B, T = x_seq.shape[:2]
        device = x_seq.device

        if state is None:
            (h1, c1), (h2, c2) = self._init_state(B, device)
        else:
            (h1, c1), (h2, c2) = state

        outputs = []
        for t in range(T):
            frame = x_seq[:, t]  # (B, 1, H, W)

            # Encode current frame with shared weights
            x1 = self.inc(frame)    # (B, 64,  256, 64)
            x2 = self.down1(x1)     # (B, 128, 128, 32)
            x3 = self.down2(x2)     # (B, 256,  64, 16)
            x4 = self.down3(x3)     # (B, 512,  32,  8)
            x5 = self.down4(x4)     # (B, 512,  16,  4)

            # ConvLSTM cell 1: bottleneck (x5)
            x5_proj = self.proj_in1(x5)              # (B, 256, 16, 4)
            h1, c1  = self.convlstm1(x5_proj, h1, c1)
            x5_out  = self.proj_out1(h1.to(x5.dtype))  # (B, 512, 16, 4)

            # ConvLSTM cell 2: deepest skip (x4)
            x4_proj = self.proj_in2(x4)              # (B, 256, 32, 8)
            h2, c2  = self.convlstm2(x4_proj, h2, c2)
            x4_out  = self.proj_out2(h2.to(x4.dtype))  # (B, 512, 32, 8)

            # Decode using temporal features + current-frame skips
            x = self.up1(x5_out, x4_out)   # (B, 256, 32,  8)
            x = self.up2(x, x3)             # (B, 128, 64, 16)
            x = self.up3(x, x2)             # (B,  64, 128, 32)
            x = self.up4(x, x1)             # (B,  64, 256, 64)
            x = self.up5(x)                 # (B,  64, 256, 128)
            x = self.up6(x)                 # (B,  64, 256, 256)
            x = self.up7(x)                 # (B,  64, 256, 512)
            outputs.append(self.final_sigmoid(self.outc(x)))  # (B, 1, 256, 512)

        out = torch.stack(outputs, dim=1)       # (B, T, 1, 256, 512)
        state_out = ((h1, c1), (h2, c2))
        return out, state_out
