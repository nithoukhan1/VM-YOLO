import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SS2D(nn.Module):
    """SS2D: Cross-Scan Mechanism for VMamba. Scans the feature map in 4 directions to capture global context. Includes
    CPU Bypass for safe YOLO initialization.
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input Projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolution (Local Context)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.act = nn.SiLU()

        # The Mamba Engine
        self.mamba = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)

        # Output Projection
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # x: [Batch, Channels, Height, Width]
        B, C, H, W = x.shape

        # 1. Reshape for Linear Projection
        x = x.permute(0, 2, 3, 1)
        x_proj = self.in_proj(x)
        (x_in, z) = x_proj.chunk(2, dim=-1)

        # 2. Local Convolution
        x_conv = x_in.permute(0, 3, 1, 2).contiguous()
        x_conv = self.conv2d(x_conv)
        x_conv = self.act(x_conv)

        # --- 3. SS2D: Cross-Scan (CPU SAFEGUARD) ---
        if not x.is_cuda:
            # 🛡️ BYPASS: If input is on CPU (during init), SKIP Mamba kernel.
            # We flatten and pass it through to satisfy shape requirements.
            x_ss2d = x_conv.flatten(2).transpose(1, 2)
            print("SS2D Mamba bypassed on CPU input.")
        else:
            # 🚀 RUN: If on GPU (Training), run full Cross-Scan Mamba

            # Direction 1: Forward (Top-Left -> Bottom-Right)
            x_fwd = x_conv.flatten(2).transpose(1, 2)
            out_fwd = self.mamba(x_fwd)

            # Direction 2: Backward (Bottom-Right -> Top-Left)
            x_bwd = x_conv.flatten(2).transpose(1, 2).flip([1])
            out_bwd = self.mamba(x_bwd).flip([1])

            # Direction 3: Transposed Forward
            x_t_fwd = x_conv.transpose(2, 3).flatten(2).transpose(1, 2)
            out_t_fwd = self.mamba(x_t_fwd).transpose(1, 2).view(B, -1, W, H).transpose(2, 3).flatten(2).transpose(1, 2)

            # Direction 4: Transposed Backward
            x_t_bwd = x_conv.transpose(2, 3).flatten(2).transpose(1, 2).flip([1])
            out_t_bwd = (
                self.mamba(x_t_bwd)
                .flip([1])
                .transpose(1, 2)
                .view(B, -1, W, H)
                .transpose(2, 3)
                .flatten(2)
                .transpose(1, 2)
            )

            # Merge the 4 scans
            x_ss2d = (out_fwd + out_bwd + out_t_fwd + out_t_bwd) / 4.0

        # 4. Gating and Output
        x_out = x_ss2d * F.silu(z.flatten(1, 2))
        x_out = self.out_norm(x_out)
        x_out = self.out_proj(x_out)
        x_out = self.dropout(x_out)

        return x_out.transpose(1, 2).view(B, C, H, W)


class VSSBlock(nn.Module):
    """The Main VSS Block Wrapper. Replaces C3k2/Bottleneck in YOLO.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c2 = c2
        self.proj = nn.Conv2d(c1, c2, 1, 1, 0) if c1 != c2 else nn.Identity()
        self.ss2d = SS2D(d_model=c2, d_state=16, d_conv=3, expand=2)

    def forward(self, x):
        x_proj = self.proj(x)
        x_out = self.ss2d(x_proj)
        return x_proj + x_out


class DySample(nn.Module):
    """Dynamic Upsampling (Fixed for 512-channel compatibility)."""

    def __init__(self, in_channels, scale=2, style="lp", groups=4):
        super().__init__()
        self.scale = scale
        # Generate a correction map with the SAME number of channels as input
        self.gen_offset = nn.Conv2d(in_channels, in_channels, 1)

        # Zero initialization ensures we start with standard bilinear behavior
        nn.init.constant_(self.gen_offset.weight, 0)
        nn.init.constant_(self.gen_offset.bias, 0)

    def forward(self, x):
        # 1. Standard Bilinear Upsampling
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)

        # 2. Learn the Correction Map (Low Res)
        offset = self.gen_offset(x)

        # 3. Upsample the Correction to match the Base size
        correction = F.interpolate(offset, scale_factor=self.scale, mode="bilinear", align_corners=False)

        # 4. Add them together
        return base + correction
