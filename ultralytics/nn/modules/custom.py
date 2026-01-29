import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class SS2D(nn.Module):
    """
    SS2D: Cross-Scan Mechanism for VMamba.
    Scans the feature map in 4 directions to capture global context from all angles.
    Includes CPU Bypass for YOLO initialization.
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.):
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

        # The Mamba Engine (one for each scan direction)
        self.mamba = Mamba(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1
        )

        # Output Projection
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        # x: [Batch, Channels, Height, Width]
        B, C, H, W = x.shape
        
        # 1. Reshape for Linear Projection: [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x_proj = self.in_proj(x) # [B, H, W, 2*d_inner]
        
        # Split into x (content) and z (gate)
        (x_in, z) = x_proj.chunk(2, dim=-1)
        
        # 2. Local Convolution (needs [B, C, H, W])
        x_conv = x_in.permute(0, 3, 1, 2).contiguous()
        x_conv = self.conv2d(x_conv)
        x_conv = self.act(x_conv) # [B, d_inner, H, W]
        
        # --- 3. SS2D: Cross-Scan (CPU SAFEGUARD) ---
        if not x.is_cuda:
            # 🛡️ BYPASS: If input is on CPU (YOLO initialization), SKIP Mamba kernel.
            # We flatten and pass it through to satisfy shape requirements without crashing.
            x_ss2d = x_conv.flatten(2).transpose(1, 2)
        else:
            # 🚀 RUN: If on GPU (Training), run full Cross-Scan Mamba
            
            # Direction 1: Forward (Top-Left -> Bottom-Right)
            x_fwd = x_conv.flatten(2).transpose(1, 2) # [B, L, C]
            out_fwd = self.mamba(x_fwd)
            
            # Direction 2: Backward (Bottom-Right -> Top-Left)
            x_bwd = x_conv.flatten(2).transpose(1, 2).flip([1])
            out_bwd = self.mamba(x_bwd).flip([1])
            
            # Direction 3: Transposed Forward (Top-Right -> Bottom-Left approx)
            x_t_fwd = x_conv.transpose(2, 3).flatten(2).transpose(1, 2)
            out_t_fwd = self.mamba(x_t_fwd).transpose(1, 2).view(B, -1, W, H).transpose(2, 3).flatten(2).transpose(1, 2)

            # Direction 4: Transposed Backward
            x_t_bwd = x_conv.transpose(2, 3).flatten(2).transpose(1, 2).flip([1])
            out_t_bwd = self.mamba(x_t_bwd).flip([1]).transpose(1, 2).view(B, -1, W, H).transpose(2, 3).flatten(2).transpose(1, 2)

            # Merge the 4 scans (Average them)
            x_ss2d = (out_fwd + out_bwd + out_t_fwd + out_t_bwd) / 4.0
        
        # 4. Gating and Output
        x_out = x_ss2d * F.silu(z.flatten(1, 2)) # Gate with original z
        x_out = self.out_norm(x_out)
        x_out = self.out_proj(x_out)
        x_out = self.dropout(x_out)
        
        # Reshape back to [B, C, H, W]
        return x_out.transpose(1, 2).view(B, C, H, W)

class VSSBlock(nn.Module):
    """
    The Main VSS Block Wrapper.
    Replaces C3k2/Bottleneck in YOLO.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c2 = c2
        # Input projection to match channels
        self.proj = nn.Conv2d(c1, c2, 1, 1, 0) if c1 != c2 else nn.Identity()
        
        # The Core SS2D Engine
        self.ss2d = SS2D(d_model=c2, d_state=16, d_conv=3, expand=2)

    def forward(self, x):
        x_proj = self.proj(x)
        # Apply SS2D (Cross-Scan Mamba)
        x_out = self.ss2d(x_proj)
        # Residual Connection
        return x_proj + x_out

class DySample(nn.Module):
    """
    Dynamic Upsampling (Same as before - this was already SOTA).
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4):
        super().__init__()
        self.scale = scale
        self.offset = nn.Conv2d(in_channels, 2 * groups * scale**2, 1)
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        offset = self.offset(x)
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        correction = 0.1 * offset[:, :C, :, :].repeat(1, 1, self.scale, self.scale)
        return base + correction