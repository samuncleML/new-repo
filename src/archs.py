"""
KMP-UNet: A parallel UNet integrating KAN and Mamba for medical image segmentation.

Faithful, self-contained re-implementation of the architecture described in:
"A parallel UNet integrating KAN and mamba for medical image segmentation"
(Liu, Wu, Xu, Shi & Zheng, Scientific Reports, 2026).

This file plugs into the KM-UNet training pipeline (dataset.py / losses.py /
metrics.py / utils.py / train.py) that already ships with the BUSI dataset
pre-processed under data/busi/{images,masks/0}.

Design notes / simplifications made to keep the model dependency-free
(no mamba_ssm / causal_conv1d / custom CUDA kernels, no torchvision ops):

  * LDConv (linear deformable convolution): implemented as a depthwise
    convolution whose sampling locations are predicted per-pixel and looked
    up with `grid_sample`. The number of learnable parameters grows
    linearly with the number of sampling points (kernel_size**2), matching
    the "linear" growth motivation described in the paper for LDConv.

  * Mamba branch (forward/backward selective SSM): implemented as a
    minimal selective state-space scan in pure PyTorch (sequential
    recurrence over the token sequence). This is the same recurrence used
    by the original Mamba block, just without the fused CUDA kernel, so it
    is slower but numerically equivalent for the small token counts that
    appear in this compact 1M-parameter network.

  * KAN block: uses the real spline-based KANLinear layer (Kolmogorov-Arnold
    Network) from kan.py, applied pixel-wise across the channel dimension,
    followed by a depthwise conv + LayerNorm as described in the paper.

  * ESCA channel-attention branch: the paper's Eq. (6) computes multi-head
    scaled dot-product attention on a 1x1 global-average-pooled token. We
    implement this literally by reshaping the pooled C-length token into
    (heads, head_dim) and running self-attention over the head_dim axis,
    which is the only reading of Eq. (6) that is dimensionally consistent
    with a 1x1 spatial token.

Author: reimplementation for training on BUSI.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kan import KANLinear

__all__ = ['KMPUNet']


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #

def _init_linear_like(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) tensor."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# --------------------------------------------------------------------------- #
# LDConv: linear deformable convolution (Conv-Block core, Fig. 2b)
# --------------------------------------------------------------------------- #

class LDConv(nn.Module):
    """Depthwise convolution with learned per-point sampling offsets.

    Parameter count grows linearly with the number of sampling points
    (num_points = k*k), which is the property LDConv is designed for
    (as opposed to standard deformable conv whose param count for the
    regular conv part grows quadratically with channels*k*k).
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.k = kernel_size
        self.num_points = kernel_size * kernel_size

        # base regular sampling grid, centered at 0
        ys, xs = torch.meshgrid(
            torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size),
            torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size),
            indexing='ij',
        )
        base_grid = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # (num_points, 2)
        self.register_buffer('base_grid', base_grid)

        # predict per-pixel offsets for every sampling point (linear in channels)
        self.offset_conv = nn.Conv2d(channels, 2 * self.num_points, kernel_size=3,
                                      padding=1, groups=1)
        # per-point, per-channel mixing weight (depthwise, linear in channels)
        self.point_weight = nn.Parameter(torch.randn(channels, self.num_points) * 0.02)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        offsets = self.offset_conv(x)  # (B, 2*num_points, H, W)
        offsets = offsets.view(B, self.num_points, 2, H, W)

        # base normalized pixel grid
        device = x.device
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        base_xy = torch.stack([xx, yy], dim=0)  # (2, H, W)

        out = torch.zeros_like(x)
        for p in range(self.num_points):
            dx, dy = self.base_grid[p, 0], self.base_grid[p, 1]
            off_x = offsets[:, p, 0]  # (B, H, W)
            off_y = offsets[:, p, 1]
            sample_x = base_xy[0].unsqueeze(0) + dx + off_x
            sample_y = base_xy[1].unsqueeze(0) + dy + off_y
            # normalize to [-1, 1] for grid_sample
            norm_x = 2.0 * sample_x / max(W - 1, 1) - 1.0
            norm_y = 2.0 * sample_y / max(H - 1, 1) - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, H, W, 2)
            sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros',
                                     align_corners=True)  # (B, C, H, W)
            w = self.point_weight[:, p].view(1, C, 1, 1)
            out = out + sampled * w

        out = self.bn(out)
        out = self.act(out)
        return out


class ConvBlock(nn.Module):
    """Fig. 2b: 3x3 conv -> LDConv -> 3x3 conv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ldconv = LDConv(out_ch, kernel_size=3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.ldconv(x)
        x = self.conv2(x)
        return x


# --------------------------------------------------------------------------- #
# KAN block (Fig. 2c)
# --------------------------------------------------------------------------- #

class KANBlock(nn.Module):
    """Kolmogorov-Arnold nonlinear channel mixing + local spatial conv."""

    def __init__(self, channels, grid_size=5, spline_order=3):
        super().__init__()
        self.channels = channels
        self.kan = KANLinear(
            channels, channels,
            grid_size=grid_size, spline_order=spline_order,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1],
        )
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = LayerNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2).reshape(B * H * W, C)  # (B*HW, C)
        tokens = self.kan(tokens)
        x_hat = tokens.reshape(B, H * W, C).transpose(1, 2).reshape(B, C, H, W).contiguous()
        x_hat = self.dwconv(x_hat)
        x_hat = self.norm(x_hat)
        return x_hat + x  # residual for stable optimization


# --------------------------------------------------------------------------- #
# Minimal selective-scan Mamba (pure PyTorch, no custom kernels)
# --------------------------------------------------------------------------- #

class MinimalMamba1D(nn.Module):
    """A minimal, dependency-free selective SSM block operating on (B, L, D)."""

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank or max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = dt.clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        # x: (B, L, D)
        B, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.transpose(1, 2)  # (B, d_inner, L)
        x_in = self.conv1d(x_in)[:, :, :L]
        x_in = self.act(x_in)
        x_in = x_in.transpose(1, 2)  # (B, L, d_inner)

        x_dbl = self.x_proj(x_in)  # (B, L, dt_rank + 2*d_state)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # discretize: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * Bm.unsqueeze(2)  # (B, L, d_inner, d_state)
        dBx = dB * x_in.unsqueeze(-1)  # (B, L, d_inner, d_state)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            y_t = torch.einsum('bdn,bn->bd', h, Cm[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + x_in * self.D

        y = y * self.act(z)
        out = self.out_proj(y)
        return out


# --------------------------------------------------------------------------- #
# CMB / PCMB (Fig. 3)
# --------------------------------------------------------------------------- #

class CMB(nn.Module):
    """Forward Mamba + Backward Mamba + Conv + Residual (Eq. 4)."""

    def __init__(self, channels, d_state=16):
        super().__init__()
        self.channels = channels
        self.mamba_fw = MinimalMamba1D(channels, d_state=d_state)
        self.mamba_bw = MinimalMamba1D(channels, d_state=d_state)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)

        fw = self.mamba_fw(tokens)
        bw = self.mamba_bw(torch.flip(tokens, dims=[1]))
        bw = torch.flip(bw, dims=[1])

        conv_feat = self.conv(x).flatten(2).transpose(1, 2)  # (B, HW, C)

        fused = self.fuse(torch.cat([fw, bw, conv_feat], dim=-1))
        out = fused + tokens
        out = out.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return out


class PCMB(nn.Module):
    """Parallel Convolution Mamba Block: group-wise CMB with feature decoupling."""

    def __init__(self, channels, groups=4, d_state=16):
        super().__init__()
        self.groups = groups
        self.channels = channels
        self.norm = LayerNorm2d(channels)

        base = channels // groups
        rem = channels - base * groups
        self.split_sizes = [base + (1 if i < rem else 0) for i in range(groups)]

        self.cmbs = nn.ModuleList([CMB(sz, d_state=d_state) for sz in self.split_sizes])
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        chunks = torch.split(x, self.split_sizes, dim=1)
        outs = [cmb(c) for cmb, c in zip(self.cmbs, chunks)]
        out = torch.cat(outs, dim=1)  # (B, C, H, W)

        B, C, H, W = out.shape
        tokens = out.flatten(2).transpose(1, 2)
        tokens = self.mlp(tokens)
        out = tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()
        out = self.proj(out)
        return out + residual


# --------------------------------------------------------------------------- #
# ESCA: Enhanced Spatial-Channel Attention (Fig. 4)
# --------------------------------------------------------------------------- #

class MSDWConv1D(nn.Module):
    """Multi-scale depthwise 1D convs applied on 4 channel groups, kernel
    sizes {3,5,7,9}, concatenated back together."""

    def __init__(self, channels, kernel_sizes=(3, 5, 7, 9)):
        super().__init__()
        assert channels % len(kernel_sizes) == 0, \
            f"channels ({channels}) must be divisible by {len(kernel_sizes)}"
        g = channels // len(kernel_sizes)
        self.groups_ch = g
        self.convs = nn.ModuleList([
            nn.Conv1d(g, g, kernel_size=k, padding=k // 2, groups=g)
            for k in kernel_sizes
        ])

    def forward(self, x):  # x: (B, C, L)
        chunks = torch.split(x, self.groups_ch, dim=1)
        outs = [conv(c) for conv, c in zip(self.convs, chunks)]
        return torch.cat(outs, dim=1)


class ESCA(nn.Module):
    """Enhanced Spatial-Channel Attention for skip-connection refinement."""

    def __init__(self, channels, num_heads=4, drop=0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0

        # spatial attention branch
        self.ms_h = MSDWConv1D(channels)
        self.ms_w = MSDWConv1D(channels)
        self.norm_h = nn.GroupNorm(4, channels)
        self.norm_w = nn.GroupNorm(4, channels)

        # channel attention branch
        self.q_proj = nn.Conv2d(channels, channels, 1, groups=channels)
        self.k_proj = nn.Conv2d(channels, channels, 1, groups=channels)
        self.v_proj = nn.Conv2d(channels, channels, 1, groups=channels)
        self.chan_norm = nn.LayerNorm(channels)

        # gated fusion
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2),
        )
        self.drop = nn.Dropout(drop)
        self.ln = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def _spatial_attention(self, x):
        B, C, H, W = x.shape
        x_h = x.mean(dim=3)  # (B, C, H)
        x_w = x.mean(dim=2)  # (B, C, W)

        a_h = torch.sigmoid(self.norm_h(self.ms_h(x_h)))  # (B, C, H)
        a_w = torch.sigmoid(self.norm_w(self.ms_w(x_w)))  # (B, C, W)

        a_h = a_h.unsqueeze(-1)  # (B, C, H, 1)
        a_w = a_w.unsqueeze(2)   # (B, C, 1, W)
        return x * a_h * a_w

    def _channel_attention(self, x):
        B, C, H, W = x.shape
        token = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        q = self.q_proj(token).view(B, self.num_heads, C // self.num_heads)
        k = self.k_proj(token).view(B, self.num_heads, C // self.num_heads)
        v = self.v_proj(token).view(B, self.num_heads, C // self.num_heads)

        # self-attention across the head_dim axis (only spatial-degenerate
        # reading of Eq. 6 that is dimensionally consistent with a 1x1 token)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(
            torch.einsum('bhd,bhe->bhde', q, k) * scale, dim=-1
        )
        out = torch.einsum('bhde,bhe->bhd', attn, v)
        out = out.reshape(B, C)
        gate = torch.sigmoid(out).view(B, C, 1, 1)
        return x * gate

    def forward(self, x):
        xs = self._spatial_attention(x)
        xc = self._channel_attention(x)

        B, C, H, W = x.shape
        gap_s = xs.mean(dim=[2, 3])
        gap_c = xc.mean(dim=[2, 3])
        alpha = torch.softmax(self.gate_mlp(torch.cat([gap_s, gap_c], dim=-1)), dim=-1)
        alpha_s = alpha[:, 0].view(B, 1, 1, 1)
        alpha_c = alpha[:, 1].view(B, 1, 1, 1)

        xf = alpha_s * xs + alpha_c * xc
        y = x + self.drop(xf)

        y_tok = y.permute(0, 2, 3, 1)
        z = y_tok + self.ffn(self.ln(y_tok))
        z = z.permute(0, 3, 1, 2).contiguous()
        return z


# --------------------------------------------------------------------------- #
# Parallel PCMB + KAN stage
# --------------------------------------------------------------------------- #

class ParallelPKStage(nn.Module):
    """One 'parallel' encoder/decoder stage: PCMB(x) and KAN(x) computed in
    parallel on the same input and fused with a learned 1x1 conv."""

    def __init__(self, channels, groups=4, d_state=16):
        super().__init__()
        self.pcmb = PCMB(channels, groups=groups, d_state=d_state)
        self.kan = KANBlock(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        p = self.pcmb(x)
        k = self.kan(x)
        out = self.fuse(torch.cat([p, k], dim=1))
        return out + x


class ParallelPKBlockPair(nn.Module):
    """Two stacked ParallelPKStage layers (best per the paper's ablation
    on number of parallel layers, see Fig. 8 left)."""

    def __init__(self, channels, groups=4, d_state=16):
        super().__init__()
        self.stage1 = ParallelPKStage(channels, groups=groups, d_state=d_state)
        self.stage2 = ParallelPKStage(channels, groups=groups, d_state=d_state)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x


# --------------------------------------------------------------------------- #
# KMP-UNet
# --------------------------------------------------------------------------- #

class KMPUNet(nn.Module):
    """KMP-UNet, a compact parallel UNet integrating KAN and Mamba.

    Architecture (mirrors Fig. 2a of the paper, at a scale suited to a
    ~1M-parameter model trained at 256x256 on BUSI):

        input
          -> Conv-Block x3 (stem, /8 spatial, channels: 3->8->16->32)
          -> [ParallelPKBlockPair @ 32ch]  -> maxpool /16, 32->64
          -> [ParallelPKBlockPair @ 64ch]  -> maxpool /32, 64->128  (bottleneck)
          -> upsample /16, 128->64 -> [ParallelPKBlockPair @ 64ch]  (ESCA skip #2)
          -> upsample /8,  64->32  -> [ParallelPKBlockPair @ 32ch]  (ESCA skip #1)
          -> Conv-Block x3 (decoder stem, /1 spatial, channels: 32->16->8)
          -> 1x1 conv -> num_classes
    """

    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False,
                 base_channels=8, groups=4, d_state=16, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        c0, c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, \
            base_channels * 8, base_channels * 16

        # ---------------- Encoder stem: 3 Conv-Blocks + maxpool ----------------
        self.enc_conv1 = ConvBlock(input_channels, c0)   # 3 -> 8
        self.enc_conv2 = ConvBlock(c0, c1)               # 8 -> 16
        self.enc_conv3 = ConvBlock(c1, c2)               # 16 -> 32
        self.pool = nn.MaxPool2d(2, 2)

        # ---------------- Encoder parallel stages ----------------
        self.enc_pk1 = ParallelPKBlockPair(c2, groups=groups, d_state=d_state)   # 32ch, /8
        self.down1 = nn.Sequential(nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.GELU())  # 32->64
        self.enc_pk2 = ParallelPKBlockPair(c3, groups=groups, d_state=d_state)   # 64ch, /16
        self.down2 = nn.Sequential(nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.GELU())  # 64->128

        # ---------------- Bottleneck ----------------
        self.bottleneck = ParallelPKBlockPair(c4, groups=groups, d_state=d_state)  # 128ch, /32

        # ---------------- ESCA skip connections ----------------
        self.esca_inner = ESCA(c3)   # fuses 64-ch skip (post enc_pk2 / pre down2)
        self.esca_outer = ESCA(c2)   # fuses 32-ch skip (post enc_pk1 / pre down1)

        # ---------------- Decoder parallel stages ----------------
        self.up2 = nn.Sequential(nn.Conv2d(c4, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.GELU())   # 128->64
        self.dec_pk2 = ParallelPKBlockPair(c3, groups=groups, d_state=d_state)  # 64ch
        self.up1 = nn.Sequential(nn.Conv2d(c3, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.GELU())   # 64->32
        self.dec_pk1 = ParallelPKBlockPair(c2, groups=groups, d_state=d_state)  # 32ch

        # ---------------- Decoder stem: 3 Conv-Blocks + upsample ----------------
        self.dec_conv3 = ConvBlock(c2, c1)   # 32 -> 16
        self.dec_conv2 = ConvBlock(c1, c0)   # 16 -> 8
        self.dec_conv1 = ConvBlock(c0, c0)   # 8 -> 8

        self.final = nn.Conv2d(c0, num_classes, kernel_size=1)

        self.apply(_init_linear_like)

    def forward(self, x):
        # ---------------- Encoder stem ----------------
        e1 = self.enc_conv1(x)
        e1p = self.pool(e1)
        e2 = self.enc_conv2(e1p)
        e2p = self.pool(e2)
        e3 = self.enc_conv3(e2p)
        e3p = self.pool(e3)               # (B, 32, H/8, W/8)  -- outer skip source

        # ---------------- Encoder parallel stages ----------------
        s1 = self.enc_pk1(e3p)            # (B, 32, H/8, W/8)  -- outer skip
        s1d = self.pool(self.down1(s1))   # (B, 64, H/16, W/16)
        s2 = self.enc_pk2(s1d)            # (B, 64, H/16, W/16) -- inner skip
        s2d = self.pool(self.down2(s2))   # (B, 128, H/32, W/32)

        # ---------------- Bottleneck ----------------
        b = self.bottleneck(s2d)          # (B, 128, H/32, W/32)

        # ---------------- Decoder parallel stages ----------------
        u2 = F.interpolate(self.up2(b), scale_factor=2, mode='bilinear', align_corners=False)
        skip_inner = self.esca_inner(s2 + u2) if s2.shape == u2.shape else self.esca_inner(u2)
        d2 = self.dec_pk2(skip_inner)     # (B, 64, H/16, W/16)

        u1 = F.interpolate(self.up1(d2), scale_factor=2, mode='bilinear', align_corners=False)
        skip_outer = self.esca_outer(s1 + u1) if s1.shape == u1.shape else self.esca_outer(u1)
        d1 = self.dec_pk1(skip_outer)     # (B, 32, H/8, W/8)

        # ---------------- Decoder stem ----------------
        o3 = F.interpolate(self.dec_conv3(d1), scale_factor=2, mode='bilinear', align_corners=False)
        o2 = F.interpolate(self.dec_conv2(o3), scale_factor=2, mode='bilinear', align_corners=False)
        o1 = F.interpolate(self.dec_conv1(o2), scale_factor=2, mode='bilinear', align_corners=False)

        out = self.final(o1)
        return out
