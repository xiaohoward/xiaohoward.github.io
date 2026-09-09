"""
22 — Diffusion UNet building blocks: time-conditioned ResBlock, Down/Upsample, tiny 2-level UNet

Implement the residual block used by DDPM / ADM / Stable Diffusion UNets, conditioned on a timestep embedding via
scale-shift (FiLM / "adaptive GroupNorm"), the stride-2 downsample and nearest-2x upsample layers, and the
forward pass of a tiny 2-level UNet with skip concatenation. The module constructors are GIVEN; you write the
forward passes.

ResBlock(in_ch, out_ch, t_dim):
    h = conv3x3( SiLU( GN(x) ) )
    scale, shift = Linear(t_dim, 2*out_ch)( SiLU(t_emb) ).chunk(2)
    h = GN(h) * (1 + scale) + shift                      # scale-shift conditioning (ADM style)
    h = conv3x3_zero_init( Dropout( SiLU(h) ) )
    out = h + skip(x)                                    # skip = 1x1 conv if in_ch != out_ch else identity
Downsample: conv3x3 stride 2 pad 1  -> H_out = (H + 1) // 2 (so odd sizes round UP).
Upsample  : nearest x2 (or to an explicit `out_hw`) then conv3x3 pad 1.
TinyUNet  : in_conv -> Res(base) --skip--> Down -> Res(base->2base) -> Res(2base) -> Up(to skip size) ->
            cat([up, skip]) -> Res(3base -> base) -> GN -> SiLU -> out_conv.

Signatures (methods to implement):
    ResBlock.forward(x, t_emb) ; Downsample.forward(x) ; Upsample.forward(x, out_hw=None) ; TinyUNet.forward(x, t_emb)

Constraints: torch (CPU) only. Because the last conv of every ResBlock is zero-initialised, at init each ResBlock
equals its skip path (identity when in_ch == out_ch) and is independent of t — the tests check that, and check
that after randomising the weights the output depends on t and gradients reach every parameter.

Interview budget: 25 min

Discussion follow-ups:
  * Why zero-init the last conv (and the output conv) of a residual block? (Network starts as identity; stable
    deep training — same trick as Fixup / ControlNet zero-convs.)
  * Scale-shift (FiLM) vs additive time embedding: why does ADM find scale-shift better? Relation to adaLN-Zero
    in DiT.
  * GroupNorm vs BatchNorm in diffusion UNets: why not BatchNorm (batch statistics vs noise level mixing)?
  * How does the UNet's skip concatenation compare to DiT's lack of skips? What did U-ViT / Hunyuan add back?
  * Odd resolutions: where does the UNet break and how do production models handle it (pad to multiples of 2^L,
    or crop after upsampling)?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__implement__ = ["ResBlock.forward", "Downsample.forward", "Upsample.forward", "TinyUNet.forward"]


def group_count(ch, max_groups=32):
    """Largest number of groups <= max_groups that divides ch (given helper)."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Time-conditioned residual block (given constructor; attribute names: norm1, conv1, t_proj, norm2, dropout,
    conv2, skip)."""

    def __init__(self, in_ch, out_ch, t_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(group_count(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        """
        x     : (B, in_ch, H, W) float tensor.
        t_emb : (B, t_dim) timestep embedding (already sinusoidal/MLP-embedded; apply SiLU before t_proj).
        Returns (B, out_ch, H, W):
            h = conv1(silu(norm1(x)))
            scale, shift = t_proj(silu(t_emb)).chunk(2, dim=1)      # each (B, out_ch), broadcast over H, W
            h = norm2(h) * (1 + scale) + shift
            h = conv2(dropout(silu(h)))
            return h + skip(x)
        Spatial size is preserved (3x3 convs with padding 1).
        """
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.t_proj(F.silu(t_emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Stride-2 3x3 conv (given constructor; attribute: conv)."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        """x : (B, ch, H, W) -> (B, ch, (H + 1) // 2, (W + 1) // 2) via the stride-2, padding-1 3x3 conv."""
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest 2x upsampling followed by a 3x3 conv (given constructor; attribute: conv)."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, out_hw=None):
        """
        x      : (B, ch, H, W).
        out_hw : optional (H_out, W_out) target size; if None, upsample to (2H, 2W).
        Returns (B, ch, H_out, W_out): nearest-neighbour interpolation (F.interpolate mode="nearest") to the target
        size, then the 3x3 conv with padding 1. Passing out_hw lets the UNet match an odd-sized skip tensor.
        """
        if out_hw is None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            x = F.interpolate(x, size=tuple(out_hw), mode="nearest")
        return self.conv(x)


class TinyUNet(nn.Module):
    """Two-level UNet without attention (given constructor).

    Attributes: in_conv (in_ch -> base), res_down (base -> base), down (Downsample(base)),
    res_mid1 (base -> 2*base), res_mid2 (2*base -> 2*base), up (Upsample(2*base)),
    res_up (3*base -> base), out_norm (GroupNorm on base), out_conv (base -> out_ch)."""

    def __init__(self, in_ch, out_ch, base, t_dim, dropout=0.0):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.res_down = ResBlock(base, base, t_dim, dropout)
        self.down = Downsample(base)
        self.res_mid1 = ResBlock(base, 2 * base, t_dim, dropout)
        self.res_mid2 = ResBlock(2 * base, 2 * base, t_dim, dropout)
        self.up = Upsample(2 * base)
        self.res_up = ResBlock(3 * base, base, t_dim, dropout)
        self.out_norm = nn.GroupNorm(group_count(base), base)
        self.out_conv = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x, t_emb):
        """
        x     : (B, in_ch, H, W), any H, W >= 2 (odd sizes allowed).
        t_emb : (B, t_dim).
        Returns (B, out_ch, H, W):
            h0 = in_conv(x)
            s  = res_down(h0, t)                     # skip tensor, (B, base, H, W)
            h  = down(s)                             # (B, base, ceil(H/2), ceil(W/2))
            h  = res_mid2(res_mid1(h, t), t)         # (B, 2base, ...)
            h  = up(h, out_hw=s.shape[-2:])          # back to (H, W) even when H or W is odd
            h  = res_up(cat([h, s], dim=1), t)       # (B, base, H, W)
            return out_conv(silu(out_norm(h)))
        Concatenation order is [upsampled, skip] along channels.
        """
        h0 = self.in_conv(x)
        s = self.res_down(h0, t_emb)
        h = self.down(s)
        h = self.res_mid2(self.res_mid1(h, t_emb), t_emb)
        h = self.up(h, out_hw=s.shape[-2:])
        h = self.res_up(torch.cat([h, s], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))
