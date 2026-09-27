"""DiT block with adaLN-Zero + sinusoidal timestep embedding

Implement the core block of a Diffusion Transformer (Peebles & Xie 2023) on token sequences x (B, N, D)
conditioned on a vector c (B, D) (timestep embedding, optionally + class/text pooled embedding):
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = adaLN(c).chunk(6)     adaLN = Linear(SiLU(c)) -> 6D
    x = x + gate_a * Attn( modulate(LN1(x), shift_a, scale_a) )
    x = x + gate_m * MLP ( modulate(LN2(x), shift_m, scale_m) )
    modulate(h, shift, scale) = h * (1 + scale[:, None]) + shift[:, None]
"adaLN-Zero": the adaLN linear layer is zero-initialised (weight AND bias) so every block is the identity at init
and the residual stream is untouched until training moves the gates. LayerNorms have NO affine parameters
(elementwise_affine=False, eps=1e-6). Attention is standard multi-head self-attention (fused qkv linear, output
projection); the MLP is Linear -> GELU(tanh approx) -> Linear with hidden = mlp_ratio * D.
Also implement the sinusoidal timestep embedding used to build c.

Signatures:
    timestep_embedding(t, dim, max_period=10000) -> (B, dim)
    modulate(x, shift, scale) -> (B, N, D)
    Attention.forward(x) -> (B, N, D)
    DiTBlock.__init__(dim, num_heads, mlp_ratio=4.0)
    DiTBlock.modulation(c) -> tuple of 6 tensors (B, D)
    DiTBlock.forward(x, c) -> (B, N, D)

Constraints: torch CPU float32, no external libs. You may use F.scaled_dot_product_attention. Attention.__init__,
Mlp and TimestepEmbedder are given; keep their attribute names (qkv, proj, fc1, fc2, ...) since tests read them.

Interview budget: 30 min

Discussion follow-ups:
  - Why zero-init the gate (and what happens if you zero-init only the last MLP layer instead)? Relate to
    ReZero / Fixup and to training stability of very deep DiTs.
  - Cost of adaLN parameters: 6 D^2 per block. What do SD3 / FLUX / WAN do differently (shared adaLN, "single
    stream" vs "double stream" blocks, text tokens joining self-attention)?
  - How would you add positional information for video tokens (3D RoPE vs learned abs pos) and what breaks when
    you change resolution/frames at inference (extrapolation, NTK scaling)?
  - Attention cost is O(N^2 D): options to cut it for 4D video tokens (windowed / factorised / token reduction).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__implement__ = ["timestep_embedding", "modulate", "Attention.forward", "DiTBlock.__init__", "DiTBlock.modulation",
                 "DiTBlock.forward"]


# ----------------------------------------------------------------------------- given helpers
class Mlp(nn.Module):
    """Given helper: fc1 -> GELU(tanh) -> fc2."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TimestepEmbedder(nn.Module):
    """Given helper: c = MLP(sinusoidal(t)). Uses your timestep_embedding."""

    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim))


class Attention(nn.Module):
    """Multi-head self-attention. __init__ is given; implement forward."""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """Multi-head self-attention (no mask, no dropout).

        qkv = self.qkv(x) -> split the LAST dim into (q, k, v) in that order, each (B, N, D); reshape each to
        (B, H, N, head_dim); softmax(q k^T / sqrt(head_dim)) v; merge heads back to (B, N, D); apply self.proj.

        Args:
            x: (B, N, D) float32.
        Returns:
            (B, N, D) float32.
        """
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.view(B, N, self.num_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


# ----------------------------------------------------------------------------- to implement
def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of (possibly fractional) timesteps, as in DiT / OpenAI guided-diffusion.

    half = dim // 2;  freqs[i] = exp(-ln(max_period) * i / half),  i = 0..half-1;  args = t[:, None] * freqs[None];
    return cat([cos(args), sin(args)], dim=-1) (cos FIRST). If dim is odd, append one zero column.

    Args:
        t: (B,) int64 or float32 tensor. dim: int. max_period: float.
    Returns:
        (B, dim) float32.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t.to(torch.float32)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x, shift, scale):
    """x * (1 + scale) + shift with per-sample shift/scale broadcast over tokens.

    Args:
        x: (B, N, D). shift, scale: (B, D).
    Returns:
        (B, N, D).
    """
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        """Build: self.norm1, self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6);
        self.attn = Attention(dim, num_heads); self.mlp = Mlp(dim, int(dim * mlp_ratio));
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim)) with the Linear's weight and
        bias zero-initialised (adaLN-Zero).
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def modulation(self, c):
        """Compute the six modulation vectors from the conditioning.

        Args:
            c: (B, D).
        Returns:
            (shift_a, scale_a, gate_a, shift_m, scale_m, gate_m), each (B, D), = adaLN_modulation(c).chunk(6, dim=-1).
        """
        return self.adaLN_modulation(c).chunk(6, dim=-1)

    def forward(self, x, c):
        """adaLN-Zero block:
            x = x + gate_a[:, None] * attn(modulate(norm1(x), shift_a, scale_a))
            x = x + gate_m[:, None] * mlp (modulate(norm2(x), shift_m, scale_m))

        Args:
            x: (B, N, D) tokens. c: (B, D) conditioning.
        Returns:
            (B, N, D).
        """
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(c)
        x = x + gate_a[:, None] * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m[:, None] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x
