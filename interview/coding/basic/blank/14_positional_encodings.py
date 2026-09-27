"""
14 — Positional encodings: sinusoidal table and RoPE

Implement (a) the sinusoidal positional-encoding table from Vaswani et al. and (b) Rotary Position Embedding
(Su et al., RoFormer) applied to query/key tensors.

Signatures:
    def sinusoidal_table(max_len, d_model, base=10000.0) -> Tensor (max_len, d_model)
    def rope_cos_sin(positions, head_dim, base=10000.0) -> (cos, sin) each (..., head_dim)
    def apply_rope(x, positions, base=10000.0) -> Tensor  same shape as x

Convention (state it!): HALF-SPLIT (LLaMA / HF style). For head_dim = D, frequency i in [0, D/2) has
inv_freq_i = base^(-2i/D) and rotates the PAIR (x[i], x[i + D/2]) by angle m * inv_freq_i, i.e.
    out = x * cos + rotate_half(x) * sin,   rotate_half(x) = cat(-x2, x1) with x1, x2 = x.chunk(2, -1).
(The other common convention, interleaved pairs (x[2i], x[2i+1]) as in the original paper / GPT-NeoX-
"rotate every two", is equivalent up to a fixed permutation of the feature dim.)

Constraints: torch (CPU) only. head_dim must be even. Positions may be any integer (or float) tensor whose
shape broadcasts with x's sequence dim. Everything in float32; tests use 1e-5 tolerances.

Interview budget: 20 min

Discussion follow-ups:
  * Prove q_m . k_n depends only on m - n (2x2 rotation blocks: R_m^T R_n = R_{n-m}). Why does that make RoPE
    "relative" while still being applied absolutely — and why is it applied to q and k but NOT v?
  * Length extrapolation: what breaks beyond the training length? Position interpolation, NTK-aware scaling,
    YaRN — what does each change in inv_freq?
  * 2D/3D RoPE for images and video (split head_dim into axes; FLUX / WAN / Foveated Diffusion mixed-resolution
    tokens: what position do you give a coarse token?). Why sinusoidal tables are added and RoPE is multiplied.
  * Cost: cos/sin cache size, fusing RoPE into the attention kernel, fp16 precision at large positions
    (m * inv_freq loses precision -> compute angles in fp32).
"""
import torch

__implement__ = ["sinusoidal_table", "rope_cos_sin", "apply_rope"]


def sinusoidal_table(max_len, d_model, base=10000.0):
    """
    Returns PE of shape (max_len, d_model), float32, with (Vaswani et al. 2017)
        PE[pos, 2i]   = sin(pos / base^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / base^(2i/d_model))
    for pos in [0, max_len), i in [0, d_model/2). d_model must be even (raise ValueError otherwise).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def rope_cos_sin(positions, head_dim, base=10000.0):
    """
    positions : tensor of any shape (...,), integer or float positions m.
    head_dim  : D (even).
    Returns (cos, sin), each of shape (..., D), float32, where for i in [0, D/2):
        inv_freq[i] = base^(-2i/D),  angle[..., i] = positions * inv_freq[i],
        cos = cat(cos(angle), cos(angle)) along the last dim (and likewise sin), so that cos/sin line up with the
        half-split layout used by apply_rope. Raise ValueError if head_dim is odd.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def apply_rope(x, positions, base=10000.0):
    """
    x         : (..., L, D) queries or keys, e.g. (B, H, L, D). D even.
    positions : (L,) or any shape broadcastable to x.shape[:-1] (e.g. (B, 1, L) for per-sample offsets),
                giving the absolute position of each token along the sequence dim.
    Returns x rotated, same shape/dtype as x, using the half-split convention:
        x1, x2 = x[..., :D/2], x[..., D/2:]
        out = x * cos + cat(-x2, x1) * sin       with cos, sin from rope_cos_sin(positions, D).
    Properties the tests check: ||out|| == ||x|| per token; <apply_rope(q, m), apply_rope(k, n)> depends only
    on m - n.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
