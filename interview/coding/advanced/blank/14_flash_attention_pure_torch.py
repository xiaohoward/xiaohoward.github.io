"""FlashAttention in pure PyTorch: tiled forward with online softmax, and backward by recomputation

Write attention that never materialises the (n x n) score matrix. Q, K, V have shape (B, H, N, D); the softmax scale
is 1/sqrt(D). Tile queries in blocks of Br rows and keys/values in blocks of Bc columns:
  Forward: for every Q block keep a running row max m, running row sum l and an unnormalised accumulator acc. For every
    K/V block compute the (Br x Bc) scores, update (m, l, acc) with the online-softmax rescaling, and at the end write
    O = acc / l and the row logsumexp L = m + log(l). Support a causal mask (skip tiles entirely above the diagonal,
    mask partial tiles) and N not divisible by Br / Bc.
  Backward (FlashAttention-2 style): given dO, O, L (no saved probabilities): Delta_i = rowsum(dO_i * O_i). For every
    K/V block j and Q block i: recompute S = Q_i K_j^T * scale, P = exp(S - L_i); dV_j += P^T dO_i; dP = dO_i V_j^T;
    dS = P * (dP - Delta_i); dQ_i += dS K_j * scale; dK_j += dS^T Q_i * scale.
Signatures:
    flash_attention_forward(q, k, v, Br, Bc, causal=False) -> (o, lse)   o (B,H,N,D), lse (B,H,N) natural log
    flash_attention_backward(q, k, v, o, lse, do, Br, Bc, causal=False) -> (dq, dk, dv)
Constraints: torch CPU, float32, no autograd inside (the backward is manual). Never create a tensor with its last two
dims equal to (N, N); the largest intermediate is (B, H, Br, Bc). Reference: F.scaled_dot_product_attention.

Interview budget: 45 min

Discussion follow-ups:
  - Why does storing the logsumexp (N floats per head) instead of P (N^2) make the backward cheaper in memory but ~2x
    the FLOPs? What is the arithmetic intensity argument (SRAM vs HBM) that makes this a net win on GPU?
  - How do you pick Br, Bc on an H100 (SRAM 228 KB/SM, head dim 64/128, register pressure)? Why is FA-2's loop order
    (outer over Q blocks in the forward, outer over K/V blocks in the backward) chosen, and how is work split across
    warps? What does FA-3 add (warp specialisation, FP8, async wgmma)?
  - Extending to a block-sparse / causal / sliding-window mask: which tiles are skipped and what speedup follows?
  - Numerics: why is m initialised to -inf (or a large negative finite value) and what goes wrong with bf16 for l?
"""
import math

import torch
import torch.nn.functional as F

__implement__ = ["flash_attention_forward", "flash_attention_backward"]


def make_qkv(B=1, H=2, N=64, D=16, seed=0):
    """Given helper: random Q, K, V of shape (B, H, N, D), float32."""
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, H, N, D, generator=g) for _ in range(3))


def reference_attention(q, k, v, causal=False):
    """Given helper: dense reference via F.scaled_dot_product_attention."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def reference_lse(q, k, causal=False):
    """Given helper: dense row logsumexp of the scaled scores, (B, H, N)."""
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), float("-inf"))
    return torch.logsumexp(s, dim=-1)


def _tile_mask(i0, i1, j0, j1):
    """Causal mask for query rows [i0, i1) x key cols [j0, j1): True where col <= row."""
    rows = torch.arange(i0, i1)[:, None]
    cols = torch.arange(j0, j1)[None, :]
    return cols <= rows


def flash_attention_forward(q, k, v, Br, Bc, causal=False):
    """Tiled attention forward with online softmax.

    Args:
        q, k, v: (B, H, N, D) float32. Br, Bc: query / key block sizes (need not divide N).
        causal: query i attends keys j <= i.
    Returns:
        o: (B, H, N, D) attention output; lse: (B, H, N) row logsumexp of the scaled (and masked) scores, natural log.
    Implementation notes: initialise m = -inf as a finite large negative (-1e30) or handle exp(-inf - -inf); for the
    causal case skip K/V tiles with j0 > i1 - 1 and mask tiles that straddle the diagonal. Do not build any (N, N)
    tensor; the biggest intermediate is (B, H, Br, Bc).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def flash_attention_backward(q, k, v, o, lse, do, Br, Bc, causal=False):
    """Tiled attention backward by recomputation from the saved logsumexp.

    Args:
        q, k, v, o, do: (B, H, N, D) float32; lse: (B, H, N) from flash_attention_forward.
    Returns:
        (dq, dk, dv): gradients of sum(o * do) w.r.t. q, k, v, each (B, H, N, D).
    Delta = (do * o).sum(-1) (B, H, N). Outer loop over K/V blocks j (accumulate dK_j, dV_j in registers), inner loop
    over Q blocks i (accumulate dQ_i into the output tensor). Same causal tile skipping / masking as the forward.
    No (N, N) tensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
