"""
11 — Scaled dot-product attention + multi-head split/merge

Implement attention as in "Attention Is All You Need", with an optional causal mask and an optional key
padding mask, plus the head split/merge used by multi-head attention.

Signatures:
    def softmax(x, dim=-1) -> Tensor                                  # numerically stable, no torch.softmax
    def build_mask(Lq, Lk, causal=False, key_padding_mask=None) -> BoolTensor or None   # True = may attend
    def scaled_dot_product_attention(q, k, v, attn_mask=None) -> (out, probs)
    def split_heads(x, num_heads) -> Tensor     (B, L, D) -> (B, H, L, D/H)
    def merge_heads(x) -> Tensor                (B, H, L, Dh) -> (B, L, H*Dh)
    def multi_head_attention(x_q, x_kv, Wq, Wk, Wv, Wo, num_heads, attn_mask=None) -> Tensor

Constraints: torch (CPU) only. Do NOT call F.scaled_dot_product_attention, nn.MultiheadAttention,
torch.softmax / F.softmax. Masks are boolean with True = attend (torch SDPA convention). Match
F.scaled_dot_product_attention to 1e-5 in float32.

Numerical stability note (say this out loud): softmax(x) = exp(x - max x) / sum exp(x - max x). Subtracting the
row max is exact (softmax is shift invariant) and prevents exp overflow; masked logits are set to -inf BEFORE the
max so exp(-inf) = 0 exactly. A row where every key is masked yields NaN (0/0) — the caller must avoid it (torch
does the same).

Interview budget: 25 min

Discussion follow-ups:
  * Cost: O(L^2 d) time and O(L^2) memory for the probs — FlashAttention avoids materialising them using online
    softmax (running max + running sum). Sketch the online-softmax update.
  * Why divide by sqrt(d)? (variance of q.k grows with d -> saturated softmax -> vanishing gradients.)
  * How do you get a causal mask to work with KV-cache decoding (query length 1, key length T)?
  * Multi-query / grouped-query attention: what changes in split_heads and why it helps decode bandwidth.
"""
import math
import torch

__implement__ = ["softmax", "build_mask", "scaled_dot_product_attention", "split_heads", "merge_heads",
                 "multi_head_attention"]


def softmax(x, dim=-1):
    """
    Numerically stable softmax along `dim`. x may contain -inf (treated as probability 0). Same shape as x.
    Must not call torch.softmax / F.softmax / torch.log_softmax. Rows that are entirely -inf give NaN (allowed).
    """
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=dim, keepdim=True)


def build_mask(Lq, Lk, causal=False, key_padding_mask=None):
    """
    Build a boolean attention mask with True = query may attend to key.

    Lq, Lk           : query / key lengths.
    causal           : if True, query i may only attend keys j with j <= i + (Lk - Lq)  (i.e. aligned to the END
                       of the key sequence, which is what you need for KV-cache decoding where Lq=1, Lk=T; for
                       Lq == Lk this is the usual lower-triangular mask).
    key_padding_mask : optional bool tensor (B, Lk) with True = this key is PADDING (must not be attended).

    Returns None if causal is False and key_padding_mask is None; otherwise a bool tensor broadcastable to
    (B, H, Lq, Lk): shape (1, 1, Lq, Lk) for causal only, (B, 1, 1, Lk) for padding only, (B, 1, Lq, Lk) for both.
    """
    mask = None
    if causal:
        offset = Lk - Lq
        idx_q = torch.arange(Lq).view(Lq, 1)
        idx_k = torch.arange(Lk).view(1, Lk)
        mask = (idx_k <= idx_q + offset).view(1, 1, Lq, Lk)
    if key_padding_mask is not None:
        pad = (~key_padding_mask).view(key_padding_mask.shape[0], 1, 1, -1)
        mask = pad if mask is None else (mask & pad)
    return mask


def scaled_dot_product_attention(q, k, v, attn_mask=None):
    """
    q : (..., Lq, d),  k : (..., Lk, d),  v : (..., Lk, dv).  Leading dims (e.g. B, H) broadcast.
    attn_mask : optional bool tensor broadcastable to (..., Lq, Lk); True = attend, False = masked out.

    Returns (out, probs): out (..., Lq, dv) = probs @ v, probs (..., Lq, Lk) = softmax(q k^T / sqrt(d)) with
    masked entries exactly 0. Masked logits must be set to -inf before the softmax (not after).
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    probs = softmax(scores, dim=-1)
    return probs @ v, probs


def split_heads(x, num_heads):
    """
    x : (B, L, D) with D % num_heads == 0. Returns (B, H, L, D//H): head h owns the contiguous feature slice
    x[..., h*Dh:(h+1)*Dh]. Raise ValueError if D is not divisible by num_heads.
    """
    B, L, D = x.shape
    if D % num_heads:
        raise ValueError(f"D={D} not divisible by num_heads={num_heads}")
    return x.view(B, L, num_heads, D // num_heads).transpose(1, 2)


def merge_heads(x):
    """
    x : (B, H, L, Dh). Returns (B, L, H*Dh), the exact inverse of split_heads (head h -> features h*Dh:(h+1)*Dh).
    Result must be contiguous.
    """
    B, H, L, Dh = x.shape
    return x.transpose(1, 2).reshape(B, L, H * Dh)


def multi_head_attention(x_q, x_kv, Wq, Wk, Wv, Wo, num_heads, attn_mask=None):
    """
    x_q : (B, Lq, D) queries' input;  x_kv : (B, Lk, D) keys/values' input (== x_q for self-attention).
    Wq, Wk, Wv, Wo : (D, D) weight matrices applied as x @ W (no biases).
    attn_mask : optional bool tensor broadcastable to (B, H, Lq, Lk), True = attend.

    Returns (B, Lq, D) = merge_heads(attention per head) @ Wo.
    """
    q = split_heads(x_q @ Wq, num_heads)
    k = split_heads(x_kv @ Wk, num_heads)
    v = split_heads(x_kv @ Wv, num_heads)
    out, _ = scaled_dot_product_attention(q, k, v, attn_mask)
    return merge_heads(out) @ Wo
