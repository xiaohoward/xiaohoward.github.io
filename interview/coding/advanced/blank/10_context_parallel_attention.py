"""Context (sequence) parallel attention: ring attention and Ulysses, with simulated ranks

Long-context video DiTs shard the token sequence across GPUs. Implement both standard schemes in one process, with
"rank r" = the r-th entry of a python list:
  (a) Ring attention. Q, K, V of shape (B, H, S, D) are split along S into N chunks. Rank r keeps its Q chunk for the
      whole computation and, over N steps, receives the K/V chunk that rank (r - step) mod N owns, sending its current
      K/V chunk to rank r+1 (a ring). Attention is accumulated with an online softmax (running max m, running
      denominator l, running numerator acc) so that no rank ever holds more than ONE K/V chunk at a time. Support a
      causal mask, implemented with the ORIGINAL token positions (so it stays correct with zig-zag assignment).
  (b) Zig-zag (load-balanced) sequence assignment for causal: split S into 2N pieces; rank r gets pieces r and
      2N-1-r, so every rank does the same amount of causal work.
  (c) Ulysses (DeepSpeed) attention: each rank starts with a sequence shard of all heads; an all-to-all turns it
      into a head shard of the full sequence; each rank runs ordinary full attention on its heads; an all-to-all
      turns the result back into a sequence shard.
Signatures:
    split_sequence(x, n_ranks, zigzag=False) -> (chunks, positions)
    merge_sequence(chunks, positions, S) -> full (B, H, S, D)
    ring_attention(q_chunks, k_chunks, v_chunks, positions, causal=False) -> (out_chunks, trace)
    all_to_all_seq_to_heads(chunks) -> list ;  all_to_all_heads_to_seq(chunks) -> list
    ulysses_attention(q_chunks, k_chunks, v_chunks, causal=False) -> out_chunks
Constraints: torch CPU. Softmax scale 1/sqrt(D). Use -inf masking with a finite running max initial value so fully
masked blocks never produce NaN. For Ulysses, H % N == 0 and S % N == 0. Reference: F.scaled_dot_product_attention.

Interview budget: 40 min

Discussion follow-ups:
  - Communication per step: ring attention moves K/V (2 * S/N * H * D per step, N-1 steps) vs Ulysses moving Q, K, V,
    O through all-to-all (4 * S/N * H * D per rank per layer). When is one better (H vs N, NVLink vs IB)?
  - Why does naive contiguous causal split leave rank N-1 doing N times the work of rank 0? Zig-zag / striped fix.
  - How does the online-softmax merge interact with the backward pass (recompute with saved logsumexp)?
  - For a 10B video DiT on 512 GPUs with 100k tokens per sample: how would you combine CP with TP, FSDP and pipeline?
"""
import math

import torch
import torch.nn.functional as F

__implement__ = [
    "split_sequence",
    "merge_sequence",
    "ring_attention",
    "all_to_all_seq_to_heads",
    "all_to_all_heads_to_seq",
    "ulysses_attention",
]


def make_qkv(B=2, H=4, S=32, D=8, seed=0):
    """Given helper: random Q, K, V of shape (B, H, S, D)."""
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, H, S, D, generator=g) for _ in range(3))


def reference_attention(q, k, v, causal=False):
    """Given helper: full attention via F.scaled_dot_product_attention."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def split_sequence(x, n_ranks, zigzag=False):
    """Split x (B, H, S, D) along S into n_ranks chunks and return (chunks, positions).

    Contiguous (zigzag=False): requires S % n_ranks == 0; rank r gets x[:, :, r*S/N:(r+1)*S/N].
    Zig-zag (zigzag=True): requires S % (2*n_ranks) == 0; split into 2N pieces of size S/(2N); rank r gets
      cat(piece r, piece 2N-1-r) along S.
    Returns:
        chunks: list of N tensors (B, H, S/N, D);
        positions: list of N 1-D long tensors of length S/N with the ORIGINAL sequence index of each token.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def merge_sequence(chunks, positions, S):
    """Inverse of split_sequence: scatter chunks back to their original positions. Returns (B, H, S, D)."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def _block_update(m, l, acc, q, k, v, mask):
    """Online-softmax update of (m, l, acc) with one K/V block. mask: (Sq, Sk) bool or None (True = attend)."""
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])          # (B, H, Sq, Sk)
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    m_new = torch.maximum(m, s.amax(-1, keepdim=True))
    alpha = torch.exp(m - m_new)
    p = torch.exp(s - m_new)
    l = alpha * l + p.sum(-1, keepdim=True)
    acc = alpha * acc + p @ v
    return m_new, l, acc


def ring_attention(q_chunks, k_chunks, v_chunks, positions, causal=False):
    """Ring attention over N simulated ranks.

    Args:
        q_chunks, k_chunks, v_chunks: lists of N tensors (B, H, S_r, D) (S_r may differ per rank).
        positions: list of N long tensors (S_r,) with original token indices (from split_sequence).
        causal: if True, query at position i attends only keys at positions j <= i (using `positions`).
    Algorithm: every rank initialises m = -1e30 (finite!), l = 0, acc = 0 of shapes (B,H,S_r,1)/(B,H,S_r,D). For step
      s = 0..N-1, rank r holds exactly one K/V chunk: the one from rank src = (r - s) mod N. It performs one
      online-softmax block update with it and then "passes" it on. Output = acc / l.
      Blocks that are entirely masked out must still be handled without NaN (they contribute nothing).
    Returns:
        out_chunks: list of N tensors (B, H, S_r, D) -- rank r's output for its own queries;
        trace: list of N lists; trace[r][s] is the int index of the K/V chunk rank r held at step s. Each inner list
        must have exactly N entries (one chunk per step) and be a permutation of range(N).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def all_to_all_seq_to_heads(chunks):
    """All-to-all: sequence-sharded -> head-sharded.

    Args:
        chunks: list of N tensors (B, H, S/N, D), rank r holds sequence shard r of all heads. H % N == 0.
    Returns:
        list of N tensors (B, H/N, S, D): rank r holds heads [r*H/N, (r+1)*H/N) for the FULL sequence, in order.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def all_to_all_heads_to_seq(chunks):
    """Inverse all-to-all: head-sharded (B, H/N, S, D) per rank -> sequence-sharded (B, H, S/N, D) per rank."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def ulysses_attention(q_chunks, k_chunks, v_chunks, causal=False):
    """Ulysses attention: all-to-all Q, K, V to head shards, full attention per rank, all-to-all the output back.

    Args:
        q_chunks, k_chunks, v_chunks: lists of N tensors (B, H, S/N, D) from a CONTIGUOUS split.
    Returns:
        list of N tensors (B, H, S/N, D): the attention output for each rank's sequence shard.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
