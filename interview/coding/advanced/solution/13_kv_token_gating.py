"""KV-cache compression by surprise: token gating for an autoregressive world(-action) model

During an autoregressive video rollout a world model "imagines" the next chunk of latents, then observes the real
chunk. Tokens whose imagined latent was already right carry little new information; the KV cache can drop them.
Implement the gating pipeline on per-chunk latents of shape (T, C) (T tokens, C VAE channels):
  (a) channel_stats(reference) -> per-channel mean / std over a reference set; standardize(x, mean, std).
  (b) surprise_scores(imagined, gt, mean, std) -> (T,) L2 distance between standardized imagined and gt tokens.
  (c) Keep masks: topk_mask(scores, frac) keeps the ceil(frac * T) highest-surprise tokens (ties -> lower index);
      threshold_mask(scores, thr) keeps scores >= thr; RunningQuantileGate(q): a cumulative pool of every score seen
      so far across chunks; a token is kept iff its score >= the q-quantile of the pool INCLUDING the current chunk
      (torch.quantile / np.quantile default linear interpolation). Distribution-free, self-calibrating.
  (d) combine_and(mask_a, mask_b) for combining surprise with a second gate (e.g. attention from the action expert).
  (e) gate_kv_cache(cache, keep) -> new cache with dropped tokens removed from k, v (shape (H, T, Dh)) and pos (T,);
      kv_size(cache) -> number of elements in k + v.
Signatures:
    channel_stats(reference) -> (mean (C,), std (C,)) ; standardize(x, mean, std) -> (T, C)
    surprise_scores(imagined, gt, mean, std) -> (T,)
    topk_mask(scores, frac) -> bool (T,) ; threshold_mask(scores, thr) -> bool (T,)
    RunningQuantileGate.update(scores) -> bool (T,)
    combine_and(a, b) -> bool (T,) ; gate_kv_cache(cache, keep) -> dict ; kv_size(cache) -> int
Constraints: torch CPU, float32; scores are 1-D. std uses the unbiased=False (population) estimator; guard zero std by
max(std, 1e-6). topk with frac = 0 keeps nothing, frac = 1 keeps all.

Interview budget: 30 min

Discussion follow-ups:
  - Why running quantile instead of a z-score / median threshold? (No Gaussian assumption; the surprise distribution
    drifts as the rollout diverges; a fixed threshold is model- and scene-specific.) What is the memory cost of the pool
    and how would you bound it (reservoir sampling, t-digest, exponential decay)?
  - What does the AND with action-expert attention protect against? Why is high-surprise-but-unattended a token you
    can still drop? What happens to RoPE / positions after dropping tokens?
  - Evaluation confounds: paired seeds, flash-attention nondeterminism, "failure episodes are easy to predict" -- how
    would you design the ablation so a PSNR change is attributable to gating?
  - How does this compare to H2O / StreamingLLM / TOVA style KV eviction for language models?
"""
import math

import torch

__implement__ = [
    "channel_stats",
    "standardize",
    "surprise_scores",
    "topk_mask",
    "threshold_mask",
    "RunningQuantileGate.update",
    "combine_and",
    "gate_kv_cache",
    "kv_size",
]


def make_chunks(n_chunks=5, T=64, C=16, H=4, Dh=8, seed=0):
    """Given helper: list of dicts with 'imagined' (T, C), 'gt' (T, C), 'attn' (T,) in [0,1] and a KV cache
    {'k': (H, T, Dh), 'v': (H, T, Dh), 'pos': (T,) long}. Ground truth = imagined + structured error (a subset of tokens
    is very wrong) so that surprise is informative."""
    g = torch.Generator().manual_seed(seed)
    chunks = []
    scale = torch.rand(C, generator=g) * 3 + 0.5
    shift = torch.randn(C, generator=g) * 2
    for i in range(n_chunks):
        imagined = torch.randn(T, C, generator=g) * scale + shift
        err = 0.1 * torch.randn(T, C, generator=g) * scale
        wrong = torch.rand(T, generator=g) < 0.25
        err[wrong] += (1.5 + i * 0.3) * torch.randn(int(wrong.sum()), C, generator=g) * scale
        chunks.append({
            "imagined": imagined,
            "gt": imagined + err,
            "attn": torch.rand(T, generator=g),
            "cache": {"k": torch.randn(H, T, Dh, generator=g), "v": torch.randn(H, T, Dh, generator=g),
                      "pos": torch.arange(i * T, (i + 1) * T)},
        })
    return chunks


def channel_stats(reference):
    """Per-channel mean and std of reference latents (N, C) (population std, unbiased=False). Returns ((C,), (C,))."""
    return reference.mean(0), reference.std(0, unbiased=False)


def standardize(x, mean, std):
    """(x - mean) / max(std, 1e-6), broadcast over the last dim. x: (T, C) -> (T, C)."""
    return (x - mean) / std.clamp_min(1e-6)


def surprise_scores(imagined, gt, mean, std):
    """Per-token surprise = ||standardize(gt) - standardize(imagined)||_2 over channels. (T, C), (T, C) -> (T,)."""
    return (standardize(gt, mean, std) - standardize(imagined, mean, std)).norm(dim=-1)


def topk_mask(scores, frac):
    """Keep exactly k = ceil(frac * T) tokens with the highest scores (frac in [0, 1]).

    Ties are broken towards the LOWER index (use a stable descending sort: argsort(-scores, stable=True)).
    Returns a bool tensor (T,) with mask.sum() == k.
    """
    T = scores.numel()
    k = int(math.ceil(frac * T))
    mask = torch.zeros(T, dtype=torch.bool)
    if k > 0:
        order = torch.argsort(-scores, stable=True)
        mask[order[:k]] = True
    return mask


def threshold_mask(scores, thr):
    """Keep tokens with scores >= thr. Returns bool (T,)."""
    return scores >= thr


class RunningQuantileGate:
    """Running-quantile gate. Attributes: q (float in [0, 1]), pool (1-D tensor of all scores seen so far, starts
    empty), last_threshold (float or None)."""

    def __init__(self, q=0.5):
        self.q = q
        self.pool = torch.empty(0)
        self.last_threshold = None

    def update(self, scores):
        """Append `scores` (T,) to the pool, set last_threshold = quantile(pool, q) (linear interpolation, same as
        np.quantile default), and return keep = scores >= last_threshold (bool (T,))."""
        self.pool = torch.cat([self.pool, scores.reshape(-1).float()])
        self.last_threshold = float(torch.quantile(self.pool, self.q))
        return scores >= self.last_threshold


def combine_and(a, b):
    """Elementwise AND of two bool masks of equal length (a token survives only if BOTH gates keep it)."""
    assert a.shape == b.shape
    return a & b


def gate_kv_cache(cache, keep):
    """Drop tokens where keep is False.

    Args:
        cache: {'k': (H, T, Dh), 'v': (H, T, Dh), 'pos': (T,)}. keep: bool (T,).
    Returns:
        a NEW dict with 'k', 'v' of shape (H, T_keep, Dh) and 'pos' (T_keep,) (original positions of kept tokens, in
        the original order). The input cache is not modified.
    """
    return {"k": cache["k"][:, keep], "v": cache["v"][:, keep], "pos": cache["pos"][keep]}


def kv_size(cache):
    """Number of stored elements: k.numel() + v.numel() (python int)."""
    return int(cache["k"].numel() + cache["v"].numel())
