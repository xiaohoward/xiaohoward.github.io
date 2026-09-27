import math

import numpy as np
import torch


def check_standardization(mod):
    chunks = mod.make_chunks(n_chunks=4, T=64, C=16)
    ref = torch.cat([c["gt"] for c in chunks])
    mean, std = mod.channel_stats(ref)
    assert mean.shape == (16,) and std.shape == (16,)
    assert torch.allclose(mean, ref.mean(0), atol=1e-6) and torch.allclose(std, ref.std(0, unbiased=False), atol=1e-6)
    z = mod.standardize(ref, mean, std)
    assert z.shape == ref.shape
    assert torch.allclose(z.mean(0), torch.zeros(16), atol=1e-5) and torch.allclose(z.std(0, unbiased=False), torch.ones(16), atol=1e-4), \
        "standardized reference must have zero mean / unit std per channel"
    # zero-std channel must not produce inf/nan
    x = torch.ones(5, 3)
    z0 = mod.standardize(x, torch.ones(3), torch.zeros(3))
    assert torch.isfinite(z0).all() and torch.all(z0 == 0)


def check_surprise(mod):
    chunks = mod.make_chunks(n_chunks=2, T=64, C=16, seed=1)
    ref = torch.cat([c["imagined"] for c in chunks])
    mean, std = mod.channel_stats(ref)
    c = chunks[0]
    s = mod.surprise_scores(c["imagined"], c["gt"], mean, std)
    assert s.shape == (64,) and (s >= 0).all()
    ref_s = ((c["gt"] - c["imagined"]) / std.clamp_min(1e-6)).norm(dim=-1)
    assert torch.allclose(s, ref_s, atol=1e-5), "surprise must be the L2 distance in standardized space"
    assert torch.all(mod.surprise_scores(c["gt"], c["gt"], mean, std) == 0), "identical latents -> zero surprise"
    # standardization matters: scale-invariant per channel
    s2 = mod.surprise_scores(c["imagined"] * 10, c["gt"] * 10, mean * 10, std * 10)
    assert torch.allclose(s, s2, atol=1e-4), "surprise must be invariant to per-channel rescaling of the latent space"


def check_topk(mod):
    g = torch.Generator().manual_seed(0)
    s = torch.rand(50, generator=g)
    for frac in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        m = mod.topk_mask(s, frac)
        k = math.ceil(frac * 50)
        assert m.dtype == torch.bool and m.shape == (50,)
        assert int(m.sum()) == k, f"frac={frac}: kept {int(m.sum())}, expected {k}"
        if 0 < k < 50:
            assert s[m].min() >= s[~m].max(), "kept tokens must have the highest scores"
    # ties: lower index wins
    m = mod.topk_mask(torch.ones(10), 0.3)
    assert m.tolist() == [True] * 3 + [False] * 7, f"tie-break must keep lower indices: {m.tolist()}"
    assert int(mod.topk_mask(torch.zeros(7), 0.5).sum()) == 4, "all-equal scores still keep exactly ceil(frac T)"


def check_threshold(mod):
    s = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert mod.threshold_mask(s, 2.0).tolist() == [False, False, True, True], "threshold keeps scores >= thr"
    assert mod.threshold_mask(torch.full((5,), 1.0), 1.0).all(), "all equal at the threshold -> all kept"
    assert not mod.threshold_mask(torch.full((5,), 1.0), 1.0001).any()


def check_running_quantile(mod):
    chunks = mod.make_chunks(n_chunks=5, T=64, C=16, seed=2)
    ref = torch.cat([c["imagined"] for c in chunks])
    mean, std = mod.channel_stats(ref)
    for q in (0.3, 0.5, 0.8):
        gate = mod.RunningQuantileGate(q=q)
        pool = []
        for c in chunks:
            s = mod.surprise_scores(c["imagined"], c["gt"], mean, std)
            keep = gate.update(s)
            pool.append(s.numpy())
            thr = np.quantile(np.concatenate(pool), q)
            assert abs(gate.last_threshold - thr) < 1e-4, f"q={q}: threshold {gate.last_threshold} != np.quantile {thr}"
            assert keep.dtype == torch.bool and keep.shape == (64,)
            assert torch.equal(keep, s >= thr), "keep must be scores >= running quantile of the cumulative pool"
        assert gate.pool.numel() == 5 * 64, "pool must accumulate every score"
    # fraction kept tracks 1 - q in the long run
    gate = mod.RunningQuantileGate(q=0.75)
    kept = 0
    for c in chunks:
        kept += int(gate.update(mod.surprise_scores(c["imagined"], c["gt"], mean, std)).sum())
    frac = kept / (5 * 64)
    assert 0.15 < frac < 0.4, f"with q=0.75 roughly 25% should be kept, got {frac}"
    # degenerate: all-equal scores -> everything kept (score == quantile)
    gate = mod.RunningQuantileGate(q=0.9)
    assert gate.update(torch.full((8,), 2.0)).all()
    # single token pool
    gate = mod.RunningQuantileGate(q=0.5)
    assert gate.update(torch.tensor([3.0])).tolist() == [True]


def check_combine_and_cache(mod):
    chunks = mod.make_chunks(n_chunks=1, T=32, C=8, H=4, Dh=8, seed=3)
    c = chunks[0]
    mean, std = mod.channel_stats(c["imagined"])
    s = mod.surprise_scores(c["imagined"], c["gt"], mean, std)
    m_s = mod.topk_mask(s, 0.5)
    m_a = mod.threshold_mask(c["attn"], 0.5)
    both = mod.combine_and(m_s, m_a)
    assert torch.equal(both, m_s & m_a), "AND combine wrong"
    assert int(both.sum()) <= min(int(m_s.sum()), int(m_a.sum()))
    before = {k: v.clone() for k, v in c["cache"].items()}
    gated = mod.gate_kv_cache(c["cache"], both)
    n = int(both.sum())
    assert gated["k"].shape == (4, n, 8) and gated["v"].shape == (4, n, 8) and gated["pos"].shape == (n,), \
        f"gated cache shapes wrong: {[t.shape for t in gated.values()]}"
    assert torch.equal(gated["pos"], torch.arange(32)[both]), "positions must be the original indices of kept tokens"
    assert torch.equal(gated["k"], c["cache"]["k"][:, both]) and torch.equal(gated["v"], c["cache"]["v"][:, both])
    assert all(torch.equal(before[k], c["cache"][k]) for k in before), "input cache must not be modified"
    assert mod.kv_size(c["cache"]) == 2 * 4 * 32 * 8 and mod.kv_size(gated) == 2 * 4 * n * 8
    assert isinstance(mod.kv_size(gated), int)
    # keep-all and keep-none
    assert mod.kv_size(mod.gate_kv_cache(c["cache"], torch.ones(32, dtype=torch.bool))) == mod.kv_size(c["cache"])
    empty = mod.gate_kv_cache(c["cache"], torch.zeros(32, dtype=torch.bool))
    assert empty["k"].shape == (4, 0, 8) and mod.kv_size(empty) == 0


def check_end_to_end_reduction(mod):
    # the pipeline on the synthetic chunks: surprise AND attention with running quantile gives a ~40-60% reduction
    chunks = mod.make_chunks(n_chunks=6, T=64, C=16, seed=4)
    ref = torch.cat([c["imagined"] for c in chunks])
    mean, std = mod.channel_stats(ref)
    gate = mod.RunningQuantileGate(q=0.5)
    total, kept = 0, 0
    for c in chunks:
        s = mod.surprise_scores(c["imagined"], c["gt"], mean, std)
        keep = mod.combine_and(gate.update(s), mod.threshold_mask(c["attn"], 0.2))
        g = mod.gate_kv_cache(c["cache"], keep)
        total += mod.kv_size(c["cache"])
        kept += mod.kv_size(g)
    red = 1 - kept / total
    assert 0.4 < red < 0.75, f"expected a 40-75% KV reduction on the synthetic data, got {red:.2f}"


def run(mod):
    check_standardization(mod); print("  ok  channel stats / standardization")
    check_surprise(mod); print("  ok  surprise scores")
    check_topk(mod); print("  ok  top-k keeps exact count (ties -> lower index)")
    check_threshold(mod); print("  ok  fixed threshold")
    check_running_quantile(mod); print("  ok  running quantile matches np.quantile of the cumulative pool")
    check_combine_and_cache(mod); print("  ok  AND combine and KV cache gating")
    check_end_to_end_reduction(mod); print("  ok  end-to-end KV reduction")
