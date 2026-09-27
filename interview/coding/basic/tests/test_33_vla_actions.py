import math
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_normalizer(mod):
    acts = mod.make_demo_actions(512, 7, seed=0)
    for mode in ("meanstd", "minmax"):
        nz = mod.ActionNormalizer(mode=mode, eps=1e-6).fit(acts)
        n = nz.normalize(acts)
        assert n.shape == acts.shape
        back = nz.unnormalize(n)
        assert torch.allclose(back, acts, atol=1e-4), f"{mode}: unnormalize(normalize(a)) must invert (max err {(back - acts).abs().max():.2e})"
        if mode == "meanstd":
            assert torch.allclose(n[:, :6].mean(0), torch.zeros(6), atol=1e-4), "meanstd: normalized mean must be 0"
            assert torch.allclose(n[:, :6].std(0, unbiased=False), torch.ones(6), atol=1e-3), "meanstd: normalized std must be 1"
            assert torch.allclose(n[:, 6], torch.zeros(512)), "constant dim must normalize to 0 (eps keeps it finite)"
        else:
            assert n.min() >= -1 - 1e-5 and n.max() <= 1 + 1e-5, "minmax: normalized must lie in [-1, 1]"
            assert torch.allclose(n[:, :6].min(0).values, -torch.ones(6), atol=1e-4)
            assert torch.allclose(n[:, :6].max(0).values, torch.ones(6), atol=1e-4)
            assert torch.allclose(n[:, 6], -torch.ones(512)), "constant dim must normalize to -1 under minmax"
        # batched / leading dims broadcast
        x = acts[:8].view(2, 4, 7)
        assert torch.allclose(nz.normalize(x), n[:8].view(2, 4, 7))
        # stats are from the fit set, not the batch
        assert torch.allclose(nz.normalize(acts[:1]), n[:1])


def check_binning(mod):
    g = torch.Generator().manual_seed(1)
    a = torch.rand(1000, 7, generator=g) * 2 - 1
    tok = mod.tokenize_actions(a, 256)
    assert tok.dtype == torch.int64 and tok.shape == a.shape
    assert tok.min() >= 0 and tok.max() <= 255
    err = (mod.detokenize_actions(tok, 256) - a).abs().max().item()
    assert err <= 1 / 256 + 1e-6, f"round-trip error {err:.4e} exceeds half a bin width {1 / 256:.4e}"
    # edges and clipping
    edge = torch.tensor([-2.0, -1.0, -1 + 1e-7, 0.0, 1 - 1e-7, 1.0, 3.0])
    te = mod.tokenize_actions(edge, 256)
    assert te.tolist() == [0, 0, 0, 128, 255, 255, 255], f"edge tokens {te.tolist()}"
    # monotone
    s = torch.linspace(-1.5, 1.5, 4001)
    ts = mod.tokenize_actions(s, 256)
    assert bool((ts[1:] >= ts[:-1]).all()), "tokenization must be monotone in the action value"
    assert len(torch.unique(ts)) == 256, "a dense sweep must hit every one of the 256 bins"
    # de-tokenize gives centers: bin 0 -> -1 + 1/256, bin 255 -> 1 - 1/256
    c = mod.detokenize_actions(torch.tensor([0, 255, 128]), 256)
    assert torch.allclose(c, torch.tensor([-1 + 1 / 256, 1 - 1 / 256, 1 / 256]), atol=1e-6)
    # other bin counts
    assert torch.allclose(mod.detokenize_actions(mod.tokenize_actions(a, 16), 16), a, atol=1 / 16 + 1e-6)


def check_chunking(mod):
    T, D = 20, 3
    traj = torch.arange(T * D, dtype=torch.float32).view(T, D)
    for H, s in ((5, 1), (5, 3), (4, 4), (20, 1), (6, 7)):
        chunks, starts = mod.chunk_trajectory(traj, H, s)
        N = (T - H) // s + 1
        assert chunks.shape == (N, H, D), f"H={H}, s={s}: chunks {tuple(chunks.shape)} != {(N, H, D)}"
        assert starts.tolist() == [i * s for i in range(N)]
        for i in range(N):
            assert torch.equal(chunks[i], traj[i * s: i * s + H]), f"chunk {i} content mismatch"
    # stride 1 covers every timestep: count of chunks covering t
    chunks, starts = mod.chunk_trajectory(traj, 5, 1)
    for t in range(T):
        cov = ((starts <= t) & (t < starts + 5)).sum().item()
        assert cov == min(t + 1, 5, T - t, T - 4), f"coverage at t={t} is {cov}"


def check_temporal_ensemble(mod):
    # 3 chunks of H=3, D=2; at t=2 all three cover it with ages 2, 1, 0
    preds = torch.tensor([[[1., 1.], [2., 2.], [3., 3.]],
                          [[10., 0.], [20., 0.], [30., 0.]],
                          [[100., -1.], [200., -1.], [300., -1.]]])
    starts = torch.tensor([0, 1, 2])
    m = 0.5
    w = torch.tensor([math.exp(-m * 2), math.exp(-m * 1), 1.0])
    w = w / w.sum()
    ref = w[0] * preds[0, 2] + w[1] * preds[1, 1] + w[2] * preds[2, 0]
    out = mod.temporal_ensemble(preds, starts, 2, m)
    assert out.shape == (2,)
    assert torch.allclose(out, ref, atol=1e-6), f"ensemble at t=2: {out.tolist()} != {ref.tolist()}"
    # t=0: only chunk 0 covers -> its first action
    assert torch.allclose(mod.temporal_ensemble(preds, starts, 0, m), preds[0, 0])
    # t=3: chunks 1 (age 2) and 2 (age 1)
    w2 = torch.tensor([math.exp(-2 * m), math.exp(-m)]); w2 = w2 / w2.sum()
    assert torch.allclose(mod.temporal_ensemble(preds, starts, 3, m), w2[0] * preds[1, 2] + w2[1] * preds[2, 1], atol=1e-6)
    # m=0 is a plain mean
    assert torch.allclose(mod.temporal_ensemble(preds, starts, 2, 0.0), (preds[0, 2] + preds[1, 1] + preds[2, 0]) / 3)
    # variance reduction: noisy chunks around a true trajectory
    g = torch.Generator().manual_seed(0)
    T, H, D = 40, 8, 4
    true = torch.sin(torch.arange(T, dtype=torch.float32)[:, None] * 0.3 + torch.arange(D)[None, :])
    chunks, starts = mod.chunk_trajectory(true, H, 1)
    err_ens, err_single = 0.0, 0.0
    trials = 30
    for _ in range(trials):
        noisy = chunks + 0.5 * torch.randn(chunks.shape, generator=g)
        for t in range(H - 1, T - H + 1):
            ens = mod.temporal_ensemble(noisy, starts, t, 0.1)
            newest = noisy[t, 0]                       # chunk predicted at t, age 0
            err_ens += ((ens - true[t]) ** 2).sum().item()
            err_single += ((newest - true[t]) ** 2).sum().item()
    assert err_ens < 0.35 * err_single, f"ensembling should cut squared error of noisy predictions (ens {err_ens:.1f} vs single {err_single:.1f})"


def check_executor(mod):
    H, n_exec, D = 6, 2, 3
    calls = []

    def policy(obs):
        calls.append(float(obs))
        return obs + torch.arange(H, dtype=torch.float32)[:, None] * torch.ones(1, D)   # chunk[j] = obs + j

    ex = mod.RecedingHorizonExecutor(policy, H, n_exec)
    out = [ex.act(torch.tensor(float(10 * t))) for t in range(7)]
    out = torch.stack(out)
    # t=0: query at obs 0 -> emit 0,1 ; t=2: query at obs 20 -> 20,21 ; t=4: obs 40 -> 40,41 ; t=6: obs 60 -> 60
    ref = torch.tensor([0., 1., 20., 21., 40., 41., 60.])[:, None].expand(7, D)
    assert torch.allclose(out, ref), f"executor sequence {out[:, 0].tolist()} != {ref[:, 0].tolist()}"
    assert calls == [0.0, 20.0, 40.0, 60.0], f"policy must be queried every n_exec steps, got obs {calls}"
    assert ex.n_calls == 4
    # n_exec = H: open loop over the whole chunk
    calls.clear()
    ex = mod.RecedingHorizonExecutor(policy, H, H)
    out = torch.stack([ex.act(torch.tensor(float(t))) for t in range(H + 1)])
    assert torch.allclose(out[:, 0], torch.tensor([0., 1., 2., 3., 4., 5., 6.])) and calls == [0.0, 6.0]


def run(mod):
    check_normalizer(mod);         print("  ok  action normalizer (meanstd / minmax) round-trips, stats from demo set")
    check_binning(mod);            print("  ok  256-bin action tokens: error <= half bin, monotone, clipped edges")
    check_chunking(mod);           print("  ok  chunk shapes, contents and coverage")
    check_temporal_ensemble(mod);  print("  ok  temporal ensemble = hand-computed weighted average; reduces noise")
    check_executor(mod);           print("  ok  receding-horizon executor emits the right sequence")
