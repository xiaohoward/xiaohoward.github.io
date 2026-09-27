import numpy as np
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_condition_dropout(mod):
    B, L, D = 20_000, 4, 8
    g = torch.Generator().manual_seed(0)
    cond = torch.randn(B, L, D, generator=g)
    null = torch.full((L, D), -7.0)
    cond_copy = cond.clone()
    for p in (0.1, 0.5):
        g = torch.Generator().manual_seed(1)
        out, mask = mod.condition_dropout(cond, null, p, g)
        assert out.shape == cond.shape and mask.shape == (B,) and mask.dtype == torch.bool, "shapes / mask dtype"
        assert torch.equal(cond, cond_copy), "cond must not be modified in place"
        frac = mask.float().mean().item()
        assert abs(frac - p) < 0.012, f"dropout frequency {frac:.3f} != p={p}"
        assert torch.equal(out[mask], null.expand(int(mask.sum()), L, D)), "dropped rows must equal null exactly"
        assert torch.equal(out[~mask], cond[~mask]), "kept rows must equal cond exactly"
        # whole-sample dropout: every element in a row shares the decision
        is_null_row = (out == -7.0).reshape(B, -1).all(dim=1)
        assert torch.equal(is_null_row, mask)
        # mask must come from torch.rand(B, generator) < p: reproduce with the same seed
        g2 = torch.Generator().manual_seed(1)
        assert torch.equal(mask, torch.rand(B, generator=g2) < p), "mask must be torch.rand(B, generator=g) < p"
    out0, m0 = mod.condition_dropout(cond, null, 0.0, torch.Generator().manual_seed(3))
    assert torch.equal(out0, cond) and not m0.any(), "p = 0 must keep everything"
    out1, m1 = mod.condition_dropout(cond, null, 1.0, torch.Generator().manual_seed(3))
    assert m1.all() and torch.equal(out1, null.expand_as(cond)), "p = 1 must drop everything"
    # different generators give different masks; same seed gives the same mask
    _, ma = mod.condition_dropout(cond, null, 0.5, torch.Generator().manual_seed(5))
    _, mb = mod.condition_dropout(cond, null, 0.5, torch.Generator().manual_seed(6))
    _, mc = mod.condition_dropout(cond, null, 0.5, torch.Generator().manual_seed(5))
    assert not torch.equal(ma, mb) and torch.equal(ma, mc)
    # 1-D condition (class embedding) also works
    outc, _ = mod.condition_dropout(torch.randn(16, 5), torch.zeros(5), 0.5, torch.Generator().manual_seed(0))
    assert outc.shape == (16, 5)


def check_combine(mod):
    torch.manual_seed(0)
    u = torch.randn(3, 2, 4, 4)
    c = torch.randn(3, 2, 4, 4)
    assert torch.allclose(mod.cfg_combine(u, c, 1.0), c, atol=1e-6), "scale 1 must give cond"
    assert torch.equal(mod.cfg_combine(u, c, 0.0), u), "scale 0 must give uncond exactly"
    assert torch.allclose(mod.cfg_combine(u, c, 7.5), u + 7.5 * (c - u), atol=1e-6)
    assert torch.allclose(mod.cfg_combine(u, c, 7.5), 7.5 * c - 6.5 * u, atol=1e-5), "w c - (w-1) u form"
    # linear in scale
    assert torch.allclose(mod.cfg_combine(u, c, 2.0) - mod.cfg_combine(u, c, 1.0), c - u, atol=1e-6)
    # per-sample scale
    w = torch.tensor([0.0, 1.0, 3.0])
    out = mod.cfg_combine(u, c, w)
    assert torch.equal(out[0], u[0]) and torch.allclose(out[1], c[1], atol=1e-6) and torch.allclose(out[2], u[2] + 3 * (c[2] - u[2]), atol=1e-6)


def check_rescale(mod):
    torch.manual_seed(1)
    u = torch.randn(4, 3, 8, 8)
    c = torch.randn(4, 3, 8, 8) * 1.5 + 0.2
    guided = mod.cfg_combine(u, c, 9.0)
    assert guided.reshape(4, -1).std(dim=1).min().item() > c.reshape(4, -1).std(dim=1).max().item(), \
        "(sanity) high-scale CFG inflates the std"
    r0 = mod.cfg_rescale(guided, c, 0.0)
    assert torch.equal(r0, guided) or torch.allclose(r0, guided, atol=1e-6), "phi = 0 must return plain CFG"
    r1 = mod.cfg_rescale(guided, c, 1.0)
    assert r1.shape == guided.shape
    assert torch.allclose(r1.reshape(4, -1).std(dim=1, unbiased=False), c.reshape(4, -1).std(dim=1, unbiased=False), rtol=1e-4), \
        "phi = 1 must give a per-sample std equal to std(cond)"
    # rescale is a per-sample scalar multiplication (direction unchanged)
    ratio = r1 / guided
    assert torch.allclose(ratio.reshape(4, -1), ratio.reshape(4, -1)[:, :1].expand(-1, 3 * 64), rtol=1e-4), \
        "rescale must multiply each sample by a single scalar"
    # phi = 0.7 is a convex blend
    r7 = mod.cfg_rescale(guided, c, 0.7)
    assert torch.allclose(r7, 0.7 * r1 + 0.3 * guided, atol=1e-5)
    # per-sample: rescale factor of sample b depends only on sample b
    r1b = mod.cfg_rescale(guided[1:2], c[1:2], 1.0)
    assert torch.allclose(r1b[0], r1[1], atol=1e-6)


def check_interval(mod):
    torch.manual_seed(2)
    u = torch.randn(5, 6)
    c = torch.randn(5, 6)
    t = torch.tensor([0.05, 0.2, 0.5, 0.8, 0.95])
    out = mod.cfg_with_interval(u, c, 4.0, t, 0.2, 0.8)
    full = mod.cfg_combine(u, c, 4.0)
    for b in (1, 2, 3):
        assert torch.allclose(out[b], full[b], atol=1e-6), f"t={t[b]} inside [lo, hi] must be guided (inclusive bounds)"
    for b in (0, 4):
        assert torch.allclose(out[b], c[b], atol=1e-6), f"t={t[b]:.2f} outside the interval must return the plain cond output"
    assert torch.allclose(mod.cfg_with_interval(u, c, 4.0, t, 0.0, 1.0), full, atol=1e-6), "interval [0,1] = plain CFG"
    assert torch.allclose(mod.cfg_with_interval(u, c, 4.0, t, 2.0, 3.0), c, atol=1e-6), "empty interval = cond everywhere"


def check_batched_forward(mod):
    torch.manual_seed(3)
    B, D, Dc = 5, 8, 6
    model = mod.TinyCondModel(D, Dc, seed=0)
    x = torch.randn(B, D)
    t = torch.rand(B)
    cond = torch.randn(B, Dc)
    null = torch.zeros(Dc) + 0.5
    model.calls = 0
    out = mod.batched_cfg_forward(model, x, t, cond, null, 3.0)
    assert model.calls == 1, f"must call the model exactly once, got {model.calls}"
    assert out.shape == (B, D)
    u = model(x, t, null.expand(B, Dc))
    c = model(x, t, cond)
    ref = u + 3.0 * (c - u)
    assert torch.allclose(out, ref, atol=1e-5), f"batched forward must equal two separate forwards + combine; max err {(out - ref).abs().max():.2e}"
    assert torch.allclose(mod.batched_cfg_forward(model, x, t, cond, null, 1.0), c, atol=1e-6), "scale 1 = cond forward"
    assert torch.allclose(mod.batched_cfg_forward(model, x, t, cond, null, 0.0), u, atol=1e-6), "scale 0 = uncond forward"
    # ordering check: a model that returns its condition reveals which half was which
    captured = {}
    def spy(xx, tt, cc):
        captured["c"] = cc.clone(); captured["x"] = xx.clone(); captured["t"] = tt.clone()
        return xx
    mod.batched_cfg_forward(spy, x, t, cond, null, 2.0)
    assert captured["c"].shape == (2 * B, Dc) and captured["x"].shape == (2 * B, D) and captured["t"].shape == (2 * B,)
    assert torch.equal(captured["c"][:B], null.expand(B, Dc)) and torch.equal(captured["c"][B:], cond), \
        "unconditional half must come first, conditional second"
    assert torch.equal(captured["x"][:B], x) and torch.equal(captured["x"][B:], x) and torch.equal(captured["t"][B:], t)


def run(mod):
    check_condition_dropout(mod); print("  ok  condition dropout: frequency ≈ p, null rows exact, per-sample from generator")
    check_combine(mod);           print("  ok  cfg_combine identities (scale 0 / 1 / per-sample)")
    check_rescale(mod);           print("  ok  cfg_rescale: phi=1 matches cond std, phi=0 is plain CFG")
    check_interval(mod);          print("  ok  guidance interval logic")
    check_batched_forward(mod);   print("  ok  batched CFG forward == two separate forwards, uncond first, one call")
