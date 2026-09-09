import numpy as np
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_timesteps(mod):
    for n in (1, 4, 50):
        ts = mod.make_timesteps(n)
        assert ts.shape == (n + 1,), f"make_timesteps({n}) must have n+1 entries"
        assert ts[0].item() == 1.0 and ts[-1].item() == 0.0, "endpoints must be exactly 1 and 0"
        assert torch.all(ts[1:] < ts[:-1]), "timesteps must be strictly decreasing"
        assert torch.allclose(ts, torch.linspace(1, 0, n + 1)), "shift=1 must be uniform"
    ts = mod.make_timesteps(4, shift=3.0)
    assert ts[0].item() == 1.0 and ts[-1].item() == 0.0, "shift must keep the endpoints"
    assert torch.allclose(ts, torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0]), atol=1e-6), \
        f"shift=3 on [1, .75, .5, .25, 0] must give [1, .9, .75, .5, 0], got {ts.tolist()}"
    assert torch.all(ts[1:] < ts[:-1])
    ts5 = mod.make_timesteps(50, shift=3.0)
    uni = mod.make_timesteps(50)
    assert torch.all(ts5[1:-1] > uni[1:-1]), "shift > 1 must push interior timesteps toward 1"
    assert abs((ts5 > 0.5).float().mean().item() - 0.75) < 0.03, "with shift 3, t' > 0.5 iff t > 0.25: ~75% of steps at high noise"
    ts_lo = mod.make_timesteps(50, shift=0.5)
    assert torch.all(ts_lo[1:-1] < uni[1:-1]), "shift < 1 pushes toward 0"


def check_euler_step(mod):
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    v = torch.randn(3, 4)
    assert torch.allclose(mod.euler_step(x, v, 1.0, 0.75), x - 0.25 * v, atol=1e-6)
    assert torch.equal(mod.euler_step(x, v, 0.5, 0.5), x), "zero-length step is the identity"
    assert torch.allclose(mod.euler_step(x, v, torch.tensor(0.4), torch.tensor(0.1)), x - 0.3 * v, atol=1e-6)


def check_one_step(mod):
    torch.manual_seed(1)
    model = mod.GaussianVelocityModel(mu=0.3, sigma=0.5)
    x1 = torch.randn(8, 5)
    out = mod.sample_euler(model, x1, mod.make_timesteps(1))
    ref = x1 - model(x1, torch.ones(8))
    assert torch.allclose(out, ref, atol=1e-6), "1-step Euler must be exactly x1 - v(x1, 1)"
    # for the Gaussian velocity, v(x, 1) = x - mu so the 1-step sample is the mean
    assert torch.allclose(out, torch.full_like(out, 0.3), atol=1e-5)
    # a per-step spy sees n calls with t = the schedule's first n entries, in order
    seen = []
    def spy(x, t):
        seen.append(t[0].item()); return torch.zeros_like(x)
    ts = mod.make_timesteps(5, shift=2.0)
    out = mod.sample_euler(spy, x1, ts)
    assert torch.equal(out, x1) and len(seen) == 5 and np.allclose(seen, ts[:-1].tolist(), atol=1e-6), \
        "sampler must call the model once per step at t_i for i < n"


def check_sampler_statistics(mod):
    mu, sigma = 0.5, 0.7
    model = mod.GaussianVelocityModel(mu, sigma)
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(4000, 8, generator=g)
    for shift in (1.0, 3.0):
        out = mod.sample_euler(model, x1, mod.make_timesteps(200, shift=shift))
        assert out.shape == x1.shape and torch.isfinite(out).all()
        m, s = out.mean().item(), out.std().item()
        assert abs(m - mu) < 0.03 * max(1, abs(mu)) + 0.01, f"sample mean {m:.4f} != {mu} (shift {shift})"
        assert abs(s - sigma) / sigma < 0.03, f"sample std {s:.4f} != {sigma} within 3% (shift {shift})"
    # per-dimension isotropy and the map is (nearly) affine in x1: corr(x1, out) ~ 1
    out = mod.sample_euler(model, x1, mod.make_timesteps(200))
    assert (out.std(dim=0) - sigma).abs().max().item() < 0.05
    c = torch.corrcoef(torch.stack([x1[:, 0], out[:, 0]]))[0, 1].item()
    assert c > 0.999, f"Gaussian->Gaussian flow is affine; corr(x1, out) = {c:.4f}"
    # more steps is more accurate (std error shrinks)
    e50 = abs(mod.sample_euler(model, x1, mod.make_timesteps(50)).std().item() - sigma)
    e400 = abs(mod.sample_euler(model, x1, mod.make_timesteps(400)).std().item() - sigma)
    assert e400 < e50, "Euler error must shrink with more steps"
    # image-shaped batch works
    out_img = mod.sample_euler(model, torch.randn(2, 3, 4, 4, generator=g), mod.make_timesteps(10))
    assert out_img.shape == (2, 3, 4, 4)


def check_clamped_sampler(mod):
    model = mod.GaussianVelocityModel(0.0, 0.8)     # data well outside [-1, 1] sometimes -> the clamp is active
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(2000, 4, generator=g)
    ts = mod.make_timesteps(40)
    out, hist = mod.sample_with_x0_clamp(model, x1, ts, -1.0, 1.0, return_x0_history=True)
    assert hist.shape == (40, 2000, 4), f"x0 history shape {tuple(hist.shape)}"
    assert hist.min().item() >= -1.0 and hist.max().item() <= 1.0, "every x0 estimate must lie in [lo, hi]"
    assert out.min().item() >= -1.0 and out.max().item() <= 1.0, "final sample (t=0 -> x0_hat) must lie in [lo, hi]"
    assert (hist == 1.0).any() or (hist == -1.0).any(), "(sanity) the clamp should be active for sigma=0.8 data"
    plain = mod.sample_euler(model, x1, ts)
    assert plain.abs().max().item() > 1.0, "(sanity) unclamped sampler exceeds the bounds"
    assert torch.allclose(out.clamp(-1, 1), out), "sample_with_x0_clamp output must be within bounds"
    # with a wide clamp it reduces exactly to plain Euler (same trajectory)
    wide = mod.sample_with_x0_clamp(model, x1, ts, -1e6, 1e6)
    assert torch.allclose(wide, plain, atol=1e-4), "with an inactive clamp the sampler must equal plain Euler"
    # x0_hat at the last step equals the output (t_next = 0)
    assert torch.allclose(hist[-1], out, atol=1e-6)
    # x0 estimate formula: first step x0_hat = clamp(x1 - 1 * v(x1, 1))
    assert torch.allclose(hist[0], (x1 - model(x1, torch.ones(2000))).clamp(-1, 1), atol=1e-6)
    # custom bounds
    out2 = mod.sample_with_x0_clamp(model, x1, ts, -0.5, 0.5)
    assert out2.min().item() >= -0.5 and out2.max().item() <= 0.5


def run(mod):
    check_timesteps(mod);           print("  ok  timestep schedule: endpoints, monotone, shift(3) values")
    check_euler_step(mod);          print("  ok  euler_step = x + (t_next - t_cur) v")
    check_one_step(mod);            print("  ok  1-step Euler = x1 - v(x1, 1); one model call per step")
    check_sampler_statistics(mod);  print("  ok  200-step Euler with the analytic Gaussian velocity matches (mu, sigma) within 3%")
    check_clamped_sampler(mod);     print("  ok  x0-clamped sampler keeps every x0 estimate within bounds")
