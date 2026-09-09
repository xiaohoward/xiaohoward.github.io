import torch

# tiny tensors: intra-op threading is pure overhead (and can be 10x slower on many-core boxes)
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_interpolant_and_target(mod):
    torch.manual_seed(0)
    x0, eps = torch.randn(32, 2), torch.randn(32, 2)
    t = torch.rand(32)
    xt = mod.interpolate(x0, eps, t)
    assert xt.shape == (32, 2)
    assert torch.allclose(xt, (1 - t)[:, None] * x0 + t[:, None] * eps, atol=1e-6)
    assert torch.allclose(mod.interpolate(x0, eps, torch.zeros(32)), x0), "t = 0 must be data"
    assert torch.allclose(mod.interpolate(x0, eps, torch.ones(32)), eps), "t = 1 must be noise"
    u = mod.velocity_target(x0, eps)
    assert torch.allclose(u, eps - x0), "target must be eps - x0"
    # finite-difference check: d x_t / d t == target
    h = 1e-3
    fd = (mod.interpolate(x0, eps, t + h) - mod.interpolate(x0, eps, t - h)) / (2 * h)
    assert torch.allclose(fd, u, atol=1e-2), "velocity target must equal d x_t / dt"


def check_loss(mod):
    torch.manual_seed(0)
    x0, eps = torch.randn(64, 2), torch.randn(64, 2)
    t, cond = torch.rand(64), torch.randint(0, 8, (64,))
    loss = mod.flow_matching_loss(lambda x, t, c: torch.zeros_like(x), x0, eps, t, cond)
    assert loss.ndim == 0 and torch.allclose(loss, ((eps - x0) ** 2).mean()), "zero model -> mean||eps - x0||^2"
    seen = {}
    def spy(x, tt, c):
        seen["x"], seen["t"], seen["c"] = x, tt, c
        return eps - x0
    assert torch.allclose(mod.flow_matching_loss(spy, x0, eps, t, cond), torch.zeros(()), atol=1e-7)
    assert torch.allclose(seen["x"], mod.interpolate(x0, eps, t)) and torch.equal(seen["c"], cond)
    # short training run must reduce the loss
    model = mod.TinyCondVelocity()
    data, labels = mod.make_eight_gaussians(1024)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for it in range(200):
        idx = torch.randint(0, len(data), (128,))
        l = mod.flow_matching_loss(model, data[idx], torch.randn(128, 2), torch.rand(128), labels[idx])
        opt.zero_grad(); l.backward(); opt.step(); losses.append(l.item())
    assert sum(losses[-30:]) / 30 < 0.7 * sum(losses[:30]) / 30, "training did not reduce the loss"


def check_euler_oracle(mod):
    mu, sigma = torch.tensor([1.5, -0.5]), 0.7
    oracle = mod.GaussianOracleVelocity(mu, sigma)
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(4000, 2, generator=g)
    x0 = mod.sample_euler(oracle, x1, mod.uniform_timesteps(400))
    assert x0.shape == (4000, 2) and torch.isfinite(x0).all()
    assert torch.allclose(x0.mean(0), mu, atol=0.05), f"Euler mean {x0.mean(0)} vs {mu}"
    assert torch.allclose(x0.std(0), torch.full((2,), sigma), atol=0.05), f"Euler std {x0.std(0)} vs {sigma}"
    # the exact flow map for this Gaussian path is x0 = mu + sigma x1
    exact = mu + sigma * x1
    assert (x0 - exact).abs().max() < 0.05, "Euler with 400 steps should track the exact flow map"
    # direction of integration: with 1 step Euler gives x1 - v(x1, 1) = x1 - (x1 - mu) = mu
    one = mod.sample_euler(oracle, x1[:8], torch.tensor([1.0, 0.0]))
    assert torch.allclose(one, mu.expand(8, 2), atol=1e-5), "1-step Euler from t=1 must give x1 + (0-1) v(x1, 1)"
    # velocity_fn must receive t as a (B,) tensor
    def strict(x, t):
        assert t.shape == (x.shape[0],), "t must be passed as a (B,) tensor"
        return oracle(x, t)
    mod.sample_euler(strict, x1[:8], mod.uniform_timesteps(3))


def check_heun_beats_euler(mod):
    mu, sigma = torch.tensor([1.5, -0.5]), 0.7
    oracle = mod.GaussianOracleVelocity(mu, sigma)
    g = torch.Generator().manual_seed(1)
    x1 = torch.randn(512, 2, generator=g)
    exact = mu + sigma * x1
    ts = mod.uniform_timesteps(10)
    calls = {"n": 0}
    def counting(x, t):
        calls["n"] += 1
        return oracle(x, t)
    e_err = (mod.sample_euler(counting, x1, ts) - exact).norm(dim=1).mean()
    n_euler = calls["n"]; calls["n"] = 0
    h_err = (mod.sample_heun(counting, x1, ts) - exact).norm(dim=1).mean()
    n_heun = calls["n"]
    assert n_euler == 10, f"Euler must use 1 eval per step, used {n_euler}"
    assert n_heun == 20, f"Heun must use 2 evals per step, used {n_heun}"
    assert h_err < 0.3 * e_err, f"Heun error {h_err:.4f} should be well below Euler {e_err:.4f} at 10 steps"
    # order check: halving the step (20 -> 40) should cut Heun error ~4x, Euler ~2x
    h20 = (mod.sample_heun(oracle, x1, mod.uniform_timesteps(20)) - exact).norm(dim=1).mean()
    h40 = (mod.sample_heun(oracle, x1, mod.uniform_timesteps(40)) - exact).norm(dim=1).mean()
    e20 = (mod.sample_euler(oracle, x1, mod.uniform_timesteps(20)) - exact).norm(dim=1).mean()
    e40 = (mod.sample_euler(oracle, x1, mod.uniform_timesteps(40)) - exact).norm(dim=1).mean()
    assert h20 / h40 > 3.0, f"Heun should be ~2nd order: ratio {h20 / h40:.2f}"
    assert 1.5 < e20 / e40 < 3.0, f"Euler should be ~1st order: ratio {e20 / e40:.2f}"


def check_cfg(mod):
    torch.manual_seed(0)
    model = mod.TinyCondVelocity()
    x, t = torch.randn(16, 2), torch.rand(16)
    cond = torch.randint(0, 8, (16,))
    null = torch.full((16,), model.null_class, dtype=torch.long)
    with torch.no_grad():
        v_c, v_u = model(x, t, cond), model(x, t, null)
        v1 = mod.cfg_velocity(model, x, t, cond, null, 1.0)
        v0 = mod.cfg_velocity(model, x, t, cond, null, 0.0)
        v4 = mod.cfg_velocity(model, x, t, cond, null, 4.0)
    assert v1.shape == (16, 2)
    assert torch.allclose(v1, v_c, atol=1e-6), "scale 1 must equal the conditional velocity"
    assert torch.allclose(v0, v_u, atol=1e-6), "scale 0 must equal the unconditional velocity"
    assert torch.allclose(v4, v_u + 4.0 * (v_c - v_u), atol=1e-5), "wrong guidance formula"
    # single batched call of size 2B
    calls = []
    def spy(xx, tt, cc):
        calls.append((xx.shape[0], cc.clone()))
        return model(xx, tt, cc)
    with torch.no_grad():
        mod.cfg_velocity(spy, x, t, cond, null, 2.0)
    assert len(calls) == 1 and calls[0][0] == 32, "CFG should evaluate the model once on the doubled batch"
    assert torch.equal(calls[0][1][:16], cond) and torch.equal(calls[0][1][16:], null), "order must be [cond; null]"


def run(mod):
    check_interpolant_and_target(mod); print("  ok  interpolant / velocity target")
    check_loss(mod); print("  ok  flow-matching loss + short training run")
    check_euler_oracle(mod); print("  ok  Euler with oracle velocity recovers N(mu, sigma^2)")
    check_heun_beats_euler(mod); print("  ok  Heun 2nd order, beats Euler at equal steps")
    check_cfg(mod); print("  ok  classifier-free guidance")
