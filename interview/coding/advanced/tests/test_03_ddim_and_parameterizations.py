import torch

# tiny tensors: intra-op threading is pure overhead (and can be 10x slower on many-core boxes)
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_conversions_roundtrip(mod):
    torch.manual_seed(0)
    x0, eps = torch.randn(32, 2), torch.randn(32, 2)
    for abar in [0.999, 0.5, 0.01, torch.rand(32, 1) * 0.98 + 0.01]:
        a = torch.as_tensor(abar)
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps
        assert torch.allclose(mod.x0_from_eps(x_t, eps, abar), x0, atol=1e-4), "x0_from_eps"
        assert torch.allclose(mod.eps_from_x0(x_t, x0, abar), eps, atol=1e-4), "eps_from_x0"
        v = mod.v_from_x0_eps(x0, eps, abar)
        assert torch.allclose(v, a.sqrt() * eps - (1 - a).sqrt() * x0, atol=1e-6), "v definition"
        x0r, epsr = mod.x0_eps_from_v(x_t, v, abar)
        assert torch.allclose(x0r, x0, atol=1e-5) and torch.allclose(epsr, eps, atol=1e-5), "x0_eps_from_v"
        # round trips through every pair
        assert torch.allclose(mod.x0_from_eps(x_t, mod.eps_from_x0(x_t, x0, abar), abar), x0, atol=1e-4)
        assert torch.allclose(mod.v_from_x0_eps(*mod.x0_eps_from_v(x_t, v, abar), abar), v, atol=1e-5)
    # v is the velocity of the "angle" parameterization: at abar -> 1, v -> eps; at abar -> 0, v -> -x0
    assert torch.allclose(mod.v_from_x0_eps(x0, eps, 1.0), eps) and torch.allclose(mod.v_from_x0_eps(x0, eps, 0.0), -x0)


def check_sigma(mod):
    ab = mod.make_alphas_cumprod(T=200)
    ts = mod.strided_timesteps(200, 20)
    for i in range(len(ts) - 1):
        a_t, a_p = ab[ts[i]], ab[ts[i + 1]]
        s0 = torch.as_tensor(mod.ddim_sigma(a_t, a_p, 0.0))
        s1 = torch.as_tensor(mod.ddim_sigma(a_t, a_p, 1.0))
        assert torch.allclose(s0, torch.zeros(())), "eta = 0 must give sigma = 0"
        ref = torch.sqrt((1 - a_p) / (1 - a_t)) * torch.sqrt(1 - a_t / a_p)
        assert torch.allclose(s1, ref, atol=1e-7), "sigma formula"
        assert torch.allclose(torch.as_tensor(mod.ddim_sigma(a_t, a_p, 0.5)), 0.5 * ref, atol=1e-7), "linear in eta"
    assert torch.allclose(torch.as_tensor(mod.ddim_sigma(ab[5], 1.0, 1.0)), torch.zeros(())), "abar_prev = 1 -> 0"


def check_eta1_matches_ddpm_posterior(mod):
    """With consecutive timesteps, eta = 1 must reproduce q(x_{t-1} | x_t, x0_hat): mean AND variance."""
    torch.manual_seed(0)
    betas = torch.linspace(1e-3, 0.08, 200)
    alphas = 1 - betas
    ab = torch.cumprod(alphas, 0)
    x_t, eps = torch.randn(16, 2), torch.randn(16, 2)
    for t in [1, 50, 199]:
        a_t, a_p = ab[t], ab[t - 1]
        var = torch.as_tensor(mod.ddim_sigma(a_t, a_p, 1.0)) ** 2
        post_var = betas[t] * (1 - a_p) / (1 - a_t)
        assert torch.allclose(var, post_var, atol=1e-7), f"t={t}: eta=1 variance {var} vs DDPM {post_var}"
        x_prev = mod.ddim_step(x_t, eps, a_t, a_p, eta=1.0, noise=torch.zeros_like(x_t))
        x0_hat = (x_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        post_mean = (betas[t] * a_p.sqrt() / (1 - a_t)) * x0_hat + ((1 - a_p) * alphas[t].sqrt() / (1 - a_t)) * x_t
        assert torch.allclose(x_prev, post_mean, atol=1e-4), f"t={t}: eta=1 mean must equal DDPM posterior mean"
    # noise is added with the right scale
    z = torch.randn(16, 2)
    a_t, a_p = ab[50], ab[49]
    d = mod.ddim_step(x_t, eps, a_t, a_p, 1.0, noise=z) - mod.ddim_step(x_t, eps, a_t, a_p, 1.0, noise=torch.zeros_like(z))
    assert torch.allclose(d, torch.as_tensor(mod.ddim_sigma(a_t, a_p, 1.0)) * z, atol=1e-6), "noise scale"


def check_ddim_step_eta0(mod):
    torch.manual_seed(0)
    x0, eps = torch.randn(8, 2), torch.randn(8, 2)
    a_t, a_p = 0.3, 0.6
    x_t = a_t ** 0.5 * x0 + (1 - a_t) ** 0.5 * eps
    x_prev = mod.ddim_step(x_t, eps, a_t, a_p, eta=0.0)
    # with the TRUE eps, DDIM eta=0 moves along the same (x0, eps) line: x_prev = sqrt(a_p) x0 + sqrt(1-a_p) eps
    ref = a_p ** 0.5 * x0 + (1 - a_p) ** 0.5 * eps
    assert torch.allclose(x_prev, ref, atol=1e-5), "eta=0 step with exact eps must keep (x0, eps) fixed"
    # last step to abar_prev = 1 returns x0_hat exactly
    assert torch.allclose(mod.ddim_step(x_t, eps, a_t, 1.0, eta=0.0), x0, atol=1e-5), "abar_prev = 1 -> x0_hat"
    assert torch.allclose(mod.ddim_step(x_t, eps, a_t, 1.0, eta=1.0, noise=torch.ones(8, 2)), x0, atol=1e-5), \
        "abar_prev = 1 -> x0_hat even with eta > 0 (sigma = 0)"


def check_ddim_sampler_oracle(mod):
    ab = mod.make_alphas_cumprod(T=200)
    mu, sigma = torch.tensor([1.0, -2.0]), 0.6
    oracle = mod.GaussianOracle(ab, mu, sigma)
    ts = mod.strided_timesteps(200, 50)
    assert ts[0] == 199 and len(ts) == 50 and (ts[:-1] > ts[1:]).all()
    g = torch.Generator().manual_seed(0)
    x_T = torch.randn(4000, 2, generator=g)
    a = mod.ddim_sample(oracle, ab, ts, x_T, eta=0.0)
    b = mod.ddim_sample(oracle, ab, ts, x_T, eta=0.0, generator=torch.Generator().manual_seed(123))
    assert torch.equal(a, b), "eta = 0 must be deterministic (no dependence on the generator)"
    assert a.shape == (4000, 2) and torch.isfinite(a).all()
    assert torch.allclose(a.mean(0), mu, atol=0.05), f"DDIM eta=0 mean {a.mean(0)} vs {mu}"
    assert torch.allclose(a.std(0), torch.full((2,), sigma), atol=0.08), f"DDIM eta=0 std {a.std(0)} vs {sigma}"
    # eta = 0 is a 1st-order PF-ODE solver: the exact flow map for this Gaussian path is x0 = mu + sigma x_T,
    # and the error must shrink as the grid is refined
    exact = mu + sigma * x_T
    err = [(mod.ddim_sample(oracle, ab, mod.strided_timesteps(200, n), x_T, eta=0.0) - exact).abs().max()
           for n in (20, 100)]
    assert err[1] < 0.5 * err[0] and err[1] < 0.1, f"DDIM eta=0 must converge to the PF-ODE flow map: {err}"
    # eta = 1 on the full 200-step grid = DDPM ancestral sampling; also correct, and stochastic
    full = torch.arange(199, -1, -1, dtype=torch.long)
    c = mod.ddim_sample(oracle, ab, full, x_T, eta=1.0, generator=torch.Generator().manual_seed(1))
    d = mod.ddim_sample(oracle, ab, full, x_T, eta=1.0, generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(c, d), "eta = 1 must be stochastic"
    assert torch.allclose(c.mean(0), mu, atol=0.05), f"DDIM eta=1 mean {c.mean(0)} vs {mu}"
    assert torch.allclose(c.std(0), torch.full((2,), sigma), atol=0.06), f"DDIM eta=1 std {c.std(0)} vs {sigma}"
    # model must be called with the int64 index of the current step
    seen = []
    def spy(x, t):
        seen.append(t.clone()); return oracle(x, t)
    mod.ddim_sample(spy, ab, ts, x_T[:4], eta=0.0)
    assert len(seen) == 50 and all(s.shape == (4,) and s.dtype == torch.long and int(s[0]) == int(ts[i]) for i, s in enumerate(seen))


def run(mod):
    check_conversions_roundtrip(mod); print("  ok  eps / x0 / v conversions round-trip")
    check_sigma(mod); print("  ok  ddim sigma(eta)")
    check_eta1_matches_ddpm_posterior(mod); print("  ok  eta = 1 reproduces the DDPM posterior mean and variance")
    check_ddim_step_eta0(mod); print("  ok  eta = 0 step")
    check_ddim_sampler_oracle(mod); print("  ok  DDIM sampler with oracle eps (deterministic, correct moments)")
