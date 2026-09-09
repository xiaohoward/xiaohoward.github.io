import torch

# tiny tensors: intra-op threading is pure overhead (and can be 10x slower on many-core boxes)
torch.set_num_threads(min(torch.get_num_threads(), 8))


def _ref_schedule(betas):
    alphas = 1 - betas
    ab = torch.cumprod(alphas, 0)
    ab_prev = torch.cat([torch.ones(1), ab[:-1]])
    return alphas, ab, ab_prev


def check_schedule(mod):
    betas = mod.make_linear_betas(T=50)
    s = mod.make_schedule(betas)
    alphas, ab, ab_prev = _ref_schedule(betas)
    for k in ["betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
              "sqrt_one_minus_alphas_cumprod", "posterior_variance", "posterior_log_variance_clipped",
              "posterior_mean_coef1", "posterior_mean_coef2"]:
        assert k in s and s[k].shape == (50,), f"schedule['{k}'] missing or wrong shape"
    assert torch.allclose(s["alphas_cumprod"], ab), "alphas_cumprod wrong"
    assert torch.allclose(s["alphas_cumprod_prev"], ab_prev), "alphas_cumprod_prev wrong (abar_{-1} must be 1)"
    assert s["alphas_cumprod_prev"][0] == 1.0
    pv = betas * (1 - ab_prev) / (1 - ab)
    assert torch.allclose(s["posterior_variance"], pv, atol=1e-7), "posterior_variance wrong"
    assert s["posterior_variance"][0] == 0.0, "beta_tilde_0 must be 0"
    assert torch.isfinite(s["posterior_log_variance_clipped"]).all(), "log variance must be clipped at t=0"
    assert torch.allclose(s["posterior_log_variance_clipped"][0], torch.log(pv[1]))
    assert torch.allclose(s["posterior_mean_coef1"], betas * ab_prev.sqrt() / (1 - ab), atol=1e-7)
    assert torch.allclose(s["posterior_mean_coef2"], (1 - ab_prev) * alphas.sqrt() / (1 - ab), atol=1e-7)


def check_q_sample_statistics(mod):
    torch.manual_seed(0)
    betas = mod.make_linear_betas(T=100)
    s = mod.make_schedule(betas)
    _, ab, _ = _ref_schedule(betas)
    x0 = torch.tensor([[1.5, -2.0]]).expand(20000, 2)
    for ti in [0, 30, 99]:
        t = torch.full((20000,), ti, dtype=torch.long)
        eps = torch.randn(20000, 2)
        xt = mod.q_sample(s, x0, t, eps)
        assert xt.shape == (20000, 2)
        m, v = xt.mean(0), xt.var(0)
        assert torch.allclose(m, ab[ti].sqrt() * x0[0], atol=0.03), f"t={ti}: mean {m} vs {ab[ti].sqrt() * x0[0]}"
        assert torch.allclose(v, (1 - ab[ti]).expand(2), atol=0.03), f"t={ti}: var {v} vs {1 - ab[ti]}"
    # per-sample t: gathering must be elementwise
    t = torch.tensor([0, 50, 99])
    xt = mod.q_sample(s, torch.ones(3, 2), t, torch.zeros(3, 2))
    assert torch.allclose(xt[:, 0], ab[t].sqrt()), "q_sample must gather coefficients per sample"


def check_loss(mod):
    torch.manual_seed(0)
    s = mod.make_schedule(mod.make_linear_betas(T=100))
    x0, eps = torch.randn(64, 2), torch.randn(64, 2)
    t = torch.randint(0, 100, (64,))
    loss = mod.ddpm_loss(lambda x, t: torch.zeros_like(x), s, x0, t, eps)
    assert loss.ndim == 0 and torch.allclose(loss, (eps ** 2).mean()), "zero model -> loss = mean(eps^2)"
    # the model must be evaluated at q_sample(x0, t, eps)
    seen = {}
    def spy(x, tt):
        seen["x"] = x
        return eps
    loss = mod.ddpm_loss(spy, s, x0, t, eps)
    assert torch.allclose(loss, torch.zeros(())), "perfect model -> zero loss"
    assert torch.allclose(seen["x"], mod.q_sample(s, x0, t, eps)), "model must be called on x_t"
    # a few Adam steps on the tiny MLP must reduce the loss
    model = mod.TinyDenoiser(T=100)
    data = mod.make_two_moons(1024)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for it in range(200):
        idx = torch.randint(0, len(data), (128,))
        tt = torch.randint(0, 100, (128,))
        l = mod.ddpm_loss(model, s, data[idx], tt, torch.randn(128, 2))
        opt.zero_grad(); l.backward(); opt.step()
        losses.append(l.item())
    assert sum(losses[-50:]) / 50 < 0.7 * sum(losses[:50]) / 50, "training did not reduce the loss"


def check_posterior_formulas(mod):
    torch.manual_seed(1)
    betas = mod.make_linear_betas(T=100)
    s = mod.make_schedule(betas)
    alphas, ab, ab_prev = _ref_schedule(betas)
    x_t, eps = torch.randn(16, 2), torch.randn(16, 2)
    t = torch.randint(0, 100, (16,))
    mean, var, log_var, x0_hat = mod.p_mean_variance(s, x_t, t, eps)
    assert mean.shape == (16, 2) and x0_hat.shape == (16, 2)
    assert var.shape in [(16, 1), (16, 2)] and log_var.shape in [(16, 1), (16, 2)]
    abt = ab[t][:, None]
    x0_ref = (x_t - (1 - abt).sqrt() * eps) / abt.sqrt()
    assert torch.allclose(x0_hat, x0_ref, atol=1e-5), "x0_hat wrong"
    # Bayes posterior q(x_{t-1} | x_t, x0) written directly from the paper (Ho et al. eq. 7)
    m_ref = (betas[t][:, None] * ab_prev[t][:, None].sqrt() / (1 - abt)) * x0_ref + \
            ((1 - ab_prev[t][:, None]) * alphas[t][:, None].sqrt() / (1 - abt)) * x_t
    v_ref = betas[t][:, None] * (1 - ab_prev[t][:, None]) / (1 - abt)
    assert torch.allclose(mean, m_ref, atol=1e-5), "posterior mean wrong"
    assert torch.allclose(var.expand(16, 2), v_ref.expand(16, 2), atol=1e-7), "posterior variance wrong"
    mask = t > 0
    assert torch.allclose(log_var.expand(16, 2)[mask], v_ref.expand(16, 2)[mask].log(), atol=1e-5)
    # clipping
    _, _, _, x0c = mod.p_mean_variance(s, x_t * 50, t, eps, clip_x0=1.0)
    assert x0c.abs().max() <= 1.0 + 1e-6, "clip_x0 must clamp x0_hat"


def check_p_sample_no_noise_at_zero(mod):
    s = mod.make_schedule(mod.make_linear_betas(T=100))
    x_t = torch.randn(8, 2)
    model = lambda x, t: torch.zeros_like(x)
    t0 = torch.zeros(8, dtype=torch.long)
    a = mod.p_sample(model, s, x_t, t0, generator=torch.Generator().manual_seed(0))
    b = mod.p_sample(model, s, x_t, t0, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(a, b), "no noise may be added at t = 0"
    t5 = torch.full((8,), 5, dtype=torch.long)
    a = mod.p_sample(model, s, x_t, t5, generator=torch.Generator().manual_seed(0))
    b = mod.p_sample(model, s, x_t, t5, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(a, b), "noise must be added for t > 0"
    mixed = torch.tensor([0, 5, 0, 5, 0, 5, 0, 5])
    a = mod.p_sample(model, s, x_t, mixed, generator=torch.Generator().manual_seed(0))
    b = mod.p_sample(model, s, x_t, mixed, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(a[::2], b[::2]) and not torch.allclose(a[1::2], b[1::2]), "t==0 mask must be per-sample"


def check_oracle_sampler(mod):
    betas = mod.make_linear_betas(T=200)
    s = mod.make_schedule(betas)
    mu, sigma = torch.tensor([2.0, -1.0]), 0.5
    oracle = mod.GaussianOracle(s["alphas_cumprod"], mu, sigma)
    g = torch.Generator().manual_seed(0)
    x = mod.sample(oracle, s, (4000, 2), generator=g)
    assert x.shape == (4000, 2) and torch.isfinite(x).all()
    m, sd = x.mean(0), x.std(0)
    assert torch.allclose(m, mu, atol=0.05), f"sample mean {m} vs {mu}"
    assert torch.allclose(sd, torch.full((2,), sigma), atol=0.06), f"sample std {sd} vs {sigma}"
    # determinism given the generator seed
    y = mod.sample(oracle, s, (16, 2), generator=torch.Generator().manual_seed(3))
    z = mod.sample(oracle, s, (16, 2), generator=torch.Generator().manual_seed(3))
    assert torch.equal(y, z), "sampling must be deterministic given the generator"


def run(mod):
    check_schedule(mod); print("  ok  schedule quantities")
    check_q_sample_statistics(mod); print("  ok  q_sample mean/variance")
    check_loss(mod); print("  ok  eps-prediction loss + short training run")
    check_posterior_formulas(mod); print("  ok  posterior mean/variance vs analytic")
    check_p_sample_no_noise_at_zero(mod); print("  ok  p_sample noise masking")
    check_oracle_sampler(mod); print("  ok  ancestral sampler with oracle eps recovers N(mu, sigma^2)")
