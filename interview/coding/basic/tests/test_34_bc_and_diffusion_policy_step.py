import math
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_heads(mod):
    g = torch.Generator().manual_seed(0)
    mean = torch.randn(16, 3, generator=g)
    log_std = torch.randn(16, 3, generator=g) * 0.5
    target = torch.randn(16, 3, generator=g)
    assert torch.allclose(mod.mse_loss(mean, target), torch.nn.functional.mse_loss(mean, target))
    ref = -torch.distributions.Normal(mean, log_std.exp()).log_prob(target).mean()
    nll = mod.gaussian_nll_loss(mean, log_std, target)
    assert nll.shape == (), "losses are scalars"
    assert torch.allclose(nll, ref, atol=1e-5), f"Gaussian NLL {nll.item():.6f} != torch.distributions {ref.item():.6f}"
    # clamping: huge log_std is clamped to log_std_max and gets zero gradient
    big = torch.full((16, 3), 10.0, requires_grad=True)
    nll_big = mod.gaussian_nll_loss(mean, big, target, log_std_max=2.0)
    ref_big = -torch.distributions.Normal(mean, torch.full((16, 3), math.e ** 2)).log_prob(target).mean()
    assert torch.allclose(nll_big, ref_big, atol=1e-5), "log_std must be clamped to log_std_max before use"
    nll_big.backward()
    assert torch.all(big.grad == 0), "clamped log_std must receive zero gradient"
    small = torch.full((16, 3), -20.0)
    assert torch.isfinite(mod.gaussian_nll_loss(mean, small, target)), "clamp keeps the NLL finite"


def check_bc_step(mod):
    obs, act = mod.make_multimodal_dataset(512, seed=0)
    for head, out_dim in (("mse", 2), ("gaussian", 4)):
        policy = mod.MLP(2, out_dim, hidden=64, seed=1)
        opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
        losses = [mod.bc_step(policy, opt, obs, act, head=head) for _ in range(150)]
        assert isinstance(losses[0], float)
        # the MSE floor on this bimodal set is ~0.36 (variance of the two modes), so check an absolute drop
        assert losses[-1] < losses[0] - 0.15, f"{head}: BC loss did not drop ({losses[0]:.3f} -> {losses[-1]:.3f})"
    # multimodality: the MSE head regresses to the mean of the two modes, i.e. it predicts a_y ~ 0 for every obs
    pred = policy(obs)[:, :2] if head == "gaussian" else policy(obs)
    mse_policy = mod.MLP(2, 2, hidden=64, seed=1)
    opt = torch.optim.Adam(mse_policy.parameters(), lr=3e-3)
    for _ in range(300):
        mod.bc_step(mse_policy, opt, obs, act, head="mse")
    with torch.no_grad():
        py = mse_policy(obs)[:, 1]
    assert py.abs().mean() < 0.35 and act[:, 1].abs().mean() > 0.5, "MSE head must average the two modes (mode collapse to the mean)"


def check_flow_loss(mod):
    g = torch.Generator().manual_seed(0)
    B, H, D = 8, 4, 3
    obs = torch.randn(B, 5, generator=g)
    a0 = torch.randn(B, H, D, generator=g)
    eps = torch.randn(B, H, D, generator=g)
    t = torch.rand(B, generator=g)
    oracle = lambda o, a_t, tt: eps - a0
    assert mod.flow_matching_loss(oracle, obs, a0, t, eps).abs() < 1e-7, "loss on the true velocity must be 0"
    # the network must be evaluated at the interpolant a_t = (1-t) a0 + t eps with per-sample t
    seen = {}
    def spy(o, a_t, tt):
        seen["a_t"], seen["t"] = a_t.clone(), tt.clone()
        return torch.zeros_like(a_t)
    loss = mod.flow_matching_loss(spy, obs, a0, t, eps)
    assert torch.allclose(seen["a_t"], (1 - t.view(B, 1, 1)) * a0 + t.view(B, 1, 1) * eps, atol=1e-6)
    assert torch.allclose(seen["t"].reshape(-1), t)
    assert torch.allclose(loss, ((eps - a0) ** 2).mean())
    # the given net trains: loss decreases on the multimodal data as chunks of H=1
    obs, act = mod.make_multimodal_dataset(256, seed=0)
    net = mod.FlowPolicyNet(2, 1, 2, hidden=128, seed=0)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(0)
    first = sum(mod.diffusion_policy_step(net, opt, obs, act[:, None, :], gen) for _ in range(10)) / 10
    for _ in range(200):
        last = mod.diffusion_policy_step(net, opt, obs, act[:, None, :], gen)
    last = sum(mod.diffusion_policy_step(net, opt, obs, act[:, None, :], gen) for _ in range(10)) / 10
    assert last < 0.7 * first, f"flow loss did not drop ({first:.3f} -> {last:.3f})"
    assert isinstance(last, float)


def check_sampler(mod):
    # Analytic case: a0 ~ N(mu, sigma^2) per dim, eps ~ N(0, 1). Then a_t ~ N((1-t) mu, s^2), s^2 = (1-t)^2 sigma^2 + t^2
    # and the optimal velocity is E[eps - a0 | a_t] = (t - (1-t) sigma^2) / s^2 * (a_t - (1-t) mu) - mu.
    mu = torch.tensor([[1.0, -2.0], [0.5, 3.0]])           # (H, D) = (2, 2)
    sigma = torch.tensor([[0.5, 1.0], [2.0, 0.3]])

    def v_star(obs, a_t, t):
        tt = t.view(-1, 1, 1)
        s2 = (1 - tt) ** 2 * sigma ** 2 + tt ** 2
        return (tt - (1 - tt) * sigma ** 2) / s2 * (a_t - (1 - tt) * mu) - mu

    B = 4000
    gen = torch.Generator().manual_seed(0)
    a = mod.sample_actions(v_star, torch.zeros(B, 1), (2, 2), K=200, gen=gen)
    assert a.shape == (B, 2, 2)
    m, s = a.mean(0), a.std(0)
    assert torch.allclose(m, mu, atol=0.08), f"sample mean {m.tolist()} != {mu.tolist()}"
    assert torch.allclose(s, sigma, atol=0.1), f"sample std {s.tolist()} != {sigma.tolist()}"
    # K=1 from pure noise with the exact field lands exactly on the noise-free one-step Euler prediction
    gen = torch.Generator().manual_seed(1)
    a1 = mod.sample_actions(v_star, torch.zeros(8, 1), (2, 2), K=1, gen=gen)
    gen = torch.Generator().manual_seed(1)
    noise = torch.randn((8, 2, 2), generator=gen)
    ref = noise - v_star(None, noise, torch.ones(8))
    assert torch.allclose(a1, ref, atol=1e-6), "K=1 Euler: a = eps - v(eps, t=1)"
    # sampler must also step in decreasing t: t_k = 1 - k/K
    ts = []
    spy = lambda o, a_t, t: (ts.append(t[0].item()), torch.zeros_like(a_t))[1]
    mod.sample_actions(spy, torch.zeros(2, 1), (1, 1), K=4, gen=torch.Generator().manual_seed(0))
    assert [round(x, 6) for x in ts] == [1.0, 0.75, 0.5, 0.25], f"Euler timesteps {ts}"


def check_wilson(mod):
    for k, n, lo, hi in ((8, 10, 0.4902, 0.9433), (0, 10, 0.0, 0.2775), (30, 30, 0.8865, 1.0), (50, 100, 0.4038, 0.5962)):
        p, l, h = mod.wilson_interval(k, n)
        assert abs(p - k / n) < 1e-12
        assert abs(l - lo) < 6e-4 and abs(h - hi) < 6e-4, f"Wilson {k}/{n}: got ({l:.4f}, {h:.4f}) expected ({lo}, {hi})"
    # direct formula check with a different z
    k, n, z = 7, 20, 2.5758
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hh = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    _, l, h = mod.wilson_interval(k, n, z=z)
    assert abs(l - (c - hh)) < 1e-9 and abs(h - (c + hh)) < 1e-9
    out = mod.evaluate_success(torch.tensor([True] * 8 + [False] * 2))
    assert out["n"] == 10 and abs(out["rate"] - 0.8) < 1e-9 and abs(out["lo"] - 0.4902) < 6e-4 and abs(out["hi"] - 0.9433) < 6e-4
    out = mod.evaluate_success([False] * 10)
    assert out["rate"] == 0.0 and out["lo"] == 0.0 and abs(out["hi"] - 0.2775) < 6e-4


def run(mod):
    check_heads(mod);      print("  ok  MSE / clamped Gaussian NLL heads match torch.distributions")
    check_bc_step(mod);    print("  ok  BC step reduces loss; MSE head averages the two modes")
    check_flow_loss(mod);  print("  ok  flow-matching loss: 0 on the true velocity, uses the interpolant, trains")
    check_sampler(mod);    print("  ok  Euler sampler matches the analytic Gaussian case")
    check_wilson(mod);     print("  ok  Wilson interval matches hand-computed values")
