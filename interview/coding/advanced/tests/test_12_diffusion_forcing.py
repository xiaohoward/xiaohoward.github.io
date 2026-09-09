import torch


def check_causal_mask(mod):
    m = mod.causal_frame_mask(5)
    assert m.dtype == torch.bool and m.shape == (5, 5)
    assert torch.equal(m, torch.tril(torch.ones(5, 5, dtype=torch.bool))), "mask[i, j] must be j <= i"
    assert mod.causal_frame_mask(1).item() is True
    # changing future frames must not change earlier frames' outputs
    model = mod.TinyFrameDenoiser(D=8, seed=0)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 6, 8, generator=g)
    t = torch.rand(2, 6, generator=g)
    with torch.no_grad():
        y = model(x, t, mod.causal_frame_mask(6))
        x2 = x.clone()
        x2[:, 3:] = torch.randn(2, 3, 8, generator=g)
        t2 = t.clone()
        t2[:, 3:] = 0.1
        y2 = model(x2, t2, mod.causal_frame_mask(6))
        y_nomask = model(x2, t2, None)
    assert torch.allclose(y[:, :3], y2[:, :3], atol=1e-6), "frames < 3 must be unaffected by changes to frames >= 3"
    assert not torch.allclose(y[:, 3:], y2[:, 3:], atol=1e-3), "changed frames must change their own output"
    assert not torch.allclose(y_nomask[:, :3], y2[:, :3], atol=1e-4), "without the mask the past sees the future"


def check_loss(mod):
    model = mod.TinyFrameDenoiser(D=8, seed=1)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(3, 5, 8, generator=g)
    loss = mod.diffusion_forcing_loss(model, x0, torch.Generator().manual_seed(4))
    assert loss.dim() == 0 and torch.isfinite(loss) and loss.item() > 0, "loss must be a positive finite scalar"
    assert loss.requires_grad, "loss must be differentiable w.r.t. the model"
    loss.backward()
    assert model.qkv.weight.grad is not None and torch.isfinite(model.qkv.weight.grad).all()
    # exact value against a re-derivation with the same generator draws
    gen = torch.Generator().manual_seed(4)
    t = torch.rand(3, 5, generator=gen)
    eps = torch.randn(3, 5, 8, generator=gen)
    x_t = (1 - t)[..., None] * x0 + t[..., None] * eps
    with torch.no_grad():
        ref = ((model(x_t, t, torch.tril(torch.ones(5, 5, dtype=torch.bool))) - (eps - x0)) ** 2).mean()
    assert abs(loss.item() - ref.item()) < 1e-6, f"loss {loss.item()} != reference {ref.item()}"
    # per-frame independent noise levels: an oracle that returns exactly eps - x0 gives zero loss regardless of t
    class Oracle:
        def __init__(self):
            self.seen_t = None

        def __call__(self, x_t, t, mask):
            self.seen_t = t
            return eps - x0  # the exact target
    o = Oracle()
    z = mod.diffusion_forcing_loss(o, x0, torch.Generator().manual_seed(4))
    assert z.item() < 1e-10, "an oracle predicting eps - x0 must give zero loss"
    assert o.seen_t.shape == (3, 5), "t must be per (batch, frame)"
    assert o.seen_t.std(dim=1).min() > 0, "noise levels must differ across frames (independent per frame)"
    assert (o.seen_t >= 0).all() and (o.seen_t <= 1).all()


def check_sampler_shape_and_determinism(mod):
    model = mod.TinyFrameDenoiser(D=8, seed=2)
    with torch.no_grad():
        a = mod.sample_autoregressive(model, 2, 4, 8, n_steps=3, gen=torch.Generator().manual_seed(0))
        b = mod.sample_autoregressive(model, 2, 4, 8, n_steps=3, gen=torch.Generator().manual_seed(0))
        c = mod.sample_autoregressive(model, 2, 4, 8, n_steps=3, gen=torch.Generator().manual_seed(1))
    assert a.shape == (2, 4, 8), f"shape {a.shape}"
    assert torch.isfinite(a).all()
    assert torch.equal(a, b), "same seed must give the same rollout"
    assert not torch.equal(a, c), "different seeds must differ"


def check_sampler_oracle_exact(mod):
    g = torch.Generator().manual_seed(0)
    targets = torch.randn(2, 5, 8, generator=g)
    for n_steps in (1, 2, 4, 7):
        oracle = mod.OracleDenoiser(targets=targets)
        out = mod.sample_autoregressive(oracle, 2, 5, 8, n_steps=n_steps, gen=torch.Generator().manual_seed(3))
        assert out.shape == (2, 5, 8)
        assert torch.allclose(out, targets, atol=1e-4), f"n_steps={n_steps}: Euler along the exact straight path must land on the target"
        ns = [n for n, _ in oracle.calls]
        assert ns == sorted(ns) and ns == [i for i in range(5) for _ in range(n_steps)], "context size must be i for every step of frame i"
        ts = [t for _, t in oracle.calls][:n_steps]
        assert all(abs(t - (1 - k / n_steps)) < 1e-6 for k, t in enumerate(ts)), f"timesteps must be 1 - k/n: {ts}"


def check_teacher_forcing(mod):
    # oracle target for frame i = last context frame + 1 (0 for frame 0)
    oracle = mod.OracleDenoiser(from_context=True)
    free = mod.sample_autoregressive(oracle, 1, 4, 3, n_steps=2, gen=torch.Generator().manual_seed(0))
    expected_free = torch.arange(4, dtype=torch.float32)[None, :, None].expand(1, 4, 3)
    assert torch.allclose(free, expected_free, atol=1e-4), f"free-running rollout must be 0,1,2,3 per frame: {free[0, :, 0]}"
    teacher = torch.full((1, 4, 3), 10.0)
    tf = mod.sample_autoregressive(oracle, 1, 4, 3, n_steps=2, gen=torch.Generator().manual_seed(0), teacher_frames=teacher)
    expected_tf = torch.tensor([0.0, 11.0, 11.0, 11.0])[None, :, None].expand(1, 4, 3)
    assert torch.allclose(tf, expected_tf, atol=1e-4), f"teacher-forced rollout must condition on ground-truth frames: {tf[0, :, 0]}"


def check_sampler_consistent_with_forward(mod):
    # with the real model: in teacher-forced mode, the velocity used for frame i at the first step equals the
    # batched causal forward on [teacher[:, :i], x_i] -- i.e. the KV-cache path is equivalent to the full forward.
    model = mod.TinyFrameDenoiser(D=8, seed=5)
    g = torch.Generator().manual_seed(0)
    teacher = torch.randn(2, 4, 8, generator=g)
    x_i = torch.randn(2, 1, 8, generator=g)
    with torch.no_grad():
        K, V = model.context_kv(teacher[:, :3])
        v_cache = model.denoise_with_context(x_i, torch.ones(2, 1), K, V)
        x_full = torch.cat([teacher[:, :3], x_i], dim=1)
        t_full = torch.cat([torch.zeros(2, 3), torch.ones(2, 1)], dim=1)
        v_full = model(x_full, t_full, mod.causal_frame_mask(4))[:, 3:]
    assert torch.allclose(v_cache, v_full, atol=1e-5), "KV-cache path must equal the masked full forward"


def run(mod):
    check_causal_mask(mod); print("  ok  causal frame mask (future frames do not leak)")
    check_loss(mod); print("  ok  diffusion-forcing loss (value, per-frame t, oracle -> 0)")
    check_sampler_shape_and_determinism(mod); print("  ok  sampler shape / determinism")
    check_sampler_oracle_exact(mod); print("  ok  sampler with oracle denoiser is exact")
    check_teacher_forcing(mod); print("  ok  teacher-forced vs free-running context")
    check_sampler_consistent_with_forward(mod); print("  ok  KV-cache path equals masked full forward")
