import torch


def _fp32_reference_step(mod, master, batch, lr, weight_decay):
    """Pure-fp32 step with torch.optim.AdamW on a copy of master; returns updated params and fp32 grads."""
    ps = [p.clone().requires_grad_(True) for p in master]
    opt = torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    loss = mod.forward_loss(ps, batch)
    loss.backward()
    grads = [p.grad.detach().clone() for p in ps]
    opt.step()
    return [p.detach() for p in ps], grads, float(loss.detach())


def check_adamw_matches_torch(mod):
    torch.manual_seed(0)
    master = mod.make_model(seed=3)
    ref = [p.clone().requires_grad_(True) for p in master]
    opt = torch.optim.AdamW(ref, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    state = mod.init_adam_state(master)
    g = torch.Generator().manual_seed(5)
    for _ in range(5):
        grads = [torch.randn(p.shape, generator=g) for p in master]
        for p, gr in zip(ref, grads):
            p.grad = gr.clone()
        opt.step()
        mod.adamw_update_(master, grads, state, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    assert state["step"] == 5, "state['step'] must count applied updates"
    for p, r in zip(master, ref):
        assert torch.allclose(p, r.detach(), atol=1e-6), f"adamw_update_ mismatch vs torch.optim.AdamW: {(p - r).abs().max()}"


def check_matches_fp32_when_no_overflow(mod):
    master = mod.make_model(seed=0)
    batch = mod.make_batch(seed=1)
    lr = 1e-3
    # (1) single step from the same point: grads and the master update must match a pure-fp32 AdamW step
    ref_params, ref_grads, ref_loss = _fp32_reference_step(mod, master, batch, lr, 0.0)
    scaler = mod.DynamicLossScaler(init_scale=2.0 ** 10, growth_interval=1000)
    state = mod.init_adam_state(master)
    loss, skipped, grads32 = mod.mixed_precision_step(master, batch, scaler, state, lr=lr)
    assert not skipped, "no overflow expected at scale 2^10"
    assert abs(loss - ref_loss) < 1e-2 * max(1.0, ref_loss), f"fp16 loss {loss} vs fp32 {ref_loss}"
    for g16, g32 in zip(grads32, ref_grads):
        assert g16.dtype == torch.float32, "unscaled grads must be fp32"
        assert g16.shape == g32.shape
        assert torch.allclose(g16, g32, atol=3e-3 * g32.abs().max().item() + 1e-5), \
            f"unscaled fp16 grad differs from fp32 grad by {(g16 - g32).abs().max()}"
    for p, r in zip(master, ref_params):
        assert torch.allclose(p, r, atol=1e-5), f"master update differs from fp32 AdamW: {(p - r).abs().max()}"
    assert state["step"] == 1, "opt state step must advance on a good step"
    # (2) a 5-step trajectory stays close to a pure-fp32 AdamW trajectory (fp16 ReLU boundary flips allow ~lr drift)
    master = mod.make_model(seed=0)
    ref = [p.clone().requires_grad_(True) for p in master]
    opt = torch.optim.AdamW(ref, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scaler = mod.DynamicLossScaler(init_scale=2.0 ** 10, growth_interval=1000)
    state = mod.init_adam_state(master)
    for step in range(5):
        opt.zero_grad()
        mod.forward_loss(ref, batch).backward()
        opt.step()
        _, skipped, _ = mod.mixed_precision_step(master, batch, scaler, state, lr=lr, weight_decay=0.01)
        assert not skipped
    for p, r in zip(master, ref):
        assert torch.allclose(p, r.detach(), atol=3 * lr), f"5-step trajectory drifted by {(p - r).abs().max()}"
    assert scaler.scale == 2.0 ** 10, "scale must not change before growth_interval good steps"
    assert state["step"] == 5


def check_overflow_skips_and_halves(mod):
    master = mod.make_model(seed=0)
    batch = mod.make_batch(seed=1)
    scaler = mod.DynamicLossScaler(init_scale=2.0 ** 8, growth_interval=2)
    state = mod.init_adam_state(master)
    before = [p.clone() for p in master]

    def inject_inf(grads):
        grads = [g.clone() for g in grads]
        grads[0].view(-1)[0] = float("inf")
        return grads

    loss, skipped, _ = mod.mixed_precision_step(master, batch, scaler, state, lr=1e-2, grad_hook=inject_inf)
    assert skipped is True or skipped == True, "step with an inf grad must be skipped"  # noqa: E712
    assert scaler.scale == 2.0 ** 7, f"scale must halve on overflow, got {scaler.scale}"
    assert scaler.good_steps == 0, "good_steps must reset on overflow"
    assert state["step"] == 0, "optimizer state must not advance on a skipped step"
    for p, b in zip(master, before):
        assert torch.equal(p, b), "master weights must be untouched on a skipped step"

    def inject_nan(grads):
        grads = [g.clone() for g in grads]
        grads[2].view(-1)[3] = float("nan")
        return grads

    mod.mixed_precision_step(master, batch, scaler, state, lr=1e-2, grad_hook=inject_nan)
    assert scaler.scale == 2.0 ** 6, "nan must also count as overflow"

    # genuine overflow: an absurd scale makes fp16 grads overflow without any injection
    scaler2 = mod.DynamicLossScaler(init_scale=2.0 ** 40, growth_interval=2)
    _, skipped2, _ = mod.mixed_precision_step(master, batch, scaler2, state, lr=1e-2)
    assert skipped2 and scaler2.scale == 2.0 ** 39, "scale 2^40 must overflow fp16 grads and be halved"


def check_scale_grows(mod):
    master = mod.make_model(seed=0)
    batch = mod.make_batch(seed=1)
    N = 3
    scaler = mod.DynamicLossScaler(init_scale=2.0 ** 8, growth_interval=N)
    state = mod.init_adam_state(master)
    for i in range(N - 1):
        mod.mixed_precision_step(master, batch, scaler, state, lr=1e-3)
        assert scaler.scale == 2.0 ** 8, f"scale grew too early at good step {i + 1}"
    mod.mixed_precision_step(master, batch, scaler, state, lr=1e-3)
    assert scaler.scale == 2.0 ** 9, f"scale must double after {N} good steps, got {scaler.scale}"
    assert scaler.good_steps == 0, "good_steps must reset after growth"
    for _ in range(N):
        mod.mixed_precision_step(master, batch, scaler, state, lr=1e-3)
    assert scaler.scale == 2.0 ** 10, "scale must keep growing every N good steps"


def check_scaler_unit(mod):
    sc = mod.DynamicLossScaler(init_scale=4.0, growth_interval=2)
    loss = torch.tensor(1.5, requires_grad=True)
    s = sc.scale_loss(loss)
    assert float(s.detach()) == 6.0 and s.requires_grad, "scale_loss must multiply by the scale and stay in the graph"
    g16 = [torch.tensor([8.0, -4.0], dtype=torch.float16)]
    g32, found = sc.unscale_(g16)
    assert g32[0].dtype == torch.float32 and torch.equal(g32[0], torch.tensor([2.0, -1.0])) and not found
    _, found = sc.unscale_([torch.tensor([1.0, float("inf")], dtype=torch.float16)])
    assert found, "unscale_ must flag inf"


def run(mod):
    check_scaler_unit(mod); print("  ok  scaler unit behaviour")
    check_adamw_matches_torch(mod); print("  ok  adamw_update_ matches torch.optim.AdamW")
    check_matches_fp32_when_no_overflow(mod); print("  ok  fp16 step matches fp32 reference (no overflow)")
    check_overflow_skips_and_halves(mod); print("  ok  overflow skips the step and halves the scale")
    check_scale_grows(mod); print("  ok  scale grows after N good steps")
