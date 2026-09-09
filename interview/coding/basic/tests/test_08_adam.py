import torch


def _make_params(dtype, seed=0):
    torch.manual_seed(seed)
    shapes = [(4, 3), (3,), (2, 2, 2)]
    ps = [torch.randn(s, dtype=dtype).requires_grad_(True) for s in shapes]
    return ps


def _loss(ps, x):
    # non-trivial loss coupling all params: sum of squares + a cubic term + linear term
    return sum(((p ** 2).sum() + 0.1 * (p ** 3).sum() + (p * x[i]).sum()) for i, p in enumerate(ps))


def _run_both(mod, dtype, decoupled, wd, steps=7, lr=0.05, tol=1e-6):
    ps_ref = _make_params(dtype)
    ps_mine = [p.detach().clone().requires_grad_(True) for p in ps_ref]
    xs = [torch.randn_like(p) for p in ps_ref]
    if decoupled:
        ref = torch.optim.AdamW(ps_ref, lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=wd)
    else:
        ref = torch.optim.Adam(ps_ref, lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=wd)
    mine = mod.Adam(ps_mine, lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=wd, decoupled=decoupled)
    for s in range(steps):
        for opt, ps in ((ref, ps_ref), (mine, ps_mine)):
            opt.zero_grad()
            _loss(ps, xs).backward()
            opt.step()
        for i, (a, b) in enumerate(zip(ps_ref, ps_mine)):
            err = (a - b).abs().max().item()
            assert err < tol, f"step {s} param {i}: max abs diff {err:.3e} > {tol} (decoupled={decoupled}, wd={wd})"


def check_adam_no_decay(mod):
    _run_both(mod, torch.float64, decoupled=False, wd=0.0)
    _run_both(mod, torch.float32, decoupled=False, wd=0.0, tol=1e-5)


def check_adam_l2_decay(mod):
    _run_both(mod, torch.float64, decoupled=False, wd=0.1)


def check_adamw_decoupled(mod):
    _run_both(mod, torch.float64, decoupled=True, wd=0.1)
    _run_both(mod, torch.float32, decoupled=True, wd=0.1, tol=1e-5)


def check_first_step_magnitude(mod):
    # With bias correction, first step is ~lr * sign(g) for every element.
    p = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64, requires_grad=True)
    p.grad = torch.tensor([0.5, -3.0, 1e-3], dtype=torch.float64)
    opt = mod.Adam([p], lr=0.1)
    before = p.detach().clone()
    opt.step()
    delta = (p.detach() - before)
    expected = -0.1 * torch.sign(p.grad)
    assert torch.allclose(delta, expected, atol=1e-6), f"first step should be -lr*sign(g), got {delta}"


def check_skips_none_grad(mod):
    a = torch.ones(2, requires_grad=True)
    b = torch.ones(2, requires_grad=True)
    a.grad = torch.ones(2)
    opt = mod.Adam([a, b], lr=0.1)
    opt.step()
    assert torch.equal(b.detach(), torch.ones(2)), "param with grad=None must be untouched"
    assert not torch.equal(a.detach(), torch.ones(2)), "param with grad must move"


def run(mod):
    check_adam_no_decay(mod);        print("  ok  adam matches torch.optim.Adam (no decay)")
    check_adam_l2_decay(mod);        print("  ok  adam matches torch.optim.Adam (L2 decay)")
    check_adamw_decoupled(mod);      print("  ok  adamw matches torch.optim.AdamW (decoupled decay)")
    check_first_step_magnitude(mod); print("  ok  first step is -lr*sign(g) (bias correction)")
    check_skips_none_grad(mod);      print("  ok  params with grad=None are skipped")
