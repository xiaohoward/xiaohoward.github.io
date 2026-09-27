import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_nonsat_losses(mod):
    real = torch.tensor([0.0, 2.0, -1.0])
    fake = torch.tensor([0.0, -3.0, 1.0])
    # closed form: softplus(-r) + softplus(f), means
    sp = lambda v: math.log1p(math.exp(v))
    d_ref = sum(sp(-r) for r in real.tolist()) / 3 + sum(sp(f) for f in fake.tolist()) / 3
    g_ref = sum(sp(-f) for f in fake.tolist()) / 3
    d = mod.d_loss_nonsat(real, fake)
    g = mod.g_loss_nonsat(fake)
    assert d.ndim == 0 and g.ndim == 0
    assert abs(d.item() - d_ref) < 1e-6, f"D nonsat {d.item():.6f} != {d_ref:.6f}"
    assert abs(g.item() - g_ref) < 1e-6, f"G nonsat {g.item():.6f} != {g_ref:.6f}"
    # at logits 0 every term is ln 2
    zero = torch.zeros(4)
    assert abs(mod.d_loss_nonsat(zero, zero).item() - 2 * math.log(2)) < 1e-6
    assert abs(mod.g_loss_nonsat(zero).item() - math.log(2)) < 1e-6
    # matches BCE-with-logits formulation
    torch.manual_seed(0)
    r, f = torch.randn(16), torch.randn(16)
    bce = F.binary_cross_entropy_with_logits
    assert torch.allclose(mod.d_loss_nonsat(r, f), bce(r, torch.ones(16)) + bce(f, torch.zeros(16)), atol=1e-6)
    assert torch.allclose(mod.g_loss_nonsat(f), bce(f, torch.ones(16)), atol=1e-6)
    # numerical stability at huge logits
    big = torch.tensor([1e4, -1e4])
    d_big = mod.d_loss_nonsat(big, -big)
    assert torch.isfinite(d_big) and abs(d_big.item() - 1e4) < 1.0, "must be stable for |logit| = 1e4"
    assert abs(mod.g_loss_nonsat(torch.tensor([-1e4])).item() - 1e4) < 1.0
    assert abs(mod.g_loss_nonsat(torch.tensor([1e4])).item()) < 1e-3
    # (B,1) shaped logits also fine
    assert torch.allclose(mod.d_loss_nonsat(r[:, None], f[:, None]), mod.d_loss_nonsat(r, f))


def check_hinge_losses(mod):
    real = torch.tensor([0.5, 2.0, -1.0, 1.0])
    fake = torch.tensor([-0.5, -3.0, 1.0, -1.0])
    d_ref = ((1 - real).clamp(min=0).mean() + (1 + fake).clamp(min=0).mean()).item()
    assert abs(mod.d_loss_hinge(real, fake).item() - d_ref) < 1e-6, "hinge D must be mean relu(1-r) + mean relu(1+f)"
    # hand numbers: relu(1-r) = [.5, 0, 2, 0] -> .625 ; relu(1+f) = [.5, 0, 2, 0] -> .625 ; total 1.25
    assert abs(mod.d_loss_hinge(real, fake).item() - 1.25) < 1e-6
    assert abs(mod.g_loss_hinge(fake).item() - (-fake.mean().item())) < 1e-6, "hinge G must be -mean(fake)"
    # perfectly separated with margin -> zero D loss
    assert mod.d_loss_hinge(torch.tensor([1.0, 5.0]), torch.tensor([-1.0, -9.0])).item() == 0.0
    # gradient of hinge D is zero beyond the margin
    r = torch.tensor([3.0, 0.2], requires_grad=True)
    mod.d_loss_hinge(r, torch.zeros(2)).backward()
    assert r.grad[0] == 0.0 and abs(r.grad[1].item() + 0.5) < 1e-6


class _QuadD(nn.Module):
    """D(x) = 0.5 * x^T A x + b^T x  -> grad = A x + b (A symmetric)."""

    def __init__(self, A, b):
        super().__init__()
        self.A = nn.Parameter(A.clone())
        self.b = nn.Parameter(b.clone())

    def forward(self, x):
        return 0.5 * ((x @ self.A) * x).sum(-1) + x @ self.b


def check_r1(mod):
    torch.manual_seed(1)
    A = torch.randn(3, 3); A = 0.5 * (A + A.T)
    b = torch.randn(3)
    D = _QuadD(A, b)
    x = torch.randn(8, 3)
    gamma = 10.0
    r1 = mod.r1_penalty(D, x, gamma)
    assert r1.ndim == 0
    grad_true = x @ A + b                                    # (8, 3)
    ref = 0.5 * gamma * grad_true.pow(2).sum(-1).mean()
    assert torch.allclose(r1, ref, atol=1e-5), f"R1 {r1.item():.5f} != analytic {ref.item():.5f}"
    # finite-difference check of ||grad D||^2 for one sample
    eps = 1e-3
    x0 = x[:1]
    fd = torch.stack([(D(x0 + eps * torch.eye(3)[i]) - D(x0 - eps * torch.eye(3)[i])) / (2 * eps) for i in range(3)]).flatten()
    r1_single = mod.r1_penalty(D, x0, 2.0)                 # gamma=2 -> exactly ||grad||^2
    assert abs(r1_single.item() - fd.pow(2).sum().item()) < 1e-3, "R1 vs finite differences"
    # create_graph: backward must reach D's parameters. d/dA of ||A x + b||^2 is nonzero
    D.zero_grad()
    mod.r1_penalty(D, x, gamma).backward()
    assert D.A.grad is not None and D.A.grad.abs().sum() > 0, "R1 must be differentiable wrt D params (create_graph=True)"
    assert D.b.grad is not None and D.b.grad.abs().sum() > 0
    # x_real without requires_grad must be handled; x_real itself must not receive grads / be modified
    x_plain = torch.randn(4, 3)
    _ = mod.r1_penalty(D, x_plain, 1.0)
    assert not x_plain.requires_grad
    # gamma=0 -> 0
    assert mod.r1_penalty(D, x, 0.0).item() == 0.0
    # also on the given MLP D
    G, Dm = mod.make_models(seed=0)
    v = mod.r1_penalty(Dm, mod.sample_real(16, torch.Generator().manual_seed(0)), 1.0)
    assert v.item() > 0 and torch.isfinite(v)


def _flat(m):
    return torch.cat([p.detach().flatten().clone() for p in m.parameters()])


def check_gan_step(mod):
    for loss in ("nonsat", "hinge"):
        G, D = mod.make_models(z_dim=4, seed=0)
        opt_g = torch.optim.Adam(G.parameters(), lr=1e-3)
        opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(0)
        x = mod.sample_real(32, g)
        g0, d0 = _flat(G), _flat(D)
        out = mod.gan_step(G, D, x, opt_g, opt_d, z_dim=4, loss=loss, r1_gamma=1.0, generator=g)
        assert set(out) >= {"loss_d", "loss_g", "r1"} and all(isinstance(v, float) for v in out.values())
        assert not torch.equal(_flat(D), d0), "D step must update D"
        assert not torch.equal(_flat(G), g0), "G step must update G"
        # Now isolate: D step only touches D. Use a no-op G optimiser (lr=0) and check G unchanged; and vice versa.
        G2, D2 = mod.make_models(z_dim=4, seed=1)
        g2, d2 = _flat(G2), _flat(D2)
        mod.gan_step(G2, D2, x, torch.optim.SGD(G2.parameters(), lr=0.0), torch.optim.SGD(D2.parameters(), lr=0.1),
                     z_dim=4, loss=loss, r1_gamma=0.0, generator=g)
        assert torch.equal(_flat(G2), g2) and not torch.equal(_flat(D2), d2), "with G lr=0 only D may change"
        G3, D3 = mod.make_models(z_dim=4, seed=1)
        g3, d3 = _flat(G3), _flat(D3)
        mod.gan_step(G3, D3, x, torch.optim.SGD(G3.parameters(), lr=0.1), torch.optim.SGD(D3.parameters(), lr=0.0),
                     z_dim=4, loss=loss, r1_gamma=0.0, generator=g)
        assert torch.equal(_flat(D3), d3) and not torch.equal(_flat(G3), g3), "with D lr=0 only G may change"
    # explicit detach check: when opt_g.zero_grad is called (start of the G step) G must hold no gradient from the D step
    G, D = mod.make_models(z_dim=4, seed=2)
    opt_d = torch.optim.SGD(D.parameters(), lr=0.1)

    class _Probe(torch.optim.SGD):
        def zero_grad(self, set_to_none=True):
            assert all(p.grad is None or p.grad.abs().sum() == 0 for p in G.parameters()), \
                "the D step leaked gradients into G (fake batch must be detached)"
            super().zero_grad(set_to_none=set_to_none)

    mod.gan_step(G, D, mod.sample_real(16, torch.Generator().manual_seed(3)), _Probe(G.parameters(), lr=0.1), opt_d,
                 z_dim=4, loss="nonsat", r1_gamma=0.5, generator=torch.Generator().manual_seed(4))
    # r1 reported as 0 when disabled
    out = mod.gan_step(G, D, mod.sample_real(16), torch.optim.SGD(G.parameters(), lr=0.0),
                       torch.optim.SGD(D.parameters(), lr=0.0), z_dim=4, r1_gamma=0.0)
    assert out["r1"] == 0.0


def check_training_moves_generator(mod):
    G, D = mod.make_models(z_dim=4, seed=0)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.5, 0.999))
    g = torch.Generator().manual_seed(0)
    target = torch.tensor([2.0, -1.0])
    z = torch.randn(256, 4, generator=torch.Generator().manual_seed(9))
    d_before = (G(z).mean(0) - target).norm().item()
    for _ in range(200):
        mod.gan_step(G, D, mod.sample_real(64, g), opt_g, opt_d, z_dim=4, loss="nonsat", r1_gamma=1.0, generator=g)
    d_after = (G(z).mean(0) - target).norm().item()
    assert d_after < 0.5 * d_before, f"generator mean must move toward the data mean: {d_before:.3f} -> {d_after:.3f}"


def run(mod):
    check_nonsat_losses(mod);           print("  ok  non-saturating D/G losses: closed form, BCE form, stability")
    check_hinge_losses(mod);            print("  ok  hinge D/G losses")
    check_r1(mod);                      print("  ok  R1 penalty: analytic quadratic D, finite differences, create_graph")
    check_gan_step(mod);                print("  ok  alternating step updates only D then only G; fake detached")
    check_training_moves_generator(mod); print("  ok  200 steps move G toward the data distribution")
