import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


def check_kl(mod):
    torch.manual_seed(0)
    mu = torch.randn(8, 5) * 2
    logvar = torch.randn(8, 5)
    kl = mod.kl_standard_normal(mu, logvar)
    assert kl.shape == (8,), f"kl must be per-sample (B,), got {tuple(kl.shape)}"
    ref = kl_divergence(Normal(mu, torch.exp(0.5 * logvar)), Normal(torch.zeros_like(mu), torch.ones_like(mu))).sum(-1)
    assert torch.allclose(kl, ref, atol=1e-5), f"KL mismatch vs torch.distributions, max err {(kl - ref).abs().max():.2e}"
    assert (kl >= 0).all(), "KL must be non-negative"
    z = torch.zeros(3, 4)
    assert torch.allclose(mod.kl_standard_normal(z, z), torch.zeros(3), atol=1e-7), "KL(N(0,I)||N(0,I)) must be 0"
    # differentiable
    mu.requires_grad_(True); logvar.requires_grad_(True)
    mod.kl_standard_normal(mu, logvar).sum().backward()
    assert torch.allclose(mu.grad, mu.detach()), "dKL/dmu = mu"
    assert torch.allclose(logvar.grad, 0.5 * (logvar.detach().exp() - 1)), "dKL/dlogvar = 0.5*(exp(logvar)-1)"


def check_reparameterize(mod):
    g = torch.Generator().manual_seed(1)
    mu = torch.tensor([[1.0, -2.0, 0.5]]).expand(20000, 3).contiguous().requires_grad_(True)
    logvar = torch.tensor([[0.0, 2.0, -1.0]]).expand(20000, 3).contiguous().requires_grad_(True)
    z = mod.reparameterize(mu, logvar, generator=g)
    assert z.shape == (20000, 3)
    assert torch.allclose(z.mean(0), mu[0].detach(), atol=0.05), f"sample mean {z.mean(0)} != mu"
    assert torch.allclose(z.std(0), torch.exp(0.5 * logvar[0].detach()), atol=0.05), \
        f"sample std {z.std(0)} != exp(0.5*logvar)"
    # differentiability: grad wrt mu is 1 per element, grad wrt logvar is 0.5*sigma*eps (nonzero)
    z.sum().backward()
    assert mu.grad is not None and torch.allclose(mu.grad, torch.ones_like(mu)), "dz/dmu must be 1"
    assert logvar.grad is not None and logvar.grad.abs().sum() > 0, "gradient must flow to logvar"
    # generator determinism
    z1 = mod.reparameterize(mu.detach(), logvar.detach(), generator=torch.Generator().manual_seed(5))
    z2 = mod.reparameterize(mu.detach(), logvar.detach(), generator=torch.Generator().manual_seed(5))
    assert torch.equal(z1, z2), "same generator seed must give the same sample"
    # logvar = -inf-ish -> z == mu
    z0 = mod.reparameterize(mu.detach(), torch.full_like(logvar.detach(), -80.0), generator=g)
    assert torch.allclose(z0, mu.detach(), atol=1e-6)


def check_loss(mod):
    torch.manual_seed(2)
    B, D, Z = 6, 10, 3
    logits, x = torch.randn(B, D), (torch.rand(B, D) < 0.5).float()
    mu, logvar = torch.randn(B, Z), torch.randn(B, Z)
    parts = mod.vae_loss(logits, x, mu, logvar, beta=1.0)
    for k in ("loss", "recon", "kl"):
        assert k in parts and parts[k].ndim == 0, f"'{k}' must be a 0-dim tensor"
    recon_ref = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / B
    kl_ref = mod.kl_standard_normal(mu, logvar).mean()
    assert torch.allclose(parts["recon"], recon_ref, atol=1e-5), "recon must be BCE summed over pixels, mean over batch"
    assert torch.allclose(parts["kl"], kl_ref, atol=1e-6)
    assert torch.allclose(parts["loss"], recon_ref + kl_ref, atol=1e-5)
    p2 = mod.vae_loss(logits, x, mu, logvar, beta=4.0)
    assert torch.allclose(p2["loss"], recon_ref + 4.0 * kl_ref, atol=1e-5), "beta must scale the KL"
    p0 = mod.vae_loss(logits, x, mu, logvar, beta=0.0)
    assert torch.allclose(p0["loss"], recon_ref, atol=1e-6), "beta=0 must ignore the KL"
    # beta=0 -> no gradient reaches mu through the loss
    mu_g = mu.clone().requires_grad_(True)
    l0 = mod.vae_loss(logits, x, mu_g, logvar, beta=0.0)["loss"]
    if l0.requires_grad:
        (gm,) = torch.autograd.grad(l0, mu_g, allow_unused=True)
        assert gm is None or gm.abs().sum() == 0, "with beta=0 the loss must not depend on mu"


def check_training(mod):
    x = mod.make_binary_data(n=512, d_in=16, seed=0)
    model = mod.TinyVAE(d_in=16, hidden=64, z_dim=2, seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    first = mod.train_step(model, x, opt, beta=1.0, generator=g)
    assert set(first) >= {"loss", "recon", "kl"} and all(isinstance(v, float) for v in first.values())
    hist = [first["loss"]]
    for _ in range(199):
        hist.append(mod.train_step(model, x, opt, beta=1.0, generator=g)["loss"])
    last = sum(hist[-10:]) / 10
    assert last < 0.8 * hist[0], f"loss must decrease over 200 steps: {hist[0]:.3f} -> {last:.3f}"
    assert last < 16 * 0.6931 * 0.9, f"loss {last:.3f} must beat the trivial p=0.5 decoder (16*ln2 = 11.09)"
    # KL stays finite / positive, and the posterior is not collapsed to the prior
    parts = mod.train_step(model, x, opt, beta=1.0, generator=g)
    assert parts["kl"] > 0.1, "posterior collapsed (KL ~ 0)"
    # parameters actually changed
    model2 = mod.TinyVAE(d_in=16, hidden=64, z_dim=2, seed=0)
    diff = sum((p - q).abs().sum().item() for p, q in zip(model.parameters(), model2.parameters()))
    assert diff > 1e-3


def run(mod):
    check_kl(mod);             print("  ok  closed-form KL matches torch.distributions + gradients")
    check_reparameterize(mod); print("  ok  reparameterise: mean/std statistics, gradients to mu and logvar")
    check_loss(mod);           print("  ok  ELBO loss parts, beta scaling, beta=0 ignores KL")
    check_training(mod);       print("  ok  loss decreases over 200 steps on synthetic 2D-latent data")
