import math
import numpy as np
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_q_sample_stats(mod):
    torch.manual_seed(0)
    N = 100_000
    c = 0.7
    x0 = torch.full((N, 3), c)
    eps = torch.randn(N, 3)
    # DDPM
    ab = torch.full((N,), 0.36)
    xt = mod.q_sample_ddpm(x0, eps, ab)
    assert xt.shape == (N, 3)
    assert abs(xt.mean().item() - math.sqrt(0.36) * c) < 5e-3, f"DDPM q_sample mean {xt.mean():.4f} != sqrt(ᾱ) x0 = {0.6 * c:.4f}"
    assert abs(xt.var().item() - 0.64) < 8e-3, f"DDPM q_sample var {xt.var():.4f} != 1-ᾱ = 0.64"
    assert torch.equal(mod.q_sample_ddpm(x0, eps, torch.ones(N)), x0), "ᾱ = 1 must return x0 exactly"
    assert torch.equal(mod.q_sample_ddpm(x0, eps, torch.zeros(N)), eps), "ᾱ = 0 must return ε exactly"
    # flow
    t = torch.full((N,), 0.3)
    xt = mod.q_sample_flow(x0, eps, t)
    assert abs(xt.mean().item() - 0.7 * c) < 5e-3, f"flow q_sample mean {xt.mean():.4f} != (1-t) x0"
    assert abs(xt.var().item() - 0.09) < 3e-3, f"flow q_sample var {xt.var():.4f} != t^2 = 0.09"
    assert torch.equal(mod.q_sample_flow(x0, eps, torch.zeros(N)), x0) and torch.equal(mod.q_sample_flow(x0, eps, torch.ones(N)), eps)
    # per-sample coefficients broadcast on image-shaped input
    x0i = torch.randn(4, 3, 8, 8)
    epsi = torch.randn(4, 3, 8, 8)
    abi = torch.tensor([1.0, 0.5, 0.25, 0.0])
    xti = mod.q_sample_ddpm(x0i, epsi, abi)
    assert torch.equal(xti[0], x0i[0]) and torch.equal(xti[3], epsi[3])
    assert torch.allclose(xti[2], 0.5 * x0i[2] + math.sqrt(0.75) * epsi[2])
    xtf = mod.q_sample_flow(x0i, epsi, torch.tensor([0.0, 0.25, 0.5, 1.0]))
    assert torch.equal(xtf[0], x0i[0]) and torch.allclose(xtf[1], 0.75 * x0i[1] + 0.25 * epsi[1])


def check_targets_and_roundtrip(mod):
    torch.manual_seed(1)
    B = 6
    x0 = torch.randn(B, 3, 8, 8)
    eps = torch.randn(B, 3, 8, 8)
    ab = torch.tensor([0.999, 0.9, 0.5, 0.3, 0.1, 0.01])
    xt = mod.q_sample_ddpm(x0, eps, ab)
    # targets
    assert torch.equal(mod.make_target(x0, eps, "eps"), eps)
    assert torch.equal(mod.make_target(x0, eps, "x0"), x0)
    v = mod.make_target(x0, eps, "v", alphas_cumprod_t=ab)
    v_ref = ab.sqrt().view(-1, 1, 1, 1) * eps - (1 - ab).sqrt().view(-1, 1, 1, 1) * x0
    assert torch.allclose(v, v_ref, atol=1e-6), "v target must be sqrt(ᾱ) ε - sqrt(1-ᾱ) x0"
    assert torch.allclose(mod.make_target(x0, eps, "flow", t=torch.rand(B)), eps - x0), "flow target must be ε - x0"
    try:
        mod.make_target(x0, eps, "score")
        assert False, "unknown kind must raise ValueError"
    except ValueError:
        pass
    # round trips from every DDPM parameterization
    for kind, pred in (("eps", eps), ("x0", x0), ("v", v)):
        x0h, epsh = mod.ddpm_pred_to_x0_eps(pred, xt, ab, kind)
        assert x0h.shape == x0.shape and epsh.shape == eps.shape
        assert torch.allclose(x0h, x0, atol=2e-4), f"{kind}-pred -> x0 round trip failed, max err {(x0h - x0).abs().max():.2e}"
        assert torch.allclose(epsh, eps, atol=2e-4), f"{kind}-pred -> eps round trip failed, max err {(epsh - eps).abs().max():.2e}"
    # v-pred consistency: (x0, eps) -> v -> (x0, eps) -> v again
    x0h, epsh = mod.ddpm_pred_to_x0_eps(v, xt, ab, "v")
    assert torch.allclose(mod.make_target(x0h, epsh, "v", alphas_cumprod_t=ab), v, atol=1e-5)
    # v is unit-variance for unit-variance x0 and eps (orthogonal rotation of (x0, eps))
    assert abs(v.var().item() - 1.0) < 0.1, "v has the same variance as x0/ε (it is a rotation of the pair)"
    # flow round trip
    t = torch.tensor([0.05, 0.2, 0.5, 0.7, 0.9, 1.0])
    xtf = mod.q_sample_flow(x0, eps, t)
    x0h, epsh = mod.flow_pred_to_x0_eps(eps - x0, xtf, t)
    assert torch.allclose(x0h, x0, atol=1e-5) and torch.allclose(epsh, eps, atol=1e-5), "flow velocity -> (x0, eps) round trip failed"
    # cross-check DDPM eps->x0 against a manual formula at one sample
    x0m = (xt[2] - math.sqrt(0.5) * eps[2]) / math.sqrt(0.5)
    assert torch.allclose(mod.ddpm_pred_to_x0_eps(eps, xt, ab, "eps")[0][2], x0m, atol=1e-5)


def check_min_snr(mod):
    snr = torch.tensor([0.5, 5.0, 50.0])
    g = 5.0
    assert torch.allclose(mod.min_snr_weight(snr, g, "eps"), torch.tensor([1.0, 1.0, 0.1])), "eps weights = min(SNR,γ)/SNR"
    assert torch.allclose(mod.min_snr_weight(snr, g, "x0"), torch.tensor([0.5, 5.0, 5.0])), "x0 weights = min(SNR,γ)"
    assert torch.allclose(mod.min_snr_weight(snr, g, "v"), torch.tensor([0.5 / 1.5, 5.0 / 6.0, 5.0 / 51.0])), \
        "v weights = min(SNR,γ)/(SNR+1)"
    # equivalence: weight_kind * (implicit x0-weight of kind) == min(SNR, γ) for every kind
    implicit = {"eps": snr, "x0": torch.ones(3), "v": snr + 1}
    for k, w in implicit.items():
        assert torch.allclose(mod.min_snr_weight(snr, g, k) * w, torch.clamp(snr, max=g))
    # gamma -> inf recovers the unweighted loss for eps-pred
    assert torch.allclose(mod.min_snr_weight(snr, 1e9, "eps"), torch.ones(3))
    try:
        mod.min_snr_weight(snr, g, "flow")
        assert False, "unknown kind must raise ValueError"
    except ValueError:
        pass


def check_weighted_mse(mod):
    torch.manual_seed(2)
    pred = torch.randn(4, 3, 5, 5)
    tgt = torch.randn(4, 3, 5, 5)
    w = torch.tensor([1.0, 2.0, 0.0, 0.5])
    loss = mod.weighted_mse(pred, tgt, w)
    assert loss.dim() == 0, "weighted_mse must return a scalar"
    ref = (w * ((pred - tgt) ** 2).mean(dim=(1, 2, 3))).mean()
    assert torch.allclose(loss, ref, atol=1e-6), "weighted_mse must be mean_b(w_b * per-sample MSE)"
    assert torch.allclose(mod.weighted_mse(pred, tgt, torch.ones(4)), torch.nn.functional.mse_loss(pred, tgt), atol=1e-6), \
        "unit weights must give plain MSE"
    # loss on the true target is zero, for every parameterization
    x0 = torch.randn(4, 3, 5, 5); eps = torch.randn(4, 3, 5, 5); ab = torch.tensor([0.9, 0.5, 0.2, 0.05])
    snr = ab / (1 - ab)
    for kind in ("eps", "x0", "v"):
        t = mod.make_target(x0, eps, kind, alphas_cumprod_t=ab)
        assert mod.weighted_mse(t, t, mod.min_snr_weight(snr, 5.0, kind)).item() == 0.0
    assert mod.weighted_mse(eps - x0, mod.make_target(x0, eps, "flow"), torch.ones(4)).item() == 0.0
    # gradient flows to pred
    p = pred.clone().requires_grad_(True)
    mod.weighted_mse(p, tgt, w).backward()
    assert p.grad is not None and p.grad[2].abs().max().item() == 0.0, "zero weight sample must get zero gradient"


def run(mod):
    check_q_sample_stats(mod);         print("  ok  q_sample (DDPM / flow) mean and variance match closed form on 1e5 samples")
    check_targets_and_roundtrip(mod);  print("  ok  eps / x0 / v / flow targets; all conversions round-trip to (x0, eps)")
    check_min_snr(mod);                print("  ok  min-SNR-γ weights match hand values for eps / x0 / v")
    check_weighted_mse(mod);           print("  ok  weighted_mse; loss on the true target is zero")
