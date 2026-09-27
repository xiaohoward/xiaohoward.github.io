import math
import numpy as np
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))


# --- numpy references (DDPM / improved-diffusion) -------------------------------------------------------------
def _ref_cosine_betas(T, s=0.008, max_beta=0.999):
    f = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return np.array([min(1 - f((i + 1) / T) / f(i / T), max_beta) for i in range(T)])


def check_linear(mod):
    b = mod.linear_betas(1000)
    assert b.shape == (1000,) and b.dtype == torch.float64, "linear_betas must be (T,) float64"
    assert b[0].item() == 1e-4 and abs(b[-1].item() - 0.02) < 1e-12, "endpoints must be beta_start, beta_end"
    assert torch.all(b[1:] > b[:-1]), "linear betas must be strictly increasing"
    ab = mod.alphas_cumprod_from_betas(b)
    ref = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
    assert np.allclose(ab.numpy(), ref, rtol=1e-12), "alphas_cumprod mismatch vs numpy cumprod"
    assert torch.all(ab[1:] < ab[:-1]) and 0 < ab[-1] < 1e-4, f"linear ᾱ_T should be ~4e-5, got {ab[-1]:.3e}"
    assert abs(ab[-1].item() - 4.0357e-5) < 2e-8, f"DDPM ᾱ_999 = 4.0357e-05 expected, got {ab[-1]:.5e}"
    b2 = mod.linear_betas(10, 0.1, 0.5)
    assert torch.allclose(b2, torch.linspace(0.1, 0.5, 10, dtype=torch.float64))


def check_cosine(mod):
    ab0 = mod.cosine_alpha_bar(torch.tensor(0.0))
    ab1 = mod.cosine_alpha_bar(torch.tensor(1.0))
    assert abs(ab0.item() - 1.0) == 0.0, f"cosine ᾱ(0) must be exactly 1, got {ab0}"
    assert ab1.item() < 1e-30, f"cosine ᾱ(1) must be ~0, got {ab1}"
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    ab = mod.cosine_alpha_bar(t)
    assert torch.all(ab[1:] < ab[:-1]), "cosine ᾱ(t) must be strictly decreasing"
    s = 0.008
    ref = np.cos((t.numpy() + s) / (1 + s) * np.pi / 2) ** 2 / math.cos(s / (1 + s) * math.pi / 2) ** 2
    assert np.allclose(ab.numpy(), ref, atol=1e-14), "cosine_alpha_bar mismatch"
    # midpoint hand value: f(0.5)/f(0) = cos(0.508/1.008 * pi/2)^2 / cos(0.008/1.008 * pi/2)^2
    mid = math.cos(0.508 / 1.008 * math.pi / 2) ** 2 / math.cos(0.008 / 1.008 * math.pi / 2) ** 2
    assert abs(mod.cosine_alpha_bar(0.5).item() - mid) < 1e-12
    for T in (10, 1000):
        b = mod.cosine_betas(T)
        assert b.shape == (T,) and b.dtype == torch.float64
        assert np.allclose(b.numpy(), _ref_cosine_betas(T), atol=1e-12), f"cosine_betas mismatch for T={T}"
        assert b.max().item() <= 0.999 + 1e-15, "betas must be clipped at max_beta"
    b = mod.cosine_betas(1000)
    assert b[-1].item() == 0.999, "for T=1000 the last cosine beta hits the 0.999 clip"
    assert torch.all(b[1:-1] >= b[:-2]), "cosine betas are non-decreasing before the clip"
    abT = mod.alphas_cumprod_from_betas(b)
    assert abT[0].item() == 1 - b[0].item() and abT[-1].item() < 1e-3
    assert abs(mod.cosine_betas(10, max_beta=0.5).max().item() - 0.5) < 1e-15, "max_beta must be honored"


def check_terms_and_snr(mod):
    ab = torch.tensor([0.99, 0.5, 0.2, 0.01], dtype=torch.float64)
    d = mod.ddpm_terms(ab)
    for k in ("sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod", "snr", "log_snr"):
        assert k in d and d[k].shape == (4,), f"ddpm_terms missing/mis-shaped key {k}"
    assert torch.allclose(d["sqrt_alphas_cumprod"], ab.sqrt())
    assert torch.allclose(d["sqrt_one_minus_alphas_cumprod"], (1 - ab).sqrt())
    assert torch.allclose(d["snr"], torch.tensor([99.0, 1.0, 0.25, 1 / 99], dtype=torch.float64)), "SNR = ᾱ/(1-ᾱ)"
    assert torch.allclose(d["log_snr"], d["snr"].log()), "log_snr must equal log(snr)"
    assert torch.allclose(d["sqrt_alphas_cumprod"] ** 2 + d["sqrt_one_minus_alphas_cumprod"] ** 2,
                          torch.ones(4, dtype=torch.float64)), "variance preserving: sqrt terms square-sum to 1"
    # SNR is monotone decreasing along the linear schedule and log_snr is finite
    abl = mod.alphas_cumprod_from_betas(mod.linear_betas(1000))
    snr = mod.ddpm_terms(abl)["snr"]
    assert torch.all(snr[1:] < snr[:-1]) and torch.isfinite(mod.ddpm_terms(abl)["log_snr"]).all()
    # flow SNR
    t = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64)
    fs = mod.flow_snr(t)
    assert torch.allclose(fs, torch.tensor([81.0, 9.0, 1.0, 1 / 9, 1 / 81], dtype=torch.float64)), "flow SNR = ((1-t)/t)^2"
    assert torch.all(fs[1:] < fs[:-1]), "flow SNR must decrease with t"
    assert abs(mod.flow_snr(0.5).item() - 1.0) < 1e-12 and abs(mod.flow_snr(torch.tensor(1.0)).item()) < 1e-15


def check_shift(mod):
    assert abs(mod.shift_timestep(torch.tensor(0.5), 3.0).item() - 0.75) < 1e-12, "shift(0.5, s=3) must be 0.75"
    assert abs(mod.shift_timestep(0.5, 3.0).item() - 0.75) < 1e-12, "must accept python floats"
    t = torch.linspace(0, 1, 201, dtype=torch.float64)
    for s in (0.5, 1.0, 3.0, 4.0):
        ts = mod.shift_timestep(t, s)
        assert ts.shape == t.shape
        assert ts[0].item() == 0.0 and abs(ts[-1].item() - 1.0) < 1e-15, "shift must fix 0 and 1"
        assert torch.all(ts[1:] > ts[:-1]), "shift must be strictly increasing (bijection on [0,1])"
        assert torch.allclose(mod.unshift_timestep(ts, s), t, atol=1e-12), f"unshift(shift(t)) != t for s={s}"
        assert torch.allclose(mod.shift_timestep(mod.unshift_timestep(t, s), s), t, atol=1e-12)
        assert torch.allclose(mod.unshift_timestep(t, s), mod.shift_timestep(t, 1.0 / s), atol=1e-12), \
            "unshift(., s) must equal shift(., 1/s)"
    assert torch.allclose(mod.shift_timestep(t, 1.0), t), "s = 1 is the identity"
    assert torch.all(mod.shift_timestep(t[1:-1], 3.0) > t[1:-1]), "s > 1 pushes t toward 1 (more noise)"
    # SNR mapping identity: flow_snr(shift(t, s)) == flow_snr(t) / s^2, and s = sqrt(k) for k× pixels
    tt = t[1:-1]
    for k in (1.0, 2.0, 4.0, 16.0):
        s = mod.shift_for_resolution(k)
        assert abs(s - math.sqrt(k)) < 1e-12, f"shift_for_resolution({k}) must be sqrt(k)"
        assert torch.allclose(mod.flow_snr(mod.shift_timestep(tt, s)), mod.flow_snr(tt) / k, rtol=1e-10), \
            f"SNR at shifted t must equal SNR(t)/k for k={k}"
    assert abs(mod.shift_for_resolution(9.0) - 3.0) < 1e-12, "SD3 at 1024^2 vs ~341^2 -> s=3"


def check_zero_terminal_snr(mod):
    b = mod.linear_betas(1000)
    ab_old = mod.alphas_cumprod_from_betas(b)
    b_new = mod.enforce_zero_terminal_snr(b)
    assert b_new.shape == (1000,) and b_new.dtype == torch.float64
    ab_new = mod.alphas_cumprod_from_betas(b_new)
    assert ab_new[-1].item() == 0.0, f"zero-terminal-SNR rescale must give ᾱ_T = 0 exactly, got {ab_new[-1]:.3e}"
    assert b_new[-1].item() == 1.0, "last beta must be 1 (final step is pure noise)"
    assert abs(ab_new[0].item() - ab_old[0].item()) < 1e-12, "ᾱ_0 must be preserved"
    assert torch.all(ab_new[1:] < ab_new[:-1]), "rescaled ᾱ must still be strictly decreasing"
    assert torch.all(b_new[:-1] > 0) and torch.all(b_new <= 1)
    # direct check of the algorithm: sqrt(ᾱ_new) = (a - a_T) * a_0 / (a_0 - a_T)
    a = ab_old.sqrt()
    a_ref = (a - a[-1]) * a[0] / (a[0] - a[-1])
    assert torch.allclose(ab_new.sqrt(), a_ref, atol=1e-9), "rescaled sqrt(ᾱ) must match Lin et al. Algorithm 1"
    # the middle of the schedule moves only slightly
    assert (ab_new[500] - ab_old[500]).abs().item() < 0.01
    # idempotent: applying it twice changes nothing
    assert torch.allclose(mod.enforce_zero_terminal_snr(b_new), b_new, atol=1e-9)
    # also works on the cosine schedule (which already has ᾱ_T ~ 1e-3 at T=1000 due to the clip)
    bc = mod.enforce_zero_terminal_snr(mod.cosine_betas(1000))
    assert mod.alphas_cumprod_from_betas(bc)[-1].item() == 0.0


def run(mod):
    check_linear(mod);            print("  ok  linear betas / alphas_cumprod match DDPM (ᾱ_999 = 4.04e-5)")
    check_cosine(mod);            print("  ok  cosine ᾱ(0)=1, ᾱ(1)=0, betas match improved-diffusion with 0.999 clip")
    check_terms_and_snr(mod);     print("  ok  sqrt terms, SNR = ᾱ/(1-ᾱ), logSNR, flow SNR = ((1-t)/t)^2")
    check_shift(mod);             print("  ok  timestep shift is a bijection, shift(0.5,3)=0.75, SNR(shift)=SNR/k with s=sqrt(k)")
    check_zero_terminal_snr(mod); print("  ok  zero-terminal-SNR rescale: ᾱ_T = 0 exactly, ᾱ_0 preserved")
