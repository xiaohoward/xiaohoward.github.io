import math

import numpy as np
import torch


def check_dct(mod):
    for n in (1, 4, 7, 16):
        C = mod.dct_matrix(n)
        assert C.shape == (n, n) and C.dtype == torch.float32
        assert torch.allclose(C @ C.T, torch.eye(n), atol=1e-5), f"DCT matrix must be orthogonal (n={n})"
        assert torch.allclose(C[0], torch.full((n,), 1 / math.sqrt(n))), "DC row must be 1/sqrt(n)"
    # against the definition
    n = 8
    C = mod.dct_matrix(n)
    i, k = 3, 2
    expected = math.sqrt(2 / n) * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    assert abs(C[k, i].item() - expected) < 1e-6, "DCT-II entry does not match definition"
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 12, 10, generator=g)
    y = mod.dct2d(x)
    assert y.shape == x.shape
    assert torch.allclose(mod.idct2d(y), x, atol=1e-5), "idct2d(dct2d(x)) must equal x"
    # Parseval: orthonormal transform preserves energy
    assert abs((y ** 2).sum().item() - (x ** 2).sum().item()) < 1e-3 * (x ** 2).sum().item(), "Parseval violated"
    # a constant image has all energy in the DC coefficient
    const = torch.full((1, 1, 12, 10), 2.0)
    yc = mod.dct2d(const)
    assert abs(yc[0, 0, 0, 0].item() - 2.0 * math.sqrt(120)) < 1e-4 and (yc.abs().sum() - yc[0, 0, 0, 0].abs()).item() < 1e-3
    # white noise has unit power per bin (the reason P = 1 is the SNR = 1 line)
    noise = torch.randn(64, 4, 32, 32, generator=g)
    f, P = mod.radial_power_spectrum(noise)
    assert torch.all((P > 0.85) & (P < 1.15)), f"white noise must have ~unit power per DCT bin, got range {P.min()}..{P.max()}"


def check_spectrum_shape_and_binning(mod):
    x = mod.make_power_law_latents(B=2, C=2, H=32, W=48, beta=2.0, seed=1)
    f, P = mod.radial_power_spectrum(x)
    assert f.dim() == 1 and P.dim() == 1 and f.shape == P.shape, "f and P must be 1-D of the same length"
    assert torch.all(f[1:] > f[:-1]), "frequencies must be increasing"
    assert torch.all(P > 0) and torch.isfinite(P).all()
    assert f[0].item() > 0, "DC must be excluded"
    assert abs(f[0].item() - 0.5 / 32) < 1e-6, "first bin center must be (0 + 0.5) / min(H, W)"
    assert f[-1].item() < math.sqrt(2) + 0.5 / 32
    # mean power per coefficient (not sum): a per-bin recomputation for bin 3
    y2 = (mod.dct2d(x) ** 2).mean(dim=(0, 1))
    fg = mod.radial_frequency_grid(32, 48)
    idx = torch.floor(fg * 32).long()
    sel = (idx == 3)
    sel[0, 0] = False
    ref = y2[sel].mean()
    j = int(torch.argmin((f - 3.5 / 32).abs()))
    assert abs(P[j].item() - ref.item()) < 1e-4 * max(1.0, ref.item()), "P must be the mean power per coefficient in the bin"


def check_power_law_recovery(mod):
    for beta, A in [(1.92, 1.0), (2.42, 3.0), (1.5, 0.5)]:
        x = mod.make_power_law_latents(B=8, C=4, H=64, W=64, A=A, beta=beta, seed=int(beta * 10))
        f, P = mod.radial_power_spectrum(x)
        A_hat, beta_hat = mod.fit_power_law(f, P, f_min=0.1, f_max=0.7)
        assert abs(beta_hat - beta) < 0.15, f"beta {beta}: fit gave {beta_hat}"
        assert abs(math.log(A_hat / A)) < 0.25, f"A {A}: fit gave {A_hat}"
    # exact fit on a noiseless power law
    f = torch.linspace(0.05, 0.9, 40)
    P = 2.5 * f ** (-2.2)
    A_hat, beta_hat = mod.fit_power_law(f, P, 0.1, 0.8)
    assert abs(A_hat - 2.5) < 1e-3 and abs(beta_hat - 2.2) < 1e-4, f"noiseless fit inexact: {A_hat}, {beta_hat}"


def check_activation_time(mod):
    assert abs(mod.activation_time(1.0) - 0.5) < 1e-9, "P = 1 must activate at t* = 0.5"
    assert abs(mod.activation_time(4.0) - 2.0 / 3.0) < 1e-9
    assert mod.activation_time(0.0) == 0.0
    f = torch.linspace(0.05, 1.4, 50)
    P = 2.0 * f ** (-2.0)
    t = mod.activation_time(P)
    assert isinstance(t, torch.Tensor) and t.shape == f.shape
    assert torch.all(t[1:] < t[:-1]), "t* must be monotone decreasing in f for a decreasing spectrum"
    assert torch.all((t > 0) & (t < 1))
    # SNR = 1 exactly at t*
    snr = (1 - t) ** 2 * P / t ** 2
    assert torch.allclose(snr, torch.ones_like(snr), atol=1e-5), "(1-t*)^2 P must equal t*^2"


def check_stage_transition(mod):
    ts = torch.linspace(1.0, 0.0, 51)[:-1]  # 50 steps, t of each step's input: 1.0, 0.98, ..., 0.02
    A, beta = 1.0, 2.0
    fc = 0.5
    t_star = mod.activation_time(A * fc ** (-beta))  # P = 4 -> t* = 2/3
    i = mod.stage_transition_step(ts, fc, A, beta, delta=0.0)
    assert isinstance(i, int)
    assert ts[i] <= t_star and ts[i - 1] > t_star, f"transition index {i} (t={ts[i]}) is not the first step with t <= t*"
    i_early = mod.stage_transition_step(ts, fc, A, beta, delta=0.1)
    assert i_early < i, "a positive delta must switch earlier"
    assert ts[i_early] <= t_star + 0.1 and ts[i_early - 1] > t_star + 0.1
    # a finer stage (higher cutoff -> less power) transitions later
    assert mod.stage_transition_step(ts, 0.9, A, beta) > i
    # impossible cutoff: never activates within the schedule
    assert mod.stage_transition_step(ts, 100.0, 1e-9, beta) == len(ts)


def check_spectral_expand(mod):
    g = torch.Generator().manual_seed(0)
    x_low = torch.randn(2, 4, 16, 12, generator=g)
    t = 0.6
    out = mod.spectral_expand(x_low, t, 32, 24, torch.Generator().manual_seed(1))
    assert out.shape == (2, 4, 32, 24), f"shape {out.shape}"
    Y_low = mod.dct2d(x_low)
    Y_out = mod.dct2d(out)
    assert torch.allclose(Y_out[..., :16, :12], Y_low, atol=1e-4), "low-frequency corner must be preserved exactly"
    high = torch.ones(32, 24, dtype=torch.bool)
    high[:16, :12] = False
    hf = Y_out[..., high]
    assert abs(hf.std().item() - t) < 0.05 * t, f"high-frequency coefficients must have std ~ t={t}, got {hf.std()}"
    assert abs(hf.mean().item()) < 0.05
    # determinism
    out2 = mod.spectral_expand(x_low, t, 32, 24, torch.Generator().manual_seed(1))
    assert torch.equal(out, out2), "same generator seed must give the same expansion"
    # t = 0: pure zero-padding in frequency, a constant image stays exactly constant (scaled by sqrt(hw/HW))
    const = torch.full((1, 1, 8, 8), 3.0)
    o0 = mod.spectral_expand(const, 0.0, 16, 16, torch.Generator().manual_seed(2))
    assert torch.allclose(o0, torch.full((1, 1, 16, 16), 3.0 * math.sqrt(64 / 256)), atol=1e-5), \
        "constant image with t=0 must expand to a constant (attenuated by sqrt(hw/HW))"
    # t > 0: constant + zero-mean noise with the right energy
    o1 = mod.spectral_expand(const, 0.5, 16, 16, torch.Generator().manual_seed(2))
    assert abs(o1.mean().item() - 1.5) < 1e-5, "mean is the DC coefficient, which the noise does not touch"
    resid = o1 - o1.mean()
    expected_std = 0.5 * math.sqrt((256 - 64) / 256)
    assert abs(resid.std().item() - expected_std) < 0.25 * expected_std, "residual std must be t * sqrt(1 - hw/HW)"
    # same-size expansion is the identity
    same = mod.spectral_expand(x_low, 0.9, 16, 12, torch.Generator().manual_seed(3))
    assert torch.allclose(same, x_low, atol=1e-5), "expanding to the same size must be the identity"


def run(mod):
    check_dct(mod); print("  ok  orthonormal DCT-II, inverse, Parseval, unit noise power")
    check_spectrum_shape_and_binning(mod); print("  ok  radial spectrum shape / binning")
    check_power_law_recovery(mod); print("  ok  power-law fit recovers beta within 0.15")
    check_activation_time(mod); print("  ok  activation time (t* = 0.5 at P = 1, monotone)")
    check_stage_transition(mod); print("  ok  stage transition index")
    check_spectral_expand(mod); print("  ok  spectral expansion")
