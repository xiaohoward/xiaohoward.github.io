"""Spectral progressive diffusion: power spectrum, activation times and DCT spectral expansion

Progressive-resolution samplers (SPEED-style) denoise at low resolution first and grow the latent when the next band of
frequencies becomes "active". Implement the four building blocks:
  (a) Orthonormal DCT-II on the last two dims and the radial power spectrum of a batch of latents: mean |X[ky,kx]|^2 per
      radial bin in Nyquist-normalised frequency f = sqrt((ky/H)^2 + (kx/W)^2)  (f = 1 is Nyquist on an axis).
  (b) Power-law fit log P = log A - beta log f by least squares over f in [f_min, f_max].
  (c) Activation time. Under the interpolant x_t = (1 - t) x_0 + t eps, with an ORTHONORMAL transform every noise bin
      has unit power, so bin f has signal power (1-t)^2 P(f) and noise power t^2. It "activates" (SNR = 1) at
      (1-t)^2 P = t^2  ->  (1-t) sqrt(P) = t  ->  t* = sqrt(P) / (1 + sqrt(P)).  Sampling runs t: 1 -> 0, so a frequency
      with lower P activates later (smaller t*). A stage whose Nyquist frequency is f_c can be left once t <= t*(f_c) +
      delta (delta >= 0 = "switch a bit early").
  (d) Spectral expansion at a stage transition: take the low-res latent x_low (h x w) at time t, DCT it, embed the
      coefficients in the low-frequency corner of an H x W DCT grid, fill every new coefficient with t * DCT(noise)
      (i.e. exactly the noise power that a fresh x_t would have there), inverse DCT.
Signatures:
    dct_matrix(n) -> (n, n) ; dct2d(x) ; idct2d(y)
    radial_power_spectrum(x) -> (f_centers, P)
    fit_power_law(f, P, f_min, f_max) -> (A, beta)
    activation_time(P) -> t*           ;  stage_transition_step(timesteps, f_cutoff, A, beta, delta) -> int
    spectral_expand(x_low, t, H, W, generator) -> (B, C, H, W)
Constraints: torch CPU only, float32. DCT-II orthonormal: C[k, i] = s_k cos(pi (2i + 1) k / (2n)), s_0 = sqrt(1/n),
s_k = sqrt(2/n); dct2d(x) = C_H x C_W^T ; idct2d = C_H^T y C_W (C is orthogonal so idct2d(dct2d(x)) == x).

Interview budget: 40 min

Discussion follow-ups:
  - Why does embedding an h x w spectrum into H x W attenuate the pixel-domain signal by r_eff = sqrt(HW / hw) under an
    orthonormal transform, and how do you fix the SNR (rescale the low-freq block by r_eff, or re-align t)?
  - Why does a per-step brick-wall low-pass blur (late frequencies get too few steps, ringing) while a staged reveal at
    a transition is sharp? Why does a staged low-pass at full resolution give no speedup?
  - Video: the temporal axis is ~50x more energetic than spatial at the same normalised frequency; what does that do
    to the order of spatial vs temporal upsampling stages?
  - Multistep solvers (UniPC, DPM++) keep a history of past model outputs -- what must you do at a resolution change?
"""
import math

import torch

__implement__ = [
    "dct_matrix",
    "dct2d",
    "idct2d",
    "radial_power_spectrum",
    "fit_power_law",
    "activation_time",
    "stage_transition_step",
    "spectral_expand",
]


def dct_matrix(n):
    """Orthonormal DCT-II matrix C of shape (n, n), float32: (C @ x)[k] = s_k sum_i x[i] cos(pi (2i+1) k / (2n)).

    s_0 = sqrt(1/n), s_k = sqrt(2/n) for k >= 1. C @ C.T == I.
    """
    i = torch.arange(n, dtype=torch.float64)
    k = torch.arange(n, dtype=torch.float64)[:, None]
    C = torch.cos(math.pi * (2 * i[None, :] + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    C[0] *= math.sqrt(0.5)
    return C.float()


def dct2d(x):
    """2-D orthonormal DCT-II over the last two dims of x (..., H, W): C_H @ x @ C_W^T. Returns same shape."""
    H, W = x.shape[-2:]
    return dct_matrix(H) @ x @ dct_matrix(W).T


def idct2d(y):
    """Inverse of dct2d over the last two dims: C_H^T @ y @ C_W."""
    H, W = y.shape[-2:]
    return dct_matrix(H).T @ y @ dct_matrix(W)


def radial_frequency_grid(H, W):
    """Given helper: (H, W) tensor of Nyquist-normalised radial frequency f = sqrt((ky/H)^2 + (kx/W)^2)."""
    ky = torch.arange(H, dtype=torch.float32)[:, None] / H
    kx = torch.arange(W, dtype=torch.float32)[None, :] / W
    return torch.sqrt(ky ** 2 + kx ** 2)


def make_power_law_latents(B=4, C=4, H=64, W=64, A=1.0, beta=2.0, seed=0):
    """Given helper: latents whose DCT coefficients are N(0, A f^-beta) (DC = 0). Returns (B, C, H, W)."""
    g = torch.Generator().manual_seed(seed)
    f = radial_frequency_grid(H, W)
    amp = torch.zeros(H, W)
    amp[f > 0] = torch.sqrt(A * f[f > 0] ** (-beta))
    coef = torch.randn(B, C, H, W, generator=g) * amp
    return idct2d(coef)


def radial_power_spectrum(x):
    """Radially binned power spectrum of a batch of latents.

    Args:
        x: (B, C, H, W) float32.
    Method: y = dct2d(x); power = y^2 averaged over B and C -> (H, W). Let n = min(H, W); bin index of a coefficient
      is floor(f * n) where f is from radial_frequency_grid. P[b] is the MEAN power per coefficient in bin b (not the
      sum), f_center[b] = (b + 0.5) / n. Skip the DC coefficient (ky = kx = 0) and any empty bin.
    Returns:
        (f_centers, P): 1-D float32 tensors of equal length, increasing f.
    """
    B, C, H, W = x.shape
    n = min(H, W)
    power = (dct2d(x) ** 2).mean(dim=(0, 1))
    f = radial_frequency_grid(H, W)
    idx = torch.floor(f * n).long()
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[0, 0] = False
    nb = int(idx.max().item()) + 1
    sums = torch.zeros(nb).index_add_(0, idx[valid], power[valid])
    counts = torch.zeros(nb).index_add_(0, idx[valid], torch.ones(int(valid.sum())))
    keep = counts > 0
    P = sums[keep] / counts[keep]
    f_centers = (torch.arange(nb, dtype=torch.float32)[keep] + 0.5) / n
    return f_centers, P


def fit_power_law(f, P, f_min=0.1, f_max=0.7):
    """Least-squares fit of log P = log A - beta log f on bins with f_min <= f <= f_max.

    Args:
        f, P: 1-D tensors (from radial_power_spectrum), P > 0.
    Returns:
        (A, beta) as python floats.
    """
    sel = (f >= f_min) & (f <= f_max)
    lf = torch.log(f[sel]).double()
    lp = torch.log(P[sel]).double()
    X = torch.stack([torch.ones_like(lf), -lf], dim=1)
    sol = torch.linalg.lstsq(X, lp[:, None]).solution[:, 0]
    return float(math.exp(sol[0])), float(sol[1])


def activation_time(P):
    """Flow time at which a bin with signal power P (unit noise power) reaches SNR = 1 under x_t = (1-t) x0 + t eps.

    (1-t)^2 P = t^2  =>  t* = sqrt(P) / (1 + sqrt(P)).  Accepts a float or tensor; returns the same type. P = 1 -> 0.5,
    P -> inf -> 1 (always active), P -> 0 -> 0 (never active).
    """
    if isinstance(P, torch.Tensor):
        s = torch.sqrt(P)
    else:
        s = math.sqrt(P)
    return s / (1.0 + s)


def stage_transition_step(timesteps, f_cutoff, A, beta, delta=0.0):
    """Index of the first sampler step at which the stage with Nyquist frequency f_cutoff should be left.

    Args:
        timesteps: 1-D tensor of the sampler's t values, strictly decreasing from ~1 to ~0 (t of each step's input).
        f_cutoff: normalised frequency of the current stage's Nyquist (e.g. 0.5 when running at half resolution).
        A, beta: power-law fit, so P(f_cutoff) = A f_cutoff^-beta.
        delta: margin >= 0; transition when t <= t*(f_cutoff) + delta (larger delta = earlier switch).
    Returns:
        int i: smallest index with timesteps[i] <= t* + delta; len(timesteps) if no step qualifies.
    """
    t_star = activation_time(A * f_cutoff ** (-beta))
    hit = torch.nonzero(timesteps <= t_star + delta)
    return int(hit[0].item()) if hit.numel() > 0 else int(len(timesteps))


def spectral_expand(x_low, t, H, W, generator):
    """Expand a low-res latent at flow time t to H x W in the DCT domain.

    Args:
        x_low: (B, C, h, w) with h <= H, w <= W. t: python float in [0, 1]. generator: torch.Generator for the noise.
    Steps: Y = dct2d(x_low); Z = zeros (B, C, H, W); Z[..., :h, :w] = Y; noise = torch.randn(B, C, H, W, generator);
      Zn = dct2d(noise); every coefficient OUTSIDE the [:h, :w] corner gets t * Zn; return idct2d(Z).
    Returns:
        (B, C, H, W) float32. dct2d(out)[..., :h, :w] == Y exactly (up to float error); the other coefficients are
        N(0, t^2).
    """
    B, C, h, w = x_low.shape
    Y = dct2d(x_low)
    Z = torch.zeros(B, C, H, W)
    Zn = dct2d(torch.randn(B, C, H, W, generator=generator))
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[:h, :w] = True
    Z[..., mask] = Y.reshape(B, C, -1)
    Z[..., ~mask] = t * Zn[..., ~mask]
    return idct2d(Z)
