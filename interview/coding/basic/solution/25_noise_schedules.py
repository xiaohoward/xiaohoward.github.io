"""
25 — Noise schedules: DDPM betas (linear / cosine), SNR, flow-matching SNR, timestep shift, zero-terminal SNR

The bookkeeping every diffusion codebase gets wrong once. Build the standard discrete DDPM schedules, derive the
alphas_cumprod / sqrt terms / SNR from them, write the flow-matching (rectified-flow) SNR, the SD3 / FLUX resolution
timestep shift with its inverse, and the "zero terminal SNR" rescale of Lin et al. 2023. Every formula must be exact.

Signatures:
    def linear_betas(T, beta_start=1e-4, beta_end=0.02) -> Tensor (T,)
    def cosine_alpha_bar(t, s=0.008) -> Tensor                    continuous ᾱ(t) for t in [0, 1], ᾱ(0) = 1
    def cosine_betas(T, s=0.008, max_beta=0.999) -> Tensor (T,)
    def alphas_cumprod_from_betas(betas) -> Tensor (T,)
    def ddpm_terms(alphas_cumprod) -> dict   sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, snr, log_snr
    def flow_snr(t) -> Tensor                 SNR of x_t = (1-t) x0 + t eps   = ((1-t)/t)^2
    def shift_timestep(t, s) -> Tensor        t' = s t / (1 + (s-1) t)
    def unshift_timestep(t_shifted, s) -> Tensor   exact inverse
    def shift_for_resolution(k) -> float      shift factor so SNR at t' equals SNR(t)/k for k× more pixels: sqrt(k)
    def enforce_zero_terminal_snr(betas) -> Tensor (T,)   rescaled betas with ᾱ_T = 0 exactly, ᾱ_0 unchanged

Constraints: torch (CPU) only; do schedule math in float64 (that is what DDPM / diffusers do, and it matters for the
cumulative products). Conventions: DDPM index t = 0..T-1, ᾱ_t = prod_{i<=t}(1 - β_i); SNR_t = ᾱ_t / (1 - ᾱ_t).
Flow-matching t in [0, 1] with t = 0 clean data, t = 1 pure noise (SD3 / FLUX / WAN convention, so "shift" pushes
t toward 1 = more noise for s > 1).

Interview budget: 20 min

Discussion follow-ups:
  * Why does the linear schedule with T=1000 leave ᾱ_T ≈ 4e-5 (not 0), and what visible artifact does that cause
    at inference (Lin et al.: "the model never saw pure noise", mean-brightness bias)? Why does the fix require
    v-prediction rather than ε-prediction at the last step?
  * Derive the sqrt(k) resolution shift: k pixels of iid unit noise averaged together have variance 1/k, while a
    smooth signal keeps its amplitude, so a k× larger image at the same t has k× higher effective SNR. Where does
    this reasoning break (non-flat spectra — cf. SPEED's power-law argument)?
  * The DDPM cosine schedule and the flow interpolant are both "SNR schedules". Show that any two schedules with
    the same logSNR(t) up to reparameterization of t train the same denoiser (Kingma et al. VDM) — so what does the
    time parameterization actually change? (Where the loss weight / where sampler steps land.)
"""
import math
import torch

__implement__ = ["linear_betas", "cosine_alpha_bar", "cosine_betas", "alphas_cumprod_from_betas", "ddpm_terms",
                 "flow_snr", "shift_timestep", "unshift_timestep", "shift_for_resolution",
                 "enforce_zero_terminal_snr"]


def linear_betas(T, beta_start=1e-4, beta_end=0.02):
    """
    Ho et al. 2020 linear schedule: T evenly spaced betas from beta_start to beta_end inclusive.
    Returns (T,) float64 tensor. (These values are for T = 1000; for other T the DDPM/ADM codebases rescale
    beta_start/beta_end by 1000/T — do NOT do that here, just linspace the given endpoints.)
    """
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def cosine_alpha_bar(t, s=0.008):
    """
    Nichol & Dhariwal 2021 continuous cosine schedule: with f(t) = cos((t + s) / (1 + s) * pi/2)^2,
        ᾱ(t) = f(t) / f(0),   t in [0, 1]   (t is the FRACTION of the way through diffusion, not an integer step).
    t : float tensor of any shape (or python float). Returns same-shape float64 tensor. ᾱ(0) = 1 exactly, ᾱ(1) = 0.
    """
    t = torch.as_tensor(t, dtype=torch.float64)
    f = lambda u: torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
    return f(t) / f(torch.zeros((), dtype=torch.float64))


def cosine_betas(T, s=0.008, max_beta=0.999):
    """
    Discrete betas from the cosine schedule (improved-diffusion `betas_for_alpha_bar`):
        β_i = min(1 - ᾱ((i+1)/T) / ᾱ(i/T), max_beta),   i = 0..T-1.
    Returns (T,) float64 tensor. The clip at max_beta = 0.999 prevents the singularity as ᾱ -> 0 at the end.
    """
    i = torch.arange(T, dtype=torch.float64)
    ab = cosine_alpha_bar(i / T, s)
    ab_next = cosine_alpha_bar((i + 1) / T, s)
    return torch.clamp(1 - ab_next / ab, max=max_beta)


def alphas_cumprod_from_betas(betas):
    """betas : (T,). Returns ᾱ_t = prod_{i<=t} (1 - β_i), shape (T,), same dtype."""
    return torch.cumprod(1.0 - betas, dim=0)


def ddpm_terms(alphas_cumprod):
    """
    alphas_cumprod : (T,) tensor ᾱ_t. Returns dict with keys (all shape (T,), same dtype):
        "sqrt_alphas_cumprod"           sqrt(ᾱ_t)                  (coefficient of x0 in q(x_t | x0))
        "sqrt_one_minus_alphas_cumprod" sqrt(1 - ᾱ_t)              (coefficient of ε)
        "snr"                           ᾱ_t / (1 - ᾱ_t)
        "log_snr"                       log(ᾱ_t) - log(1 - ᾱ_t)
    """
    ab = alphas_cumprod
    return {
        "sqrt_alphas_cumprod": torch.sqrt(ab),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - ab),
        "snr": ab / (1.0 - ab),
        "log_snr": torch.log(ab) - torch.log(1.0 - ab),
    }


def flow_snr(t):
    """
    SNR of the flow-matching / rectified-flow interpolant x_t = (1 - t) x0 + t ε with unit-variance data and noise:
        SNR(t) = ((1 - t) / t)^2   (signal amplitude (1-t) over noise amplitude t, squared).
    t : tensor (any shape) or float in (0, 1]. Returns same-shape tensor. SNR(0.5) = 1; SNR(t) -> inf as t -> 0.
    """
    t = torch.as_tensor(t)
    return ((1.0 - t) / t) ** 2


def shift_timestep(t, s):
    """
    SD3 / FLUX timestep shift: t' = s t / (1 + (s - 1) t).  t : tensor or float in [0, 1]; s > 0 (s = 3 for SD3
    at 1024^2, FLUX uses a resolution-dependent s). Fixes 0 and 1; for s > 1 pushes t toward 1 (more noise).
    Returns same shape as t. shift_timestep(0.5, 3) = 0.75.
    """
    t = torch.as_tensor(t)
    return s * t / (1.0 + (s - 1.0) * t)


def unshift_timestep(t_shifted, s):
    """
    Inverse of shift_timestep: t = t' / (s - (s - 1) t'). unshift(shift(t, s), s) == t for all t in [0, 1].
    (Equivalently shift_timestep(t', 1/s).)
    """
    t_shifted = torch.as_tensor(t_shifted)
    return t_shifted / (s - (s - 1.0) * t_shifted)


def shift_for_resolution(k):
    """
    k : ratio of pixel (token) counts new/reference, e.g. 4.0 for going 512^2 -> 1024^2.
    Returns the shift s such that flow_snr(shift_timestep(t, s)) == flow_snr(t) / k for every t, i.e. the shifted
    timestep on the k× larger image has the SAME per-pixel-block SNR as t on the reference image.
    Derivation: k iid unit-variance noise pixels averaged keep the signal but shrink noise variance by 1/k, so the
    larger image has k× the effective SNR; demanding (1-t')/t' = (1-t)/(t sqrt(k)) and solving gives
        t' = sqrt(k) t / (1 + (sqrt(k) - 1) t)   =>   s = sqrt(k)   (SD3 eq. 23: alpha_m = sqrt(m / n)).
    """
    return math.sqrt(k)


def enforce_zero_terminal_snr(betas):
    """
    Lin et al. 2023 "Common Diffusion Noise Schedules and Sample Steps are Flawed", Algorithm 1.
    betas : (T,) float64. Returns new betas (T,) such that, with a = sqrt(ᾱ):
        a_new = (a - a_T) * a_0 / (a_0 - a_T)        (subtract terminal, rescale so a_0 is preserved)
        ᾱ_new = a_new^2 ;  α_new_0 = ᾱ_new_0 ;  α_new_t = ᾱ_new_t / ᾱ_new_{t-1} ;  β_new = 1 - α_new
    Result: ᾱ_new_T = 0 EXACTLY (so β_new_T = 1, the last step is pure noise), ᾱ_new_0 = ᾱ_0 unchanged.
    """
    a = torch.sqrt(alphas_cumprod_from_betas(betas))
    a0, aT = a[0].clone(), a[-1].clone()
    a = (a - aT) * (a0 / (a0 - aT))
    ab = a ** 2
    alphas = torch.cat([ab[:1], ab[1:] / ab[:-1]])
    return 1.0 - alphas
