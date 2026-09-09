"""DDPM from betas: forward process, epsilon-prediction loss, ancestral sampler

You are given a beta schedule beta_1..beta_T and a tiny MLP denoiser. Implement the DDPM machinery
(Ho et al. 2020) end to end on 2-D toy data:
  (1) make_schedule(betas): all derived quantities (alphas, alpha-bar, posterior mean/variance coefficients);
  (2) q_sample: the closed-form forward marginal x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps;
  (3) ddpm_loss: the simplified epsilon-prediction training objective E ||eps - eps_theta(x_t, t)||^2;
  (4) p_mean_variance + p_sample + sample: the ancestral sampler that uses the posterior q(x_{t-1} | x_t, x0_hat)
      with x0_hat recovered from the predicted epsilon, and adds no noise at the final step (t = 0).
Timesteps are 0-indexed: t in {0, ..., T-1}, where t = 0 is the least noisy level (beta_1 in paper notation).

Signatures:
    make_schedule(betas: torch.Tensor) -> dict[str, torch.Tensor]
    q_sample(sched, x0, t, eps) -> x_t
    ddpm_loss(model, sched, x0, t, eps) -> scalar tensor
    p_mean_variance(sched, x_t, t, eps_pred, clip_x0=None) -> (mean, var, log_var, x0_hat)
    p_sample(model, sched, x_t, t, generator=None, clip_x0=None) -> x_{t-1}
    sample(model, sched, shape, generator=None, clip_x0=None) -> x_0

Constraints: torch CPU only, float32 everywhere. All tensors are (B, D) (D = 2 here); t is an int64 (B,) tensor and
may differ per element of the batch. Gather per-sample coefficients with sched[name][t][:, None]. Models are called
as model(x_t, t) with the same shapes. Do not use torch.distributions.

Interview budget: 30 min

Discussion follow-ups:
  - Why does the "simple" loss drop the per-timestep weighting of the ELBO? What weighting does it correspond to
    in the SNR-weighting view (Kingma et al. 2021, "min-SNR" tricks)?
  - Posterior variance: beta_t vs beta_tilde_t, and why learning it (Improved DDPM) helps likelihood but not FID.
  - Why is the forward marginal closed form essential for training (vs. simulating the Markov chain)?
  - How would you go from this ancestral sampler to DDIM / a probability-flow ODE, and what do you gain?
"""
import math

import numpy as np
import torch
import torch.nn as nn

__implement__ = ["make_schedule", "q_sample", "ddpm_loss", "p_mean_variance", "p_sample", "sample"]


# ----------------------------------------------------------------------------- given helpers
def make_linear_betas(T=200, beta_start=1e-3, beta_end=0.08):
    """Given helper: linear beta schedule as a float32 tensor of shape (T,). With the defaults, abar_T ~ 3e-4."""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


class TinyDenoiser(nn.Module):
    """Given helper: MLP eps-predictor. Called as model(x, t) with x (B, D) float32, t (B,) int64 -> (B, D)."""

    def __init__(self, dim=2, hidden=128, T=200):
        super().__init__()
        self.T = T
        self.temb = nn.Embedding(T, hidden)
        self.net = nn.Sequential(nn.Linear(dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], dim=-1))


class GaussianOracle:
    """Given helper: the Bayes-optimal eps-predictor for data x0 ~ N(mu, sigma^2 I).

    x_t = sqrt(abar) x0 + sqrt(1 - abar) eps is jointly Gaussian with eps, so
        E[eps | x_t] = sqrt(1 - abar) (x_t - sqrt(abar) mu) / (abar sigma^2 + 1 - abar).
    Useful to test samplers without training anything.
    """

    def __init__(self, alphas_cumprod, mu, sigma):
        self.ab = alphas_cumprod
        self.mu = torch.as_tensor(mu, dtype=torch.float32)
        self.sigma = float(sigma)

    def __call__(self, x_t, t):
        ab = self.ab[t][:, None]
        return torch.sqrt(1 - ab) * (x_t - torch.sqrt(ab) * self.mu) / (ab * self.sigma ** 2 + 1 - ab)


def make_two_moons(n=2048, noise=0.05, seed=0):
    """Given helper: two-moons toy dataset, float32 (n, 2), roughly zero mean / unit scale."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    th1 = rng.uniform(0, math.pi, n1)
    th2 = rng.uniform(0, math.pi, n - n1)
    a = np.stack([np.cos(th1), np.sin(th1)], 1)
    b = np.stack([1 - np.cos(th2), 1 - np.sin(th2) - 0.5], 1)
    x = np.concatenate([a, b], 0) + noise * rng.normal(size=(n, 2))
    x = (x - x.mean(0)) / x.std(0)
    return torch.tensor(x, dtype=torch.float32)


def _extract(a, t):
    """Given helper: gather a[t] and reshape to (B, 1) for broadcasting against (B, D)."""
    return a[t][:, None]


# ----------------------------------------------------------------------------- to implement
def make_schedule(betas):
    """Derive every per-timestep quantity the forward process, loss and sampler need.

    Args:
        betas: (T,) float32 tensor, 0 < beta_t < 1.
    Returns:
        dict with (T,) float32 tensors:
          'betas', 'alphas' (= 1 - betas), 'alphas_cumprod' (abar_t = prod_{s<=t} alpha_s),
          'alphas_cumprod_prev' (abar_{t-1}, with abar_{-1} := 1),
          'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
          'posterior_variance'   beta_tilde_t = beta_t (1 - abar_{t-1}) / (1 - abar_t),
          'posterior_log_variance_clipped'  log(max(beta_tilde_t, beta_tilde_1)) (entry 0 is clipped to entry 1
                                            because beta_tilde_0 = 0),
          'posterior_mean_coef1' = beta_t sqrt(abar_{t-1}) / (1 - abar_t)     (multiplies x0),
          'posterior_mean_coef2' = (1 - abar_{t-1}) sqrt(alpha_t) / (1 - abar_t)  (multiplies x_t).
    """
    betas = betas.to(torch.float32)
    alphas = 1.0 - betas
    ab = torch.cumprod(alphas, dim=0)
    ab_prev = torch.cat([torch.ones(1, dtype=ab.dtype), ab[:-1]])
    post_var = betas * (1.0 - ab_prev) / (1.0 - ab)
    post_log_var = torch.log(torch.cat([post_var[1:2], post_var[1:]]))
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": ab,
        "alphas_cumprod_prev": ab_prev,
        "sqrt_alphas_cumprod": torch.sqrt(ab),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - ab),
        "posterior_variance": post_var,
        "posterior_log_variance_clipped": post_log_var,
        "posterior_mean_coef1": betas * torch.sqrt(ab_prev) / (1.0 - ab),
        "posterior_mean_coef2": (1.0 - ab_prev) * torch.sqrt(alphas) / (1.0 - ab),
    }


def q_sample(sched, x0, t, eps):
    """Closed-form forward marginal q(x_t | x0).

    Args:
        sched: dict from make_schedule. x0: (B, D). t: (B,) int64 in [0, T). eps: (B, D) standard normal.
    Returns:
        x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, shape (B, D).
    """
    return _extract(sched["sqrt_alphas_cumprod"], t) * x0 + _extract(sched["sqrt_one_minus_alphas_cumprod"], t) * eps


def ddpm_loss(model, sched, x0, t, eps):
    """Simplified DDPM objective: mean squared error between eps and model(q_sample(x0, t, eps), t).

    Args:
        model: callable (x_t, t) -> eps_pred (B, D). sched: schedule dict. x0: (B, D). t: (B,) int64. eps: (B, D).
    Returns:
        0-d tensor, mean over batch and dimensions.
    """
    x_t = q_sample(sched, x0, t, eps)
    return torch.mean((model(x_t, t) - eps) ** 2)


def p_mean_variance(sched, x_t, t, eps_pred, clip_x0=None):
    """Mean and variance of p_theta(x_{t-1} | x_t) obtained by plugging x0_hat into the tractable posterior.

    x0_hat = (x_t - sqrt(1 - abar_t) eps_pred) / sqrt(abar_t), optionally clamped to [-clip_x0, clip_x0].
    mean   = coef1_t x0_hat + coef2_t x_t,  var = beta_tilde_t,  log_var = posterior_log_variance_clipped_t.

    Args:
        sched: schedule dict. x_t: (B, D). t: (B,) int64. eps_pred: (B, D). clip_x0: None or float > 0.
    Returns:
        (mean (B, D), var (B, 1), log_var (B, 1), x0_hat (B, D)).
    """
    x0_hat = (x_t - _extract(sched["sqrt_one_minus_alphas_cumprod"], t) * eps_pred) / _extract(
        sched["sqrt_alphas_cumprod"], t)
    if clip_x0 is not None:
        x0_hat = x0_hat.clamp(-clip_x0, clip_x0)
    mean = _extract(sched["posterior_mean_coef1"], t) * x0_hat + _extract(sched["posterior_mean_coef2"], t) * x_t
    var = _extract(sched["posterior_variance"], t)
    log_var = _extract(sched["posterior_log_variance_clipped"], t)
    return mean, var, log_var, x0_hat


def p_sample(model, sched, x_t, t, generator=None, clip_x0=None):
    """One ancestral step x_t -> x_{t-1}.

    Draw z ~ N(0, I) with torch.randn(..., generator=generator) and return mean + exp(0.5 log_var) * z, except
    that samples whose t == 0 get NO noise (return the mean). Run the model under torch.no_grad().

    Args:
        model: (x_t, t) -> eps_pred. sched: schedule dict. x_t: (B, D). t: (B,) int64. generator: torch.Generator
        or None. clip_x0: passed to p_mean_variance.
    Returns:
        (B, D) tensor.
    """
    with torch.no_grad():
        eps_pred = model(x_t, t)
    mean, _, log_var, _ = p_mean_variance(sched, x_t, t, eps_pred, clip_x0)
    z = torch.randn(x_t.shape, generator=generator, dtype=x_t.dtype)
    nonzero = (t != 0).to(x_t.dtype)[:, None]
    return mean + nonzero * torch.exp(0.5 * log_var) * z


def sample(model, sched, shape, generator=None, clip_x0=None):
    """Full ancestral sampling: x_{T-1} ~ N(0, I), then p_sample for t = T-1, ..., 0.

    Args:
        model, sched, generator, clip_x0: as in p_sample. shape: tuple (B, D).
    Returns:
        (B, D) tensor of samples.
    """
    T = sched["betas"].shape[0]
    x = torch.randn(shape, generator=generator)
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, dtype=torch.long)
        x = p_sample(model, sched, x, t, generator, clip_x0)
    return x
