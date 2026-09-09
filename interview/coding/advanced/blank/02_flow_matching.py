"""Flow matching: linear interpolant, velocity loss, Euler / Heun samplers, classifier-free guidance

Rectified-flow style flow matching on 2-D toy data. Convention (the one used by SD3 / FLUX / WAN):
    x_t = (1 - t) x0 + t eps,   t in [0, 1],   t = 0 is DATA, t = 1 is NOISE,   eps ~ N(0, I).
The model predicts the velocity v_theta(x_t, t, c) and is trained to match the conditional velocity
    u_t = d x_t / d t = eps - x0.
Sampling integrates dx/dt = v_theta(x, t) from t = 1 down to t = 0 with a fixed time grid.
Implement:
  (1) interpolate / velocity_target / flow_matching_loss;
  (2) sample_euler and sample_heun over a strictly decreasing time grid (default: uniform from 1 to 0);
  (3) cfg_velocity: classifier-free guidance v = v_uncond + scale * (v_cond - v_uncond).
A tiny conditional MLP (class label with a "null" class for CFG) and an analytic oracle velocity for Gaussian data
are given so you can test without long training.

Signatures:
    interpolate(x0, eps, t) -> x_t                      # t is (B,) float32
    velocity_target(x0, eps) -> u
    flow_matching_loss(model, x0, eps, t, cond) -> scalar tensor
    cfg_velocity(model, x, t, cond, null_cond, scale) -> v
    sample_euler(velocity_fn, x1, timesteps) -> x0     # velocity_fn(x, t) -> v; timesteps (S+1,) from 1 to 0
    sample_heun(velocity_fn, x1, timesteps) -> x0

Constraints: torch CPU, float32. x tensors are (B, D); t is (B,) float32 (broadcast it yourself). velocity_fn takes
t as a (B,) tensor. timesteps is a 1-D tensor with timesteps[0] = 1, timesteps[-1] = 0, strictly decreasing.
Heun must use exactly 2 velocity evaluations per step (no special-casing of the final step is required).

Interview budget: 30 min

Discussion follow-ups:
  - Show that the marginal velocity E[eps - x0 | x_t] is what the MSE loss recovers, and why the conditional
    target is unbiased. Why does the ODE with that field transport N(0, I) to the data marginal?
  - Time-step sampling: uniform vs logit-normal, and the "shift" used for high-res / video (SD3 shift=3).
  - CFG in velocity space vs epsilon space: are they the same? What does scale > 1 do to the marginal?
  - Euler is first order; Heun second order at 2x cost. When is Heun worth it? What about DPM-Solver / UniPC, and
    why must multistep solvers reset their history if the state is changed mid-trajectory (e.g. a resolution change)?
"""
import math

import numpy as np
import torch
import torch.nn as nn

__implement__ = ["interpolate", "velocity_target", "flow_matching_loss", "cfg_velocity", "sample_euler",
                 "sample_heun"]


# ----------------------------------------------------------------------------- given helpers
class TinyCondVelocity(nn.Module):
    """Given helper: conditional MLP velocity model. Called as model(x, t, cond):
    x (B, D) float32, t (B,) float32 in [0, 1], cond (B,) int64 class labels; label `null_class` is the
    unconditional token used for classifier-free guidance. Returns (B, D)."""

    def __init__(self, dim=2, hidden=128, n_classes=8):
        super().__init__()
        self.null_class = n_classes
        self.cemb = nn.Embedding(n_classes + 1, hidden)
        self.tproj = nn.Linear(1, hidden)
        self.net = nn.Sequential(nn.Linear(dim + 2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x, t, cond):
        h = torch.cat([x, torch.sin(self.tproj(t[:, None]) * 4.0), self.cemb(cond)], dim=-1)
        return self.net(h)


class GaussianOracleVelocity:
    """Given helper: exact marginal velocity for x0 ~ N(mu, sigma^2 I), eps ~ N(0, I).

    With x_t = (1 - t) x0 + t eps, the pair (x0, eps, x_t) is jointly Gaussian and
        Var(x_t) = (1 - t)^2 sigma^2 + t^2 =: V,   c = x_t - (1 - t) mu,
        E[x0 | x_t]  = mu + (1 - t) sigma^2 c / V,   E[eps | x_t] = t c / V,
        E[eps - x0 | x_t] = (t - (1 - t) sigma^2) c / V - mu.
    The exact ODE flow map from t = 1 to t = 0 is x0 = mu + sigma * x1 (isotropic Gaussian path).
    Called as oracle(x, t) with t (B,) float32.
    """

    def __init__(self, mu, sigma):
        self.mu = torch.as_tensor(mu, dtype=torch.float32)
        self.sigma = float(sigma)

    def __call__(self, x, t):
        t = t[:, None]
        V = (1 - t) ** 2 * self.sigma ** 2 + t ** 2
        c = x - (1 - t) * self.mu
        return (t - (1 - t) * self.sigma ** 2) * c / V - self.mu


def make_eight_gaussians(n=2048, std=0.1, seed=0):
    """Given helper: 8-Gaussians on a circle of radius 2. Returns (x (n, 2) float32, labels (n,) int64 in [0, 8))."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 8, n)
    ang = lab * (2 * math.pi / 8)
    centers = np.stack([2 * np.cos(ang), 2 * np.sin(ang)], 1)
    x = centers + std * rng.normal(size=(n, 2))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(lab, dtype=torch.long)


def uniform_timesteps(n_steps):
    """Given helper: (n_steps + 1,) float32 grid from 1.0 down to 0.0."""
    return torch.linspace(1.0, 0.0, n_steps + 1, dtype=torch.float32)


# ----------------------------------------------------------------------------- to implement
def interpolate(x0, eps, t):
    """Linear interpolant x_t = (1 - t) x0 + t eps.

    Args:
        x0: (B, D) data. eps: (B, D) noise. t: (B,) float32 in [0, 1] (per-sample).
    Returns:
        (B, D) tensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def velocity_target(x0, eps):
    """Conditional velocity d x_t / d t = eps - x0 (independent of t for the linear path).

    Args:
        x0, eps: (B, D).
    Returns:
        (B, D) tensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def flow_matching_loss(model, x0, eps, t, cond):
    """Conditional flow-matching loss: mean over batch and dims of ||model(x_t, t, cond) - (eps - x0)||^2.

    Args:
        model: callable (x_t, t, cond) -> (B, D). x0, eps: (B, D). t: (B,) float32. cond: (B,) int64.
    Returns:
        0-d tensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def cfg_velocity(model, x, t, cond, null_cond, scale):
    """Classifier-free guidance in velocity space.

    v = v_u + scale * (v_c - v_u) with v_c = model(x, t, cond), v_u = model(x, t, null_cond). Evaluate the model
    ONCE on the concatenated batch [x; x] with labels [cond; null_cond] (this is how CFG is batched in practice),
    then split. scale = 1 must return exactly v_c; scale = 0 returns v_u.

    Args:
        model: (x, t, cond) -> (B, D). x: (B, D). t: (B,). cond, null_cond: (B,) int64. scale: float.
    Returns:
        (B, D) tensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def sample_euler(velocity_fn, x1, timesteps):
    """Explicit Euler integration of dx/dt = v(x, t) from timesteps[0] = 1 to timesteps[-1] = 0.

    For each consecutive pair (t_i, t_{i+1}): x <- x + (t_{i+1} - t_i) * v(x, t_i). Note dt < 0. Run under
    torch.no_grad(). velocity_fn expects t as a (B,) tensor (expand the scalar grid value).

    Args:
        velocity_fn: (x (B, D), t (B,)) -> (B, D). x1: (B, D) initial noise. timesteps: (S + 1,) decreasing.
    Returns:
        (B, D) tensor at t = 0.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def sample_heun(velocity_fn, x1, timesteps):
    """Heun's method (explicit trapezoid, 2nd order) on the same grid, 2 velocity evaluations per step:
        v1 = v(x, t_i);  x_e = x + dt v1;  v2 = v(x_e, t_{i+1});  x <- x + dt (v1 + v2) / 2,  dt = t_{i+1} - t_i.

    Args:
        velocity_fn, x1, timesteps: as in sample_euler.
    Returns:
        (B, D) tensor at t = 0.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
