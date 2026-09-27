"""Prediction parameterizations (eps / x0 / v) and the DDIM sampler with eta

A variance-preserving diffusion model can output any of three equivalent targets given x_t = sqrt(abar) x0 +
sqrt(1 - abar) eps:
    eps-prediction,  x0-prediction,  v-prediction with  v = sqrt(abar) eps - sqrt(1 - abar) x0   (Salimans & Ho 2022).
Implement the exact conversions between them, then the DDIM update (Song et al. 2021) with the stochasticity knob eta:
    x0_hat   = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t)
    sigma_t  = eta * sqrt((1 - abar_prev) / (1 - abar_t)) * sqrt(1 - abar_t / abar_prev)
    x_prev   = sqrt(abar_prev) x0_hat + sqrt(1 - abar_prev - sigma_t^2) eps + sigma_t z,   z ~ N(0, I)
eta = 0 is the deterministic DDIM (probability-flow) sampler; eta = 1 recovers the DDPM ancestral posterior.
Sampling runs on a strided subsequence of timesteps (e.g. 20 of 200) and must reach t = 0 with abar_prev = 1.

Signatures:
    x0_from_eps(x_t, eps, abar) -> x0        eps_from_x0(x_t, x0, abar) -> eps
    v_from_x0_eps(x0, eps, abar) -> v        x0_eps_from_v(x_t, v, abar) -> (x0, eps)
    ddim_sigma(abar_t, abar_prev, eta) -> sigma
    ddim_step(x_t, eps, abar_t, abar_prev, eta=0.0, noise=None) -> x_prev
    ddim_sample(model, alphas_cumprod, timesteps, x_T, eta=0.0, generator=None) -> x_0

Constraints: torch CPU float32. x tensors are (B, D); abar may be a python float, a 0-d tensor or a (B, 1) tensor —
write the formulas so they broadcast. `alphas_cumprod` is the full (T,) schedule (0-indexed, index 0 = least noisy);
`timesteps` is a decreasing int64 tensor of indices to visit, e.g. [199, 189, ..., 9]; after the last one the
"previous" alpha-bar is 1 (clean data). model is called as model(x_t, t) with t an int64 (B,) tensor and returns eps.

Interview budget: 30 min

Discussion follow-ups:
  - Why does v-prediction behave better than eps-prediction at high noise (abar -> 0) and than x0-prediction at
    low noise? Relate to the SNR and the "angle" phi = arctan(sqrt(1 - abar)/sqrt(abar)).
  - Show that eta = 1 reproduces DDPM's posterior mean AND variance, and that eta = 0 is a first-order ODE solver
    in the (x/sqrt(abar), sqrt((1 - abar)/abar)) coordinates (Song et al. "DDIM is Euler on the PF-ODE").
  - Rectified-flow / v-prediction / EDM preconditioning: how do these relate (they are all reweightings and
    reparameterizations of the same denoiser)?
  - Stride schedules: uniform vs quadratic vs "trailing" timestep spacing (Lin et al. 2023, zero-terminal-SNR).
"""
import torch

__implement__ = ["x0_from_eps", "eps_from_x0", "v_from_x0_eps", "x0_eps_from_v", "ddim_sigma", "ddim_step",
                 "ddim_sample"]


# ----------------------------------------------------------------------------- given helpers
def make_alphas_cumprod(T=200, beta_start=1e-3, beta_end=0.08):
    """Given helper: abar for a linear beta schedule, float32 (T,)."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    return torch.cumprod(1.0 - betas, dim=0)


def strided_timesteps(T, n_steps):
    """Given helper: decreasing int64 indices with uniform stride, ending at the smallest visited index.
    e.g. T=200, n_steps=20 -> [190, 180, ..., 0] + 9 = [199, 189, ..., 9]."""
    stride = T // n_steps
    return torch.arange(T - 1, -1, -stride, dtype=torch.long)[:n_steps]


class GaussianOracle:
    """Given helper: exact eps-predictor for x0 ~ N(mu, sigma^2 I) (see 01_ddpm)."""

    def __init__(self, alphas_cumprod, mu, sigma):
        self.ab = alphas_cumprod
        self.mu = torch.as_tensor(mu, dtype=torch.float32)
        self.sigma = float(sigma)

    def __call__(self, x_t, t):
        ab = self.ab[t][:, None]
        return torch.sqrt(1 - ab) * (x_t - torch.sqrt(ab) * self.mu) / (ab * self.sigma ** 2 + 1 - ab)


def _as_t(a):
    return torch.as_tensor(a, dtype=torch.float32)


# ----------------------------------------------------------------------------- to implement
def x0_from_eps(x_t, eps, abar):
    """x0 = (x_t - sqrt(1 - abar) eps) / sqrt(abar).  Shapes: x_t, eps (B, D); abar broadcastable to (B, D)."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def eps_from_x0(x_t, x0, abar):
    """eps = (x_t - sqrt(abar) x0) / sqrt(1 - abar).  Shapes as in x0_from_eps."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def v_from_x0_eps(x0, eps, abar):
    """v = sqrt(abar) eps - sqrt(1 - abar) x0.  Shapes: x0, eps (B, D); abar broadcastable."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def x0_eps_from_v(x_t, v, abar):
    """Invert v-prediction using x_t: x0 = sqrt(abar) x_t - sqrt(1 - abar) v,  eps = sqrt(1 - abar) x_t + sqrt(abar) v.

    Returns:
        (x0, eps), each (B, D).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def ddim_sigma(abar_t, abar_prev, eta):
    """DDIM noise level sigma_t = eta * sqrt((1 - abar_prev) / (1 - abar_t)) * sqrt(1 - abar_t / abar_prev).

    Args:
        abar_t, abar_prev: floats / tensors (broadcastable). eta: float >= 0.
    Returns:
        tensor broadcastable with x (0 when eta = 0 or abar_prev = 1).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def ddim_step(x_t, eps, abar_t, abar_prev, eta=0.0, noise=None):
    """One DDIM update from level abar_t to abar_prev given the predicted eps.

    x_prev = sqrt(abar_prev) x0_hat + sqrt(1 - abar_prev - sigma^2) eps + sigma * noise
    with x0_hat = x0_from_eps(x_t, eps, abar_t). `noise` is (B, D) standard normal; if None and eta > 0, draw
    it with torch.randn_like. Clamp 1 - abar_prev - sigma^2 at 0 before the sqrt for numerical safety.

    Args:
        x_t, eps: (B, D). abar_t, abar_prev: floats or broadcastable tensors. eta: float. noise: (B, D) or None.
    Returns:
        (B, D) tensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def ddim_sample(model, alphas_cumprod, timesteps, x_T, eta=0.0, generator=None):
    """Run ddim_step over `timesteps` (decreasing indices), starting from x_T at timesteps[0].

    For step i: abar_t = alphas_cumprod[timesteps[i]], abar_prev = alphas_cumprod[timesteps[i+1]] for i < last,
    and abar_prev = 1.0 for the last step. Call the model under torch.no_grad() with t = timesteps[i] expanded to
    (B,). Noise (when eta > 0) is drawn with torch.randn(x.shape, generator=generator).

    Args:
        model: (x_t, t) -> eps. alphas_cumprod: (T,). timesteps: (S,) int64 decreasing. x_T: (B, D). eta: float.
        generator: torch.Generator or None.
    Returns:
        (B, D) tensor at abar = 1.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
