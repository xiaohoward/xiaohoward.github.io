"""
34 — Behavior cloning heads, a flow-matching diffusion-policy step + sampler, and a Wilson success-rate CI

Imitation learning for manipulation in three flavours: (i) regress the action from the observation with an MSE head
or a Gaussian head with a learned, clamped log-std; (ii) diffusion policy in ACTION space — learn a velocity field
over action chunks with the flow-matching interpolant a_t = (1 - t) a0 + t eps and sample a chunk with K Euler
steps; (iii) report success rate over n rollouts with a 95% Wilson confidence interval.

Given: MLP (Linear-ReLU-Linear), FlowPolicyNet(obs, a_t, t) -> velocity, make_multimodal_dataset.
Signatures:
    def mse_loss(pred, target) -> scalar
    def gaussian_nll_loss(mean, log_std, target, log_std_min=-5.0, log_std_max=2.0) -> scalar
    def bc_step(policy, opt, obs, actions, head) -> float           one optimizer step, head in {"mse", "gaussian"}
    def flow_matching_loss(net, obs, a0, t, eps) -> scalar          a_t = (1-t) a0 + t eps, target v = eps - a0
    def diffusion_policy_step(net, opt, obs, a0, gen) -> float      samples t ~ U(0,1), eps ~ N(0,I), steps
    def sample_actions(net, obs, chunk_shape, K, gen) -> Tensor     K Euler steps from t=1 (noise) to t=0 (data)
    def wilson_interval(k, n, z=1.96) -> (p_hat, lo, hi)
    def evaluate_success(outcomes) -> dict(rate, lo, hi, n)

Constraints: torch (CPU) only. Losses are means over batch AND action dims. log_std is clamped BEFORE use (so the
gradient is zero outside the clamp range). t is per-sample (B,) and broadcasts over the chunk (B, H, D). Flow
convention: t = 0 is clean data, t = 1 is pure noise, the velocity is d a_t / d t = eps - a0. Euler goes from
t = 1 to t = 0 with step 1/K: a <- a - (1/K) * v(a, t).

Interview budget: 25 min

Discussion follow-ups:
  * Why does an MSE / unimodal Gaussian head fail on multimodal demos (two ways around an obstacle), and what do
    diffusion / flow policies, mixture heads, and action tokens each do about it?
  * A diffusion policy predicts a chunk of H actions. Why does chunking help (compounding error, idle behaviour,
    non-Markovian demos) and what does it cost in reactivity? How does receding-horizon execution recover it?
  * Velocity vs epsilon vs x0 parameterization: which is best-conditioned near t = 0 and t = 1, and how does the
    choice change the sampler (Euler on the ODE vs DDIM)?
  * Success rate 8/10: why is +-1.96*sqrt(p(1-p)/n) (Wald) wrong here, how many rollouts do you need to separate a
    70% from an 80% policy, and how do paired seeds help?
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__implement__ = ["mse_loss", "gaussian_nll_loss", "bc_step", "flow_matching_loss", "diffusion_policy_step",
                 "sample_actions", "wilson_interval", "evaluate_success"]


class MLP(nn.Module):
    """Given: Linear-ReLU-Linear-ReLU-Linear."""

    def __init__(self, in_dim, out_dim, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


class FlowPolicyNet(nn.Module):
    """
    Given: velocity network for action chunks. forward(obs (B, obs_dim), a_t (B, H, D), t (B,)) -> (B, H, D).
    Flattens the chunk, concatenates [obs, a_t, t] and runs an MLP.
    """

    def __init__(self, obs_dim, H, D, hidden=128, seed=0):
        super().__init__()
        self.H, self.D = H, D
        self.mlp = MLP(obs_dim + H * D + 1, H * D, hidden, seed)

    def forward(self, obs, a_t, t):
        B = obs.shape[0]
        h = torch.cat([obs, a_t.reshape(B, -1), t.reshape(B, 1).to(obs.dtype)], dim=-1)
        return self.mlp(h).view(B, self.H, self.D)


def make_multimodal_dataset(n=512, seed=0):
    """
    Given: obs (n, 2) ~ U(-1, 1)^2, actions (n, 2). Each demo goes around the obstacle either way:
    a = [obs_x + 0.5 * s, s * (1 - obs_y^2)] with s = +-1 chosen at random, plus small noise. Bimodal in s.
    """
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(n, 2, generator=g) * 2 - 1
    s = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    act = torch.stack([obs[:, 0] + 0.5 * s, s * (1 - obs[:, 1] ** 2)], dim=-1)
    act = act + 0.02 * torch.randn(n, 2, generator=g)
    return obs, act


def mse_loss(pred, target):
    """pred, target : (B, D) (or any equal shapes). Returns mean over all elements of (pred - target)^2."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def gaussian_nll_loss(mean, log_std, target, log_std_min=-5.0, log_std_max=2.0):
    """
    Diagonal Gaussian negative log-likelihood with a learned, clamped log-std.
    mean, log_std, target : (B, D). log_std is clamped to [log_std_min, log_std_max] first (torch.clamp, so no
    gradient flows outside the range). Returns the mean over B and D of
        0.5 * ((target - mean) / std)^2 + log_std + 0.5 * log(2 pi),   std = exp(log_std)
    i.e. -Normal(mean, std).log_prob(target).mean().
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def bc_step(policy, opt, obs, actions, head="mse"):
    """
    One behavior-cloning gradient step. obs : (B, obs_dim), actions : (B, D).
    head = "mse":      policy(obs) -> (B, D) mean; loss = mse_loss.
    head = "gaussian": policy(obs) -> (B, 2D) = [mean | log_std] split along the last dim; loss = gaussian_nll_loss
                       with default clamps.
    Zeroes grads, backprops, steps opt. Returns the loss as a Python float.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def flow_matching_loss(net, obs, a0, t, eps):
    """
    Flow-matching (rectified-flow) loss in action space.
    obs : (B, obs_dim); a0 : (B, H, D) clean action chunk; t : (B,) in [0, 1]; eps : (B, H, D) ~ N(0, I).
    a_t = (1 - t) a0 + t eps (t broadcast to (B, 1, 1)); target velocity v* = eps - a0;
    returns mean over B, H, D of (net(obs, a_t, t) - v*)^2. If net returns exactly eps - a0 the loss is 0.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def diffusion_policy_step(net, opt, obs, a0, gen):
    """
    One training step of the flow-matching diffusion policy. obs : (B, obs_dim), a0 : (B, H, D), gen : torch.Generator.
    Samples t ~ U(0, 1) of shape (B,) and eps ~ N(0, I) of shape a0.shape using gen (t first, then eps),
    computes flow_matching_loss, zero_grad / backward / step. Returns the loss as a float.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


@torch.no_grad()
def sample_actions(net, obs, chunk_shape, K, gen):
    """
    Sample an action chunk per observation with K Euler steps of the probability-flow ODE.
    obs : (B, obs_dim); chunk_shape : (H, D); K : int steps; gen : torch.Generator.
    a <- randn((B, H, D), gen) at t = 1; for k = 0..K-1: t_k = 1 - k/K,  a <- a - (1/K) * net(obs, a, t_k * ones(B)).
    Returns a : (B, H, D) at t = 0. No gradient is needed.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def wilson_interval(k, n, z=1.96):
    """
    Wilson score interval for a binomial proportion. k successes out of n trials (n >= 1), z the normal quantile
    (1.96 for 95%). Returns (p_hat, lo, hi) as Python floats with p_hat = k / n and
        centre = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
        half   = z / (1 + z^2 / n) * sqrt(p_hat (1 - p_hat) / n + z^2 / (4 n^2))
        lo, hi = centre -+ half   (never outside [0, 1]; k = 0 gives lo = 0, k = n gives hi = 1).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def evaluate_success(outcomes):
    """
    outcomes : bool tensor / list of n rollout outcomes. Returns {"rate": k/n, "lo": lo, "hi": hi, "n": n} using the
    95% Wilson interval (z = 1.96).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
