"""Diffusion forcing: causal video training step and frame-by-frame autoregressive sampler

A causal video model is trained with an INDEPENDENT noise level per frame (diffusion forcing) so that at inference it
can denoise frame i while conditioning on already-clean frames < i, exactly like an autoregressive world model.
You get a tiny per-frame denoiser (one token per frame, one causal attention layer over frames). Implement:
  (a) causal_frame_mask(F): bool (F, F) attention mask, True where frame i may attend frame j (j <= i).
  (b) diffusion_forcing_loss(model, x0, gen): sample t_f ~ U(0,1) independently per (batch, frame), eps ~ N(0, I),
      x_t = (1 - t) x0 + t eps, flow-matching target v = eps - x0, loss = mean over everything of (v_hat - v)^2,
      where v_hat = model(x_t, t, causal_frame_mask(F)).
  (c) sample_autoregressive(model, B, F, D, n_steps, gen, teacher_frames=None): generate frames one by one. Frame i
      starts from noise at t = 1 and takes n_steps Euler steps (t_k = 1 - k / n_steps, dt = 1 / n_steps) with the
      velocity from model.denoise_with_context(x_i, t_i, K_ctx, V_ctx), where (K_ctx, V_ctx) = model.context_kv(past
      clean frames) is a KV cache of CLEAN frames (t = 0). Free-running: the context is the frames generated so far.
      Teacher-forced: the context for frame i is teacher_frames[:, :i] (ground truth), and frames are still generated.
Signatures:
    causal_frame_mask(F) -> BoolTensor (F, F)
    diffusion_forcing_loss(model, x0, gen) -> 0-d tensor
    sample_autoregressive(model, B, F, D, n_steps, gen, teacher_frames=None) -> (B, F, D)
Constraints: torch CPU. x0: (B, F, D). The model interface (given): model(x, t, mask) -> v (B, F, D) for x (B, F, D),
t (B, F); model.context_kv(frames (B, n, D)) -> (K, V) each (B, n, D) (n may be 0); model.denoise_with_context(x_i
(B, 1, D), t_i (B, 1), K, V) -> v (B, 1, D). Use the passed torch.Generator for ALL randomness.

Interview budget: 35 min

Discussion follow-ups:
  - Why independent per-frame noise levels instead of one t per clip? What does it enable at inference (variable
    horizon rollout, guidance from the past, "noisy history" for robustness)?
  - Exposure bias / error accumulation over long rollouts: how does training with noisy past frames help? What does
    teacher forcing hide?
  - KV cache: which tensors are cached per frame in a real DiT, memory for 1k frames x 4k tokens? How would you
    compress the cache (surprise-based token dropping, sliding window, sink tokens)?
  - How does this relate to CausVid / Self-Forcing distillation and to world-action models?
"""
import math

import torch
import torch.nn as nn

__implement__ = ["causal_frame_mask", "diffusion_forcing_loss", "sample_autoregressive"]


def timestep_embedding(t, dim):
    """Given helper: sinusoidal embedding of t (B, F) in [0, 1] -> (B, F, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t[..., None] * 1000.0 * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TinyFrameDenoiser(nn.Module):
    """Given scaffold: one token per frame; time-conditioned embedding, one masked attention layer over frames, MLP.

    forward(x, t, mask): x (B, F, D), t (B, F), mask (F, F) bool (True = may attend) -> v (B, F, D).
    context_kv(frames): clean frames (B, n, D) at t = 0 -> (K, V) each (B, n, D). n may be 0.
    denoise_with_context(x_i, t_i, K, V): one frame (B, 1, D) at time t_i (B, 1) attending to cached K, V plus
        itself -> v (B, 1, D). Equals forward(...)[:, i] when K, V come from the clean frames before i.
    """

    def __init__(self, D=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.D = D
        self.inp = nn.Linear(2 * D, D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.out = nn.Linear(D, D)

    def _embed(self, x, t):
        return self.inp(torch.cat([x, timestep_embedding(t, self.D)], dim=-1))

    def _attend(self, q, k, v, mask):
        s = q @ k.transpose(-1, -2) / math.sqrt(self.D)  # (B, Fq, Fk)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        return torch.softmax(s, dim=-1) @ v

    def _tail(self, h, a):
        h = h + self.proj(a)
        h = h + self.mlp(self.norm2(h))
        return self.out(h)

    def forward(self, x, t, mask=None):
        h = self._embed(x, t)
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        return self._tail(h, self._attend(q, k, v, mask))

    def context_kv(self, frames):
        B, n, D = frames.shape
        if n == 0:
            return frames.new_zeros(B, 0, D), frames.new_zeros(B, 0, D)
        h = self._embed(frames, torch.zeros(B, n))
        _, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        return k, v

    def denoise_with_context(self, x_i, t_i, K, V):
        h = self._embed(x_i, t_i)
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        a = self._attend(q, torch.cat([K, k], dim=1), torch.cat([V, v], dim=1), None)
        return self._tail(h, a)


class OracleDenoiser:
    """Given helper (for tests): knows the target frames and returns the exact straight-path velocity.

    For frame i with target x0* = target(i, context): v = (x_t - x0*) / t, which is exactly eps - x0* when
    x_t = (1 - t) x0* + t eps. If `from_context` is True, the target for frame i is (last context frame + 1) (and
    zeros for frame 0), so teacher-forced and free-running rollouts differ in a checkable way.
    """

    def __init__(self, targets=None, from_context=False):
        self.targets = targets
        self.from_context = from_context
        self.calls = []

    def __call__(self, x, t, mask=None):
        raise RuntimeError("the sampler must use context_kv / denoise_with_context")

    def context_kv(self, frames):
        return frames.clone(), frames.clone()  # the "cache" is just the frames themselves

    def denoise_with_context(self, x_i, t_i, K, V):
        n = K.shape[1]
        self.calls.append((n, float(t_i[0, 0])))
        if self.from_context:
            x0 = torch.zeros_like(x_i) if n == 0 else K[:, -1:] + 1.0
        else:
            x0 = self.targets[:, n:n + 1]
        return (x_i - x0) / t_i[..., None]


def causal_frame_mask(F):
    """Bool (F, F) tensor with mask[i, j] = True iff j <= i (frame i may attend frames up to and including itself)."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def diffusion_forcing_loss(model, x0, gen):
    """Diffusion-forcing flow-matching loss with independent per-frame noise levels.

    Args:
        model: callable model(x_t, t, mask) -> v_hat (B, F, D).
        x0: (B, F, D) clean frames. gen: torch.Generator.
    Draw t = torch.rand(B, F, generator=gen), eps = torch.randn(B, F, D, generator=gen) (in THIS order);
    x_t = (1 - t)[..., None] x0 + t[..., None] eps; v = eps - x0; mask = causal_frame_mask(F);
    Returns: 0-d tensor mean((model(x_t, t, mask) - v)^2).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def sample_autoregressive(model, B, F, D, n_steps, gen, teacher_frames=None):
    """Frame-by-frame Euler sampling with a KV cache of clean past frames.

    For i in 0..F-1:
        context = generated[:, :i] (free-running) or teacher_frames[:, :i] (teacher-forced)
        K, V = model.context_kv(context)                       (recomputed here; a real system caches it)
        x = torch.randn(B, 1, D, generator=gen)                 (fresh noise per frame, drawn in frame order)
        for k in 0..n_steps-1: t_k = 1 - k / n_steps ; v = model.denoise_with_context(x, t_k * ones(B, 1), K, V)
                               x = x - v / n_steps
        generated[:, i] = x
    Returns: (B, F, D) generated frames (even in teacher-forced mode the OUTPUT is what the model generated).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
