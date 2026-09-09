"""
33 — VLA action pipeline: normalization, 256-bin action tokens, chunking, temporal ensembling, receding horizon

The action side of a vision-language-action policy (RT-2 / OpenVLA / ACT / pi0). Actions from demonstrations are
normalized with per-dimension dataset statistics, optionally discretized into 256 uniform bins so an LLM can emit
them as tokens, predicted in chunks of H future steps, and executed either by replanning every n_exec steps
(receding horizon) or by temporally ensembling every chunk that covers the current step (ACT).

Signatures:
    class ActionNormalizer(mode="meanstd" | "minmax", eps=1e-6)
        .fit(actions)          (N, D) demo actions -> stores per-dim stats, returns self
        .normalize(a)          (..., D) -> (..., D)
        .unnormalize(a)        inverse
    def tokenize_actions(a, n_bins=256) -> LongTensor        (..., D) in [-1, 1] -> bin ids in [0, n_bins-1]
    def detokenize_actions(tok, n_bins=256) -> Tensor        bin ids -> bin centers in (-1, 1)
    def chunk_trajectory(traj, H, stride) -> (chunks, starts) (T, D) -> (N, H, D), (N,)
    def temporal_ensemble(preds, starts, t, m) -> Tensor      ACT weighted average for timestep t, (D,)
    class RecedingHorizonExecutor(policy, H, n_exec).act(obs) -> (D,)

Constraints: torch (CPU) only. All functions must broadcast over leading dims. Normalizer stats are computed once
from the demo set (never per batch). Binning is uniform on [-1, 1]; values outside are clipped first. Chunks are
taken with a fixed stride and only complete chunks are kept (no padding). Ensemble weights are exp(-m * k) over the
chunks covering step t, normalized to sum to 1.

Interview budget: 25 min

Discussion follow-ups:
  * Mean/std vs 1st-99th percentile min/max (OpenVLA uses quantiles): what does a single outlier demo do to each,
    and what does it do to the 256-bin resolution near the typical action?
  * 256 uniform bins give a worst-case error of 1/256 of the range. When is that too coarse for a gripper vs a
    joint velocity, and what do FAST tokens (DCT + BPE) buy over per-step uniform bins?
  * Temporal ensembling with m: the ACT code indexes exp(-m*i) from the OLDEST chunk (i=0 gets weight 1), which is
    the opposite of the "age" convention here. Which one do you want when the policy is reacting to a new
    observation, and why does the paper say a smaller m "incorporates new observations faster"?
  * Receding horizon with n_exec < H: what latency budget does replanning every n_exec steps impose on the model,
    and how does a diffusion policy with K denoising steps fit into a 10 Hz control loop?
"""
import math
import torch

__implement__ = ["ActionNormalizer.fit", "ActionNormalizer.normalize", "ActionNormalizer.unnormalize",
                 "tokenize_actions", "detokenize_actions", "chunk_trajectory", "temporal_ensemble",
                 "RecedingHorizonExecutor.act"]


class ActionNormalizer:
    """
    Per-dimension action normalizer with statistics fit once on a demonstration dataset.
      mode = "meanstd":  norm(a) = (a - mean) / (std + eps)             (std is the population std, unbiased=False)
      mode = "minmax":   norm(a) = 2 * (a - min) / (max - min + eps) - 1  (maps the demo range onto [-1, 1])
    eps keeps constant dimensions (e.g. an unused gripper) finite: their normalized value is 0 (meanstd) or -1
    (minmax) and they invert exactly.
    """

    def __init__(self, mode="meanstd", eps=1e-6):
        assert mode in ("meanstd", "minmax")
        self.mode, self.eps = mode, eps
        self.mean = self.std = self.min = self.max = None

    def fit(self, actions):
        """
        actions : (N, D) float tensor of demo actions. Stores self.mean, self.std (D,) for "meanstd" or self.min,
        self.max (D,) for "minmax". Returns self.
        """
        actions = actions.float()
        if self.mode == "meanstd":
            self.mean = actions.mean(dim=0)
            self.std = actions.std(dim=0, unbiased=False)
        else:
            self.min = actions.min(dim=0).values
            self.max = actions.max(dim=0).values
        return self

    def normalize(self, a):
        """a : (..., D) raw actions -> (..., D) normalized (see class docstring). Requires fit() first."""
        if self.mode == "meanstd":
            return (a - self.mean) / (self.std + self.eps)
        return 2.0 * (a - self.min) / (self.max - self.min + self.eps) - 1.0

    def unnormalize(self, a):
        """a : (..., D) normalized actions -> (..., D) raw. Exact inverse of normalize (up to float error)."""
        if self.mode == "meanstd":
            return a * (self.std + self.eps) + self.mean
        return (a + 1.0) / 2.0 * (self.max - self.min + self.eps) + self.min


def tokenize_actions(a, n_bins=256):
    """
    a : (..., D) normalized actions. Clip to [-1, 1], then bin uniformly: bin edges are -1 + 2*i/n_bins,
    i = 0..n_bins, token = floor((a + 1) / 2 * n_bins) clamped to [0, n_bins - 1] (so a = +1 lands in the last bin).
    Returns int64 tensor of the same shape with values in [0, n_bins - 1]. Monotone in a.
    """
    a = a.clamp(-1.0, 1.0)
    tok = torch.floor((a + 1.0) / 2.0 * n_bins).long()
    return tok.clamp(0, n_bins - 1)


def detokenize_actions(tok, n_bins=256):
    """
    tok : (...,) int64 bin ids in [0, n_bins - 1]. Returns float32 bin CENTERS: -1 + (tok + 0.5) * 2 / n_bins.
    For any a in [-1, 1], |detokenize(tokenize(a)) - a| <= 1 / n_bins (half a bin width).
    """
    return -1.0 + (tok.float() + 0.5) * (2.0 / n_bins)


def chunk_trajectory(traj, H, stride):
    """
    traj : (T, D) one trajectory. Returns (chunks, starts):
      chunks : (N, H, D) with chunks[i] = traj[starts[i] : starts[i] + H]
      starts : (N,) int64, starts[i] = i * stride, N = (T - H) // stride + 1 (only complete chunks; T >= H).
    With stride = 1 every timestep t is covered by the chunks with starts in [t - H + 1, t] (clipped to valid).
    Implement with unfold or indexing, no Python loop over T.
    """
    T, D = traj.shape
    assert T >= H, f"trajectory of length {T} shorter than horizon {H}"
    N = (T - H) // stride + 1
    starts = torch.arange(N) * stride
    idx = starts[:, None] + torch.arange(H)[None, :]          # (N, H)
    return traj[idx], starts


def temporal_ensemble(preds, starts, t, m):
    """
    ACT-style temporal ensembling for executing chunked predictions.
    preds  : (N, H, D) action chunks; preds[i, j] is the prediction for absolute timestep starts[i] + j.
    starts : (N,) int64, the timestep each chunk was predicted at (its first action).
    t      : int, the current timestep. m : float >= 0, temperature.
    Consider the chunks that cover t: starts[i] <= t < starts[i] + H. Chunk i contributes preds[i, t - starts[i]]
    with weight w_i = exp(-m * k_i), k_i = t - starts[i] (the chunk's age in steps: the newest chunk has k = 0 and
    weight 1). Returns the weighted average (D,) with weights normalized to sum to 1 (so m = 0 is a plain mean).
    Raises if no chunk covers t.
    """
    N, H, D = preds.shape
    ages = t - starts                                       # (N,)
    cover = (ages >= 0) & (ages < H)
    assert bool(cover.any()), f"no chunk covers timestep {t}"
    idx = cover.nonzero(as_tuple=True)[0]
    k = ages[idx]
    acts = preds[idx, k]                                    # (n_cover, D)
    w = torch.exp(-m * k.to(preds.dtype))
    w = w / w.sum()
    return (w[:, None] * acts).sum(dim=0)


class RecedingHorizonExecutor:
    """
    Executes a chunked policy in receding-horizon fashion. policy(obs) -> (H, D) chunk of the next H actions.
    On the first call and whenever the queue is empty, query the policy and enqueue the first n_exec actions of the
    chunk (1 <= n_exec <= H); every act(obs) pops and returns one action (D,). The observation is only used when the
    policy is queried, i.e. every n_exec steps. self.n_calls counts policy queries.
    """

    def __init__(self, policy, H, n_exec):
        assert 1 <= n_exec <= H
        self.policy, self.H, self.n_exec = policy, H, n_exec
        self.queue = []
        self.n_calls = 0

    def act(self, obs):
        """obs : whatever policy accepts. Returns the next action (D,) following the receding-horizon schedule."""
        if not self.queue:
            chunk = self.policy(obs)
            assert chunk.shape[0] == self.H
            self.queue = list(chunk[: self.n_exec])
            self.n_calls += 1
        return self.queue.pop(0)


def make_demo_actions(n=512, D=7, seed=0):
    """Given helper: synthetic demo actions with different scales per dim and a constant (gripper) dim."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([0.05, 0.5, 2.0, 10.0, 1.0, 0.1, 1.0])[:D]
    offset = torch.tensor([0.0, 1.0, -3.0, 5.0, 0.0, 0.2, 0.0])[:D]
    a = torch.randn(n, D, generator=g) * scale + offset
    if D >= 7:
        a[:, 6] = 1.0                                   # constant gripper dim
    return a
