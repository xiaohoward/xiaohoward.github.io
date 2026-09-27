"""Ring all-reduce with gradient bucketing, and a ZeRO-1 sharded optimizer (simulated ranks)

We simulate N data-parallel "ranks" in one process: rank r's tensors live in the r-th entry of a python list. There is
no torch.distributed. Implement:
  (a) ring_allreduce(bufs, comm) -> reduced bufs. Every rank holds a 1-D fp32 buffer of the same length M. Do the
      bandwidth-optimal ring: split each buffer into N chunks, N-1 reduce-scatter steps (rank r sends chunk (r - s) mod
      N to rank r+1 and adds the received chunk into its own copy), then N-1 all-gather steps. Every transfer must go
      through comm.send(src, dst, tensor), which records the number of elements sent per rank. Pad M to a multiple of N.
  (b) flatten / unflatten a list of gradient tensors into one flat buffer ("bucketing") so many small grads are
      reduced with a single ring pass.
  (c) ZeRO-1 (optimizer-state sharding): all ranks hold identical full fp32 parameters and their own gradients. After
      averaging grads with the ring, rank r owns ONLY the contiguous shard r of the flat parameter vector: it keeps
      Adam moments for that shard, updates it, and the updated shards are all-gathered so every rank ends with the same
      full parameters. Result must equal a single-process torch.optim.Adam step on the averaged gradient.
Signatures:
    shard_bounds(numel, world_size) -> list[(start, end)]
    flatten(tensors) -> flat ; unflatten(flat, like) -> list of tensors
    ring_allreduce(bufs, comm) -> list of reduced 1-D tensors (identical on every rank)
    class Zero1Adam: step(rank_grads) -> None   (updates self.rank_params in place)
Constraints: torch CPU + stdlib. N in [1, 8]. Chunks must be contiguous slices; the last shard may be smaller. Sum
order must be the ring order (rank 0's copy of chunk c ends up as ((x_{c+1} + x_{c+2}) + ...) etc.) - tests use atol.

Interview budget: 35 min

Discussion follow-ups:
  - Why is ring all-reduce bandwidth-optimal (each rank sends 2(N-1)/N * M elements) and what is its latency term
    (2(N-1) steps)? When do tree / 2D-torus / reduce-scatter+all-gather (SHARP) beat it? Bucket size vs overlap.
  - ZeRO-1/2/3 memory per parameter (16 B -> 4 B + 12/N B ...). What extra communication does ZeRO-3 add per step?
    Why is FSDP's reduce-scatter of grads "free" compared with DDP all-reduce?
  - For a 10B-param video DiT on 512 GPUs, size the per-step gradient traffic in bf16 and the time at 400 GB/s.
  - How would you overlap the reduce-scatter with the backward pass (hooks, bucket ordering reversed)?
"""
import math

import torch

__implement__ = ["shard_bounds", "flatten", "unflatten", "ring_allreduce", "Zero1Adam.step"]


class Comm:
    """Given helper: a fake network. send(src, dst, t) 'delivers' a copy of t and records elements sent by src."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.sent = [0] * world_size          # elements sent per rank
        self.log = []                          # (src, dst, numel)

    def send(self, src, dst, t):
        assert 0 <= src < self.world_size and 0 <= dst < self.world_size and src != dst
        self.sent[src] += t.numel()
        self.log.append((src, dst, t.numel()))
        return t.clone()


def make_param_shapes(seed=0):
    """Given helper: a small 'model' as a list of parameter shapes (odd sizes on purpose)."""
    return [(7, 5), (5,), (13, 3), (11,), (2, 3, 4)]


def shard_bounds(numel, world_size):
    """Contiguous partition of range(numel) into world_size shards.

    Shard size S = ceil(numel / world_size); shard r is [r*S, min((r+1)*S, numel)). Later shards may be empty (start ==
    end) when numel is small. Returns a list of world_size (start, end) int tuples.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def flatten(tensors):
    """Concatenate tensors (any shapes, same dtype) into one contiguous 1-D tensor, in order. Returns a new tensor."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def unflatten(flat, like):
    """Inverse of flatten: split 1-D `flat` into tensors with the shapes of `like` (list of tensors), in order.

    Returns a list of views into `flat` (no copy), so writes to them update `flat`.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def ring_allreduce(bufs, comm):
    """Ring all-reduce (sum) of N 1-D tensors of equal length M through comm.send.

    Args:
        bufs: list of N 1-D tensors (rank r's buffer). Not modified.
        comm: Comm; every transfer must be exactly one comm.send(src, dst, chunk) with dst = (src + 1) % N.
    Algorithm: pad each buffer with zeros to length N * C, C = ceil(M / N); chunk c of rank r is x_r[c*C:(c+1)*C].
      Reduce-scatter (N-1 steps): at step s rank r sends chunk (r - s) mod N to rank r+1, which adds it into its own
        chunk (r - s) mod N. After this, rank r holds the full sum of chunk (r + 1) mod N.
      All-gather (N-1 steps): at step s rank r sends its complete chunk (r + 1 - s) mod N to rank r+1, which overwrites
        its copy.
    Returns: list of N 1-D tensors of length M, each equal to sum_r bufs[r] (up to fp summation order). N == 1 returns
    copies without any send.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def bucketed_allreduce_mean(rank_grads, comm):
    """Given helper: flatten each rank's grad list, ring all-reduce, divide by N, unflatten. Returns list-of-lists."""
    N = len(rank_grads)
    flats = [flatten(g) for g in rank_grads]
    reduced = ring_allreduce(flats, comm)
    return [unflatten(r / N, rank_grads[i]) for i, r in enumerate(reduced)]


class Zero1Adam:
    """ZeRO-1 Adam over simulated ranks.

    Attributes set in __init__ (given):
        world_size N; rank_params: list over ranks of lists of fp32 tensors (identical across ranks at init);
        bounds = shard_bounds(numel, N); m, v: per-rank moment tensors of the shard size (zeros); step_count = 0;
        lr, betas, eps; comm: Comm.
    """

    def __init__(self, rank_params, comm, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.world_size = len(rank_params)
        self.rank_params = rank_params
        self.comm = comm
        self.lr, self.betas, self.eps = lr, betas, eps
        numel = sum(p.numel() for p in rank_params[0])
        self.bounds = shard_bounds(numel, self.world_size)
        self.m = [torch.zeros(e - s) for s, e in self.bounds]
        self.v = [torch.zeros(e - s) for s, e in self.bounds]
        self.step_count = 0

    def step(self, rank_grads):
        """One ZeRO-1 Adam step.

        Args:
            rank_grads: list over ranks of lists of grads (same shapes as rank_params[r]).
        Steps:
          1. avg = bucketed_allreduce_mean(rank_grads, self.comm)  (every rank gets the mean grad).
          2. self.step_count += 1. For each rank r with shard [s, e): take the flat param slice and flat mean-grad slice,
             update self.m[r], self.v[r] with Adam (bias-corrected, eps added after sqrt, like torch.optim.Adam):
                 m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g^2
                 p -= lr * (m / (1-b1^t)) / (sqrt(v / (1-b2^t)) + eps)
          3. All-gather: rank r sends its updated shard to every other rank via comm.send (N-1 sends per rank, any
             order); every rank writes ALL shards into its own flat params, and copies the flat result back into its
             rank_params tensors in place (use unflatten views or copy_).
        Afterwards, all ranks' params are identical and equal a single-process Adam step on the mean grad.
        """
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError
