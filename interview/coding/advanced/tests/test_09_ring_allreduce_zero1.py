import math

import torch


def _rand_params(shapes, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*s, generator=g) for s in shapes]


def check_shards_partition(mod):
    for numel, N in [(10, 4), (12, 4), (5, 8), (1, 1), (100, 7), (0, 3)]:
        b = mod.shard_bounds(numel, N)
        assert len(b) == N, f"need {N} shards"
        assert b[0][0] == 0 and b[-1][1] == numel, f"shards must cover [0, {numel}): {b}"
        for (s0, e0), (s1, e1) in zip(b[:-1], b[1:]):
            assert e0 == s1, f"shards must be contiguous and non-overlapping: {b}"
        for s, e in b:
            assert 0 <= s <= e <= numel
        sizes = [e - s for s, e in b]
        S = math.ceil(numel / N) if numel else 0
        assert sizes == [max(0, min(S, numel - r * S)) for r in range(N)], f"shard sizes {sizes} for numel={numel}, N={N}"
        assert sum(sizes) == numel


def check_flatten_roundtrip(mod):
    ps = _rand_params(mod.make_param_shapes(), 0)
    flat = mod.flatten(ps)
    assert flat.dim() == 1 and flat.numel() == sum(p.numel() for p in ps)
    back = mod.unflatten(flat, ps)
    assert all(torch.equal(a, b) for a, b in zip(back, ps)), "unflatten(flatten(x)) must equal x"
    back[0].fill_(3.0)
    assert torch.all(flat[: ps[0].numel()] == 3.0), "unflatten must return views into flat"


def check_allreduce_sum_and_bytes(mod):
    for N, M in [(4, 64), (4, 65), (3, 10), (8, 8), (5, 1), (2, 7), (1, 9)]:
        g = torch.Generator().manual_seed(N * 100 + M)
        bufs = [torch.randn(M, generator=g) for _ in range(N)]
        saved = [b.clone() for b in bufs]
        comm = mod.Comm(N)
        out = mod.ring_allreduce(bufs, comm)
        ref = torch.stack(bufs).sum(0)
        assert len(out) == N
        for r in range(N):
            assert out[r].shape == (M,), f"rank {r} output shape {out[r].shape}"
            assert torch.allclose(out[r], ref, atol=1e-5), f"N={N} M={M} rank {r}: allreduce != sum"
        C = math.ceil(M / N)
        expected = 2 * (N - 1) * C
        for r in range(N):
            assert comm.sent[r] == expected, f"N={N} M={M}: rank {r} sent {comm.sent[r]} elements, expected {expected}"
        assert abs(expected - 2 * (N - 1) / N * M) <= 2 * (N - 1), "bytes per rank must be ~2(N-1)/N * M"
        for src, dst, _ in comm.log:
            assert dst == (src + 1) % N, "ring: every send must go to the next rank"
        assert all(torch.equal(b, b2) for b, b2 in zip(bufs, saved)), "inputs must not be modified" 


def check_allreduce_no_shortcut(mod):
    # With N = 4 there are exactly 2*(N-1)*N sends in total and each has the chunk size.
    N, M = 4, 40
    bufs = [torch.arange(M, dtype=torch.float32) * (r + 1) for r in range(N)]
    comm = mod.Comm(N)
    mod.ring_allreduce(bufs, comm)
    assert len(comm.log) == 2 * (N - 1) * N, f"expected {2 * (N - 1) * N} sends, got {len(comm.log)}"
    assert all(n == M // N for _, _, n in comm.log), "every send must be exactly one chunk"


def check_bucketed_mean(mod):
    N = 3
    shapes = mod.make_param_shapes()
    rank_grads = [_rand_params(shapes, 10 + r) for r in range(N)]
    comm = mod.Comm(N)
    out = mod.bucketed_allreduce_mean(rank_grads, comm)
    for i, s in enumerate(shapes):
        ref = torch.stack([rank_grads[r][i] for r in range(N)]).mean(0)
        for r in range(N):
            assert out[r][i].shape == torch.Size(s)
            assert torch.allclose(out[r][i], ref, atol=1e-6), "bucketed mean wrong"
    total = sum(math.prod(s) for s in shapes)
    assert comm.sent[0] == 2 * (N - 1) * math.ceil(total / N), "bucketing must do a single ring pass over the flat buffer"


def check_zero1_matches_adam(mod):
    N = 4
    shapes = mod.make_param_shapes()
    base = _rand_params(shapes, 0)
    rank_params = [[p.clone() for p in base] for _ in range(N)]
    ref = [p.clone().requires_grad_(True) for p in base]
    opt = torch.optim.Adam(ref, lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    comm = mod.Comm(N)
    z = mod.Zero1Adam(rank_params, comm, lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    numel = sum(p.numel() for p in base)
    assert [e - s for s, e in z.bounds] == [e - s for s, e in mod.shard_bounds(numel, N)]
    for step in range(4):
        rank_grads = [_rand_params(shapes, 100 + 10 * step + r) for r in range(N)]
        for i in range(len(shapes)):
            ref[i].grad = torch.stack([rank_grads[r][i] for r in range(N)]).mean(0)
        opt.step()
        z.step(rank_grads)
        for r in range(N):
            for i in range(len(shapes)):
                assert rank_params[r][i].shape == torch.Size(shapes[i])
                assert torch.allclose(rank_params[r][i], ref[i].detach(), atol=1e-6), \
                    f"step {step} rank {r} param {i}: ZeRO-1 differs from Adam by {(rank_params[r][i] - ref[i]).abs().max()}"
        for r in range(1, N):
            for i in range(len(shapes)):
                assert torch.equal(rank_params[r][i], rank_params[0][i]), "all ranks must hold identical params"
    # optimizer state really is sharded: moment tensors sum to numel, not N * numel
    assert sum(m.numel() for m in z.m) == numel and sum(v.numel() for v in z.v) == numel, "moments must be sharded"
    # all-gather traffic: each rank sends its shard to N-1 peers per step, plus the ring for grads
    C = math.ceil(numel / N)
    per_step = 2 * (N - 1) * C + (N - 1) * (z.bounds[0][1] - z.bounds[0][0])
    assert comm.sent[0] == 4 * per_step, f"rank 0 traffic {comm.sent[0]} != 4 * {per_step}"


def check_zero1_single_rank(mod):
    shapes = mod.make_param_shapes()
    base = _rand_params(shapes, 1)
    rank_params = [[p.clone() for p in base]]
    ref = [p.clone().requires_grad_(True) for p in base]
    opt = torch.optim.Adam(ref, lr=5e-3)
    z = mod.Zero1Adam(rank_params, mod.Comm(1), lr=5e-3)
    grads = _rand_params(shapes, 2)
    for p, g in zip(ref, grads):
        p.grad = g.clone()
    opt.step()
    z.step([grads])
    for a, b in zip(rank_params[0], ref):
        assert torch.allclose(a, b.detach(), atol=1e-6), "N=1 must reduce to plain Adam"


def run(mod):
    check_shards_partition(mod); print("  ok  shard_bounds partitions exactly")
    check_flatten_roundtrip(mod); print("  ok  flatten / unflatten round trip (views)")
    check_allreduce_sum_and_bytes(mod); print("  ok  ring all-reduce equals sum, 2(N-1)/N * M elements per rank")
    check_allreduce_no_shortcut(mod); print("  ok  ring structure (2(N-1) chunk sends per rank)")
    check_bucketed_mean(mod); print("  ok  bucketed all-reduce mean in one ring pass")
    check_zero1_matches_adam(mod); print("  ok  ZeRO-1 matches single-process torch.optim.Adam (atol 1e-6)")
    check_zero1_single_rank(mod); print("  ok  ZeRO-1 with N=1")
