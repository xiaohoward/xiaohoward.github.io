import torch
from torch.overrides import TorchFunctionMode


class ShapeRecorder(TorchFunctionMode):
    """Records the shape of every tensor produced by any torch function / operator (including `@`)."""

    def __init__(self):
        super().__init__()
        self.shapes = []
        self.matmul_shapes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._record(out, "matmul" in getattr(func, "__name__", "") or "bmm" in getattr(func, "__name__", ""))
        return out

    def _record(self, out, is_matmul):
        if isinstance(out, torch.Tensor):
            self.shapes.append(tuple(out.shape))
            if is_matmul:
                self.matmul_shapes.append(tuple(out.shape))
        elif isinstance(out, (tuple, list)):
            for o in out:
                self._record(o, is_matmul)


def _autograd_reference(mod, q, k, v, do, causal):
    q_, k_, v_ = (t.clone().requires_grad_(True) for t in (q, k, v))
    o = mod.reference_attention(q_, k_, v_, causal=causal)
    (o * do).sum().backward()
    return o.detach(), q_.grad, k_.grad, v_.grad


def check_forward_matches_reference(mod):
    for (B, H, N, D, Br, Bc) in [(1, 2, 64, 16, 16, 16), (2, 3, 50, 8, 16, 8), (1, 1, 37, 32, 8, 16), (1, 2, 5, 4, 8, 8),
                                 (1, 2, 33, 8, 32, 32)]:
        for causal in (False, True):
            q, k, v = mod.make_qkv(B, H, N, D, seed=N + Br)
            o, lse = mod.flash_attention_forward(q, k, v, Br, Bc, causal=causal)
            assert o.shape == (B, H, N, D), f"o shape {o.shape}"
            assert lse.shape == (B, H, N), f"lse shape {lse.shape}"
            ref = mod.reference_attention(q, k, v, causal=causal)
            assert torch.isfinite(o).all() and torch.isfinite(lse).all()
            err = (o - ref).abs().max().item()
            assert err < 1e-5, f"N={N} Br={Br} Bc={Bc} causal={causal}: forward differs from SDPA by {err}"
            lse_ref = mod.reference_lse(q, k, causal=causal)
            lerr = (lse - lse_ref).abs().max().item()
            assert lerr < 1e-4, f"N={N} causal={causal}: logsumexp differs by {lerr}"


def check_backward_matches_autograd(mod):
    for (B, H, N, D, Br, Bc) in [(1, 2, 64, 16, 16, 16), (2, 2, 45, 8, 16, 8), (1, 1, 30, 32, 8, 16), (1, 2, 7, 4, 8, 8)]:
        for causal in (False, True):
            q, k, v = mod.make_qkv(B, H, N, D, seed=7 * N + Bc)
            g = torch.Generator().manual_seed(N)
            do = torch.randn(B, H, N, D, generator=g)
            o, lse = mod.flash_attention_forward(q, k, v, Br, Bc, causal=causal)
            dq, dk, dv = mod.flash_attention_backward(q, k, v, o, lse, do, Br, Bc, causal=causal)
            o_ref, dq_ref, dk_ref, dv_ref = _autograd_reference(mod, q, k, v, do, causal)
            for name, a, b in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)):
                assert a.shape == b.shape, f"{name} shape {a.shape} != {b.shape}"
                err = (a - b).abs().max().item()
                assert err < 1e-4, f"N={N} Br={Br} Bc={Bc} causal={causal}: {name} differs from autograd by {err}"


def check_no_nxn_intermediate(mod):
    B, H, N, D, Br, Bc = 1, 2, 64, 16, 16, 8
    q, k, v = mod.make_qkv(B, H, N, D, seed=1)
    g = torch.Generator().manual_seed(2)
    do = torch.randn(B, H, N, D, generator=g)
    for causal in (False, True):
        rec = ShapeRecorder()
        with rec:
            o, lse = mod.flash_attention_forward(q, k, v, Br, Bc, causal=causal)
            mod.flash_attention_backward(q, k, v, o, lse, do, Br, Bc, causal=causal)
        assert len(rec.shapes) > 0, "recorder saw no torch ops"
        bad = [s for s in rec.shapes if len(s) >= 2 and s[-1] >= N and s[-2] >= N]
        assert not bad, f"an (N, N)-sized intermediate was created: {bad[:3]}"
        # nothing bigger than a tile in BOTH of its last two dims (D < Br, Bc here so (.., N, D) slices are fine)
        big = [s for s in rec.shapes if len(s) >= 2 and s[-1] > max(Br, Bc) and s[-2] > max(Br, Bc)]
        assert not big, f"intermediate larger than a (Br, Bc) tile in both trailing dims: {big[:3]}"


def check_block_size_independence(mod):
    q, k, v = mod.make_qkv(1, 2, 40, 8, seed=9)
    o1, l1 = mod.flash_attention_forward(q, k, v, 8, 8, causal=True)
    o2, l2 = mod.flash_attention_forward(q, k, v, 16, 4, causal=True)
    o3, l3 = mod.flash_attention_forward(q, k, v, 64, 64, causal=True)
    assert torch.allclose(o1, o2, atol=1e-6) and torch.allclose(o2, o3, atol=1e-6), "result must not depend on the tiling"
    assert torch.allclose(l1, l2, atol=1e-5) and torch.allclose(l2, l3, atol=1e-5)


def check_causal_skips_tiles(mod):
    # with causal masking the number of score tiles computed must be ~half of the non-causal count
    B, H, N, D, Br, Bc = 1, 1, 64, 8, 16, 16
    q, k, v = mod.make_qkv(B, H, N, D, seed=3)
    counts = {}
    for causal in (False, True):
        rec = ShapeRecorder()
        with rec:
            mod.flash_attention_forward(q, k, v, Br, Bc, causal=causal)
        counts[causal] = sum(1 for s in rec.matmul_shapes if s[-2:] == (Br, Bc))
    assert counts[False] == (N // Br) * (N // Bc), f"non-causal must compute every score tile once: {counts}"
    assert counts[True] == (N // Br) * (N // Br + 1) // 2, f"causal must compute only the lower-triangular tiles: {counts}"


def run(mod):
    check_forward_matches_reference(mod); print("  ok  forward + logsumexp match reference (incl. non-divisible N, causal)")
    check_backward_matches_autograd(mod); print("  ok  backward by recomputation matches autograd (atol 1e-4)")
    check_no_nxn_intermediate(mod); print("  ok  no (N, N) intermediate; tiles bounded by (Br, Bc)")
    check_block_size_independence(mod); print("  ok  tiling-independent result")
    check_causal_skips_tiles(mod); print("  ok  causal skips tiles above the diagonal")
