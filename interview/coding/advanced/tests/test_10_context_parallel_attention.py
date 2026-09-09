import torch


def check_split_merge(mod):
    q, _, _ = mod.make_qkv(B=1, H=2, S=16, D=4)
    for zz in (False, True):
        chunks, pos = mod.split_sequence(q, 4, zigzag=zz)
        assert len(chunks) == 4 and len(pos) == 4
        assert all(c.shape == (1, 2, 4, 4) for c in chunks), [c.shape for c in chunks]
        allpos = torch.cat(pos).sort().values
        assert torch.equal(allpos, torch.arange(16)), "positions must be a permutation of range(S)"
        for c, p in zip(chunks, pos):
            assert torch.equal(c, q[:, :, p]), "chunk content must be q at its positions"
        back = mod.merge_sequence(chunks, pos, 16)
        assert torch.equal(back, q), "merge(split(x)) must equal x"
    _, pos = mod.split_sequence(q, 4, zigzag=True)
    assert pos[0].tolist() == [0, 1, 14, 15] and pos[1].tolist() == [2, 3, 12, 13], f"zig-zag assignment wrong: {pos}"
    _, pos = mod.split_sequence(q, 4, zigzag=False)
    assert pos[3].tolist() == [12, 13, 14, 15], "contiguous assignment wrong"


def check_ring_matches_reference(mod):
    for causal in (False, True):
        for zz in (False, True):
            for N, S in [(4, 32), (3, 24), (1, 8)]:
                q, k, v = mod.make_qkv(B=2, H=3, S=S, D=8, seed=S + N)
                qc, pos = mod.split_sequence(q, N, zigzag=zz)
                kc, _ = mod.split_sequence(k, N, zigzag=zz)
                vc, _ = mod.split_sequence(v, N, zigzag=zz)
                outs, trace = mod.ring_attention(qc, kc, vc, pos, causal=causal)
                assert len(outs) == N and all(o.shape == qc[i].shape for i, o in enumerate(outs))
                full = mod.merge_sequence(outs, pos, S)
                ref = mod.reference_attention(q, k, v, causal=causal)
                assert torch.isfinite(full).all(), "NaN/inf in ring attention output (fully masked block?)"
                err = (full - ref).abs().max().item()
                assert err < 1e-5, f"causal={causal} zigzag={zz} N={N}: ring attention differs from SDPA by {err}"


def check_ring_structure(mod):
    N, S = 4, 32
    q, k, v = mod.make_qkv(B=1, H=2, S=S, D=8, seed=3)
    qc, pos = mod.split_sequence(q, N)
    kc, _ = mod.split_sequence(k, N)
    vc, _ = mod.split_sequence(v, N)
    _, trace = mod.ring_attention(qc, kc, vc, pos, causal=True)
    assert len(trace) == N, "trace must have one entry per rank"
    for r in range(N):
        assert len(trace[r]) == N, f"rank {r} must hold exactly one K/V chunk per step (N steps): {trace[r]}"
        assert all(isinstance(int(c), int) and 0 <= int(c) < N for c in trace[r]), "trace entries are chunk indices"
        assert sorted(int(c) for c in trace[r]) == list(range(N)), f"rank {r} must see every chunk exactly once: {trace[r]}"
        assert int(trace[r][0]) == r, "at step 0 each rank holds its own chunk"
    for s in range(N):
        held = [int(trace[r][s]) for r in range(N)]
        assert sorted(held) == list(range(N)), f"at step {s} chunks must be spread one-per-rank: {held}"
        if s > 0:
            prev = [int(trace[r][s - 1]) for r in range(N)]
            assert all(held[r] == prev[(r - 1) % N] for r in range(N)), "K/V must move around the ring to rank r+1"


def check_ring_variable_chunks(mod):
    # ragged chunk sizes: ring attention must not assume equal S_r
    q, k, v = mod.make_qkv(B=1, H=2, S=20, D=8, seed=11)
    pos = [torch.arange(0, 7), torch.arange(7, 12), torch.arange(12, 20)]
    qc = [q[:, :, p] for p in pos]
    kc = [k[:, :, p] for p in pos]
    vc = [v[:, :, p] for p in pos]
    for causal in (False, True):
        outs, _ = mod.ring_attention(qc, kc, vc, pos, causal=causal)
        full = mod.merge_sequence(outs, pos, 20)
        ref = mod.reference_attention(q, k, v, causal=causal)
        assert torch.allclose(full, ref, atol=1e-5), "ragged chunk sizes must still match reference"


def check_all_to_all(mod):
    q, _, _ = mod.make_qkv(B=2, H=4, S=16, D=8, seed=5)
    N = 4
    chunks, _ = mod.split_sequence(q, N)
    heads = mod.all_to_all_seq_to_heads(chunks)
    assert len(heads) == N and all(h.shape == (2, 1, 16, 8) for h in heads), [h.shape for h in heads]
    for r in range(N):
        assert torch.equal(heads[r], q[:, r:r + 1]), f"rank {r} must hold head {r} for the full sequence"
    back = mod.all_to_all_heads_to_seq(heads)
    assert all(torch.equal(a, b) for a, b in zip(back, chunks)), "heads->seq must invert seq->heads"
    # N = 2 with 2 heads per rank
    chunks2, _ = mod.split_sequence(q, 2)
    heads2 = mod.all_to_all_seq_to_heads(chunks2)
    assert heads2[1].shape == (2, 2, 16, 8) and torch.equal(heads2[1], q[:, 2:4])


def check_ulysses_matches_reference_and_ring(mod):
    for causal in (False, True):
        N = 4
        q, k, v = mod.make_qkv(B=2, H=8, S=32, D=8, seed=7)
        qc, pos = mod.split_sequence(q, N)
        kc, _ = mod.split_sequence(k, N)
        vc, _ = mod.split_sequence(v, N)
        u = mod.ulysses_attention(qc, kc, vc, causal=causal)
        assert len(u) == N and all(x.shape == (2, 8, 8, 8) for x in u)
        full_u = mod.merge_sequence(u, pos, 32)
        ref = mod.reference_attention(q, k, v, causal=causal)
        assert torch.allclose(full_u, ref, atol=1e-5), f"causal={causal}: Ulysses differs from SDPA"
        r_out, _ = mod.ring_attention(qc, kc, vc, pos, causal=causal)
        full_r = mod.merge_sequence(r_out, pos, 32)
        assert torch.allclose(full_u, full_r, atol=1e-5), "Ulysses and ring attention must agree"


def run(mod):
    check_split_merge(mod); print("  ok  split / merge (contiguous and zig-zag)")
    check_ring_matches_reference(mod); print("  ok  ring attention matches SDPA (non-causal, causal, zig-zag)")
    check_ring_structure(mod); print("  ok  ring structure: one K/V chunk per rank per step, rotating")
    check_ring_variable_chunks(mod); print("  ok  ragged chunk sizes")
    check_all_to_all(mod); print("  ok  all-to-all seq<->heads")
    check_ulysses_matches_reference_and_ring(mod); print("  ok  Ulysses matches SDPA and ring attention")
