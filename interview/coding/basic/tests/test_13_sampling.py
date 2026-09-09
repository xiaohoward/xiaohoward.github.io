import math
import torch


def _finite_set(row):
    return set(torch.nonzero(torch.isfinite(row)).flatten().tolist())


def check_temperature(mod):
    x = torch.tensor([[1.0, 2.0, 4.0]])
    assert torch.allclose(mod.apply_temperature(x, 2.0), x / 2)
    assert torch.equal(mod.apply_temperature(x, 1.0), x)
    # lower temperature -> peakier softmax
    p_hot = torch.softmax(mod.apply_temperature(x, 0.5), -1)[0, 2]
    p_cold = torch.softmax(mod.apply_temperature(x, 2.0), -1)[0, 2]
    assert p_hot > p_cold
    try:
        mod.apply_temperature(x, 0.0); raise AssertionError("temperature 0 must raise in apply_temperature")
    except ValueError:
        pass


def check_top_k(mod):
    logits = torch.tensor([[0.1, 5.0, 3.0, -1.0, 4.0],
                           [2.0, 2.0, 1.0, 2.0, 0.0]])
    out = mod.top_k_filter(logits, 2)
    assert _finite_set(out[0]) == {1, 4}, f"row0 top-2 should keep {{1,4}}, kept {_finite_set(out[0])}"
    assert _finite_set(out[1]) == {0, 1, 3}, "ties at the k-th value must all be kept"
    assert torch.equal(out[0, [1, 4]], logits[0, [1, 4]]), "kept logits unchanged"
    assert torch.equal(mod.top_k_filter(logits, 10), logits), "k >= V must be a no-op"
    assert _finite_set(mod.top_k_filter(logits, 1)[0]) == {1}
    assert torch.equal(logits[0], torch.tensor([0.1, 5.0, 3.0, -1.0, 4.0])), "input must not be modified in place"


def check_top_p(mod):
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    # cum_before = [0, .5, .8, .95]
    assert _finite_set(mod.top_p_filter(logits, 0.8)[0]) == {0, 1}, "p=0.8: keep prefix {0,1} (0.5+0.3 reaches 0.8)"
    assert _finite_set(mod.top_p_filter(logits, 0.81)[0]) == {0, 1, 2}, "p=0.81: token 2 crosses the threshold, keep it"
    assert _finite_set(mod.top_p_filter(logits, 0.3)[0]) == {0}, "p below top-1 prob keeps only top-1"
    assert _finite_set(mod.top_p_filter(logits, 1.0)[0]) == {0, 1, 2, 3}, "p=1 keeps all"
    # unsorted order + batch
    perm = torch.tensor([2, 0, 3, 1])
    lp = logits[:, perm]                      # probs now [.15, .5, .05, .3]
    out = mod.top_p_filter(torch.cat([lp, logits]), 0.8)
    assert _finite_set(out[0]) == {1, 3}, f"permuted row: expected {{1,3}}, got {_finite_set(out[0])}"
    assert _finite_set(out[1]) == {0, 1}
    # already -inf entries stay -inf and are ignored in the cumulative sum
    lg = torch.tensor([[math.log(0.5), float("-inf"), math.log(0.3), math.log(0.2)]])
    assert _finite_set(mod.top_p_filter(lg, 0.79)[0]) == {0, 2}


def check_sampling_respects_filters(mod):
    torch.manual_seed(0)
    V, B, R = 50, 4, 200
    logits = torch.randn(B, V) * 3
    big = logits.repeat(R, 1)                      # (R*B, V): R independent draws per row, one batched call
    g = torch.Generator().manual_seed(0)
    allowed_k = [_finite_set(r) for r in mod.top_k_filter(logits, 5)]
    s = mod.sample(big, temperature=1.0, top_k=5, generator=g)
    assert s.shape == (R * B,) and s.dtype == torch.long
    for i in range(R * B):
        assert s[i].item() in allowed_k[i % B], "top-k sample outside allowed set"
    allowed_p = [_finite_set(r) for r in mod.top_p_filter(mod.top_k_filter(logits / 0.7, 10), 0.9)]
    s = mod.sample(big, temperature=0.7, top_k=10, top_p=0.9, generator=g)
    for i in range(R * B):
        assert s[i].item() in allowed_p[i % B], "top-k+top-p sample outside allowed set"
    # greedy
    assert torch.equal(mod.sample(logits, temperature=0.0), logits.argmax(-1))
    # reproducible
    a = mod.sample(logits, 1.0, 5, 0.9, torch.Generator().manual_seed(42))
    b = mod.sample(logits, 1.0, 5, 0.9, torch.Generator().manual_seed(42))
    assert torch.equal(a, b), "same generator seed must give the same samples"
    # empirical frequency matches the filtered softmax
    lg = torch.tensor([[0.0, math.log(3.0), math.log(6.0), -20.0]])   # probs .1 .3 .6 ~0
    n = 4000
    s = mod.sample(lg.repeat(n, 1), generator=torch.Generator().manual_seed(1))
    freq = torch.bincount(s, minlength=4).float() / n
    assert (freq - torch.tensor([0.1, 0.3, 0.6, 0.0])).abs().max() < 0.03, f"empirical freq {freq} off"


def run(mod):
    check_temperature(mod);               print("  ok  temperature")
    check_top_k(mod);                     print("  ok  top-k filtering (ties, k>=V, no in-place)")
    check_top_p(mod);                     print("  ok  top-p filtering (threshold crossing, permutation, -inf)")
    check_sampling_respects_filters(mod); print("  ok  sampling only returns allowed indices; greedy; reproducible; frequencies")
