import math
import torch


def check_sinusoidal(mod):
    L, D = 50, 16
    pe = mod.sinusoidal_table(L, D)
    assert pe.shape == (L, D) and pe.dtype == torch.float32
    # reference formula, scalar loop
    ref = torch.zeros(L, D)
    for pos in range(L):
        for i in range(D // 2):
            div = 10000.0 ** (2 * i / D)
            ref[pos, 2 * i] = math.sin(pos / div)
            ref[pos, 2 * i + 1] = math.cos(pos / div)
    assert torch.allclose(pe, ref, atol=1e-5), f"sinusoidal table mismatch {(pe-ref).abs().max():.2e}"
    assert torch.allclose(pe[0, 0::2], torch.zeros(D // 2)) and torch.allclose(pe[0, 1::2], torch.ones(D // 2)), "pos 0 = (0,1,0,1,...)"
    # first pair has the highest frequency (period 2pi), last pair the lowest
    assert abs(pe[1, 0] - math.sin(1.0)) < 1e-6
    assert (pe[:, 1] >= -1).all() and (pe[:, -2].diff() > 0).all(), "lowest-frequency sin should be monotone over 50 positions"
    try:
        mod.sinusoidal_table(4, 5); raise AssertionError("odd d_model should raise")
    except ValueError:
        pass


def check_rope_cos_sin(mod):
    pos = torch.arange(6)
    cos, sin = mod.rope_cos_sin(pos, 8)
    assert cos.shape == (6, 8) and sin.shape == (6, 8)
    assert torch.allclose(cos[:, :4], cos[:, 4:]) and torch.allclose(sin[:, :4], sin[:, 4:]), "half-split: two halves equal"
    inv = 10000.0 ** (-torch.arange(0, 8, 2).float() / 8)
    assert torch.allclose(cos[:, :4], torch.cos(pos.float()[:, None] * inv), atol=1e-6)
    assert torch.allclose(cos[0], torch.ones(8)) and torch.allclose(sin[0], torch.zeros(8)), "position 0 is identity"
    try:
        mod.rope_cos_sin(pos, 7); raise AssertionError("odd head_dim should raise")
    except ValueError:
        pass


def check_rope_norm_and_identity(mod):
    torch.manual_seed(0)
    B, H, L, D = 2, 3, 10, 16
    x = torch.randn(B, H, L, D)
    pos = torch.arange(L)
    y = mod.apply_rope(x, pos)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5), "RoPE must preserve per-token norm"
    assert torch.allclose(y[..., 0, :], x[..., 0, :], atol=1e-6), "position 0 must be identity"
    # explicit 2x2 rotation check on one token / one frequency
    m, i = 7, 2
    theta = m * 10000.0 ** (-2 * i / D)
    a, b = x[0, 0, m, i].item(), x[0, 0, m, i + D // 2].item()
    exp_a = a * math.cos(theta) - b * math.sin(theta)
    exp_b = a * math.sin(theta) + b * math.cos(theta)
    assert abs(y[0, 0, m, i].item() - exp_a) < 1e-4 and abs(y[0, 0, m, i + D // 2].item() - exp_b) < 1e-4, \
        "pair (x[i], x[i+D/2]) must be rotated by m * base^(-2i/D)"
    # per-sample position offsets broadcast (B, 1, L)
    pos2 = torch.arange(L).view(1, 1, L) + torch.tensor([0, 5]).view(B, 1, 1)
    y2 = mod.apply_rope(x, pos2)
    assert torch.allclose(y2[0], y[0], atol=1e-6)
    assert torch.allclose(y2[1], mod.apply_rope(x[1:], torch.arange(5, 5 + L))[0], atol=1e-5)


def check_rope_relative(mod):
    torch.manual_seed(1)
    D = 32
    q = torch.randn(1, 1, 1, D)
    k = torch.randn(1, 1, 1, D)
    def score(m, n):
        qm = mod.apply_rope(q, torch.tensor([m]))
        kn = mod.apply_rope(k, torch.tensor([n]))
        return (qm * kn).sum().item()
    for (m, n) in [(3, 1), (10, 4), (0, 7)]:
        for s in (1, 5, 100, 1000):
            assert abs(score(m, n) - score(m + s, n + s)) < 1e-3, \
                f"q@{m}.k@{n} != q@{m+s}.k@{n+s}: RoPE score must depend only on m-n"
    # same position -> raw dot product; different offsets -> generally different
    assert abs(score(5, 5) - (q * k).sum().item()) < 1e-4
    assert abs(score(5, 3) - score(5, 4)) > 1e-3
    # full sequence: causal attention logits are shift invariant
    L = 12
    Q = torch.randn(1, 2, L, D); K = torch.randn(1, 2, L, D)
    s0 = mod.apply_rope(Q, torch.arange(L)) @ mod.apply_rope(K, torch.arange(L)).transpose(-1, -2)
    s1 = mod.apply_rope(Q, torch.arange(L) + 37) @ mod.apply_rope(K, torch.arange(L) + 37).transpose(-1, -2)
    assert torch.allclose(s0, s1, atol=1e-3), "attention logit matrix must be invariant to a global position shift"


def run(mod):
    check_sinusoidal(mod);              print("  ok  sinusoidal table matches Vaswani formula")
    check_rope_cos_sin(mod);            print("  ok  rope cos/sin table (half-split layout)")
    check_rope_norm_and_identity(mod);  print("  ok  RoPE preserves norm, pos 0 identity, explicit rotation, broadcast positions")
    check_rope_relative(mod);           print("  ok  RoPE dot product depends only on relative position")
