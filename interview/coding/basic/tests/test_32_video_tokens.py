import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))
import torch.nn.functional as F


def check_patchify(mod):
    torch.manual_seed(0)
    B, C, T, H, W = 2, 3, 4, 6, 8
    patch = (2, 3, 2)
    x = torch.randn(B, C, T, H, W)
    tok = mod.patchify_3d(x, patch)
    pt, ph, pw = patch
    t, h, w = T // pt, H // ph, W // pw
    assert tok.shape == (B, t * h * w, pt * ph * pw * C), f"patchify_3d shape {tuple(tok.shape)}"
    assert torch.equal(mod.unpatchify_3d(tok, patch, C, T, H, W), x), "unpatchify_3d(patchify_3d(x)) must be exact"
    # token order: temporal-major, then row, then column; inner (pt, ph, pw, C)
    for (tt, i, j, a, b, c, ch) in [(1, 0, 3, 1, 2, 0, 2), (0, 1, 0, 0, 0, 1, 0), (1, 1, 2, 1, 1, 1, 1)]:
        n = tt * (h * w) + i * w + j
        d = ((a * ph + b) * pw + c) * C + ch
        assert tok[1, n, d] == x[1, ch, tt * pt + a, i * ph + b, j * pw + c], f"token order wrong at {(tt, i, j, a, b, c, ch)}"
    # loop reference for the full tensor
    ref = torch.stack([torch.stack([x[:, :, tt*pt:(tt+1)*pt, i*ph:(i+1)*ph, j*pw:(j+1)*pw].permute(0, 2, 3, 4, 1).reshape(B, -1)
                                    for i in range(h) for j in range(w)], 1) for tt in range(t)], 1).reshape(B, -1, pt*ph*pw*C)
    assert torch.equal(tok, ref), "patchify_3d must match the explicit loop reference"
    # WAN-like (1, 2, 2) patch, frames stay separate tokens
    x2 = torch.randn(1, 16, 5, 8, 8)
    tok2 = mod.patchify_3d(x2, (1, 2, 2))
    assert tok2.shape == (1, 5 * 16, 64)
    assert torch.equal(tok2[0, :16], x2[0, :, 0].permute(1, 2, 0).reshape(4, 2, 4, 2, 16).permute(0, 2, 1, 3, 4).reshape(16, 64)), \
        "first 16 tokens must all come from frame 0 in row-major order"
    assert torch.equal(mod.unpatchify_3d(tok2, (1, 2, 2), 16, 5, 8, 8), x2)


def check_sincos(mod):
    dims = (8, 12, 12)
    gt, gh, gw = 3, 4, 5
    pe = mod.get_3d_sincos_pos_embed(dims, gt, gh, gw)
    assert pe.shape == (gt * gh * gw, sum(dims)), f"3D pos embed shape {tuple(pe.shape)}"
    n = lambda t, i, j: t * (gh * gw) + i * gw + j
    # per-axis reference
    def ref1d(D, p):
        half = D // 2
        om = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float64) / half))
        a = p * om
        return torch.cat([a.sin(), a.cos()]).float()
    for (t, i, j) in [(0, 0, 0), (2, 3, 4), (1, 2, 0)]:
        ref = torch.cat([ref1d(8, t), ref1d(12, i), ref1d(12, j)])
        assert torch.allclose(pe[n(t, i, j)], ref, atol=1e-6), f"3D sincos mismatch at {(t, i, j)}"
    # moving along t changes only the t slice; along h only the h slice; along w only the w slice
    base = pe[n(1, 1, 1)]
    dt_ = pe[n(2, 1, 1)] - base
    assert dt_[:8].abs().max() > 1e-3 and dt_[8:].abs().max() == 0, "changing t must touch only the first dt features"
    dh_ = pe[n(1, 2, 1)] - base
    assert dh_[8:20].abs().max() > 1e-3 and dh_[:8].abs().max() == 0 and dh_[20:].abs().max() == 0
    dw_ = pe[n(1, 1, 2)] - base
    assert dw_[20:].abs().max() > 1e-3 and dw_[:20].abs().max() == 0
    # swapping t and h coordinates: t and h slices differ, w slice identical
    a, b = pe[n(2, 1, 3)], pe[n(1, 2, 3)]
    assert (a[:8] - b[:8]).abs().max() > 1e-3 and (a[8:20] - b[8:20]).abs().max() > 1e-3 and torch.equal(a[20:], b[20:])
    # all positions distinct
    d = torch.cdist(pe, pe) + torch.eye(pe.shape[0]) * 1e9
    assert d.min() > 1e-3


def check_rope(mod):
    dims = (16, 24, 24)
    gt, gh, gw = 3, 4, 4
    cos, sin = mod.rope_3d_freqs(dims, gt, gh, gw)
    N = gt * gh * gw
    assert cos.shape == (N, 64) and sin.shape == (N, 64), f"rope table shapes {tuple(cos.shape)} {tuple(sin.shape)}"
    assert torch.allclose(cos ** 2 + sin ** 2, torch.ones(N, 64), atol=1e-5)
    assert torch.allclose(cos[0], torch.ones(64)) and torch.allclose(sin[0], torch.zeros(64)), "origin token has zero angle"
    n = lambda t, i, j: t * (gh * gw) + i * gw + j
    # explicit angle check: interleaved pairs, per-axis freqs theta^(-2k/D_a)
    t, i, j = 2, 3, 1
    row = torch.atan2(sin[n(t, i, j)], cos[n(t, i, j)])
    ft = 10000.0 ** (-torch.arange(0, 16, 2).float() / 16)
    fh = 10000.0 ** (-torch.arange(0, 24, 2).float() / 24)
    exp_ang = torch.cat([(t * ft).repeat_interleave(2), (i * fh).repeat_interleave(2), (j * fh).repeat_interleave(2)])
    exp_ang = torch.atan2(exp_ang.sin(), exp_ang.cos())     # wrap to (-pi, pi]
    assert torch.allclose(row, exp_ang, atol=1e-4), "rope angles: interleaved pairs, axis order t,h,w, freq theta^(-2k/D_a)"
    # slice ownership: changing t changes only dims [:16], h only [16:40], w only [40:]
    base = cos[n(1, 1, 1)]
    assert (cos[n(2, 1, 1)] - base)[16:].abs().max() == 0 and (cos[n(2, 1, 1)] - base)[:16].abs().max() > 1e-3
    assert (cos[n(1, 2, 1)] - base)[:16].abs().max() == 0 and (cos[n(1, 2, 1)] - base)[40:].abs().max() == 0
    assert (cos[n(1, 1, 2)] - base)[:40].abs().max() == 0 and (cos[n(1, 1, 2)] - base)[40:].abs().max() > 1e-3
    # relative-position property: rotated q.k is invariant to a joint shift of both tokens along any axis
    torch.manual_seed(0)
    q, k = torch.randn(64), torch.randn(64)
    def score(nq, nk):
        return (mod.apply_rope_interleaved(q[None], cos[nq:nq+1], sin[nq:nq+1]) *
                mod.apply_rope_interleaved(k[None], cos[nk:nk+1], sin[nk:nk+1])).sum()
    s0 = score(n(0, 1, 1), n(1, 2, 3))
    assert abs(float(score(n(1, 1, 1), n(2, 2, 3))) - float(s0)) < 1e-4, "shift along t must not change q.k"
    assert abs(float(score(n(0, 2, 0), n(1, 3, 2))) - float(s0)) < 1e-4, "shift along h and w must not change q.k"
    assert abs(float(score(n(0, 1, 1), n(2, 2, 3))) - float(s0)) > 1e-3, "different relative offset must change q.k"
    # theta matters
    cos2, _ = mod.rope_3d_freqs(dims, gt, gh, gw, theta=100.0)
    assert not torch.allclose(cos2, cos)


def check_masks(mod):
    m = mod.block_causal_mask(3, 2)
    assert m.shape == (6, 6) and m.dtype == torch.bool
    exp = torch.tensor([[1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, exp), f"block-causal mask\n{m.int()}\nexpected\n{exp.int()}"
    mw = mod.block_causal_mask(4, 2, window=2)
    expw = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(mw, expw), f"windowed mask\n{mw.int()}\nexpected\n{expw.int()}"
    assert torch.equal(mod.block_causal_mask(4, 2, window=1), torch.block_diag(*[torch.ones(2, 2, dtype=torch.bool)] * 4)), \
        "window=1 is per-frame attention"
    assert torch.equal(mod.block_causal_mask(4, 2, window=99), mod.block_causal_mask(4, 2)), "window >= T is plain block-causal"
    # larger: every row attends to itself (no NaN in softmax) and frame f sees exactly (f+1)*P keys
    T, P = 5, 7
    m = mod.block_causal_mask(T, P)
    assert m.diagonal().all()
    counts = m.sum(1).view(T, P)
    assert torch.equal(counts, ((torch.arange(T) + 1) * P)[:, None].expand(T, P))
    out = F.scaled_dot_product_attention(torch.randn(1, T*P, 4), torch.randn(1, T*P, 4), torch.randn(1, T*P, 4), attn_mask=m)
    assert torch.isfinite(out).all()


def check_frame_counts(mod):
    for T, TL in [(81, 21), (1, 1), (49, 13), (5, 2), (17, 5), (2, 1), (4, 1)]:
        assert mod.vae_latent_frames(T) == TL, f"vae_latent_frames({T}) = {mod.vae_latent_frames(T)} != {TL}"
    for TL, T in [(21, 81), (1, 1), (13, 49)]:
        assert mod.vae_pixel_frames(TL) == T
    for T in (1, 5, 9, 81, 121):
        assert mod.vae_pixel_frames(mod.vae_latent_frames(T)) == T, "round trip for T = 4k + 1"
    assert mod.vae_latent_frames(17, stride=8) == 3 and mod.vae_pixel_frames(3, stride=8) == 17
    # tokens per latent video: 81 frames, 480x832 pixels, 8x spatial VAE, (1,2,2) patch -> 21 * 30 * 52
    assert mod.vae_latent_frames(81) * (480 // 8 // 2) * (832 // 8 // 2) == 32760


def run(mod):
    check_patchify(mod);      print("  ok  3D patchify / unpatchify exact, temporal-major token order")
    check_sincos(mod);        print("  ok  factorized 3D sin-cos pos embed (t | h | w slices)")
    check_rope(mod);          print("  ok  axis-split 3D RoPE tables; relative-position invariance")
    check_masks(mod);         print("  ok  block-causal frame mask and sliding window")
    check_frame_counts(mod);  print("  ok  causal 3D VAE frame-count formula and inverse")
