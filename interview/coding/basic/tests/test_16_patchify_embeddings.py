import math
import numpy as np
import torch
import torch.nn.functional as F


# --- numpy references copied from the MAE repo (util/pos_embed.py) ---------------------------------------------
def _mae_1d(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _mae_2d(embed_dim, grid_h, grid_w):
    grid = np.meshgrid(np.arange(grid_w, dtype=np.float32), np.arange(grid_h, dtype=np.float32))
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_h, grid_w])
    emb_h = _mae_1d(embed_dim // 2, grid[0])
    emb_w = _mae_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _adm_timestep(t, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(half, dtype=np.float32) / half).astype(np.float32)
    args = (t.numpy().astype(np.float32)[:, None] * freqs[None]).astype(np.float32)
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        emb = np.concatenate([emb, np.zeros_like(emb[:, :1])], axis=-1)
    return emb


def check_patchify(mod):
    torch.manual_seed(0)
    B, C, H, W, p = 2, 4, 8, 12, 2
    x = torch.randn(B, C, H, W)
    tok = mod.patchify(x, p)
    assert tok.shape == (B, (H // p) * (W // p), p * p * C), f"patchify shape {tuple(tok.shape)}"
    # reference via F.unfold: (B, C*p*p, N) with inner order (c, ph, pw) -> reorder to (ph, pw, c)
    unf = F.unfold(x, kernel_size=p, stride=p)                       # (B, C*p*p, N)
    ref = unf.view(B, C, p, p, -1).permute(0, 4, 2, 3, 1).reshape(B, -1, p * p * C)
    assert torch.equal(tok, ref), "patchify must match unfold reference with (ph, pw, c) inner order, row-major patches"
    # explicit element check
    i, j, ph, pw, c = 2, 5, 1, 0, 3
    assert tok[1, i * (W // p) + j, (ph * p + pw) * C + c] == x[1, c, i * p + ph, j * p + pw]
    assert torch.equal(mod.unpatchify(tok, p, C, H, W), x), "unpatchify(patchify(x)) must be exact"
    # non-square patch, p=4
    x2 = torch.randn(3, 3, 16, 8)
    assert torch.equal(mod.unpatchify(mod.patchify(x2, 4), 4, 3, 16, 8), x2)
    assert mod.patchify(x2, 4).shape == (3, 8, 48)


def check_pos_embed(mod):
    for D, gh, gw in ((16, 4, 4), (32, 3, 7), (64, 8, 8)):
        pe = mod.get_2d_sincos_pos_embed(D, gh, gw)
        assert pe.shape == (gh * gw, D), f"pos embed shape {tuple(pe.shape)} != {(gh * gw, D)}"
        ref = torch.from_numpy(_mae_2d(D, gh, gw)).float()
        assert torch.allclose(pe, ref, atol=1e-6), \
            f"2D sin-cos pos embed mismatch vs MAE reference (D={D}, grid={gh}x{gw}); max err {(pe - ref).abs().max():.2e}"
    pos = torch.tensor([0.0, 1.0, 2.5, 100.0])
    pe1 = mod.get_1d_sincos_pos_embed(8, pos)
    assert pe1.shape == (4, 8)
    assert torch.allclose(pe1, torch.from_numpy(_mae_1d(8, pos.numpy())).float(), atol=1e-6)
    assert torch.allclose(pe1[0, :4], torch.zeros(4)) and torch.allclose(pe1[0, 4:], torch.ones(4)), \
        "position 0 must give sin=0, cos=1"
    # every position has a distinct code
    pe = mod.get_2d_sincos_pos_embed(32, 6, 6)
    d = torch.cdist(pe, pe) + torch.eye(36) * 1e9
    assert d.min() > 1e-3, "distinct grid positions must have distinct embeddings"


def check_timestep_embedding(mod):
    t = torch.tensor([0, 1, 10, 500, 999], dtype=torch.float32)
    for dim in (8, 64, 256, 33):
        emb = mod.timestep_embedding(t, dim)
        assert emb.shape == (5, dim) and emb.dtype == torch.float32
        ref = torch.from_numpy(_adm_timestep(t, dim)).float()
        assert torch.allclose(emb, ref, atol=1e-4), f"timestep embedding mismatch for dim={dim}, max err {(emb - ref).abs().max():.2e}"
    emb = mod.timestep_embedding(t, 64)
    assert torch.allclose(emb[0, :32], torch.ones(32)) and torch.allclose(emb[0, 32:], torch.zeros(32)), \
        "t=0 must give cos=1 (first half), sin=0 (second half)"
    d = torch.cdist(emb, emb) + torch.eye(5) * 1e9
    assert d.min() > 0.1, "different timesteps must give clearly different embeddings"
    # max_period changes the frequencies
    assert not torch.allclose(mod.timestep_embedding(t, 64, max_period=100), emb)
    # integer input accepted
    assert torch.allclose(mod.timestep_embedding(torch.tensor([3, 7]), 16), mod.timestep_embedding(torch.tensor([3.0, 7.0]), 16))


def check_embed_timestep(mod):
    mlp = mod.make_timestep_mlp(64, 32, seed=0)
    t = torch.tensor([0.0, 250.0, 999.0])
    out = mod.embed_timestep(t, 64, mlp)
    assert out.shape == (3, 32)
    ref = mlp(torch.from_numpy(_adm_timestep(t, 64)).float())
    assert torch.allclose(out, ref, atol=1e-5), "embed_timestep must be mlp(timestep_embedding(t))"
    assert (out[0] - out[1]).abs().max() > 1e-3 and (out[1] - out[2]).abs().max() > 1e-3


def run(mod):
    check_patchify(mod);            print("  ok  patchify matches unfold reference; unpatchify exact inverse")
    check_pos_embed(mod);           print("  ok  2D sin-cos pos embed matches MAE numpy reference")
    check_timestep_embedding(mod);  print("  ok  sinusoidal timestep embedding matches ADM/DiT reference")
    check_embed_timestep(mod);      print("  ok  timestep embedder = MLP(sinusoid)")
