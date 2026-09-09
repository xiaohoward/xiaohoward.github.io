import math

import torch
import torch.nn as nn

# tiny tensors: intra-op threading is pure overhead (and can be 10x slower on many-core boxes)
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_timestep_embedding(mod):
    t = torch.tensor([0, 1, 10, 999])
    e = mod.timestep_embedding(t, 64)
    assert e.shape == (4, 64) and e.dtype == torch.float32
    half = 32
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    ref = torch.cat([torch.cos(args), torch.sin(args)], -1)
    assert torch.allclose(e, ref, atol=1e-5), "sinusoidal embedding mismatch (cos first, then sin)"
    assert torch.allclose(e[0, :half], torch.ones(half)) and torch.allclose(e[0, half:], torch.zeros(half)), "t=0"
    assert e.abs().max() <= 1.0 + 1e-6
    # fractional / float timesteps and odd dims
    e2 = mod.timestep_embedding(torch.tensor([0.5, 2.25]), 33)
    assert e2.shape == (2, 33) and torch.allclose(e2[:, -1], torch.zeros(2)), "odd dim -> zero pad"
    e3 = mod.timestep_embedding(torch.tensor([3.0]), 64, max_period=100)
    assert not torch.allclose(e3, mod.timestep_embedding(torch.tensor([3.0]), 64)), "max_period must matter"
    emb = mod.TimestepEmbedder(32)
    assert emb(torch.tensor([1, 2])).shape == (2, 32)


def check_modulate(mod):
    torch.manual_seed(0)
    x, shift, scale = torch.randn(2, 5, 8), torch.randn(2, 8), torch.randn(2, 8)
    y = mod.modulate(x, shift, scale)
    assert y.shape == (2, 5, 8)
    assert torch.allclose(y, x * (1 + scale[:, None]) + shift[:, None])
    assert torch.allclose(mod.modulate(x, torch.zeros(2, 8), torch.zeros(2, 8)), x), "zero shift/scale = identity"


def check_attention_vs_manual(mod):
    torch.manual_seed(0)
    B, N, D, H = 2, 7, 32, 4
    attn = mod.Attention(D, H)
    x = torch.randn(B, N, D)
    y = attn(x)
    assert y.shape == (B, N, D)
    W, b = attn.qkv.weight, attn.qkv.bias
    qkv = x @ W.T + b
    q, k, v = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
    hd = D // H
    q = q.view(B, N, H, hd).transpose(1, 2)
    k = k.view(B, N, H, hd).transpose(1, 2)
    v = v.view(B, N, H, hd).transpose(1, 2)
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1) @ v
    ref = a.transpose(1, 2).reshape(B, N, D) @ attn.proj.weight.T + attn.proj.bias
    assert torch.allclose(y, ref, atol=1e-5), "attention output differs from manual multi-head softmax attention"
    # heads must be independent: permuting tokens permutes the output (no positional mixing)
    perm = torch.randperm(N)
    assert torch.allclose(attn(x[:, perm]), y[:, perm], atol=1e-5), "self-attention must be permutation equivariant"


def check_block_shapes_and_identity_at_init(mod):
    torch.manual_seed(0)
    B, N, D = 2, 9, 32
    blk = mod.DiTBlock(D, num_heads=4, mlp_ratio=4.0)
    x, c = torch.randn(B, N, D), torch.randn(B, D)
    y = blk(x, c)
    assert y.shape == (B, N, D)
    assert torch.allclose(y, x, atol=1e-6), "adaLN-Zero: block must be the identity at init (gates are zero)"
    mods = blk.modulation(c)
    assert isinstance(mods, (tuple, list)) and len(mods) == 6, "modulation must return 6 tensors"
    assert all(m.shape == (B, D) for m in mods), "each modulation tensor must be (B, D)"
    assert all(torch.allclose(m, torch.zeros(B, D)) for m in mods), "modulation must be zero at init"
    lin = [m for m in blk.adaLN_modulation.modules() if isinstance(m, nn.Linear)]
    assert len(lin) == 1 and lin[0].out_features == 6 * D and lin[0].in_features == D
    assert torch.all(lin[0].weight == 0) and torch.all(lin[0].bias == 0), "adaLN linear must be zero-init"
    assert blk.mlp.fc1.out_features == 128, "mlp hidden = mlp_ratio * dim"
    assert isinstance(blk.norm1, nn.LayerNorm) and blk.norm1.weight is None, "LayerNorm without affine params"


def check_gradient_flows_to_modulation(mod):
    torch.manual_seed(0)
    B, N, D = 2, 9, 32
    blk = mod.DiTBlock(D, num_heads=4)
    x, c = torch.randn(B, N, D), torch.randn(B, D)
    loss = (blk(x, c) ** 2).sum()
    loss.backward()
    lin = [m for m in blk.adaLN_modulation.modules() if isinstance(m, nn.Linear)][0]
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0, "no gradient reached adaLN weights"
    assert lin.bias.grad is not None and lin.bias.grad.abs().sum() > 0, "no gradient reached adaLN bias"
    # the gate rows (chunks 2 and 5) get gradient at init, the shift/scale rows do not (they are multiplied by 0)
    g = lin.bias.grad.view(6, D)
    assert g[2].abs().sum() > 0 and g[5].abs().sum() > 0, "gate gradients must be nonzero at init"
    assert torch.allclose(g[[0, 1, 3, 4]], torch.zeros(4, D), atol=1e-6), \
        "shift/scale gradients are zero at init because gates are zero (check the residual wiring)"
    # after one SGD step the block is no longer the identity
    with torch.no_grad():
        for p in blk.parameters():
            p -= 1e-2 * p.grad
    assert not torch.allclose(blk(x, c), x, atol=1e-4), "block should change after a step"


def check_block_matches_manual_forward(mod):
    torch.manual_seed(1)
    B, N, D = 2, 6, 16
    blk = mod.DiTBlock(D, num_heads=2)
    lin = [m for m in blk.adaLN_modulation.modules() if isinstance(m, nn.Linear)][0]
    with torch.no_grad():
        lin.weight.normal_(0, 0.2); lin.bias.normal_(0, 0.2)
    x, c = torch.randn(B, N, D), torch.randn(B, D)
    y = blk(x, c)
    m = (torch.nn.functional.silu(c) @ lin.weight.T + lin.bias).chunk(6, -1)
    ln = lambda h: torch.nn.functional.layer_norm(h, (D,), eps=1e-6)
    h = x + m[2][:, None] * blk.attn(ln(x) * (1 + m[1][:, None]) + m[0][:, None])
    h = h + m[5][:, None] * blk.mlp(ln(h) * (1 + m[4][:, None]) + m[3][:, None])
    assert torch.allclose(y, h, atol=1e-5), "block wiring differs from the adaLN-Zero reference"


def run(mod):
    check_timestep_embedding(mod); print("  ok  sinusoidal timestep embedding")
    check_modulate(mod); print("  ok  modulate")
    check_attention_vs_manual(mod); print("  ok  attention matches manual softmax attention")
    check_block_shapes_and_identity_at_init(mod); print("  ok  block shapes / identity at init / modulation shapes")
    check_gradient_flows_to_modulation(mod); print("  ok  gradient reaches adaLN params (gates first)")
    check_block_matches_manual_forward(mod); print("  ok  block matches manual adaLN-Zero forward")
