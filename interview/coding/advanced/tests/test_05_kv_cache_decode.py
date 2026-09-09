import math

import torch
import torch.nn.functional as F

# tiny tensors: intra-op threading is pure overhead (and can be 10x slower on many-core boxes)
torch.set_num_threads(min(torch.get_num_threads(), 8))


def _reference_forward(model, tokens, window=None):
    """Independent full forward with an explicit (banded) causal mask, using only the model's weights."""
    B, T = tokens.shape
    x = model.tok_emb(tokens) + model.pos_emb(torch.arange(T))[None]
    i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
    allowed = j <= i
    if window is not None:
        allowed = allowed & (j > i - window)
    for blk in model.blocks:
        h = blk.ln1(x)
        a = blk.attn
        qkv = h @ a.qkv.weight.T + a.qkv.bias
        D = h.shape[-1]
        q, k, v = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
        q, k, v = [z.view(B, T, a.n_heads, a.head_dim).transpose(1, 2) for z in (q, k, v)]
        s = q @ k.transpose(-1, -2) / math.sqrt(a.head_dim)
        s = s.masked_fill(~allowed, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + o @ a.proj.weight.T + a.proj.bias
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.ln_f(x))


def check_mask(mod):
    m = mod.causal_mask(4, 0)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool))), "plain causal mask"
    m = mod.causal_mask(2, 3)
    ref = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, ref), "new queries must see all cached keys plus causal among new"
    m = mod.causal_mask(3, 2, window=2)
    ref = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, ref), f"windowed mask wrong:\n{m.int()}"
    assert torch.equal(mod.causal_mask(1, 5, window=1), torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.bool))
    assert mod.causal_mask(1, 5).sum() == 6 and mod.causal_mask(1, 5, window=100).sum() == 6


def check_attention_with_cache(mod):
    torch.manual_seed(0)
    attn = mod.CausalSelfAttention(32, 4)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        full, (k, v) = attn(x)
        assert full.shape == (2, 6, 32) and k.shape == (2, 4, 6, 8) and v.shape == (2, 4, 6, 8)
        # reference: causal sdpa
        qkv = attn.qkv(x)
        q_, k_, v_ = [z.view(2, 6, 4, 8).transpose(1, 2) for z in qkv.chunk(3, -1)]
        ref = attn.proj(F.scaled_dot_product_attention(q_, k_, v_, is_causal=True).transpose(1, 2).reshape(2, 6, 32))
        assert torch.allclose(full, ref, atol=1e-5), "no-cache forward must equal causal attention"
        assert torch.allclose(k, k_, atol=1e-6) and torch.allclose(v, v_, atol=1e-6), "returned cache must hold K, V"
        # chunked: first 4 tokens, then 2 with the cache
        o1, c1 = attn(x[:, :4])
        o2, c2 = attn(x[:, 4:], cache=c1)
        assert c1[0].shape[2] == 4 and c2[0].shape[2] == 6, "cache must append along the sequence axis"
        assert torch.allclose(torch.cat([o1, o2], 1), full, atol=1e-5), "chunked attention with cache != full"
        # the cached K/V must be reused, not recomputed: corrupt the cache and see the output change
        bad = (c1[0] + 1.0, c1[1])
        o2b, _ = attn(x[:, 4:], cache=bad)
        assert not torch.allclose(o2, o2b), "cached K must actually be used"
        # window eviction
        o3, c3 = attn(x[:, 4:], cache=c1, window=3)
        assert c3[0].shape == (2, 4, 3, 8) and c3[1].shape == (2, 4, 3, 8), "cache must keep only the last `window`"
        assert torch.allclose(c3[0], k_[:, :, 3:], atol=1e-6), "must keep the LAST window positions"


def check_incremental_matches_full(mod):
    model = mod.make_model(seed=0)
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 64, (3, 20), generator=g)
    with torch.no_grad():
        full, _ = model(tokens)
    ref = _reference_forward(model, tokens)
    assert torch.allclose(full, ref, atol=1e-5), "full forward (no cache) differs from the reference"
    inc, caches = mod.decode_incremental(model, tokens)
    assert inc.shape == (3, 20, 64)
    err = (inc - ref).abs().max().item()
    assert err < 1e-5, f"incremental decode differs from full forward, max abs err {err:.2e}"
    assert len(caches) == len(model.blocks)
    for k, v in caches:
        assert k.shape == (3, 4, 20, 8) and v.shape == (3, 4, 20, 8), f"cache shape {k.shape}"
    # cache grows by one per step
    with torch.no_grad():
        c = None
        for t in range(7):
            _, c = model(tokens[:, t:t + 1], start_pos=t, caches=c)
            assert c[0][0].shape[2] == t + 1, "cache must grow by exactly one per decoded token"
    # prefill + decode mix must also match
    with torch.no_grad():
        l0, c = model(tokens[:, :9], start_pos=0)
        outs = [l0]
        for t in range(9, 20):
            l, c = model(tokens[:, t:t + 1], start_pos=t, caches=c)
            outs.append(l)
    assert (torch.cat(outs, 1) - ref).abs().max() < 1e-5, "prefill + incremental decode differs from full"


def check_windowed_decode(mod):
    model = mod.make_model(seed=1)
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 64, (2, 24), generator=g)
    W = 5
    ref = _reference_forward(model, tokens, window=W)
    inc, caches = mod.decode_incremental(model, tokens, window=W)
    err = (inc - ref).abs().max().item()
    assert err < 1e-5, f"windowed incremental decode differs from banded full forward, max abs err {err:.2e}"
    for k, v in caches:
        assert k.shape[2] == W and v.shape[2] == W, f"window cache must hold exactly {W} positions, got {k.shape}"
    # windowed != unwindowed once the window is exceeded, equal before
    full_ref = _reference_forward(model, tokens)
    assert torch.allclose(inc[:, :W], full_ref[:, :W], atol=1e-5), "first W positions are unaffected by the window"
    assert not torch.allclose(inc[:, W:], full_ref[:, W:], atol=1e-3), "window must change later positions"
    # multi-token forward with window (banded mask over cached + new keys) must agree too
    with torch.no_grad():
        l0, c = model(tokens[:, :10], start_pos=0, window=W)
        l1, c = model(tokens[:, 10:17], start_pos=10, caches=c, window=W)
        l2, c = model(tokens[:, 17:], start_pos=17, caches=c, window=W)
    assert (torch.cat([l0, l1, l2], 1) - ref).abs().max() < 1e-5, "chunked windowed forward differs from banded ref"


def run(mod):
    check_mask(mod); print("  ok  causal / windowed masks")
    check_attention_with_cache(mod); print("  ok  attention with cache append + eviction")
    check_incremental_matches_full(mod); print("  ok  incremental decode == full forward (atol 1e-5)")
    check_windowed_decode(mod); print("  ok  sliding-window decode == banded full forward")
