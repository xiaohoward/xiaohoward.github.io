import torch
import torch.nn.functional as F


def check_softmax(mod):
    torch.manual_seed(0)
    x = torch.randn(3, 5, 7) * 50  # large logits -> naive exp overflows in fp32? not at 50, so add 1e4
    x[0, 0, 0] = 1e4
    y = mod.softmax(x, dim=-1)
    assert torch.isfinite(y).all(), "softmax must not overflow on large logits (subtract the max)"
    assert torch.allclose(y, torch.softmax(x, -1), atol=1e-6)
    x2 = torch.tensor([[1.0, float("-inf"), 2.0]])
    y2 = mod.softmax(x2)
    assert y2[0, 1] == 0.0 and abs(y2.sum().item() - 1) < 1e-6, "-inf logits must give exactly 0 probability"


def check_matches_torch_sdpa(mod):
    torch.manual_seed(1)
    B, H, L, d = 2, 4, 9, 16
    q, k, v = torch.randn(B, H, L, d), torch.randn(B, H, L, d), torch.randn(B, H, L, d)
    out, probs = mod.scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, L, d) and probs.shape == (B, H, L, L)
    assert torch.allclose(out, ref, atol=1e-5), "unmasked attention mismatch"
    assert torch.allclose(probs.sum(-1), torch.ones(B, H, L), atol=1e-5), "probs must sum to 1"
    # causal
    m = mod.build_mask(L, L, causal=True)
    out_c, probs_c = mod.scaled_dot_product_attention(q, k, v, m)
    ref_c = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out_c, ref_c, atol=1e-5), "causal attention mismatch"
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    assert (probs_c[..., upper] == 0).all(), "future positions must have exactly zero probability"
    # cross-attention with Lq != Lk
    Lk = 13
    k2, v2 = torch.randn(B, H, Lk, d), torch.randn(B, H, Lk, d)
    out_x, _ = mod.scaled_dot_product_attention(q, k2, v2)
    assert torch.allclose(out_x, F.scaled_dot_product_attention(q, k2, v2), atol=1e-5), "Lq != Lk mismatch"


def check_padding_mask(mod):
    torch.manual_seed(2)
    B, H, L, d = 3, 2, 8, 8
    q, k, v = torch.randn(B, H, L, d), torch.randn(B, H, L, d), torch.randn(B, H, L, d)
    lengths = torch.tensor([8, 5, 3])
    key_pad = torch.arange(L).view(1, L) >= lengths.view(B, 1)     # True = padding
    m = mod.build_mask(L, L, causal=False, key_padding_mask=key_pad)
    out, probs = mod.scaled_dot_product_attention(q, k, v, m)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    assert torch.allclose(out, ref, atol=1e-5), "padding-masked attention mismatch vs torch"
    for b in range(B):
        assert (probs[b, :, :, lengths[b]:] == 0).all(), f"padded keys of batch {b} must get zero probability"
    # causal + padding combined (padding at the end so no fully-masked rows)
    m2 = mod.build_mask(L, L, causal=True, key_padding_mask=key_pad)
    assert m2.shape == (B, 1, L, L)
    out2, probs2 = mod.scaled_dot_product_attention(q, k, v, m2)
    assert torch.allclose(out2, F.scaled_dot_product_attention(q, k, v, attn_mask=m2), atol=1e-5)
    assert torch.isfinite(out2).all()
    # decode-style causal: Lq=1, Lk=T must attend everything
    m3 = mod.build_mask(1, 6, causal=True)
    assert m3.all(), "with Lq=1 (KV-cache decoding) the single query must see all keys"
    assert mod.build_mask(4, 4) is None


def check_split_merge_and_mha(mod):
    torch.manual_seed(3)
    B, L, D, Hh = 2, 6, 24, 4
    x = torch.randn(B, L, D)
    s = mod.split_heads(x, Hh)
    assert s.shape == (B, Hh, L, D // Hh)
    assert torch.equal(s[:, 1], x[..., 6:12]), "head 1 must own features 6:12"
    assert torch.equal(mod.merge_heads(s), x), "merge_heads(split_heads(x)) must be identity"
    assert mod.merge_heads(s).is_contiguous()
    try:
        mod.split_heads(x, 5); raise AssertionError("D=24, H=5 should raise ValueError")
    except ValueError:
        pass
    # MHA vs nn.MultiheadAttention (batch_first, no biases)
    mha = torch.nn.MultiheadAttention(D, Hh, bias=False, batch_first=True)
    Wq, Wk, Wv = mha.in_proj_weight.detach().chunk(3, dim=0)
    Wo = mha.out_proj.weight.detach()
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    ref, _ = mha(x, x, x, attn_mask=causal, need_weights=False)
    mine = mod.multi_head_attention(x, x, Wq.T, Wk.T, Wv.T, Wo.T, Hh, attn_mask=mod.build_mask(L, L, causal=True))
    assert mine.shape == (B, L, D)
    assert torch.allclose(mine, ref.detach(), atol=1e-5), f"MHA mismatch vs nn.MultiheadAttention {(mine-ref).abs().max():.2e}"


def run(mod):
    check_softmax(mod);              print("  ok  stable softmax (large logits, -inf)")
    check_matches_torch_sdpa(mod);   print("  ok  SDPA matches torch (plain, causal, Lq!=Lk)")
    check_padding_mask(mod);         print("  ok  padding mask / combined mask / decode causal")
    check_split_merge_and_mha(mod);  print("  ok  split/merge heads + MHA matches nn.MultiheadAttention")
