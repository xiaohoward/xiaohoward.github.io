import math
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))
import torch.nn.functional as F


def check_shift(mod):
    logits, labels, mask = mod.make_batch(B=3, L=10, V=50, seed=0)
    nll, valid = mod.next_token_nll(logits, labels)          # no mask, no ignore -> everything valid
    assert nll.shape == (3, 9) and valid.shape == (3, 9) and valid.dtype == torch.bool
    assert valid.all(), "without mask/ignore every shifted position is valid"
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, 50), labels[:, 1:].reshape(-1), reduction="none").view(3, 9)
    assert torch.allclose(nll, ref, atol=1e-6), "shift: logits[:, t] must be scored against labels[:, t+1]"
    # wrong shift direction would match this instead
    wrong = F.cross_entropy(logits[:, 1:].reshape(-1, 50), labels[:, :-1].reshape(-1), reduction="none").view(3, 9)
    assert not torch.allclose(nll, wrong), "shifted the wrong way"


def check_padding_and_ignore(mod):
    logits, labels, mask = mod.make_batch(B=3, L=10, V=50, seed=1)
    lab = labels.clone()
    lab[0, 2] = -100                                          # ignore one label inside a real sequence
    nll, valid = mod.next_token_nll(logits, lab, attention_mask=mask)
    exp_valid = torch.ones(3, 9, dtype=torch.bool)
    exp_valid[0, 1] = False                                  # target position 2 ignored
    exp_valid[1, 6:] = False                                 # seq1 has 7 real tokens: targets 7,8,9 are pad
    exp_valid[2, 3:] = False                                 # seq2 has 4 real tokens
    assert torch.equal(valid, exp_valid), f"valid mask wrong:\n{valid.int()}\nexpected\n{exp_valid.int()}"
    assert torch.equal(nll[~valid], torch.zeros(int((~valid).sum()))), "invalid positions must carry nll = 0"
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, 50), lab[:, 1:].reshape(-1), reduction="none").view(3, 9)
    assert torch.allclose(nll[valid], ref[valid], atol=1e-6)
    # altering logits at invalid positions (and the padding labels) must not change any loss
    logits2 = logits.clone()
    logits2[:, :-1][~valid] += 100.0 * torch.randn(int((~valid).sum()), 50)
    lab2 = lab.clone(); lab2[mask == 0] = 7
    nll2, valid2 = mod.next_token_nll(logits2, lab2, attention_mask=mask)
    assert torch.equal(valid2, valid) and torch.allclose(nll2, nll, atol=1e-6), "padding logits/labels must not affect the loss"
    assert torch.allclose(mod.sequence_loss(nll2, valid2), mod.sequence_loss(nll, valid))
    # reference token-mean via F.cross_entropy ignore_index on the flattened batch
    lab_ign = lab.clone(); lab_ign[mask == 0] = -100
    ref_mean = F.cross_entropy(logits[:, :-1].reshape(-1, 50), lab_ign[:, 1:].reshape(-1), ignore_index=-100)
    assert abs(float(mod.batch_loss(nll, valid, "token")) - float(ref_mean)) < 1e-5, "token-weighted batch loss mismatch"
    # left padding: context token must also be real
    maskL = torch.ones(1, 6, dtype=torch.long); maskL[0, :2] = 0
    labL = torch.randint(1, 50, (1, 6))
    _, vL = mod.next_token_nll(torch.randn(1, 6, 50), labL, attention_mask=maskL)
    assert vL[0].tolist() == [False, False, True, True, True], "left padding: predicting the first real token from a pad context is invalid"


def check_reductions_and_ppl(mod):
    nll = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    sl = mod.sequence_loss(nll, valid)
    assert torch.allclose(sl, torch.tensor([2.0, 4.0])), f"sequence loss {sl.tolist()}"
    assert abs(float(mod.batch_loss(nll, valid, "token")) - 10.0 / 4) < 1e-6
    assert abs(float(mod.batch_loss(nll, valid, "sequence")) - 3.0) < 1e-6
    assert abs(mod.perplexity(nll, valid) - math.exp(2.5)) < 1e-4
    # empty sequence -> 0 not nan
    sl0 = mod.sequence_loss(nll, torch.zeros(2, 4, dtype=torch.bool))
    assert torch.equal(sl0, torch.zeros(2)), "sequence with no valid tokens should give 0"
    # uniform model -> perplexity = V
    V = 37
    logits = torch.zeros(2, 8, V)
    labels = torch.randint(0, V, (2, 8))
    n, v = mod.next_token_nll(logits, labels)
    assert abs(mod.perplexity(n, v) - V) < 1e-3, f"uniform model perplexity {mod.perplexity(n, v)} != {V}"
    # perfect model -> perplexity 1
    logits = torch.zeros(2, 8, V)
    logits[:, :-1].scatter_(2, labels[:, 1:, None], 60.0)          # position t predicts labels[t+1]
    n, v = mod.next_token_nll(logits, labels)
    assert abs(mod.perplexity(n, v) - 1.0) < 1e-3
    try:
        mod.batch_loss(nll, valid, "foo")
        raise AssertionError("unknown reduction must raise ValueError")
    except ValueError:
        pass


def check_label_smoothing(mod):
    torch.manual_seed(0)
    B, L, V, eps = 2, 5, 11, 0.1
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    nll, valid = mod.next_token_nll(logits, labels, label_smoothing=eps)
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt_lp = logp.gather(-1, labels[:, 1:, None])[..., 0]
    ref = (1 - eps) * (-tgt_lp) + eps * (-logp).mean(-1)
    assert torch.allclose(nll, ref, atol=1e-5), "label smoothing must be (1-eps)*nll + eps*mean_v(-log p_v)"
    nll0, _ = mod.next_token_nll(logits, labels)
    assert not torch.allclose(nll0, nll), "smoothing must change the loss"
    # smoothing + ignore: ignored positions still 0
    lab = labels.clone(); lab[1, 3] = -100
    nll_s, v_s = mod.next_token_nll(logits, lab, label_smoothing=eps)
    assert not v_s[1, 2] and nll_s[1, 2] == 0


def check_packing(mod):
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10], [11, 12, 13, 14, 15]]
    tok, sid = mod.pack_sequences(seqs, block_size=8, eos_id=99, pad_id=0)
    assert tok.dtype == torch.long and sid.dtype == torch.long
    exp_tok = torch.tensor([[1, 2, 3, 99, 4, 5, 99, 0],
                            [6, 7, 8, 9, 99, 10, 99, 0],
                            [11, 12, 13, 14, 15, 99, 0, 0]])
    exp_sid = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1],
                            [2, 2, 2, 2, 2, 3, 3, -1],
                            [4, 4, 4, 4, 4, 4, -1, -1]])
    assert torch.equal(tok, exp_tok), f"packed tokens\n{tok}\nexpected\n{exp_tok}"
    assert torch.equal(sid, exp_sid), f"sequence ids\n{sid}\nexpected\n{exp_sid}"
    try:
        mod.pack_sequences([[1] * 8], 8, 99, 0)
        raise AssertionError("document longer than block (incl. eos) must raise ValueError")
    except ValueError:
        pass
    # exact fit: no wasted slot
    tok2, sid2 = mod.pack_sequences([[1, 2, 3], [4, 5, 6]], 4, 99, 0)
    assert tok2.shape == (2, 4) and (sid2 != -1).all()


def check_packed_mask(mod):
    sid = torch.tensor([[0, 0, 0, 1, 1, -1]])
    m = mod.packed_causal_mask(sid)
    assert m.shape == (1, 6, 6) and m.dtype == torch.bool
    exp = torch.tensor([[1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(m[0], exp), f"packed causal mask\n{m[0].int()}\nexpected\n{exp.int()}"
    # properties on a packed batch: causal within, block-diagonal, no cross-document leakage
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10], [11, 12, 13, 14, 15]]
    _, sid = mod.pack_sequences(seqs, 8, 99, 0)
    m = mod.packed_causal_mask(sid)
    assert m.shape == (3, 8, 8)
    assert not m.triu(1).any(), "must be causal (no future keys)"
    same = sid[:, :, None] == sid[:, None, :]
    assert not (m & ~same).any(), "no attention across documents"
    assert torch.equal(m, torch.tril(same) & (sid != -1)[:, :, None]), "must equal tril(same-doc) with pad rows cleared"
    # feeding it to SDPA: real rows all finite
    q = torch.randn(3, 8, 4)
    out = F.scaled_dot_product_attention(q, q, q, attn_mask=m)
    assert torch.isfinite(out[sid != -1]).all()


def run(mod):
    check_shift(mod);                print("  ok  next-token shift matches F.cross_entropy on hand-shifted tensors")
    check_padding_and_ignore(mod);   print("  ok  ignore_index + attention mask (right and left padding)")
    check_reductions_and_ppl(mod);   print("  ok  sequence / token / sequence-weighted reductions; uniform ppl = V")
    check_label_smoothing(mod);      print("  ok  label smoothing matches manual formula")
    check_packing(mod);              print("  ok  pack_sequences layout and sequence_ids")
    check_packed_mask(mod);          print("  ok  packed block-diagonal causal mask")
