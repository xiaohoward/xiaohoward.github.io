import math
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))
import torch.nn.functional as F


def _manual_clip_loss(img, txt, scale):
    i = img / img.norm(dim=1, keepdim=True)
    t = txt / txt.norm(dim=1, keepdim=True)
    B = img.shape[0]
    tot = 0.0
    for a in range(B):
        row = torch.stack([scale * (i[a] * t[b]).sum() for b in range(B)])
        col = torch.stack([scale * (i[b] * t[a]).sum() for b in range(B)])
        tot += -(row[a] - torch.logsumexp(row, 0)) - (col[a] - torch.logsumexp(col, 0))
    return tot / (2 * B)


def check_normalize_and_logits(mod):
    torch.manual_seed(0)
    x = torch.randn(5, 7) * 3
    n = mod.l2_normalize(x)
    assert n.shape == x.shape and torch.allclose(n.norm(dim=1), torch.ones(5), atol=1e-6)
    assert torch.allclose(mod.l2_normalize(torch.zeros(2, 4)), torch.zeros(2, 4)), "zero rows must not produce nan"
    img, txt = mod.make_paired_embeddings(8, 16)
    ls = torch.tensor(math.log(1 / 0.07))
    L = mod.clip_logits(img, txt, ls)
    assert L.shape == (8, 8)
    ref = (1 / 0.07) * F.normalize(img, dim=1) @ F.normalize(txt, dim=1).t()
    assert torch.allclose(L, ref, atol=1e-4), "logits = exp(logit_scale) * cos-sim matrix"
    assert L.abs().max() <= 1 / 0.07 + 1e-3, "cosine similarities must be in [-1, 1] before scaling"


def check_loss_vs_manual(mod):
    img, txt = mod.make_paired_embeddings(6, 12, noise=0.8, seed=1)
    ls = torch.tensor(math.log(10.0))
    loss = mod.clip_loss(img, txt, ls)
    assert loss.dim() == 0
    ref = _manual_clip_loss(img, txt, 10.0)
    assert abs(float(loss) - float(ref)) < 1e-5, f"clip loss {float(loss)} != manual {float(ref)}"
    # not symmetric-per-direction in general but symmetric under swapping img/txt
    assert abs(float(mod.clip_loss(txt, img, ls)) - float(loss)) < 1e-5, "symmetric loss must be invariant to swapping modalities"


def check_perfect_alignment(mod):
    B, D = 8, 16
    img = torch.eye(B, D)                                # orthonormal rows
    txt = img.clone()
    prev = float("inf")
    for s in (1.0, 5.0, 20.0, 100.0):
        loss = float(mod.clip_loss(img, txt, torch.tensor(math.log(s))))
        assert loss <= prev, "loss must decrease with scale on a perfectly separable batch"
        prev = loss
        # exact value: CE with one logit s and B-1 zeros
        expected = math.log(1 + (B - 1) * math.exp(-s))
        assert abs(loss - expected) < 1e-5, f"loss {loss} != log(1+(B-1)e^-s)={expected} at scale {s}"
    assert prev < 1e-6, "loss must approach 0 as scale -> infinity"
    # random assignment baseline: identical img == txt but scale 0 -> log B
    loss0 = float(mod.clip_loss(img, txt, torch.tensor(-30.0)))
    assert abs(loss0 - math.log(B)) < 1e-4, "at scale ~0 all logits are equal -> loss = log B"


def check_temperature_gradient(mod):
    B, D = 8, 16
    img = torch.eye(B, D); txt = img.clone()
    ls = torch.tensor(math.log(5.0), requires_grad=True)
    mod.clip_loss(img, txt, ls).backward()
    assert ls.grad < 0, "with correct pairs most similar, increasing the scale lowers the loss: d loss / d log_scale < 0"
    # adversarial: each image's true text is the LEAST similar one (txt rolled) -> should want a smaller scale
    txt_bad = -img + 0.1 * torch.roll(img, 1, dims=0)
    ls2 = torch.tensor(math.log(5.0), requires_grad=True)
    mod.clip_loss(img, txt_bad, ls2).backward()
    assert ls2.grad > 0, "when the positives are the least similar, the loss wants a smaller scale"
    # gradients flow to embeddings too
    img_p = img.clone().requires_grad_(True)
    mod.clip_loss(img_p, txt, torch.tensor(1.0)).backward()
    assert img_p.grad is not None and img_p.grad.abs().sum() > 0


def check_recall(mod):
    img, txt = mod.make_paired_embeddings(B=20, D=8, noise=1.0, seed=3)
    sim = F.normalize(img, dim=1) @ F.normalize(txt, dim=1).t()
    for k in (1, 3, 5):
        r_i2t, r_t2i = mod.recall_at_k(img, txt, k)
        ref_i2t = sum(int(i in sim[i].argsort(descending=True)[:k].tolist()) for i in range(20)) / 20
        ref_t2i = sum(int(i in sim[:, i].argsort(descending=True)[:k].tolist()) for i in range(20)) / 20
        assert abs(r_i2t - ref_i2t) < 1e-6, f"recall@{k} i2t {r_i2t} != brute force {ref_i2t}"
        assert abs(r_t2i - ref_t2i) < 1e-6, f"recall@{k} t2i {r_t2i} != brute force {ref_t2i}"
    assert mod.recall_at_k(img, txt, 20) == (1.0, 1.0), "recall@B must be 1"
    assert mod.recall_at_k(img, img, 1) == (1.0, 1.0), "identical modalities -> recall@1 = 1"
    assert 0.0 < mod.recall_at_k(img, txt, 1)[0] < 1.0, "noisy pairs should give a non-trivial recall@1"


def check_siglip(mod):
    img, txt = mod.make_paired_embeddings(B=7, D=10, noise=0.5, seed=2)
    ls, lb = torch.tensor(math.log(10.0)), torch.tensor(-10.0)
    loss = mod.siglip_loss(img, txt, ls, lb)
    assert loss.dim() == 0
    i = F.normalize(img, dim=1); t = F.normalize(txt, dim=1)
    tot = 0.0
    for a in range(7):
        for b in range(7):
            z = 10.0 * float((i[a] * t[b]).sum()) - 10.0
            y = 1.0 if a == b else -1.0
            tot += -math.log(1 / (1 + math.exp(-y * z)))
    ref = tot / 7
    assert abs(float(loss) - ref) < 1e-4, f"siglip loss {float(loss)} != formula {ref}"
    # perfect batch: with bias 0 the orthogonal negatives sit at z = 0 -> each costs log 2 (that is why the bias
    # exists); with a large negative bias and a large scale the loss -> 0
    eye = torch.eye(6, 8)
    l_nobias = float(mod.siglip_loss(eye, eye, torch.tensor(math.log(200.0)), torch.tensor(0.0)))
    assert abs(l_nobias - 5 * math.log(2)) < 1e-3, f"bias 0: (B-1) negatives at z=0 -> (B-1) log 2 per image, got {l_nobias}"
    assert float(mod.siglip_loss(eye, eye, torch.tensor(math.log(200.0)), torch.tensor(-20.0))) < 1e-3
    lb_p = torch.tensor(-10.0, requires_grad=True)
    mod.siglip_loss(img, txt, ls, lb_p).backward()
    assert lb_p.grad is not None and lb_p.grad.abs() > 0, "logit_bias must receive gradient"
    # numerically stable for huge logits (no exp overflow)
    big = mod.siglip_loss(img, txt, torch.tensor(math.log(1e4)), torch.tensor(0.0))
    assert torch.isfinite(big), "use logsigmoid: must be finite for large scale"


def run(mod):
    check_normalize_and_logits(mod);  print("  ok  l2 normalize + scaled cosine logits")
    check_loss_vs_manual(mod);        print("  ok  symmetric InfoNCE matches manual computation")
    check_perfect_alignment(mod);     print("  ok  loss -> 0 with scale on a perfectly separable batch; log B at scale 0")
    check_temperature_gradient(mod);  print("  ok  log-temperature gradient has the right sign")
    check_recall(mod);                print("  ok  recall@k matches brute force")
    check_siglip(mod);                print("  ok  SigLIP pairwise sigmoid loss matches formula")
