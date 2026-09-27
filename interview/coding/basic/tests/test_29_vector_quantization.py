import math
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_nearest_code(mod):
    z, cb, lab = mod.make_clustered_data(K=8, N=256, D=4, seed=0)
    idx = mod.nearest_code(z, cb)
    assert idx.shape == (256,) and idx.dtype == torch.int64, f"idx shape/dtype {tuple(idx.shape)} {idx.dtype}"
    ref = torch.stack([((z[n][None] - cb) ** 2).sum(1).argmin() for n in range(z.shape[0])])
    assert torch.equal(idx, ref), "nearest_code must match the per-point loop"
    assert (idx == lab).float().mean() > 0.95, "well-separated clusters should recover their labels"
    # unrelated random data too
    torch.manual_seed(1)
    z2, cb2 = torch.randn(50, 6), torch.randn(17, 6)
    ref2 = torch.cdist(z2, cb2).argmin(1)
    assert torch.equal(mod.nearest_code(z2, cb2), ref2), "nearest_code mismatch vs cdist on random data"


def check_straight_through(mod):
    torch.manual_seed(0)
    z = torch.randn(32, 4, requires_grad=True)
    cb = torch.randn(10, 4, requires_grad=True)
    z_q, idx = mod.quantize(z, cb)
    assert z_q.shape == z.shape
    assert torch.allclose(z_q.detach(), cb.detach()[idx], atol=1e-6), "forward value of z_q must be the selected code (z + (e - z) up to float rounding)"
    assert torch.equal(idx, mod.nearest_code(z.detach(), cb.detach()))
    g = torch.randn(32, 4)
    z_q.backward(g)
    assert torch.equal(z.grad, g), "STE: gradient w.r.t. z must be exactly the upstream gradient (identity Jacobian)"
    assert cb.grad is None or torch.count_nonzero(cb.grad) == 0, "no gradient may reach the codebook through z_q"


def check_losses(mod):
    z = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    cb = torch.tensor([[0.0, 1.0], [3.0, 3.0]], requires_grad=True)
    idx = mod.nearest_code(z.detach(), cb.detach())
    assert idx.tolist() == [0, 0], f"hand example assignments {idx.tolist()}"
    cl, cm = mod.vq_losses(z, cb, idx, beta=0.5)
    # squared errors: point0 -> (0,1): [0,1]; point1 -> (0,1): [1,0]; mean over 4 elements = 0.5
    assert abs(cl.item() - 0.5) < 1e-6, f"codebook loss {cl.item()} != 0.5"
    assert abs(cm.item() - 0.25) < 1e-6, f"commitment loss {cm.item()} != 0.5 * 0.5"
    cl.backward()
    assert z.grad is None or torch.count_nonzero(z.grad) == 0, "codebook loss must not update z"
    assert torch.count_nonzero(cb.grad) > 0, "codebook loss must update the codebook"
    # d/de mean((z - e)^2) = 2 (e - z) / (N*D), summed over the 2 points assigned to code 0
    exp0 = 2.0 * ((cb.detach()[0][None] - z.detach()).sum(0)) / 4.0
    assert torch.allclose(cb.grad[0], exp0, atol=1e-6) and torch.count_nonzero(cb.grad[1]) == 0
    z.grad = None; cb.grad = None
    cm.backward()
    assert cb.grad is None or torch.count_nonzero(cb.grad) == 0, "commitment loss must not update the codebook"
    assert torch.allclose(z.grad, 0.5 * 2.0 * (z.detach() - cb.detach()[idx]) / 4.0, atol=1e-6), "commitment grad wrong"
    # default beta
    cl2, cm2 = mod.vq_losses(z.detach(), cb.detach(), idx)
    assert abs(float(cm2) - 0.25 * 0.5) < 1e-6, "default beta should be 0.25"


def check_ema(mod):
    torch.manual_seed(0)
    K, D, N = 6, 3, 40
    cb = torch.randn(K, D)
    N0 = torch.rand(K) * 3
    m0 = torch.randn(K, D)
    z = torch.randn(N, D)
    idx = torch.randint(0, K, (N,))
    idx[idx == 5] = 0                                      # make code 5 unused this step
    decay, eps = 0.9, 1e-5
    cb_new, N_new, m_new = mod.ema_update(cb, N0, m0, z, idx, decay=decay, eps=eps)
    assert cb_new.shape == (K, D) and N_new.shape == (K,) and m_new.shape == (K, D)
    n = torch.zeros(K); s = torch.zeros(K, D)
    for i in range(N):
        n[idx[i]] += 1; s[idx[i]] += z[i]
    N_ref = decay * N0 + (1 - decay) * n
    m_ref = decay * m0 + (1 - decay) * s
    tot = N_ref.sum()
    N_s = (N_ref + eps) / (tot + K * eps) * tot
    cb_ref = m_ref / N_s[:, None]
    assert torch.allclose(N_new, N_ref, atol=1e-6), "EMA cluster_size mismatch"
    assert torch.allclose(m_new, m_ref, atol=1e-6), "EMA embed_avg mismatch"
    assert torch.allclose(cb_new, cb_ref, atol=1e-5), "EMA codebook mismatch (check Laplace smoothing)"
    assert torch.equal(N0, N0.clone()) and cb.shape == (K, D), "inputs must not be modified"
    # unused code: still finite thanks to smoothing even from zero state
    cb_z, N_z, m_z = mod.ema_update(torch.zeros(K, D), torch.zeros(K), torch.zeros(K, D), z, idx, decay=0.0, eps=eps)
    assert torch.isfinite(cb_z).all(), "Laplace smoothing must keep unused codes finite"
    # converges to the batch mean of the assigned points when decay=0 (up to smoothing)
    for k in range(5):
        assert torch.allclose(cb_z[k], z[idx == k].mean(0), atol=1e-3), "decay=0 EMA should give cluster means"


def check_usage(mod):
    K = 8
    counts, ppl = mod.codebook_usage(torch.arange(K).repeat(4), K)
    assert counts.shape == (K,) and counts.tolist() == [4] * K
    assert abs(ppl - K) < 1e-5, f"uniform usage perplexity {ppl} != {K}"
    counts, ppl = mod.codebook_usage(torch.full((30,), 3), K)
    assert counts.tolist() == [0, 0, 0, 30, 0, 0, 0, 0]
    assert abs(ppl - 1.0) < 1e-6, f"single-code perplexity {ppl} != 1"
    idx = torch.tensor([0, 0, 1, 2])
    counts, ppl = mod.codebook_usage(idx, 4)
    p = torch.tensor([0.5, 0.25, 0.25])
    assert abs(ppl - math.exp(float(-(p * p.log()).sum()))) < 1e-5, "perplexity formula wrong"


def check_dead_reset(mod):
    torch.manual_seed(0)
    K, D = 8, 4
    cb = torch.randn(K, D)
    z = torch.randn(50, D)
    counts = torch.tensor([5, 0, 3, 0, 0, 9, 1, 2])
    gen = torch.Generator().manual_seed(123)
    cb_new, dead = mod.reset_dead_codes(cb, counts, z, gen)
    assert dead.dtype == torch.bool and dead.tolist() == [False, True, False, True, True, False, False, False]
    live = ~dead
    assert torch.equal(cb_new[live], cb[live]), "live codes must be untouched"
    for k in torch.nonzero(dead).flatten().tolist():
        assert any(torch.equal(cb_new[k], z[n]) for n in range(z.shape[0])), "dead codes must be reset to rows of z"
    rows = [cb_new[k] for k in torch.nonzero(dead).flatten().tolist()]
    assert not torch.equal(rows[0], rows[1]) and not torch.equal(rows[1], rows[2]), "sample without replacement"
    assert torch.equal(cb, cb_new) is False and not cb.requires_grad
    # determinism
    gen2 = torch.Generator().manual_seed(123)
    cb_new2, _ = mod.reset_dead_codes(cb, counts, z, gen2)
    assert torch.equal(cb_new, cb_new2), "must be deterministic given the generator"
    # nothing dead -> identical
    cb_same, dead0 = mod.reset_dead_codes(cb, torch.ones(K, dtype=torch.long), z, gen)
    assert torch.equal(cb_same, cb) and not dead0.any()


def run(mod):
    check_nearest_code(mod);      print("  ok  nearest code matches loop / cdist")
    check_straight_through(mod);  print("  ok  straight-through: identity Jacobian to z, none to codebook")
    check_losses(mod);            print("  ok  codebook / commitment losses and their gradient routing")
    check_ema(mod);               print("  ok  EMA update matches manual formula with Laplace smoothing")
    check_usage(mod);             print("  ok  usage histogram and perplexity")
    check_dead_reset(mod);        print("  ok  dead-code reset touches only dead codes")
