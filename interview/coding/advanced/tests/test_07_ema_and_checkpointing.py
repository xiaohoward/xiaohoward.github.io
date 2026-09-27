import copy

import torch
import torch.nn as nn

# tiny tensors: intra-op threading is pure overhead (and can be 10x slower on many-core boxes)
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_ema_decay_schedule(mod):
    m = nn.Linear(4, 4)
    ema = mod.EMA(m, decay=0.999, warmup=True)
    assert abs(ema.get_decay() - 0.1) < 1e-12, "first decay must be 1/10"
    for n in range(50):
        assert abs(ema.get_decay() - min(0.999, (1 + n) / (10 + n))) < 1e-12
        ema.update(m)
    ema2 = mod.EMA(m, decay=0.5, warmup=True)
    for _ in range(30):
        ema2.update(m)
    assert abs(ema2.get_decay() - 0.5) < 1e-12, "decay must be capped by `decay`"
    ema3 = mod.EMA(m, decay=0.9, warmup=False)
    assert abs(ema3.get_decay() - 0.9) < 1e-12, "no warmup -> constant decay"
    ema3.update(m)
    assert abs(ema3.get_decay() - 0.9) < 1e-12


def check_ema_arithmetic(mod):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 2))
    ema = mod.EMA(m, decay=0.9, warmup=True)
    manual = [p.detach().clone() for p in m.parameters()]
    assert all(torch.equal(s, q) for s, q in zip(ema.shadow, manual)), "shadow must start as a copy"
    for t in range(25):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(m)
        d = min(0.9, (1 + t) / (10 + t))
        manual = [d * s + (1 - d) * p.detach() for s, p in zip(manual, m.parameters())]
        for s, q in zip(ema.shadow, manual):
            assert torch.allclose(s, q, atol=1e-6), f"EMA arithmetic mismatch at update {t}"
    for s in ema.shadow:
        assert not s.requires_grad, "shadow must not require grad"
    for p in m.parameters():
        assert p.grad is None and p.requires_grad, "update must not touch the model"
    # a model whose params never change keeps shadow == params for every decay
    m2 = nn.Linear(3, 3)
    e2 = mod.EMA(m2, decay=0.99, warmup=False)
    for _ in range(5):
        e2.update(m2)
    assert all(torch.allclose(s, p) for s, p in zip(e2.shadow, m2.parameters()))


def check_copy_to_restore(mod):
    torch.manual_seed(1)
    m = nn.Linear(4, 3)
    ema = mod.EMA(m, decay=0.5, warmup=False)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    live = [p.detach().clone() for p in m.parameters()]
    x = torch.randn(2, 4)
    y_live = m(x).detach().clone()
    ema.copy_to(m)
    for p, s in zip(m.parameters(), ema.shadow):
        assert torch.equal(p.detach(), s), "copy_to must write shadow values into the model"
    assert not torch.allclose(m(x), y_live), "model output should change under EMA weights"
    ema.restore(m)
    for p, l in zip(m.parameters(), live):
        assert torch.equal(p.detach(), l), "restore must bring the live weights back"
    assert torch.allclose(m(x), y_live)
    assert ema.backup is None, "restore must clear the backup"
    try:
        ema.restore(m)
        raise AssertionError("restore without copy_to must raise RuntimeError")
    except RuntimeError:
        pass
    # copy_to must not break optimizer identity: parameters are modified in place
    ids = [id(p) for p in m.parameters()]
    ema.copy_to(m); ema.restore(m)
    assert ids == [id(p) for p in m.parameters()], "parameters must be modified in place (same tensor objects)"


def _run(block, x, use_ckpt, mod, seed):
    torch.manual_seed(seed)
    block.zero_grad(set_to_none=True)
    xi = x.detach().clone().requires_grad_(True)
    h = mod.checkpoint(block, xi) if use_ckpt else block(xi)
    loss = (h ** 2).sum() + h.mean()
    loss.backward()
    return loss.detach(), xi.grad.clone(), [p.grad.clone() for p in block.parameters()]


def check_checkpoint_gradients(mod):
    torch.manual_seed(0)
    block = mod.DropBlock(d=32, p=0.0)
    x = torch.randn(8, 32)
    l0, gx0, gp0 = _run(block, x, False, mod, seed=0)
    l1, gx1, gp1 = _run(block, x, True, mod, seed=0)
    assert torch.allclose(l0, l1, atol=1e-6), "checkpointed forward must give the same output"
    assert torch.allclose(gx0, gx1, atol=1e-6), "input gradient must be identical"
    for a, b in zip(gp0, gp1):
        assert a is not None and b is not None and torch.allclose(a, b, atol=1e-6), "param gradients must match"
    # works when the input does not require grad (parameters still get gradients)
    block.zero_grad(set_to_none=True)
    mod.checkpoint(block, x.detach()).sum().backward()
    assert all(p.grad is not None for p in block.parameters()), "params need grads even if the input does not"
    # composes: two checkpointed blocks in sequence
    b2 = mod.DropBlock(d=32, p=0.0)
    xi = x.clone().requires_grad_(True)
    ref = (b2(block(xi)) ** 2).sum()
    g_ref = torch.autograd.grad(ref, xi)[0]
    xi2 = x.clone().requires_grad_(True)
    out = (mod.checkpoint(b2, mod.checkpoint(block, xi2)) ** 2).sum()
    g_ck = torch.autograd.grad(out, xi2)[0]
    assert torch.allclose(g_ref, g_ck, atol=1e-6), "chained checkpoints must back-propagate correctly"


def check_recompute_happens_and_saves_memory(mod):
    torch.manual_seed(0)
    block = mod.DropBlock(d=64, p=0.0)
    x = torch.randn(16, 64, requires_grad=True)
    block.calls = 0
    (block(x) ** 2).sum().backward()
    assert block.calls == 1
    block.calls = 0
    (mod.checkpoint(block, x) ** 2).sum().backward()
    assert block.calls == 2, f"checkpointing must run the block exactly twice (fwd + recompute), got {block.calls}"
    # forward alone must not run the block twice, and must not build a graph inside the block
    block.calls = 0
    y = mod.checkpoint(block, x)
    assert block.calls == 1 and y.requires_grad, "forward runs once; output must be part of the outer graph"
    # count tensors saved for backward in the outer graph: checkpointing must save fewer
    def count_saved(fn):
        n = [0]
        def pack(t):
            n[0] += 1; return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = fn()
        out.sum().backward()
        return n[0]
    n_plain = count_saved(lambda: block(x))
    n_ckpt = count_saved(lambda: mod.checkpoint(block, x))
    assert n_ckpt < n_plain, f"checkpoint should save fewer tensors ({n_ckpt} vs {n_plain})"


def check_rng_restored_for_dropout(mod):
    torch.manual_seed(0)
    block = mod.DropBlock(d=32, p=0.5)
    block.train()
    x = torch.randn(8, 32)
    l0, gx0, gp0 = _run(block, x, False, mod, seed=3)
    l1, gx1, gp1 = _run(block, x, True, mod, seed=3)
    assert torch.allclose(l0, l1, atol=1e-6), "same seed -> same dropout mask in forward"
    assert torch.allclose(gx0, gx1, atol=1e-6), "recompute must reuse the SAME dropout mask (restore RNG state)"
    for a, b in zip(gp0, gp1):
        assert torch.allclose(a, b, atol=1e-6), "param grads must match with dropout (RNG restored)"
    # the RNG state after backward must not be perturbed by the recompute (relative to no checkpoint)
    torch.manual_seed(5); _run(block, x, False, mod, seed=5); r_plain = torch.rand(3)
    torch.manual_seed(5); _run(block, x, True, mod, seed=5); r_ck = torch.rand(3)
    assert torch.equal(r_plain, r_ck), "backward recompute must leave the global RNG stream where it was"


def run(mod):
    check_ema_decay_schedule(mod); print("  ok  EMA warmup decay schedule")
    check_ema_arithmetic(mod); print("  ok  EMA update arithmetic vs manual")
    check_copy_to_restore(mod); print("  ok  EMA copy_to / restore")
    check_checkpoint_gradients(mod); print("  ok  checkpoint gradients identical (atol 1e-6)")
    check_recompute_happens_and_saves_memory(mod); print("  ok  recompute happens once, fewer saved tensors")
    check_rng_restored_for_dropout(mod); print("  ok  RNG state restored for dropout in recompute")
