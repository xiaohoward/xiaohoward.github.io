import numpy as np
import torch
import torch.nn.functional as F


def check_log_softmax(mod):
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(16, 7)) * 3
    ls = mod.log_softmax(Z)
    assert ls.shape == (16, 7)
    ref = F.log_softmax(torch.from_numpy(Z), dim=1).numpy()
    assert np.allclose(ls, ref, atol=1e-10), "log_softmax mismatch vs torch"
    assert np.allclose(np.exp(ls).sum(1), 1.0, atol=1e-12)
    # Stability: huge and tiny logits.
    Zbig = np.array([[1e3, 0.0, -1e3], [-1e4, -1e4 + 1, -1e4]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        lsb = mod.log_softmax(Zbig)
    assert np.all(np.isfinite(lsb)), "log_softmax must be finite for extreme logits"
    refb = F.log_softmax(torch.from_numpy(Zbig), dim=1).numpy()
    assert np.allclose(lsb, refb, atol=1e-8)


def check_loss_vs_torch(mod):
    rng = np.random.default_rng(1)
    for n, c in ((1, 3), (32, 10), (64, 1000)):
        Z = rng.normal(size=(n, c)) * 5
        y = rng.integers(c, size=n)
        L = mod.cross_entropy(Z, y)
        ref = F.cross_entropy(torch.from_numpy(Z), torch.from_numpy(y)).item()
        assert np.ndim(L) == 0
        assert abs(L - ref) < 1e-10, f"({n},{c}) loss {L} vs torch {ref}"
    # Uniform logits -> log C.
    assert abs(mod.cross_entropy(np.zeros((5, 4)), np.arange(4).repeat(2)[:5]) - np.log(4)) < 1e-12


def check_grad_vs_autograd(mod):
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(40, 12)) * 4
    y = rng.integers(12, size=40)
    G = mod.cross_entropy_grad(Z, y)
    assert G.shape == (40, 12)
    Zt = torch.from_numpy(Z).requires_grad_(True)
    F.cross_entropy(Zt, torch.from_numpy(y)).backward()
    assert np.allclose(G, Zt.grad.numpy(), atol=1e-10), "gradient mismatch vs torch autograd"
    # Rows of the gradient sum to zero (softmax sums to 1, onehot sums to 1).
    assert np.allclose(G.sum(1), 0, atol=1e-12)
    # Finite-difference spot check.
    eps = 1e-6
    i, j = 3, 5
    Zp = Z.copy(); Zp[i, j] += eps
    Zm = Z.copy(); Zm[i, j] -= eps
    fd = (mod.cross_entropy(Zp, y) - mod.cross_entropy(Zm, y)) / (2 * eps)
    assert abs(fd - G[i, j]) < 1e-7, f"finite diff {fd} vs grad {G[i, j]}"


def check_ignore_index(mod):
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(20, 6))
    y = rng.integers(6, size=20)
    y[[0, 4, 9, 15]] = -100
    L = mod.cross_entropy(Z, y)
    G = mod.cross_entropy_grad(Z, y)
    Zt = torch.from_numpy(Z).requires_grad_(True)
    Lt = F.cross_entropy(Zt, torch.from_numpy(y), ignore_index=-100)
    Lt.backward()
    assert abs(L - Lt.item()) < 1e-10, f"ignore_index loss {L} vs torch {Lt.item()}"
    assert np.allclose(G, Zt.grad.numpy(), atol=1e-10), "ignore_index gradient mismatch"
    assert np.all(G[[0, 4, 9, 15]] == 0), "ignored rows must have zero gradient"
    # Custom ignore_index value.
    y2 = y.copy(); y2[y2 == -100] = 7
    assert abs(mod.cross_entropy(Z, y2, ignore_index=7) - L) < 1e-12
    # All ignored.
    assert mod.cross_entropy(Z, np.full(20, -100)) == 0.0
    assert np.all(mod.cross_entropy_grad(Z, np.full(20, -100)) == 0)


def check_extreme_logits_grad(mod):
    Z = np.array([[1e3, 0.0, -1e3], [-500.0, -500.0, -500.0]])
    y = np.array([2, 0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        L = mod.cross_entropy(Z, y)
        G = mod.cross_entropy_grad(Z, y)
    Zt = torch.from_numpy(Z).requires_grad_(True)
    Lt = F.cross_entropy(Zt, torch.from_numpy(y)); Lt.backward()
    assert abs(L - Lt.item()) < 1e-8 and np.allclose(G, Zt.grad.numpy(), atol=1e-10)


def run(mod):
    torch.manual_seed(0)
    check_log_softmax(mod); print("  ok  stable log_softmax vs torch")
    check_loss_vs_torch(mod); print("  ok  loss vs F.cross_entropy")
    check_grad_vs_autograd(mod); print("  ok  gradient vs autograd")
    check_ignore_index(mod); print("  ok  ignore_index")
    check_extreme_logits_grad(mod); print("  ok  extreme logits")
