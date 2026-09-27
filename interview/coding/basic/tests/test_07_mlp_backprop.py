import numpy as np
import torch
import torch.nn.functional as F


def _torch_forward(params, X):
    W1 = torch.from_numpy(params["W1"]).requires_grad_(True)
    b1 = torch.from_numpy(params["b1"]).requires_grad_(True)
    W2 = torch.from_numpy(params["W2"]).requires_grad_(True)
    b2 = torch.from_numpy(params["b2"]).requires_grad_(True)
    Xt = torch.from_numpy(X).requires_grad_(True)
    Y = F.relu(Xt @ W1 + b1) @ W2 + b2
    return Y, {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "X": Xt}


def check_init_and_shapes(mod):
    p = mod.init_params(5, 16, 3, seed=0)
    assert set(p) == {"W1", "b1", "W2", "b2"}
    assert p["W1"].shape == (5, 16) and p["b1"].shape == (16,) and p["W2"].shape == (16, 3) and p["b2"].shape == (3,)
    assert np.all(p["b1"] == 0) and np.all(p["b2"] == 0)
    q = mod.init_params(5, 16, 3, seed=0)
    assert np.array_equal(p["W1"], q["W1"]) and np.array_equal(p["W2"], q["W2"]), "init must be deterministic"
    big = mod.init_params(2000, 4000, 3, seed=1)
    assert abs(big["W1"].std() - np.sqrt(2 / 2000)) < 0.002, "W1 must use He scale sqrt(2/fan_in)"
    assert abs(big["W2"].std() - np.sqrt(2 / 4000)) < 0.002, "W2 must use He scale sqrt(2/fan_in)"
    X = np.random.default_rng(1).normal(size=(7, 5))
    Y, cache = mod.forward(p, X)
    assert Y.shape == (7, 3)
    T = np.zeros((7, 3))
    g = mod.backward(p, cache, Y, T)
    assert set(g) == {"W1", "b1", "W2", "b2", "X"}
    for k in ("W1", "b1", "W2", "b2"):
        assert g[k].shape == p[k].shape, f"grad {k} shape {g[k].shape}"
    assert g["X"].shape == X.shape


def check_forward_and_loss_vs_torch(mod):
    rng = np.random.default_rng(2)
    p = mod.init_params(8, 32, 4, seed=2)
    X = rng.normal(size=(20, 8))
    T = rng.normal(size=(20, 4))
    Y, _ = mod.forward(p, X)
    Yt, _ = _torch_forward(p, X)
    assert np.allclose(Y, Yt.detach().numpy(), atol=1e-10), "forward mismatch vs torch"
    L = mod.mse_loss(Y, T)
    Lt = F.mse_loss(Yt, torch.from_numpy(T)).item()
    assert np.ndim(L) == 0 and abs(L - Lt) < 1e-10, f"loss {L} vs torch {Lt}"


def check_backward_vs_autograd(mod):
    rng = np.random.default_rng(3)
    for n, d, h, c in ((1, 3, 5, 2), (32, 10, 64, 6), (64, 128, 256, 1)):
        p = mod.init_params(d, h, c, seed=n)
        X = rng.normal(size=(n, d))
        T = rng.normal(size=(n, c))
        Y, cache = mod.forward(p, X)
        g = mod.backward(p, cache, Y, T)
        Yt, leaves = _torch_forward(p, X)
        F.mse_loss(Yt, torch.from_numpy(T)).backward()
        for k in ("W1", "b1", "W2", "b2", "X"):
            ref = leaves[k].grad.numpy()
            assert np.allclose(g[k], ref, atol=1e-9, rtol=1e-7), f"({n},{d},{h},{c}) grad {k} mismatch: max err {np.abs(g[k]-ref).max()}"


def check_finite_differences(mod):
    rng = np.random.default_rng(4)
    p = mod.init_params(4, 6, 3, seed=5)
    X = rng.normal(size=(9, 4)); T = rng.normal(size=(9, 3))
    Y, cache = mod.forward(p, X)
    g = mod.backward(p, cache, Y, T)
    eps = 1e-6
    for k in ("W1", "b1", "W2", "b2"):
        flat = p[k].reshape(-1)
        for idx in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps; Lp = mod.mse_loss(mod.forward(p, X)[0], T)
            flat[idx] = old - eps; Lm = mod.mse_loss(mod.forward(p, X)[0], T)
            flat[idx] = old
            fd = (Lp - Lm) / (2 * eps)
            assert abs(fd - g[k].reshape(-1)[idx]) < 1e-6, f"finite diff {k}[{idx}]: {fd} vs {g[k].reshape(-1)[idx]}"


def check_training_reduces_loss(mod):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(128, 3))
    T = np.sin(X) @ np.ones((3, 1)) + 0.1 * X[:, :1]
    p = mod.init_params(3, 32, 1, seed=7)
    L0 = mod.mse_loss(mod.forward(p, X)[0], T)
    for _ in range(300):
        Y, cache = mod.forward(p, X)
        g = mod.backward(p, cache, Y, T)
        for k in ("W1", "b1", "W2", "b2"):
            p[k] -= 0.05 * g[k]
    L1 = mod.mse_loss(mod.forward(p, X)[0], T)
    assert L1 < 0.25 * L0, f"GD with these gradients did not reduce loss: {L0} -> {L1}"


def run(mod):
    torch.manual_seed(0)
    check_init_and_shapes(mod); print("  ok  init / shapes")
    check_forward_and_loss_vs_torch(mod); print("  ok  forward + loss vs torch")
    check_backward_vs_autograd(mod); print("  ok  backward vs torch autograd")
    check_finite_differences(mod); print("  ok  finite differences")
    check_training_reduces_loss(mod); print("  ok  gradient descent reduces loss")
