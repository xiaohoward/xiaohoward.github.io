import numpy as np


def check_sigmoid(mod):
    z = np.array([-1e4, -50.0, -1.0, 0.0, 1.0, 50.0, 1e4])
    with np.errstate(over="raise", invalid="raise"):
        s = mod.sigmoid(z)
    assert s.shape == z.shape
    ref = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    assert np.allclose(s, ref, atol=1e-12), f"sigmoid mismatch {s} vs {ref}"
    assert s[0] == 0.0 and s[-1] == 1.0 and s[3] == 0.5
    assert np.all(s >= 0) and np.all(s <= 1)
    assert mod.sigmoid(np.zeros((2, 3))).shape == (2, 3), "sigmoid must be elementwise on any shape"


def check_loss_stable(mod):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 3)) * 1e3
    y = (rng.random(16) < 0.5).astype(np.float64)
    w = rng.normal(size=3)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        L = mod.loss(X, y, w, 0.3)
    assert np.isfinite(L), "loss must be finite for huge logits"
    # Small-logit sanity: compare against the naive formula.
    X = rng.normal(size=(50, 3))
    y = (rng.random(50) < 0.5).astype(np.float64)
    w = rng.normal(size=3) * 0.5
    b = 0.1
    p = 1 / (1 + np.exp(-(X @ w + b)))
    naive = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * 0.2 * w @ w
    assert abs(mod.loss(X, y, w, b, lam=0.2) - naive) < 1e-10, "loss disagrees with naive BCE"
    # Zero weights: loss is log 2.
    assert abs(mod.loss(X, y, np.zeros(3), 0.0) - np.log(2)) < 1e-12


def check_grad_finite_diff(mod):
    rng = np.random.default_rng(1)
    X, y = mod.make_data(n=40, d=4, seed=1)
    w = rng.normal(size=4)
    b = -0.3
    lam = 0.1
    dw, db = mod.grad(X, y, w, b, lam)
    assert dw.shape == (4,)
    eps = 1e-6
    num_w = np.zeros(4)
    for j in range(4):
        e = np.zeros(4); e[j] = eps
        num_w[j] = (mod.loss(X, y, w + e, b, lam) - mod.loss(X, y, w - e, b, lam)) / (2 * eps)
    num_b = (mod.loss(X, y, w, b + eps, lam) - mod.loss(X, y, w, b - eps, lam)) / (2 * eps)
    assert np.allclose(dw, num_w, atol=1e-6), f"dw {dw} vs finite diff {num_w}"
    assert abs(db - num_b) < 1e-6, f"db {db} vs finite diff {num_b}"


def check_fit_separable(mod):
    X, y = mod.make_data(n=300, d=2, seed=2, margin=2.5)
    w, b = mod.fit(X, y, lam=1e-3, lr=0.5, steps=500)
    p = mod.predict_proba(X, w, b)
    assert p.shape == (300,)
    acc = np.mean((p > 0.5) == (y > 0.5))
    assert acc >= 0.98, f"train accuracy {acc:.3f} < 0.98 on well-separated blobs"
    assert mod.loss(X, y, w, b, 1e-3) < 0.1, "loss should be small on separable data"
    # Held-out data from the same distribution.
    Xt, yt = mod.make_data(n=300, d=2, seed=99, margin=2.5)
    acc_t = np.mean((mod.predict_proba(Xt, w, b) > 0.5) == (yt > 0.5))
    assert acc_t >= 0.95, f"test accuracy {acc_t:.3f}"


def check_regularization_shrinks(mod):
    X, y = mod.make_data(n=200, d=3, seed=3, margin=3.0)
    w0, _ = mod.fit(X, y, lam=0.0, lr=0.5, steps=300)
    w1, _ = mod.fit(X, y, lam=1.0, lr=0.5, steps=300)
    assert np.linalg.norm(w1) < np.linalg.norm(w0), "L2 penalty should shrink weights"


def run(mod):
    check_sigmoid(mod); print("  ok  stable sigmoid")
    check_loss_stable(mod); print("  ok  stable loss")
    check_grad_finite_diff(mod); print("  ok  gradient vs finite differences")
    check_fit_separable(mod); print("  ok  fit on separable blobs")
    check_regularization_shrinks(mod); print("  ok  regularization shrinks weights")
