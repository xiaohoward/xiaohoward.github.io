import numpy as np


def check_shapes(mod):
    X, y, _, _ = mod.make_data(n=64, d=3)
    w, b = mod.fit_closed_form(X, y)
    assert w.shape == (3,), f"w shape {w.shape}"
    assert np.isscalar(b) or np.ndim(b) == 0, "b must be a scalar"
    yhat = mod.predict(X, w, b)
    assert yhat.shape == (64,), f"predict shape {yhat.shape}"
    loss = mod.mse_loss(X, y, w, b)
    assert np.ndim(loss) == 0 and np.isfinite(loss), "loss must be a finite scalar"


def check_closed_form_vs_lstsq(mod):
    X, y, _, _ = mod.make_data(n=256, d=5, noise=0.3, seed=1)
    w, b = mod.fit_closed_form(X, y, lam=0.0)
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    theta = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose(w, theta[:5], atol=1e-8), f"w mismatch vs lstsq: {w} vs {theta[:5]}"
    assert abs(b - theta[5]) < 1e-8, f"b mismatch vs lstsq: {b} vs {theta[5]}"


def check_ridge(mod):
    X, y, _, _ = mod.make_data(n=128, d=4, noise=0.3, seed=2)
    w0, _ = mod.fit_closed_form(X, y, lam=0.0)
    w1, b1 = mod.fit_closed_form(X, y, lam=1.0)
    assert np.linalg.norm(w1) < np.linalg.norm(w0), "ridge must shrink the weights"
    # Ridge solution is the stationary point of mse_loss: the gradient must vanish.
    n = len(X)
    r = X @ w1 + b1 - y
    gw = (2.0 / n) * X.T @ r + 2.0 * w1
    gb = (2.0 / n) * r.sum()
    assert np.allclose(gw, 0, atol=1e-8) and abs(gb) < 1e-8, f"ridge gradient not zero: {gw}, {gb}"
    # Bias must not be penalized: with a huge lam, w -> 0 and b -> mean(y).
    w_big, b_big = mod.fit_closed_form(X, y, lam=1e8)
    assert np.allclose(w_big, 0, atol=1e-6) and abs(b_big - y.mean()) < 1e-6, "bias must not be penalized"


def check_gd_matches_closed_form(mod):
    for lam in (0.0, 0.05):
        X, y, _, _ = mod.make_data(n=256, d=5, noise=0.1, seed=3)
        w_cf, b_cf = mod.fit_closed_form(X, y, lam=lam)
        w_gd, b_gd = mod.fit_gd(X, y, lam=lam, lr=0.05, epochs=300, batch_size=32, seed=0)
        assert np.allclose(w_gd, w_cf, atol=2e-2), f"lam={lam}: GD w {w_gd} vs closed form {w_cf}"
        assert abs(b_gd - b_cf) < 2e-2, f"lam={lam}: GD b {b_gd} vs closed form {b_cf}"
        assert mod.mse_loss(X, y, w_gd, b_gd, lam) <= mod.mse_loss(X, y, np.zeros(5), 0.0, lam), "GD did not reduce loss"


def check_determinism(mod):
    X, y, _, _ = mod.make_data(n=64, d=3)
    a = mod.fit_gd(X, y, epochs=5, seed=7)
    b = mod.fit_gd(X, y, epochs=5, seed=7)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1], "fit_gd must be deterministic given seed"


def run(mod):
    check_shapes(mod); print("  ok  shapes")
    check_closed_form_vs_lstsq(mod); print("  ok  closed form matches np.linalg.lstsq")
    check_ridge(mod); print("  ok  ridge stationarity / unpenalized bias")
    check_gd_matches_closed_form(mod); print("  ok  gradient descent matches closed form")
    check_determinism(mod); print("  ok  determinism")
