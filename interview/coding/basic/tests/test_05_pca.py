import numpy as np


def check_shapes(mod):
    X = mod.make_data(n=100, d=8, rank=3)
    p = mod.PCA(3).fit(X)
    assert p.mean_.shape == (8,)
    assert p.components_.shape == (3, 8)
    assert p.explained_variance_.shape == (3,) and p.explained_variance_ratio_.shape == (3,)
    assert p.singular_values_.shape == (3,)
    Z = p.transform(X)
    assert Z.shape == (100, 3)
    assert p.inverse_transform(Z).shape == (100, 8)


def check_vs_eigh(mod):
    X = mod.make_data(n=400, d=10, rank=3, noise=0.1, seed=1)
    k = 4
    p = mod.PCA(k).fit(X)
    Xc = X - X.mean(0)
    cov = Xc.T @ Xc / (len(X) - 1)
    evals, evecs = np.linalg.eigh(cov)
    evals, evecs = evals[::-1], evecs[:, ::-1]
    assert np.allclose(p.explained_variance_, evals[:k], rtol=1e-8), "explained_variance_ must equal top eigenvalues of covariance"
    assert np.allclose(p.explained_variance_ratio_, evals[:k] / evals.sum(), rtol=1e-8), "ratio must be relative to total variance"
    assert np.allclose(p.singular_values_ ** 2 / (len(X) - 1), evals[:k], rtol=1e-8)
    for j in range(k):
        c = p.components_[j]
        e = evecs[:, j]
        assert abs(abs(c @ e) - 1) < 1e-8, f"component {j} not parallel to eigenvector"
        assert c[np.argmax(np.abs(c))] > 0, f"component {j} sign not fixed"
    # Orthonormal rows.
    assert np.allclose(p.components_ @ p.components_.T, np.eye(k), atol=1e-10)
    assert np.all(np.diff(p.explained_variance_) <= 1e-12), "components must be sorted by decreasing variance"


def check_reconstruction(mod):
    X = mod.make_data(n=500, d=10, rank=3, noise=0.05, seed=2)
    p3 = mod.PCA(3).fit(X)
    err3 = np.mean((p3.inverse_transform(p3.transform(X)) - X) ** 2)
    assert err3 < 0.05 ** 2 * 1.5, f"rank-3 reconstruction error {err3} too large for rank-3 data"
    # Full PCA reconstructs exactly.
    p10 = mod.PCA(10).fit(X)
    assert np.allclose(p10.inverse_transform(p10.transform(X)), X, atol=1e-10), "full PCA must reconstruct exactly"
    assert abs(p10.explained_variance_ratio_.sum() - 1) < 1e-10
    # Eckart-Young: PCA reconstruction beats any other rank-3 orthonormal projection.
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.normal(size=(10, 3)))[0]
    Xc = X - X.mean(0)
    err_rand = np.mean((Xc @ Q @ Q.T - Xc) ** 2)
    assert err3 < err_rand, "PCA must minimize reconstruction error among rank-k projections"
    # Scores are centered and uncorrelated with variance = explained_variance_.
    Z = p3.transform(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-10)
    assert np.allclose(Z.T @ Z / (len(X) - 1), np.diag(p3.explained_variance_), atol=1e-8)


def check_tall_and_wide(mod):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 20))  # D > N: rank at most N - 1 = 5
    p = mod.PCA(5).fit(X)
    assert p.components_.shape == (5, 20)
    assert np.allclose(p.inverse_transform(p.transform(X)), X, atol=1e-8), "N-1 components reconstruct N points exactly"
    Xn = mod.make_data(n=50, d=4, rank=2, noise=0.0, seed=4)
    p = mod.PCA(2).fit(Xn)
    assert abs(p.explained_variance_ratio_.sum() - 1) < 1e-10, "noise-free rank-2 data: 2 components explain everything"


def run(mod):
    check_shapes(mod); print("  ok  shapes")
    check_vs_eigh(mod); print("  ok  matches np.linalg.eigh of covariance")
    check_reconstruction(mod); print("  ok  reconstruction / Eckart-Young / whitened scores")
    check_tall_and_wide(mod); print("  ok  wide and low-rank cases")
