import numpy as np


def _inertia(X, C, labels):
    return float(np.sum((X - C[labels]) ** 2))


def check_pairwise(mod):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    C = rng.normal(size=(4, 5))
    d = mod.pairwise_sq_dists(X, C)
    assert d.shape == (30, 4)
    ref = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    assert np.allclose(d, ref, atol=1e-9), "pairwise squared distances wrong"
    assert np.all(mod.pairwise_sq_dists(X, X[:3]) >= 0), "distances must be clamped >= 0"
    assert np.allclose(np.diag(mod.pairwise_sq_dists(X[:3], X[:3])), 0, atol=1e-9)


def check_init(mod):
    X, _, centers = mod.make_blobs(n_per=50, k=5, d=3, spread=0.2, seed=1)
    rng = np.random.default_rng(0)
    C = mod.kmeans_pp_init(X, 5, rng)
    assert C.shape == (5, 3)
    # Each chosen centroid is an actual data point.
    d = mod.pairwise_sq_dists(C, X)
    assert np.all(d.min(axis=1) < 1e-12), "k-means++ centroids must be data points"
    # k-means++ on well-separated blobs should pick one seed per blob (with high probability; fixed seed).
    nearest_true = np.argmin(mod.pairwise_sq_dists(C, centers), axis=1)
    assert len(set(nearest_true.tolist())) == 5, f"k-means++ seeds not spread across blobs: {nearest_true}"
    # Modifies rng deterministically.
    C2 = mod.kmeans_pp_init(X, 5, np.random.default_rng(0))
    assert np.array_equal(C, C2), "init must be deterministic given rng"


def check_assign_update(mod):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 2))
    C = rng.normal(size=(3, 2))
    labels = mod.assign(X, C)
    assert labels.shape == (40,) and labels.dtype.kind == "i"
    ref = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
    assert np.array_equal(labels, ref)
    C_new = mod.update(X, labels, C)
    for j in range(3):
        assert np.allclose(C_new[j], X[labels == j].mean(0))
    # Empty cluster keeps old centroid.
    labels_e = np.zeros(40, dtype=np.int64)
    C_e = mod.update(X, labels_e, C)
    assert np.allclose(C_e[0], X.mean(0)) and np.allclose(C_e[1:], C[1:]), "empty clusters must keep old centroid"


def check_recovery(mod):
    X, true_labels, centers = mod.make_blobs(n_per=100, k=4, d=2, spread=0.3, seed=3)
    C, labels, inertia = mod.kmeans(X, 4, seed=0)
    assert C.shape == (4, 2) and labels.shape == (400,)
    assert abs(inertia - _inertia(X, C, labels)) < 1e-6, "inertia must equal sum of squared distances to assigned centroid"
    # Each recovered centroid within 0.15 of a distinct true center.
    d = np.sqrt(mod.pairwise_sq_dists(C, centers))
    match = d.argmin(axis=1)
    assert len(set(match.tolist())) == 4, f"centroids collapsed: {match}"
    assert d.min(axis=1).max() < 0.15, f"centroid error {d.min(axis=1)}"
    # Label agreement up to permutation.
    perm_labels = match[labels]
    assert np.mean(perm_labels == true_labels) == 1.0, "cluster assignment does not recover true labels"
    # Solution is a fixed point of Lloyd's algorithm.
    assert np.array_equal(mod.assign(X, C), labels)
    assert np.allclose(mod.update(X, labels, C), C, atol=1e-8)


def check_inertia_monotone(mod):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 8))
    C = mod.kmeans_pp_init(X, 6, np.random.default_rng(1))
    prev = np.inf
    for _ in range(30):
        labels = mod.assign(X, C)
        cur = _inertia(X, C, labels)
        assert cur <= prev + 1e-9, f"inertia increased {prev} -> {cur}"
        prev = cur
        C = mod.update(X, labels, C)
    C_f, labels_f, inertia_f = mod.kmeans(X, 6, n_iter=200, seed=1)
    assert inertia_f <= prev + 1e-9, "kmeans() must be at least as good as 30 manual Lloyd steps"
    assert inertia_f < np.sum((X - X.mean(0)) ** 2), "inertia must beat a single centroid"


def check_edge_cases(mod):
    X = np.random.default_rng(5).normal(size=(20, 3))
    C, labels, inertia = mod.kmeans(X, 1, seed=0)
    assert np.allclose(C[0], X.mean(0)) and np.all(labels == 0), "k=1 must return the mean"
    C, labels, inertia = mod.kmeans(X, 20, seed=0)
    assert inertia < 1e-12 and len(set(labels.tolist())) == 20, "k=N must give zero inertia"


def run(mod):
    check_pairwise(mod); print("  ok  pairwise squared distances")
    check_init(mod); print("  ok  k-means++ init")
    check_assign_update(mod); print("  ok  assign / update")
    check_recovery(mod); print("  ok  cluster recovery on blobs")
    check_inertia_monotone(mod); print("  ok  inertia non-increasing")
    check_edge_cases(mod); print("  ok  edge cases k=1, k=N")
