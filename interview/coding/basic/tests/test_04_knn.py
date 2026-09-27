import numpy as np


def _brute_predict(X_tr, y_tr, X_te, k, C):
    out = np.zeros(len(X_te), dtype=np.int64)
    for i, x in enumerate(X_te):
        d = np.sqrt(((X_tr - x) ** 2).sum(1))
        order = np.lexsort((np.arange(len(d)), d))[:k]
        counts = np.bincount(y_tr[order], minlength=C)
        best = counts.max()
        cands = [c for c in range(C) if counts[c] == best]
        if len(cands) > 1:
            mind = [d[order][y_tr[order] == c].min() for c in cands]
            cands = [c for c, md in zip(cands, mind) if md == min(mind)]
        out[i] = min(cands)
    return out


def check_pairwise(mod):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(7, 5)); B = rng.normal(size=(9, 5))
    D = mod.pairwise_dists(A, B)
    assert D.shape == (7, 9)
    ref = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
    assert np.allclose(D, ref, atol=1e-9)
    assert np.all(np.isfinite(mod.pairwise_dists(A, A))), "self-distances must not produce nan (clamp before sqrt)"


def check_indices(mod):
    rng = np.random.default_rng(1)
    X_tr = rng.normal(size=(50, 3)); X_te = rng.normal(size=(8, 3))
    for k in (1, 5, 50):
        idx = mod.knn_indices(X_tr, X_te, k)
        assert idx.shape == (8, k) and idx.dtype.kind == "i"
        D = np.sqrt(((X_te[:, None] - X_tr[None]) ** 2).sum(-1))
        ref = np.argsort(D, axis=1, kind="stable")[:, :k]
        assert np.array_equal(idx, ref), f"k={k}: neighbour indices differ from brute force"


def check_vote_ties(mod):
    # 4 neighbours, classes [0, 1, 1, 0]: 2-2 tie -> class 0 (its nearest is at distance 0.1).
    labels = np.array([[0, 1, 1, 0], [2, 2, 0, 1], [1, 0, 0, 1]])
    dists = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.3, 0.3, 0.9, 1.0]])
    pred = mod.majority_vote(labels, dists, 3)
    assert pred.shape == (3,)
    assert pred[0] == 0, f"tie-break by nearest member failed: {pred[0]}"
    assert pred[1] == 2, "plain majority failed"
    assert pred[2] == 0, f"equal-min-distance tie must go to smaller class index: {pred[2]}"


def check_vs_brute_force(mod):
    X_tr, y_tr, X_te, y_te = mod.make_data(n_train=400, n_test=100, n_classes=4, d=6, spread=1.0, seed=2)
    for k in (1, 3, 8, 15):
        pred = mod.knn_predict(X_tr, y_tr, X_te, k, 4)
        ref = _brute_predict(X_tr, y_tr, X_te, k, 4)
        assert pred.shape == (100,)
        assert np.array_equal(pred, ref), f"k={k}: {np.sum(pred != ref)} predictions differ from brute force"
    acc = np.mean(mod.knn_predict(X_tr, y_tr, X_te, 5, 4) == y_te)
    assert acc > 0.9, f"accuracy {acc:.3f} too low on easy data"
    # k=1 on training points returns their own labels.
    assert np.array_equal(mod.knn_predict(X_tr, y_tr, X_tr[:50], 1, 4), y_tr[:50])


def check_no_loops_scale(mod):
    import time
    X_tr, y_tr, X_te, _ = mod.make_data(n_train=5000, n_test=2000, n_classes=5, d=16, seed=3)
    t = time.perf_counter()
    mod.knn_predict(X_tr, y_tr, X_te, 7, 5)
    dt = time.perf_counter() - t
    assert dt < 5.0, f"too slow ({dt:.2f}s) -- distances must be vectorized"


def run(mod):
    check_pairwise(mod); print("  ok  pairwise distances")
    check_indices(mod); print("  ok  neighbour indices")
    check_vote_ties(mod); print("  ok  majority vote + tie-break")
    check_vs_brute_force(mod); print("  ok  matches brute force")
    check_no_loops_scale(mod); print("  ok  vectorized speed")
