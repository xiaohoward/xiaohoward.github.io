"""k-nearest-neighbours classifier (vectorized)

Implement a kNN classifier. Given training data X_train (N, D) with integer labels y_train (N,) in [0, C), and
queries X_test (M, D), predict the label of each query by majority vote among its k nearest training points
(Euclidean distance). Pairwise distances must be computed with matrix operations (no Python loop over pairs of
points, no loop over queries). Ties in the vote are broken by the class whose nearest member is closest to the
query (i.e. among tied classes, the one with the smaller minimum distance wins).

Signatures:
    pairwise_dists(A, B) -> (M, N) Euclidean distances
    knn_indices(X_train, X_test, k) -> (M, k) indices of nearest neighbours, sorted by increasing distance
    majority_vote(neigh_labels, neigh_dists, n_classes) -> (M,) predicted labels
    knn_predict(X_train, y_train, X_test, k, n_classes) -> (M,) predicted labels

Constraints: numpy only. Use np.argpartition for the top-k selection (O(N) per query instead of O(N log N)),
then sort just the k selected. 1 <= k <= N.

Interview budget: 15 min

Discussion follow-ups:
  - Memory of the (M, N) distance matrix; how to chunk queries. Complexity O(M N D) vs kd-tree / ball tree
    (why they fail in high D) vs approximate NN (LSH, HNSW, IVF-PQ) used in retrieval / RAG.
  - Effect of k (bias / variance), of feature scaling, and of distance choice (cosine vs L2).
  - Why the expansion ||a||^2 + ||b||^2 - 2ab can be numerically bad, and when to use cdist directly.
  - Distance-weighted voting; kNN regression; relation to kernel density / Nadaraya-Watson.
"""
import numpy as np

__implement__ = ["pairwise_dists", "knn_indices", "majority_vote", "knn_predict"]


def make_data(n_train=300, n_test=60, n_classes=3, d=4, spread=1.0, seed=0):
    """Given helper: Gaussian class clusters. Returns (X_train, y_train, X_test, y_test)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, d)) * 3
    y_tr = rng.integers(n_classes, size=n_train)
    y_te = rng.integers(n_classes, size=n_test)
    X_tr = centers[y_tr] + spread * rng.normal(size=(n_train, d))
    X_te = centers[y_te] + spread * rng.normal(size=(n_test, d))
    return X_tr, y_tr, X_te, y_te


def pairwise_dists(A, B):
    """Euclidean distances between every row of A and every row of B.

    Args:
        A: (M, D) float64.  B: (N, D) float64.
    Returns:
        (M, N) float64, out[i, j] = ||A[i] - B[j]||_2. Use the expansion ||a||^2 + ||b||^2 - 2 a.b, clamp
        negatives to 0 before the sqrt. No Python loops.
    """
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * A @ B.T, 0.0)
    return np.sqrt(d2)


def knn_indices(X_train, X_test, k):
    """Indices of the k nearest training points for every query.

    Args:
        X_train: (N, D), X_test: (M, D), k: int with 1 <= k <= N.
    Returns:
        (M, k) int64 array; row i lists the training indices sorted by increasing distance to X_test[i]
        (ties in distance broken by lower index). Use argpartition then sort the k selected.
    """
    D = pairwise_dists(X_test, X_train)
    n = X_train.shape[0]
    if k >= n:
        part = np.tile(np.arange(n), (X_test.shape[0], 1))
    else:
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
    dsel = np.take_along_axis(D, part, axis=1)
    # Stable lexsort: primary key distance, secondary key index (for deterministic tie-breaking).
    order = np.lexsort((part, dsel), axis=1)
    return np.take_along_axis(part, order, axis=1)


def majority_vote(neigh_labels, neigh_dists, n_classes):
    """Majority vote with distance tie-break.

    Args:
        neigh_labels: (M, k) int labels of the k neighbours of each query, sorted by increasing distance.
        neigh_dists:  (M, k) float distances (same ordering).
        n_classes: int C.
    Returns:
        (M,) int64 predicted class. Winner = class with most votes; among tied classes, the one whose nearest
        neighbour (minimum distance) is smallest; remaining ties -> smallest class index. Vectorized over M.
    """
    m, k = neigh_labels.shape
    counts = np.zeros((m, n_classes), dtype=np.int64)
    np.add.at(counts, (np.arange(m)[:, None], neigh_labels), 1)
    mind = np.full((m, n_classes), np.inf)
    np.minimum.at(mind, (np.arange(m)[:, None], neigh_labels), neigh_dists)
    # Lexicographic argmax: primary counts (desc), secondary min-distance (asc), tertiary class index (asc).
    classes = np.broadcast_to(np.arange(n_classes), (m, n_classes))
    order = np.lexsort((classes, mind, -counts), axis=1)
    return order[:, 0].astype(np.int64)


def knn_predict(X_train, y_train, X_test, k, n_classes):
    """Full kNN classifier: knn_indices -> gather labels & distances -> majority_vote. Returns (M,) int64."""
    idx = knn_indices(X_train, X_test, k)
    D = pairwise_dists(X_test, X_train)
    return majority_vote(y_train[idx], np.take_along_axis(D, idx, axis=1), n_classes)
