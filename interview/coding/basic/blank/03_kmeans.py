"""K-means with k-means++ initialization

Implement Lloyd's algorithm for k-means clustering with k-means++ seeding. Given X (N, D) and k, return centroids
(k, D), integer labels (N,), and the inertia (sum of squared distances of each point to its assigned centroid).
Lloyd iterations must be vectorized (no Python loop over points).

Signatures:
    pairwise_sq_dists(X, C) -> (N, k) squared Euclidean distances
    kmeans_pp_init(X, k, rng) -> (k, D) initial centroids
    assign(X, C) -> (N,) labels
    update(X, labels, C) -> (k, D) new centroids (empty clusters keep their previous centroid)
    kmeans(X, k, n_iter=100, tol=1e-6, seed=0) -> (C, labels, inertia)

Constraints: numpy only. Use np.random.default_rng(seed) for all randomness. inertia must be non-increasing across
Lloyd iterations. Stop early when the centroid shift ||C_new - C_old||_F < tol.

Interview budget: 20 min

Discussion follow-ups:
  - Complexity per iteration O(N k D); how do you compute pairwise distances without materializing (N, k, D)?
    Cancellation issues in ||x||^2 + ||c||^2 - 2 x.c and how to clamp.
  - Why does inertia monotonically decrease (assignment step and update step each minimize it)? Does it reach a
    global optimum? What does k-means++ guarantee (O(log k)-competitive in expectation)?
  - How to choose k (elbow, silhouette); k-means as hard EM on an isotropic Gaussian mixture.
  - Mini-batch k-means for large N; use in vector quantization / VQ-VAE codebooks, dead-codebook problem.
"""
import numpy as np

__implement__ = ["pairwise_sq_dists", "kmeans_pp_init", "assign", "update", "kmeans"]


def make_blobs(n_per=100, k=4, d=2, spread=0.3, seed=0):
    """Given helper: k well-separated Gaussian blobs. Returns (X (k*n_per, D), true_labels, true_centers (k, D))."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(k, d))
    X = np.concatenate([c + spread * rng.normal(size=(n_per, d)) for c in centers], axis=0)
    labels = np.repeat(np.arange(k), n_per)
    return X, labels, centers


def pairwise_sq_dists(X, C):
    """Squared Euclidean distances between every row of X and every row of C.

    Args:
        X: (N, D) float64.  C: (k, D) float64.
    Returns:
        (N, k) float64 with d[i, j] = ||X[i] - C[j]||^2, computed via the expansion ||x||^2 + ||c||^2 - 2 x.c
        and clamped at 0 (the expansion can go slightly negative due to rounding). No loop over N or k.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def kmeans_pp_init(X, k, rng):
    """k-means++ seeding.

    Pick the first centroid uniformly at random (rng.integers(N)). For each subsequent centroid, sample a point
    with probability proportional to D(x)^2, where D(x) is the distance from x to its nearest already-chosen
    centroid (rng.choice(N, p=...)).

    Args:
        X: (N, D) float64 with N >= k.  k: int >= 1.  rng: np.random.Generator.
    Returns:
        (k, D) float64 array of chosen data points (copies).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def assign(X, C):
    """Assign every point to its nearest centroid. Returns (N,) int64 labels in [0, k)."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def update(X, labels, C):
    """Recompute centroids as the mean of their assigned points.

    Args:
        X: (N, D), labels: (N,) ints in [0, k), C: (k, D) the previous centroids.
    Returns:
        (k, D) new centroids. A cluster with no assigned points keeps its previous centroid C[j].
        Vectorized: use np.add.at / bincount rather than a Python loop over points (a loop over k is fine).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def kmeans(X, k, n_iter=100, tol=1e-6, seed=0):
    """Lloyd's algorithm with k-means++ init.

    Loop up to n_iter times: labels = assign(X, C); C_new = update(X, labels, C); stop if
    ||C_new - C||_F < tol. After the loop, recompute labels for the final centroids and the inertia.

    Args:
        X: (N, D) float64, k: int in [1, N], n_iter: int, tol: float, seed: int.
    Returns:
        (C, labels, inertia): C (k, D) float64, labels (N,) int64, inertia python float =
        sum_i ||X[i] - C[labels[i]]||^2.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
