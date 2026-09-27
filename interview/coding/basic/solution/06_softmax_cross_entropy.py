"""Softmax cross-entropy: forward and backward

Implement a numerically stable log-softmax, the mean cross-entropy loss for integer class labels, and its
gradient with respect to the logits. For logits Z (N, C) and labels y (N,) in [0, C):
    log_softmax(Z)_ic = Z_ic - logsumexp_j Z_ij
    L = -(1/N) sum_i log_softmax(Z)_{i, y_i}
    dL/dZ = (softmax(Z) - onehot(y)) / N
Optionally support an ignore_index label (rows with y == ignore_index contribute nothing and the mean is over
the remaining rows), as in torch.nn.functional.cross_entropy.

Signatures:
    log_softmax(Z) -> (N, C)
    cross_entropy(Z, y, ignore_index=-100) -> float
    cross_entropy_grad(Z, y, ignore_index=-100) -> (N, C)

Constraints: numpy only in the solution (tests compare to torch). Must not overflow for logits ~ 1e3 and must
not produce nan for rows with very negative logits. Do not compute log(softmax(Z)).

Interview budget: 15 min

Discussion follow-ups:
  - Why subtract the row max? Why is log-softmax + NLL more stable than softmax then log (underflow -> log 0)?
  - Derive dL/dZ = p - onehot: the Jacobian of softmax is diag(p) - p p^T; contract with -onehot/p.
  - Label smoothing, temperature, and their effect on the gradient. Why do fused CE kernels (e.g. in LLM training)
    avoid materializing the (N, V) logits in fp32?
  - Class imbalance: weighting, focal loss; CE as KL(onehot || p) + const.
"""
import numpy as np

__implement__ = ["log_softmax", "cross_entropy", "cross_entropy_grad"]


def log_softmax(Z):
    """Row-wise log-softmax.

    Args:
        Z: (N, C) float64 logits (any magnitude).
    Returns:
        (N, C) float64 with Z - logsumexp(Z, axis=1, keepdims=True), computed by subtracting the row max first.
        Each row satisfies logsumexp(out) == 0 (exp(out) sums to 1).
    """
    m = Z.max(axis=1, keepdims=True)
    s = Z - m
    return s - np.log(np.sum(np.exp(s), axis=1, keepdims=True))


def cross_entropy(Z, y, ignore_index=-100):
    """Mean cross-entropy loss over valid rows.

    Args:
        Z: (N, C) float64 logits. y: (N,) int labels in [0, C) or == ignore_index.
        ignore_index: rows with y == ignore_index are excluded from both the sum and the count.
    Returns:
        python float: -mean_i log_softmax(Z)[i, y_i] over valid rows. If no rows are valid return 0.0.
    """
    valid = y != ignore_index
    if not np.any(valid):
        return 0.0
    ls = log_softmax(Z[valid])
    return float(-np.mean(ls[np.arange(valid.sum()), y[valid]]))


def cross_entropy_grad(Z, y, ignore_index=-100):
    """Gradient of cross_entropy w.r.t. Z.

    Returns:
        (N, C) float64: (softmax(Z) - onehot(y)) / n_valid for valid rows, 0 for ignored rows.
        If no rows are valid return zeros.
    """
    valid = y != ignore_index
    G = np.zeros_like(Z, dtype=np.float64)
    nv = int(valid.sum())
    if nv == 0:
        return G
    P = np.exp(log_softmax(Z[valid]))
    P[np.arange(nv), y[valid]] -= 1.0
    G[valid] = P / nv
    return G
