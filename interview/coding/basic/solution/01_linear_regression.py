"""Linear regression: closed form and mini-batch gradient descent

Implement ordinary least squares with an L2 (ridge) penalty two ways:
  (1) closed form via the normal equations, and
  (2) mini-batch gradient descent on the same objective.
The model is y_hat = X @ w + b. The objective is
    L(w, b) = (1 / N) * sum_i (y_hat_i - y_i)^2 + lam * ||w||^2      (bias is NOT penalized).
Both fits should agree on the same data, and with lam = 0 the closed form must match np.linalg.lstsq.

Signatures:
    fit_closed_form(X, y, lam=0.0) -> (w, b)
    predict(X, w, b) -> y_hat
    mse_loss(X, y, w, b, lam=0.0) -> float
    fit_gd(X, y, lam=0.0, lr=0.1, epochs=200, batch_size=32, seed=0) -> (w, b)

Constraints: numpy only. X is (N, D) float64, y is (N,) float64. Handle D >= 1 and N >= D. Do not form an explicit
inverse; use np.linalg.solve (or lstsq) on the augmented system. Do not penalize the bias.

Interview budget: 20 min

Discussion follow-ups:
  - Complexity of the closed form (O(N D^2 + D^3)) vs GD (O(N D) per epoch). When would you pick each?
  - Why is solve() preferred over inv()? What does ridge do to the condition number of X^T X + lam I?
  - How does feature scaling affect GD convergence? Relate the max stable learning rate to the largest eigenvalue
    of the Hessian (2/N) X^T X + 2 lam I.
  - Why is the bias usually not regularized? What changes if you center X and y first?
"""
import numpy as np

__implement__ = ["fit_closed_form", "predict", "mse_loss", "fit_gd"]


def make_data(n=256, d=5, noise=0.1, seed=0):
    """Given helper: synthetic data y = X @ w_true + b_true + noise. Returns (X, y, w_true, b_true)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    b_true = 0.5
    y = X @ w_true + b_true + noise * rng.normal(size=n)
    return X, y, w_true, b_true


def predict(X, w, b):
    """Linear prediction.

    Args:
        X: (N, D) float64.
        w: (D,) float64 weights.
        b: python float / 0-d array bias.
    Returns:
        (N,) float64 array X @ w + b.
    """
    return X @ w + b


def mse_loss(X, y, w, b, lam=0.0):
    """Ridge-regularized mean squared error.

    L = mean((X @ w + b - y)^2) + lam * sum(w^2). The bias is NOT penalized.

    Args:
        X: (N, D), y: (N,), w: (D,), b: float, lam: float >= 0.
    Returns:
        python float.
    """
    r = predict(X, w, b) - y
    return float(np.mean(r * r) + lam * np.dot(w, w))


def fit_closed_form(X, y, lam=0.0):
    """Solve the ridge normal equations for (w, b).

    Augment X with a column of ones: A = [X, 1] of shape (N, D+1). Minimizing (1/N)||A theta - y||^2 + lam ||w||^2
    gives (A^T A + N * lam * P) theta = A^T y, where P = diag(1, ..., 1, 0) so the bias entry is not penalized.
    Use np.linalg.solve (never an explicit inverse). With lam = 0 this must equal np.linalg.lstsq(A, y).

    Args:
        X: (N, D) float64 with N >= D + 1 (full column rank when lam = 0).
        y: (N,) float64.
        lam: float >= 0.
    Returns:
        (w, b): w is (D,) float64, b is a python float.
    """
    n, d = X.shape
    A = np.concatenate([X, np.ones((n, 1))], axis=1)
    P = np.eye(d + 1)
    P[d, d] = 0.0
    theta = np.linalg.solve(A.T @ A + n * lam * P, A.T @ y)
    return theta[:d], float(theta[d])


def fit_gd(X, y, lam=0.0, lr=0.1, epochs=200, batch_size=32, seed=0):
    """Mini-batch gradient descent on mse_loss.

    Initialize w = zeros(D), b = 0. Each epoch: shuffle indices with np.random.default_rng(seed) (one generator
    for the whole run), iterate over consecutive mini-batches of size batch_size (last batch may be smaller),
    and for each batch take a step with the gradient of the batch loss:
        r = X_b @ w + b - y_b
        dw = (2 / B) * X_b^T r + 2 * lam * w
        db = (2 / B) * sum(r)
    Returns:
        (w, b): w (D,) float64, b python float. Should match fit_closed_form to ~1e-2 on well-conditioned data.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            Xb, yb = X[idx], y[idx]
            B = len(idx)
            r = Xb @ w + b - yb
            dw = (2.0 / B) * (Xb.T @ r) + 2.0 * lam * w
            db = (2.0 / B) * r.sum()
            w = w - lr * dw
            b = b - lr * db
    return w, float(b)
