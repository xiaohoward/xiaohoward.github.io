"""Logistic regression from scratch

Implement binary logistic regression with a numerically stable sigmoid, the mean binary cross-entropy loss with
optional L2 penalty, its analytic gradient, and full-batch gradient descent. Model: p = sigmoid(X @ w + b),
    L(w, b) = -(1/N) sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ] + (lam / 2) ||w||^2.

Signatures:
    sigmoid(z) -> array, same shape as z, no overflow warnings for |z| up to 1e4
    loss(X, y, w, b, lam=0.0) -> float
    grad(X, y, w, b, lam=0.0) -> (dw, db)
    fit(X, y, lam=0.0, lr=0.1, steps=500) -> (w, b)
    predict_proba(X, w, b) -> (N,) probabilities

Constraints: numpy only. X (N, D) float64, y (N,) in {0, 1}. The loss must be computed in a way that does not
produce inf/nan for large |logits| (use log-sum-exp / softplus, not log(sigmoid)).

Interview budget: 20 min

Discussion follow-ups:
  - Why is the gradient simply X^T (p - y) / N? Derive it. What is the Hessian, and is the problem convex?
  - What happens to w on linearly separable data without regularization? Why does lam fix that?
  - Newton / IRLS vs GD: cost per step and iteration count. When is IRLS impractical?
  - Relationship to a 1-hidden-unit network, and to softmax regression with 2 classes.
"""
import numpy as np

__implement__ = ["sigmoid", "loss", "grad", "fit", "predict_proba"]


def make_data(n=200, d=2, seed=0, margin=1.0):
    """Given helper: two Gaussian blobs at +-margin along a random direction. Returns (X, y) with y in {0,1}."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    y = (rng.random(n) < 0.5).astype(np.float64)
    X = rng.normal(size=(n, d)) + np.outer(2 * y - 1, margin * u)
    return X, y


def sigmoid(z):
    """Numerically stable logistic function.

    Args:
        z: float64 array of any shape (may contain values like +-1e4).
    Returns:
        Array of the same shape with values in (0, 1), computed without overflow: for z >= 0 use 1/(1+exp(-z)),
        for z < 0 use exp(z)/(1+exp(z)).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def predict_proba(X, w, b):
    """P(y = 1 | x) = sigmoid(X @ w + b). X (N, D), w (D,), b float -> (N,) float64."""
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def loss(X, y, w, b, lam=0.0):
    """Mean binary cross-entropy + (lam / 2) ||w||^2, computed stably from logits.

    Use BCE(z, y) = softplus(z) - y * z with softplus(z) = max(z, 0) + log1p(exp(-|z|)).

    Args:
        X: (N, D) float64, y: (N,) in {0, 1}, w: (D,), b: float, lam: float >= 0.
    Returns:
        python float, finite even when |logits| ~ 1e3.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def grad(X, y, w, b, lam=0.0):
    """Analytic gradient of loss().

    dw = X^T (p - y) / N + lam * w,   db = mean(p - y),   with p = sigmoid(X @ w + b).

    Returns:
        (dw, db): dw (D,) float64, db python float.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def fit(X, y, lam=0.0, lr=0.1, steps=500):
    """Full-batch gradient descent from w = 0, b = 0 for `steps` iterations with constant learning rate.

    Returns:
        (w, b): w (D,) float64, b python float.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
