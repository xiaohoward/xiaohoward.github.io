"""Two-layer MLP: forward and manual backward

Implement the forward and backward pass of a 2-layer MLP  Linear(D -> H) -> ReLU -> Linear(H -> C)  trained with
mean squared error against targets T (N, C):  L = (1 / (N C)) * ||Y - T||_F^2  (i.e. the mean over all entries,
matching torch.nn.functional.mse_loss). The backward pass must be written by hand (no autograd) and return the
gradient of L w.r.t. every parameter and w.r.t. the input.

Parameters are a dict: {"W1": (D, H), "b1": (H,), "W2": (H, C), "b2": (C,)}; forward uses X @ W1 + b1
(row-vector convention, NOT torch's x @ W.T).

Signatures:
    init_params(d_in, d_hidden, d_out, seed=0) -> dict
    forward(params, X) -> (Y (N, C), cache)
    mse_loss(Y, T) -> float
    backward(params, cache, Y, T) -> dict with keys "W1", "b1", "W2", "b2", "X" (same shapes as the corresponding arrays)

Constraints: numpy only. All arrays float64. cache may hold whatever forward needs to save (X, pre-activation,
hidden activation). Use the ReLU subgradient 1[z > 0].

Interview budget: 20 min

Discussion follow-ups:
  - Shapes bookkeeping: why dW1 = X^T dZ1 and db1 = sum over the batch. Memory: what must be cached for backward
    and how activation checkpointing trades it for recompute.
  - Cost of forward vs backward (roughly 2x forward: two matmuls per layer). FLOPs = 6 * params * tokens.
  - Dead ReLUs, initialization scale (He / Kaiming: var = 2 / fan_in) and why it matters for depth.
  - How this generalizes to a computational graph / autograd (each op stores a local Jacobian-vector product).
"""
import numpy as np

__implement__ = ["init_params", "forward", "mse_loss", "backward"]


def init_params(d_in, d_hidden, d_out, seed=0):
    """He-initialized parameters.

    Uses rng = np.random.default_rng(seed); W1 = rng.normal(size=(d_in, d_hidden)) * sqrt(2 / d_in),
    b1 = zeros(d_hidden), W2 = rng.normal(size=(d_hidden, d_out)) * sqrt(2 / d_hidden), b2 = zeros(d_out)
    (draw W1 before W2). Returns dict {"W1", "b1", "W2", "b2"} of float64 arrays.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(d_in, d_hidden)) * np.sqrt(2.0 / d_in)
    W2 = rng.normal(size=(d_hidden, d_out)) * np.sqrt(2.0 / d_hidden)
    return {"W1": W1, "b1": np.zeros(d_hidden), "W2": W2, "b2": np.zeros(d_out)}


def forward(params, X):
    """Forward pass.

    Args:
        params: dict with W1 (D, H), b1 (H,), W2 (H, C), b2 (C,).
        X: (N, D) float64.
    Returns:
        (Y, cache): Y (N, C) = relu(X @ W1 + b1) @ W2 + b2; cache is any object holding what backward needs
        (this implementation stores (X, Z1, A1) with Z1 the pre-activation and A1 = relu(Z1)).
    """
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(Z1, 0.0)
    Y = A1 @ params["W2"] + params["b2"]
    return Y, (X, Z1, A1)


def mse_loss(Y, T):
    """Mean squared error over all N*C entries: float(mean((Y - T)^2)). Y, T: (N, C)."""
    return float(np.mean((Y - T) ** 2))


def backward(params, cache, Y, T):
    """Manual backprop of mse_loss(forward(params, X)[0], T).

    Args:
        params: as in forward. cache: object returned by forward. Y: (N, C) forward output. T: (N, C) targets.
    Returns:
        dict of gradients with keys "W1" (D, H), "b1" (H,), "W2" (H, C), "b2" (C,), "X" (N, D), each dL/d(.).
        Chain: dY = 2 (Y - T) / (N C); dW2 = A1^T dY; db2 = sum_n dY; dA1 = dY W2^T; dZ1 = dA1 * 1[Z1 > 0];
        dW1 = X^T dZ1; db1 = sum_n dZ1; dX = dZ1 W1^T.
    """
    X, Z1, A1 = cache
    n, c = Y.shape
    dY = 2.0 * (Y - T) / (n * c)
    dW2 = A1.T @ dY
    db2 = dY.sum(axis=0)
    dA1 = dY @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    dX = dZ1 @ params["W1"].T
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "X": dX}
