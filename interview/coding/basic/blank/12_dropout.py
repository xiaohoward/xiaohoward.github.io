"""
12 — Inverted dropout

Implement inverted dropout: in training, zero each element independently with probability p and scale the
survivors by 1/(1-p) so that E[y] = x; in eval mode it is the identity (no rescaling at inference — that is
the whole point of the "inverted" variant).

Signature:
    def dropout(x, p, training=True, generator=None) -> (y, mask)

Constraints: torch (CPU) only; no F.dropout / nn.Dropout. Use torch.rand(..., generator=generator) so the test
can seed you. Handle p == 0 (identity) and p == 1 (all zeros, no division by zero).

Interview budget: 10 min

Discussion follow-ups:
  * Why scale at train time instead of test time (original paper scaled weights by (1-p) at test time)?
  * Variance of y vs x: Var[y] = x^2 * p/(1-p) added per element — why does that interact badly with BatchNorm
    ("variance shift")? Why do modern transformers/DiTs use little or no dropout?
  * DropPath / stochastic depth, token dropout in MAE, and classifier-free-guidance conditioning dropout in
    diffusion models: same mechanism, different granularity.
  * How do you make dropout reproducible across data-parallel ranks and with activation checkpointing (RNG
    state must be saved & restored on recompute)?
"""
import torch

__implement__ = ["dropout"]


def dropout(x, p, training=True, generator=None):
    """
    x         : float tensor, any shape.
    p         : drop probability in [0, 1].
    training  : if False, return (x, None) — exact identity, no scaling.
    generator : optional torch.Generator passed to torch.rand for reproducibility.

    Returns (y, mask):
        mask : bool tensor of x.shape, True where the element is KEPT (drawn as torch.rand(x.shape) >= p)
        y    : x * mask / (1 - p)   — same shape/dtype as x, so every kept element equals x/(1-p) exactly and
               every dropped element is 0. E[y] = x.
    Edge cases: p == 0 -> mask all True, y == x;  p == 1 -> y all zeros, mask all False (do not divide by 0).
    Raise ValueError for p outside [0, 1].
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
