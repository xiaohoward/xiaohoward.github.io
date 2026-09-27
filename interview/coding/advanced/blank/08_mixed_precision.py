"""Mixed-precision training step with dynamic loss scaling (manual AMP)

You are given a tiny 2-layer MLP expressed as a list of raw fp32 tensors (the "master weights") and a loss. Write a
manual fp16 training step in the style of NVIDIA Apex / torch.cuda.amp, without using torch.autocast or GradScaler:
  1. Make an fp16 copy of the master weights (requires_grad=True), run forward + loss in fp16.
  2. Multiply the loss by a dynamic loss scale S before backward so small fp16 gradients do not underflow.
  3. Convert the fp16 grads to fp32 and divide by S ("unscale"). If ANY grad contains inf/nan: skip the step and
     halve S. Otherwise: apply an AdamW update to the fp32 master weights. After `growth_interval` consecutive good
     steps, double S.
Signatures:
    class DynamicLossScaler:  scale_loss(loss) -> scaled loss;  unscale_(grads_fp16) -> (grads_fp32, found_inf);
                              update(found_inf) -> None   (attribute `scale` holds the current float scale)
    adamw_update_(params, grads, state, lr, betas, eps, weight_decay) -> None   (in place, decoupled decay)
    mixed_precision_step(master, batch, scaler, opt_state, lr, ...) -> (loss_value, skipped, grads_fp32)
Constraints: torch CPU only. fp16 = torch.float16 (CPU matmul/ReLU/MSE support fp16 in torch >= 2.1). The loss is
computed in fp32 from the fp16 logits (mean of squares of a large fp16 tensor can overflow). Everything must be
deterministic. Do not use torch.optim, torch.autocast or torch.cuda.amp.

Interview budget: 30 min

Discussion follow-ups:
  - Why is fp16 dynamic range (max 65504, min normal 6e-5) the problem and not precision? Why does bf16 usually not
    need loss scaling, and what does it lose instead (8-bit mantissa)?
  - What happens to Adam's second moment if a skipped step still updated `state["step"]`? Where do grad-clipping and
    the unscale go relative to each other?
  - Master weights cost 4 B/param + 8 B/param Adam state + 2 B fp16 copy + 2 B fp16 grad = 16 B/param. How does
    ZeRO shard this, and what does an fp8 (e4m3/e5m2) recipe add (per-tensor scaling, delayed scaling)?
  - The overflow check is a global all-reduce of a flag in DDP. Why must every rank agree on skipping?
"""
import math

import torch

__implement__ = [
    "DynamicLossScaler.scale_loss",
    "DynamicLossScaler.unscale_",
    "DynamicLossScaler.update",
    "adamw_update_",
    "mixed_precision_step",
]


def make_model(d_in=16, d_hidden=32, d_out=4, seed=0):
    """Given helper: list of fp32 master tensors [W1 (d_in, d_h), b1 (d_h,), W2 (d_h, d_out), b2 (d_out,)]."""
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn(d_in, d_hidden, generator=g) / math.sqrt(d_in)
    b1 = torch.zeros(d_hidden)
    W2 = torch.randn(d_hidden, d_out, generator=g) / math.sqrt(d_hidden)
    b2 = torch.zeros(d_out)
    return [W1, b1, W2, b2]


def make_batch(n=64, d_in=16, d_out=4, seed=1):
    """Given helper: (x, y) regression batch, fp32."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_in, generator=g)
    y = torch.randn(n, d_out, generator=g)
    return x, y


def forward_loss(params, batch):
    """Given helper: MLP forward and MSE loss. Runs in whatever dtype `params` are; the loss itself is fp32.

    logits = relu(x @ W1 + b1) @ W2 + b2 ; loss = mean((logits.float() - y)^2)
    """
    W1, b1, W2, b2 = params
    x, y = batch
    x = x.to(W1.dtype)
    h = torch.relu(x @ W1 + b1)
    logits = h @ W2 + b2
    return ((logits.float() - y.float()) ** 2).mean()


def init_adam_state(params):
    """Given helper: fresh AdamW state {"step": 0, "m": [zeros like p], "v": [zeros like p]}."""
    return {"step": 0, "m": [torch.zeros_like(p) for p in params], "v": [torch.zeros_like(p) for p in params]}


class DynamicLossScaler:
    """Dynamic loss scaler. Attributes: scale (float), growth_factor, backoff_factor, growth_interval, good_steps."""

    def __init__(self, init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=4):
        self.scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.good_steps = 0

    def scale_loss(self, loss):
        """Return loss * self.scale (a 0-d fp32 tensor, still attached to the autograd graph)."""
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError

    def unscale_(self, grads_fp16):
        """Convert fp16 grads to fp32 and divide by the current scale.

        Args:
            grads_fp16: list of fp16 tensors (may contain inf/nan).
        Returns:
            (grads_fp32, found_inf): list of fp32 tensors (same shapes) already divided by the scale, and a python
            bool that is True iff any element of any grad is inf or nan.
        """
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError

    def update(self, found_inf):
        """Update the scale after a step.

        If found_inf: scale *= backoff_factor and good_steps = 0.
        Else: good_steps += 1; when good_steps == growth_interval: scale *= growth_factor and good_steps = 0.
        """
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError


def adamw_update_(params, grads, state, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """One AdamW step in place on fp32 `params` (matches torch.optim.AdamW semantics).

    state["step"] += 1 ; for each p, g:
        p *= (1 - lr * weight_decay)                       (decoupled decay, applied before the Adam step)
        m = b1 m + (1 - b1) g ; v = b2 v + (1 - b2) g^2
        m_hat = m / (1 - b1^step) ; v_hat = v / (1 - b2^step)
        p -= lr * m_hat / (sqrt(v_hat) + eps)
    Args:
        params: list of fp32 tensors (modified in place). grads: list of fp32 tensors, same shapes.
        state: dict from init_adam_state (modified in place).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def mixed_precision_step(master, batch, scaler, opt_state, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                         weight_decay=0.0, grad_hook=None):
    """One fp16 training step with dynamic loss scaling and fp32 master weights.

    Steps:
      1. fp16 copies: p16 = p.detach().to(torch.float16).requires_grad_(True) for each master tensor.
      2. loss = forward_loss(fp16_params, batch)  (fp32 scalar); scaled = scaler.scale_loss(loss); scaled.backward().
      3. grads16 = [p16.grad ...]. If grad_hook is not None: grads16 = grad_hook(grads16)  (tests use this to inject
         an inf).
      4. grads32, found_inf = scaler.unscale_(grads16).
      5. If found_inf: do NOT touch master or opt_state (opt_state["step"] unchanged). Else adamw_update_(master, ...).
      6. scaler.update(found_inf).
    Args:
        master: list of fp32 tensors (updated in place). batch: (x, y). scaler: DynamicLossScaler.
        opt_state: dict from init_adam_state. grad_hook: optional callable list->list applied to the fp16 grads.
    Returns:
        (loss_value: python float of the UNscaled loss, skipped: bool, grads32: list of fp32 unscaled grads).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
