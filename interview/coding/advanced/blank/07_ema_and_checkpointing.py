"""EMA of weights with warmup, and activation checkpointing as a custom autograd.Function

Two utilities every diffusion training loop needs.
(a) EMA. Maintain shadow parameters s <- d s + (1 - d) p after every optimizer step, with the warmup schedule from
    the diffusion / TF codebases:  d_t = min(decay, (1 + t) / (10 + t))  for update number t = 0, 1, 2, ... (so the
    shadow tracks the model closely early on: d_0 = 0.1, d_1 = 2/11, ... -> decay), or the constant `decay` when
    warmup is off. Provide copy_to (swap EMA weights into the model, keeping a backup) and restore.
(b) Activation checkpointing WITHOUT torch.utils.checkpoint: a torch.autograd.Function whose forward runs the block
    under no_grad (so the block's intermediate activations are never stored) and whose backward re-runs the block
    with grad enabled, then back-propagates the incoming gradient through it. RNG state must be saved and restored
    so dropout inside the block gives identical results in the recompute.

Signatures:
    EMA.__init__(model, decay=0.999, warmup=True)          # given
    EMA.get_decay() -> float
    EMA.update(model) -> None
    EMA.copy_to(model) -> None
    EMA.restore(model) -> None
    CheckpointFunction.forward(ctx, run_function, n_args, *tensors) -> Tensor or tuple of Tensors
    CheckpointFunction.backward(ctx, *grad_outputs) -> (None, None, *grads for tensors)
    checkpoint(run_function, *args)      # given: passes the block's parameters as extra tensors (see below)

Constraints: torch CPU float32, no torch.utils.checkpoint. `tensors` = the n_args real inputs followed by the
block's parameters. The parameters are passed ONLY so that autograd sees them as inputs (otherwise the output
would not require grad when no input does, and the legacy torch.utils.checkpoint has exactly this quirk); the
recompute must call run_function on the n_args inputs only, and the parameters receive their .grad through
torch.autograd.backward inside backward (return None in their slots).

Interview budget: 35 min

Discussion follow-ups:
  - EMA of a 10B model on 512 GPUs: where does the shadow live (FSDP sharded? CPU?), how often to update, and why
    EMA acts like a learning-rate/averaging trade-off (Karras et al. 2024 "post-hoc EMA").
  - Checkpointing trades ~33% extra compute for O(sqrt(L)) or O(1) activation memory; where do you put the
    boundaries in a video DiT and how does it interact with FSDP / sequence parallelism / flash-attn recompute?
  - Why must RNG be restored? What else is stateful in a block (BatchNorm running stats, in-place ops)?
  - Selective checkpointing: recompute cheap ops (LayerNorm, GELU) but save matmul outputs.
"""
import torch
import torch.nn as nn

__implement__ = ["EMA.get_decay", "EMA.update", "EMA.copy_to", "EMA.restore", "CheckpointFunction.forward",
                 "CheckpointFunction.backward"]


# ----------------------------------------------------------------------------- given helpers
class DropBlock(nn.Module):
    """Given helper: MLP block with dropout and a forward-call counter (for tests)."""

    def __init__(self, d=32, p=0.1):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = nn.Dropout(p)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.fc2(self.drop(torch.nn.functional.gelu(self.fc1(x))))


def checkpoint(run_function, *args):
    """Given helper: apply the custom function, appending the block's parameters as extra tensor inputs."""
    params = tuple(run_function.parameters()) if isinstance(run_function, nn.Module) else ()
    return CheckpointFunction.apply(run_function, len(args), *args, *params)


class EMA:
    def __init__(self, model, decay=0.999, warmup=True):
        """Given: shadow = detached clones of model.parameters(); num_updates = 0; backup = None."""
        self.decay, self.warmup = decay, warmup
        self.shadow = [p.detach().clone() for p in model.parameters()]
        self.num_updates = 0
        self.backup = None

    # ------------------------------------------------------------------------- to implement
    def get_decay(self):
        """Decay to use for the NEXT update: min(decay, (1 + n) / (10 + n)) with n = self.num_updates if warmup,
        else self.decay. Returns a python float."""
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError

    @torch.no_grad()
    def update(self, model):
        """s <- d s + (1 - d) p for every (shadow, model parameter) pair, with d = self.get_decay(); then increment
        num_updates. Must not build autograd graph and must not modify the model."""
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError

    @torch.no_grad()
    def copy_to(self, model):
        """Save the model's current parameters into self.backup (clones) and overwrite the model's parameters
        in place with the shadow values (for evaluation / sampling)."""
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError

    @torch.no_grad()
    def restore(self, model):
        """Copy self.backup back into the model's parameters in place and clear the backup. Raise a RuntimeError
        if copy_to was not called first."""
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, n_args, *tensors):
        """Run run_function(*tensors[:n_args]) under torch.no_grad(); stash run_function, n_args, the CPU RNG
        state (torch.get_rng_state()) and the n_args input tensors (ctx.save_for_backward) for the recompute.
        Return the output(s) (a Tensor or a tuple of Tensors) - no graph is built inside.

        Args:
            run_function: callable. n_args: int. tensors: the n_args inputs followed by any number of parameter
            tensors (ignored here except that autograd tracks them).
        """
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Recompute: restore the RNG state (and put it back afterwards), detach the saved inputs (keeping their
        requires_grad flags), re-run run_function under torch.enable_grad(), then
        torch.autograd.backward(outputs, grad_outputs) so that gradients flow into the block's parameters
        (accumulated in .grad) AND into the detached inputs. Return (None, None) + input gradients (None for
        inputs that do not require grad) + (None,) * n_params.
        """
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError
