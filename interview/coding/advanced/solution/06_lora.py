"""LoRA: low-rank adapters on nn.Linear, injection by name, merging, trainable-parameter accounting

Implement Low-Rank Adaptation (Hu et al. 2021) as used for fine-tuning DiTs / LLMs:
    y = W x + b + (alpha / r) * B (A x),   A: (r, in) Kaiming-uniform init, B: (out, r) ZERO init,
so the adapted layer equals the frozen base at initialisation. Then:
  - inject_lora: replace every nn.Linear whose qualified name passes a filter with a LoRALinear (in place);
  - merge_weights: fold W <- W + (alpha / r) B A into the base so inference costs nothing extra;
  - count_trainable_params: number of elements with requires_grad=True.
A small "transformer-ish" model with named layers (attn.q, attn.k, attn.v, attn.o, mlp.fc1, mlp.fc2) is given.

Signatures:
    LoRALinear.__init__(base: nn.Linear, r: int, alpha: float, dropout: float = 0.0)
    LoRALinear.forward(x) -> y
    LoRALinear.merge_weights() -> None
    inject_lora(module, r, alpha, dropout=0.0, name_filter=None) -> list[str]   # names of replaced layers
    count_trainable_params(module) -> int

Constraints: torch CPU float32. The base weight/bias must be frozen (requires_grad False) by the wrapper; only
lora_A and lora_B train. Use nn.init.kaiming_uniform_(A, a=sqrt(5)) and zeros for B. After merge_weights, forward
must return base(x) only (no adapter path) and calling merge twice must not double-apply.

Interview budget: 30 min

Discussion follow-ups:
  - Why init B = 0 and not A = 0? What is the effective learning rate of the product B A and why does LoRA+ / rsLoRA
    (scale alpha / sqrt(r)) change the scaling?
  - Which matrices to adapt in a DiT (qkv/proj vs MLP vs adaLN) and how rank interacts with the task; memory saved
    is optimizer state + grads, NOT activations - why?
  - Merging with quantised bases (QLoRA), multi-LoRA serving (batched adapters, unmerged), DoRA.
  - Your own use: LoRA post-training for resolution-progressive / mixed-resolution generation - what changes in
    the model when you only train LoRAs (position handling, distribution shift at boundaries)?
"""
import math

import torch
import torch.nn as nn

__implement__ = ["LoRALinear.__init__", "LoRALinear.forward", "LoRALinear.merge_weights", "inject_lora",
                 "count_trainable_params"]


# ----------------------------------------------------------------------------- given helpers
class TinyAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(x.shape[-1]), -1)
        return self.o(a @ v)


class TinyMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TinyModel(nn.Module):
    """Given helper: layers named 'blocks.{i}.attn.{q,k,v,o}' and 'blocks.{i}.mlp.{fc1,fc2}', plus 'head'."""

    def __init__(self, d=32, n_layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            blk = nn.Module()
            blk.attn, blk.mlp = TinyAttn(d), TinyMLP(d)
            blk.ln1, blk.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.blocks.append(blk)
        self.head = nn.Linear(d, d)

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk.attn(blk.ln1(x))
            x = x + blk.mlp(blk.ln2(x))
        return self.head(x)


# ----------------------------------------------------------------------------- to implement
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0):
        """Wrap a frozen nn.Linear with a rank-r adapter.

        Store: self.base (the given nn.Linear, all its params set requires_grad=False), self.r, self.alpha,
        self.scaling = alpha / r, self.lora_A = nn.Parameter((r, in_features)) with kaiming_uniform_(a=sqrt(5)),
        self.lora_B = nn.Parameter((out_features, r)) zeros, self.dropout = nn.Dropout(dropout) (Identity if 0),
        self.merged = False.

        Args:
            base: nn.Linear. r: int >= 1. alpha: float. dropout: float in [0, 1).
        """
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x):
        """y = base(x) + scaling * (dropout(x) @ A^T) @ B^T; if self.merged, just base(x).

        Args:
            x: (..., in_features).
        Returns:
            (..., out_features).
        """
        y = self.base(x)
        if self.merged:
            return y
        return y + self.scaling * (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T

    @torch.no_grad()
    def merge_weights(self):
        """Fold the adapter into the base: base.weight += scaling * B @ A, set self.merged = True. Idempotent."""
        if self.merged:
            return
        self.base.weight += self.scaling * (self.lora_B @ self.lora_A)
        self.merged = True


def inject_lora(module, r, alpha, dropout=0.0, name_filter=None):
    """Replace, in place, every nn.Linear submodule of `module` whose qualified name (as in named_modules())
    passes `name_filter(name)` (all Linears if name_filter is None) with LoRALinear(linear, r, alpha, dropout).
    Do NOT wrap the root module itself even if it is a Linear, and do not wrap the `.base` Linear inside an
    existing LoRALinear (so calling inject_lora twice is a no-op the second time).

    Args:
        module: nn.Module (modified in place). r, alpha, dropout: LoRA hyper-parameters.
        name_filter: None or callable str -> bool.
    Returns:
        sorted list of qualified names that were replaced.
    """
    targets = []
    for name, m in module.named_modules():
        if not name or not isinstance(m, nn.Linear) or (name_filter is not None and not name_filter(name)):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        if isinstance(parent, LoRALinear):
            continue
        targets.append((name, parent, child))
    for name, parent, child in targets:
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    return sorted(n for n, _, _ in targets)


def count_trainable_params(module):
    """Number of scalar parameters with requires_grad=True."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
