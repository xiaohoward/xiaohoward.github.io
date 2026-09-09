"""
09 — LayerNorm and BatchNorm2d forward from scratch

Implement (a) LayerNorm forward over the last `len(normalized_shape)` dims and (b) a BatchNorm2d module with
running statistics that behaves exactly like torch.nn.BatchNorm2d in both train and eval mode.

Signatures:
    def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor
    class BatchNorm2d:
        def __init__(self, num_features, eps=1e-5, momentum=0.1)        # given
        def forward(self, x, training: bool) -> Tensor                  # implement

Constraints: torch (CPU) only; you may NOT call torch.nn.LayerNorm / BatchNorm2d / F.layer_norm / F.batch_norm.
Numerical notes: normalise with the BIASED variance (divide by N), but torch updates running_var with the
UNBIASED variance (divide by N-1). Running stats follow torch's momentum semantics:
    running = (1 - momentum) * running + momentum * batch_stat.
Match torch to 1e-5 in float32.

Interview budget: 20 min

Discussion follow-ups:
  * Why LayerNorm (not BatchNorm) in transformers? (batch-independence, variable sequence length, small
    per-device batches, autoregressive inference with batch size 1.)
  * Pre-LN vs Post-LN, RMSNorm (drop the mean, ~10% cheaper, used in LLaMA); why AdaLN/AdaLN-Zero in DiT.
  * BatchNorm train/eval discrepancy: what goes wrong with tiny batches, with distributed training
    (SyncBN), and when fine-tuning with frozen BN.
  * Numerical stability of var = E[x^2] - E[x]^2 vs the two-pass formula; Welford for streaming stats.
"""
import torch

__implement__ = ["layer_norm", "BatchNorm2d.forward"]


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """
    x                : float tensor of shape (..., *normalized_shape)
    normalized_shape : int or tuple of ints — the trailing dims to normalise over (e.g. (D,) for a
                       transformer hidden state, or (C, H, W)).
    weight, bias     : optional tensors of shape normalized_shape (affine transform, applied after normalising).
    eps              : added to the variance before the sqrt.

    Returns a tensor of the same shape/dtype as x:
        y = (x - mean) / sqrt(var + eps) * weight + bias
    where mean/var (BIASED, i.e. divide by prod(normalized_shape)) are computed per-sample over the trailing
    dims. Edge case: normalized_shape given as an int must be treated as (int,).
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    dims = tuple(range(-len(normalized_shape), 0))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = torch.ones(num_features)
        self.bias = torch.zeros(num_features)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x, training):
        """
        x        : float tensor (N, C, H, W) with C == self.num_features.
        training : True  -> normalise with the batch mean/var computed over dims (N, H, W) per channel
                            (BIASED var), then update self.running_mean / self.running_var IN PLACE with
                            running = (1 - momentum) * running + momentum * stat, where the variance used for the
                            running update is the UNBIASED batch variance (torch semantics).
                   False -> normalise with self.running_mean / self.running_var; running stats untouched.

        Returns y = (x - mean) / sqrt(var + eps) * weight + bias, shape (N, C, H, W), with per-channel
        weight/bias broadcast as (1, C, 1, 1). Edge case: in training mode with N*H*W == 1 the unbiased
        variance is undefined — torch raises; you may do the same.
        """
        dims = (0, 2, 3)
        if training:
            n = x.numel() / x.shape[1]
            if n <= 1:
                raise ValueError("BatchNorm2d needs more than one value per channel in training mode")
            mean = x.mean(dim=dims)
            var_biased = ((x - mean.view(1, -1, 1, 1)) ** 2).mean(dim=dims)
            with torch.no_grad():
                var_unbiased = var_biased * (n / (n - 1))
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var_unbiased, alpha=self.momentum)
            use_mean, use_var = mean, var_biased
        else:
            use_mean, use_var = self.running_mean, self.running_var
        shape = (1, -1, 1, 1)
        y = (x - use_mean.view(shape)) / torch.sqrt(use_var.view(shape) + self.eps)
        return y * self.weight.view(shape) + self.bias.view(shape)
