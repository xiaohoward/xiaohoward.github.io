"""
10 — Conv2d forward via im2col + matmul

Implement a 2-D convolution (really cross-correlation, like torch) the way cuDNN/GEMM-based kernels do it:
first unfold the padded input into a column matrix (im2col), then a single matmul with the flattened weights.
Support batch, multi-channel in/out, integer stride and zero padding. No dilation/groups.

Signatures:
    def im2col(x, kernel_size, stride=1, padding=0) -> Tensor  (N, C*kh*kw, L)     # == F.unfold
    def conv2d_im2col(x, weight, bias=None, stride=1, padding=0) -> Tensor (N, O, Ho, Wo)

Constraints: torch (CPU) only; you may NOT call F.conv2d / F.unfold / nn.Conv2d / Tensor.unfold.
Loops over the kernel offsets (kh*kw iterations) are fine; loops over output pixels are not (too slow).
Ho = (H + 2p - kh) // s + 1. Match F.conv2d to 1e-4 in float32.

Interview budget: 20 min

Discussion follow-ups:
  * FLOPs of the conv: 2 * N * O * Ho * Wo * C * kh * kw. Memory blow-up of im2col: kh*kw x the input — when is
    that acceptable (GEMM efficiency) vs not (large kernels, memory-bound); implicit GEMM / Winograd / FFT convs.
  * How would you get the backward pass? (dW = dY_flat @ cols^T ; dX = col2im(W^T @ dY_flat) — col2im is a
    scatter-ADD, not a scatter.)
  * Patchify in ViT/DiT is exactly a conv with kernel == stride == patch size — what's the im2col matrix then?
  * Channels-last layout, tensor-core tile shapes (why O and C*kh*kw multiples of 8/16 matter).
"""
import torch

__implement__ = ["im2col", "conv2d_im2col"]


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def im2col(x, kernel_size, stride=1, padding=0):
    """
    x           : (N, C, H, W) float tensor.
    kernel_size : int or (kh, kw);  stride: int or (sh, sw);  padding: int or (ph, pw) zero padding on each side.

    Returns cols of shape (N, C*kh*kw, L) with L = Ho*Wo, Ho = (H+2ph-kh)//sh+1, Wo = (W+2pw-kw)//sw+1,
    laid out EXACTLY like torch.nn.functional.unfold: row index = c*kh*kw + i*kw + j (channel-major, then kernel
    row i, then kernel col j), column index = oy*Wo + ox. Edge case: if the kernel does not fit (Ho or Wo < 1)
    raise ValueError.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def conv2d_im2col(x, weight, bias=None, stride=1, padding=0):
    """
    x      : (N, C, H, W);  weight: (O, C, kh, kw);  bias: optional (O,).
    stride, padding : int or pair, same meaning as torch.nn.functional.conv2d.

    Returns y of shape (N, O, Ho, Wo) equal to F.conv2d(x, weight, bias, stride, padding) (cross-correlation,
    no kernel flip), computed as  y_flat = W.reshape(O, -1) @ im2col(x)  (+ bias), then reshaped.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
