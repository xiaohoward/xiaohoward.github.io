"""
15 — Image ops: bilinear resize, separable Gaussian blur, Gaussian pyramid

Three building blocks every vision / generative-model codebase has (and that multi-resolution diffusion, foveated
rendering and antialiased downsampling all depend on). Write them by hand — no F.interpolate, no torchvision.

Signatures:
    def bilinear_resize(x, out_h, out_w) -> Tensor          # (B,C,H,W) -> (B,C,out_h,out_w), align_corners=False
    def gaussian_kernel1d(sigma, radius=None) -> Tensor     # (K,) normalised, K = 2*radius+1
    def gaussian_blur(x, sigma, radius=None) -> Tensor      # separable, reflect padding, same shape
    def gaussian_pyramid(x, n_levels, sigma=1.0) -> list    # [x, blur+down2(x), ...] n_levels tensors

Constraints: torch (CPU) only. bilinear_resize must NOT call F.interpolate / F.grid_sample; it must match
F.interpolate(mode="bilinear", align_corners=False, antialias=False) to 1e-5 for both up- and down-scaling.
gaussian_blur may use F.conv2d / F.conv1d and F.pad. Anything differentiable is fine; no Python loops over pixels.

Convention reminder (say it out loud): with align_corners=False, pixel i covers [i, i+1) and has its centre at
i + 0.5. Output centre (o + 0.5) maps to input coordinate (o + 0.5) * (H_in / H_out) - 0.5, which is then clamped to
>= 0; the two neighbours are floor(src) and min(floor(src)+1, H_in-1) with weight frac(src) on the upper one.

Interview budget: 25 min

Discussion follow-ups:
  * Why does plain bilinear downsampling by 4x alias, and what does antialias=True (or blurring first) change?
    Relate to the Gaussian pyramid: why blur BEFORE decimating.
  * Separable filtering: cost O(HW·K) vs O(HW·K²). Which other kernels are separable? (box, Gaussian, Sobel; not disk)
  * How would you make resize differentiable w.r.t. the sampling coordinates (grid_sample / spatial transformer)?
  * Laplacian pyramid: how to build it from the Gaussian pyramid and why it is used for multi-scale losses / blending.
"""
import math
import torch
import torch.nn.functional as F

__implement__ = ["bilinear_resize", "gaussian_kernel1d", "gaussian_blur", "gaussian_pyramid"]


def _src_coords(out_size, in_size):
    """Half-pixel-centre source coordinates (align_corners=False), clamped to [0, in_size-1]. -> (i0, i1, frac)."""
    scale = in_size / out_size
    src = (torch.arange(out_size, dtype=torch.float32) + 0.5) * scale - 0.5
    src = src.clamp(min=0.0)
    i0 = src.floor().long().clamp(max=in_size - 1)
    i1 = (i0 + 1).clamp(max=in_size - 1)
    frac = (src - i0.float()).clamp(0.0, 1.0)
    return i0, i1, frac


def bilinear_resize(x, out_h, out_w):
    """
    Bilinear resize with align_corners=False (half-pixel centres) and edge clamping, by hand.

    x            : (B, C, H, W) float tensor.
    out_h, out_w : ints >= 1.

    Returns (B, C, out_h, out_w), same dtype as x. Must equal
    F.interpolate(x, (out_h, out_w), mode="bilinear", align_corners=False, antialias=False) to 1e-5, for any
    scale factor (up or down; non-integer ok). Output size 1 along an axis picks the centre of the input
    (src = 0.5*H - 0.5). Use torch indexing / gather; no F.interpolate, no F.grid_sample, no per-pixel loops.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def gaussian_kernel1d(sigma, radius=None):
    """
    1D Gaussian kernel. sigma > 0 (float). radius defaults to ceil(3*sigma); K = 2*radius + 1 taps centred at 0.
    Returns float32 tensor (K,) with k[i] ∝ exp(-(i-radius)^2 / (2 sigma^2)), normalised so k.sum() == 1.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def gaussian_blur(x, sigma, radius=None):
    """
    Separable Gaussian blur of x : (B, C, H, W) with kernel gaussian_kernel1d(sigma, radius).
    Pad with mode="reflect" by `radius` on each side (so H, W must be > radius), then apply the 1D kernel along W and
    along H as two depthwise convolutions (each channel filtered independently). Returns (B, C, H, W), same dtype.
    Must equal F.conv2d with the outer-product 2D kernel on the same reflect-padded input to 1e-5.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def gaussian_pyramid(x, n_levels, sigma=1.0):
    """
    Gaussian pyramid: level 0 is x itself; level l+1 = bilinear_resize(gaussian_blur(level l, sigma),
    ceil(H_l/2), ceil(W_l/2)) i.e. blur-then-2x-downsample (blur first to avoid aliasing).
    x : (B, C, H, W); n_levels >= 1. Returns a list of n_levels tensors, shapes halving (ceil) each level.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
