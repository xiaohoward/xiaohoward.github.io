"""
23 — Orthonormal 2-D DCT-II, radial frequency grid, low-pass reconstruction, energy compaction

Implement the type-II discrete cosine transform as a matrix product (the way JPEG and spectral diffusion methods
use it), its inverse, and the two spectral utilities every "frequency-aware" generative-model paper needs: a
radial frequency grid and a low-pass reconstruction that keeps only coefficients below a cutoff.

Signatures:
    def dct_matrix(N, dtype=torch.float32) -> Tensor (N, N)      # D[k, n] = c_k cos(pi (2n+1) k / (2N))
    def dct2d(x) -> Tensor  (B, C, H, W)                          # X = D_H x D_W^T
    def idct2d(X) -> Tensor (B, C, H, W)                          # x = D_H^T X D_W
    def radial_freq_grid(H, W, normalize=True) -> Tensor (H, W)   # sqrt(kx^2 + ky^2)
    def lowpass_mask(H, W, cutoff, mode="radial") -> BoolTensor (H, W)
    def lowpass_reconstruct(x, cutoff, mode="radial") -> Tensor
    def energy_compaction(x, k) -> Tensor (B,)                   # fraction of energy in the top-left k x k block

Constraints: torch (CPU) only; no torch.fft / scipy in the solution. c_0 = sqrt(1/N), c_k = sqrt(2/N) otherwise,
so D is orthogonal (D D^T = I) and the inverse is the transpose. Build D in float64 then cast so it is orthogonal
to 1e-6 in float32. Must match scipy.fft.dctn(x, type=2, norm="ortho") on the last two axes.

Interview budget: 20 min

Discussion follow-ups:
  * Cost: O(N^2) per row via matmul vs O(N log N) with an FFT-based DCT. For latents of size 128x128 which is
    faster on a GPU and why (matmul throughput vs memory-bound FFT)?
  * DCT vs DFT for images: real output, implicit even-symmetric extension (no boundary discontinuity), better
    energy compaction — why that matters for JPEG and for progressive-resolution diffusion.
  * Downsampling by DCT truncation: what is the relationship between keeping the top-left (N/2 x N/2) block and
    an anti-aliased 2x downsample? What scale factor is needed to preserve the mean (the c_0 change)?
  * Radial vs box cutoff: which one corresponds to "resolution", which one to isotropic blur?
"""
import math
import torch

__implement__ = ["dct_matrix", "dct2d", "idct2d", "radial_freq_grid", "lowpass_mask", "lowpass_reconstruct",
                 "energy_compaction"]


def dct_matrix(N, dtype=torch.float32):
    """
    Orthonormal DCT-II matrix D of shape (N, N):
        D[k, n] = c_k * cos(pi * (2n + 1) * k / (2N)),   c_0 = sqrt(1/N),  c_k = sqrt(2/N) for k >= 1,
    so that X = D @ x is the DCT of a length-N signal and D @ D.T == I. Compute in float64, return in `dtype`.
    """
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64)
    D = torch.cos(math.pi * (2 * n[None, :] + 1) * k[:, None] / (2 * N))
    c = torch.full((N, 1), math.sqrt(2.0 / N), dtype=torch.float64)
    c[0, 0] = math.sqrt(1.0 / N)
    return (c * D).to(dtype)


def dct2d(x):
    """
    x : (B, C, H, W) float tensor. Returns the orthonormal 2-D DCT-II applied along H and W:
        X[b, c] = D_H @ x[b, c] @ D_W^T,  D_H = dct_matrix(H), D_W = dct_matrix(W)   (same dtype as x).
    X[..., 0, 0] is the DC coefficient (= mean * sqrt(H W)); larger indices are higher frequencies.
    """
    D_H = dct_matrix(x.shape[-2], x.dtype)
    D_W = dct_matrix(x.shape[-1], x.dtype)
    return D_H @ x @ D_W.T


def idct2d(X):
    """
    X : (B, C, H, W) DCT coefficients. Returns x = D_H^T @ X @ D_W (the exact inverse of dct2d, since D is
    orthogonal). Same dtype as X.
    """
    D_H = dct_matrix(X.shape[-2], X.dtype)
    D_W = dct_matrix(X.shape[-1], X.dtype)
    return D_H.T @ X @ D_W


def radial_freq_grid(H, W, normalize=True):
    """
    Returns an (H, W) float32 grid f[ky, kx] = sqrt(kx^2 + ky^2) of the radial frequency of each DCT coefficient.
    If normalize: kx = arange(W) / W and ky = arange(H) / H, so frequency 1.0 along an axis is that axis' Nyquist
    (DCT index k corresponds to k / (2N) cycles per pixel) and f ranges over [0, sqrt(2)).
    If not normalize: raw integer indices kx in [0, W), ky in [0, H).
    """
    ky = torch.arange(H, dtype=torch.float32)
    kx = torch.arange(W, dtype=torch.float32)
    if normalize:
        ky = ky / H
        kx = kx / W
    return torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def lowpass_mask(H, W, cutoff, mode="radial"):
    """
    Bool (H, W) mask of DCT coefficients to KEEP, in normalised frequency units (Nyquist = 1 per axis):
        mode="radial": sqrt(kx^2 + ky^2) <= cutoff   (isotropic disc)
        mode="box"   : max(kx, ky)       <= cutoff   (square block, i.e. "resolution reduction")
    with kx = arange(W)/W, ky = arange(H)/H. Raise ValueError for any other mode. cutoff >= sqrt(2) (radial) or
    >= 1 (box) keeps everything; cutoff = 0 keeps only DC.
    """
    ky = torch.arange(H, dtype=torch.float32)[:, None] / H
    kx = torch.arange(W, dtype=torch.float32)[None, :] / W
    if mode == "radial":
        return torch.sqrt(kx ** 2 + ky ** 2) <= cutoff
    if mode == "box":
        return torch.maximum(kx.expand(H, W), ky.expand(H, W)) <= cutoff
    raise ValueError(f"unknown mode {mode!r}")


def lowpass_reconstruct(x, cutoff, mode="radial"):
    """
    x : (B, C, H, W). Returns idct2d(dct2d(x) * lowpass_mask(H, W, cutoff, mode)) — the image with all DCT
    coefficients above the cutoff zeroed. Same shape and dtype as x.
    """
    m = lowpass_mask(x.shape[-2], x.shape[-1], cutoff, mode).to(x.dtype)
    return idct2d(dct2d(x) * m)


def energy_compaction(x, k):
    """
    x : (B, C, H, W); k : int block size. Returns (B,) float: the fraction of the total DCT energy (sum of squared
    coefficients over C, H, W) that lies in the lowest-frequency block X[..., :k, :k]. Equals 1 when k >= max(H, W).
    """
    X = dct2d(x)
    total = (X ** 2).flatten(1).sum(dim=1)
    block = (X[..., :k, :k] ** 2).flatten(1).sum(dim=1)
    return block / total
