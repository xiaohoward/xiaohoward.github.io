"""
20 — Image quality metrics: MSE / MAE, PSNR, SSIM

Implement the three metrics every image/video generation paper reports, for batches of images (B, C, H, W).
PSNR is a closed form of the MSE; SSIM (Wang et al. 2004) compares local means, variances and covariance computed
under an 11x11 Gaussian window (sigma = 1.5), per channel, then averaged.

Signatures:
    def mse(x, y) -> Tensor (B,)
    def mae(x, y) -> Tensor (B,)
    def psnr(x, y, data_range=1.0, reduce=True) -> Tensor        # 10 log10(L^2 / MSE), +inf for identical images
    def gaussian_window(size=11, sigma=1.5) -> Tensor (size, size)   # normalised, sums to 1
    def ssim(x, y, data_range=1.0, window_size=11, sigma=1.5, reduce=True) -> Tensor

Constraints: torch (CPU) only, float32. F.conv2d is allowed (use a depthwise / grouped conv for the window).
Convention (state it out loud): "valid" convolution — no padding, so the SSIM map is (H-10, W-10); the local
variance is the biased estimate E[x^2] - E[x]^2 under the window; C1 = (0.01 L)^2, C2 = (0.03 L)^2 with L the
data range. SSIM of two identical images must be exactly 1 up to float rounding.

Interview budget: 20 min

Discussion follow-ups:
  * Why does PSNR correlate poorly with perceived quality (blur vs noise at equal MSE)? What do LPIPS / FID
    measure instead, and why do generation papers report FID rather than PSNR?
  * Why is the SSIM variance estimate biased (window-weighted) and why does it not matter here? What goes wrong
    at C1 = C2 = 0 on flat regions?
  * MS-SSIM: how would you extend this to multiple scales, and why does it help for high-resolution images?
  * Video: PSNR per frame then mean, vs mean-MSE then PSNR — which is standard, and how do they differ?
"""
import math
import torch
import torch.nn.functional as F

__implement__ = ["mse", "mae", "psnr", "gaussian_window", "ssim"]


def mse(x, y):
    """
    x, y : (B, C, H, W) float tensors, same shape. Returns (B,) mean squared error per image (mean over C, H, W).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def mae(x, y):
    """
    x, y : (B, C, H, W) float tensors, same shape. Returns (B,) mean absolute error per image (mean over C, H, W).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def psnr(x, y, data_range=1.0, reduce=True):
    """
    Peak signal-to-noise ratio in dB: 10 * log10(data_range^2 / MSE) per image.

    x, y       : (B, C, H, W) float tensors.
    data_range : peak-to-peak range of the pixel values (1.0 for [0,1] images, 255 for uint8-style).
    reduce     : if True return the scalar mean over the batch (0-dim tensor); else return the (B,) per-image values.

    Edge case: identical images (MSE == 0) give +inf (no cap), and the batch mean is then +inf too. The test
    checks this convention.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def gaussian_window(size=11, sigma=1.5):
    """
    Returns a (size, size) float32 Gaussian window g[i, j] = exp(-((i-c)^2 + (j-c)^2) / (2 sigma^2)) with
    c = (size - 1) / 2, normalised so that g.sum() == 1. (Separable: build the 1-D window, take the outer product.)
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def ssim(x, y, data_range=1.0, window_size=11, sigma=1.5, reduce=True):
    """
    Structural similarity (Wang et al. 2004).

    x, y : (B, C, H, W) float tensors, H, W >= window_size.
    Per channel, under the Gaussian window w (valid convolution, output (H-ws+1, W-ws+1)):
        mu_x = w * x,  mu_y = w * y,
        var_x = w * x^2 - mu_x^2,  var_y = w * y^2 - mu_y^2,  cov = w * (x y) - mu_x mu_y,
        map = ((2 mu_x mu_y + C1) (2 cov + C2)) / ((mu_x^2 + mu_y^2 + C1) (var_x + var_y + C2)),
    with C1 = (0.01 data_range)^2, C2 = (0.03 data_range)^2.
    Per-image SSIM = mean of the map over channels and valid positions.

    Returns the scalar batch mean (0-dim) if reduce else (B,) per-image values. ssim(x, x) == 1 (up to fp32
    rounding). Use F.conv2d with groups=C so channels are filtered independently.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
