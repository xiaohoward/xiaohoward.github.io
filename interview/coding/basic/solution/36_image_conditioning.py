"""
36 — Image conditioning for diffusion: inpainting input, ControlNet zero-conv, RePaint resampling, classifier guidance

Four ways to condition a pretrained diffusion model on an image or a label without changing its backbone:
  (1) inpainting by input concatenation: the UNet sees [masked image latent | mask | noisy latent] as channels;
  (2) ControlNet: a control branch whose output is added through a ZERO-initialized 1x1 conv, so at init the model
      is exactly the base model and the branch learns from a clean starting point;
  (3) RePaint: no training at all — at every step overwrite the KNOWN region of the latent with the forward-noised
      known latent at the current t and let the model fill in the rest;
  (4) classifier guidance (Dhariwal & Nichol): steer eps with the gradient of log p(y | x_t).

Given: toy_encode (avg-pool "VAE"), TinyBlock, make_alphas_cumprod, q_sample, TinyClassifier.
Signatures:
    def downsample_mask(mask, f) -> Tensor                    (B, 1, H, W) {0,1} -> (B, 1, H/f, W/f) by max-pool
    def build_inpaint_input(image, mask, z_t, encode, f) -> Tensor   (B, 2C+1, h, w)
    def zero_conv(channels) -> nn.Conv2d                      1x1 conv with weight and bias zeroed
    class ControlledBlock.forward(x, control) -> Tensor       base(x) + zero_conv(control_branch(control))
    def repaint_step(z_pred, z0_known, mask_lat, t, noise, alphas_cumprod) -> Tensor
    def classifier_log_prob_grad(classifier, x_t, y, t) -> Tensor        d log p(y | x_t) / d x_t
    def classifier_guided_eps(eps, grad, t, alphas_cumprod, scale) -> Tensor   eps - sqrt(1 - abar_t) * s * grad

Constraints: torch (CPU) only. Mask convention everywhere: 1 = HOLE (region to generate), 0 = KNOWN. Latent
resolution is pixel resolution / f. Channel layout of the inpainting input (documented in the docstring):
[masked-image latent (C) | mask (1) | noisy latent (C)]; note Stable Diffusion inpainting orders them
[noisy latent | mask | masked-image latent] — the order is a convention, the point is to state it.

Interview budget: 25 min

Discussion follow-ups:
  * Why zero-init the last conv of each ControlNet branch rather than a small random init? What happens to the
    gradient of the zero conv's weights at init (non-zero) vs the gradient flowing INTO the control branch (zero)?
  * RePaint's known-region replacement gives a valid marginal per step but the boundary is incoherent; what does its
    resampling (jump back j steps and re-noise) fix, and why does an inpainting-trained model not need it?
  * Classifier guidance scales the gradient by sqrt(1 - abar_t): derive it from eps = -sqrt(1 - abar) grad log p(x_t)
    and the score of the conditional p(x_t | y). Why did classifier-free guidance win in practice?
  * Max-pooling the mask to latent resolution over-covers by up to f-1 pixels. When is that the right call (avoid
    leaking known pixels) and when would you dilate / soft-blend instead?
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__implement__ = ["downsample_mask", "build_inpaint_input", "zero_conv", "ControlledBlock.forward", "repaint_step",
                 "classifier_log_prob_grad", "classifier_guided_eps"]


def toy_encode(image, f):
    """Given: stand-in VAE encoder, average-pools (B, C, H, W) -> (B, C, H/f, W/f). Deterministic and linear."""
    return F.avg_pool2d(image, f)


def downsample_mask(mask, f):
    """
    mask : (B, 1, H, W) float in {0, 1} with 1 = hole (to be inpainted). Returns (B, 1, H//f, W//f) by f x f
    MAX pooling: a latent cell is a hole (1) if ANY pixel it covers is a hole. H, W divisible by f.
    """
    return F.max_pool2d(mask, f)


def build_inpaint_input(image, mask, z_t, encode, f):
    """
    image : (B, C, H, W) clean pixels; mask : (B, 1, H, W) in {0, 1} (1 = hole); z_t : (B, C, H//f, W//f) noisy
    latent at the current step; encode : callable pixels -> latent (e.g. toy_encode(x, f)); f : downsample factor.
    Steps: masked_image = image * (1 - mask)  (holes zeroed BEFORE encoding);  z_masked = encode(masked_image);
    mask_lat = downsample_mask(mask, f). Returns the channel concat
        [z_masked (C) | mask_lat (1) | z_t (C)]   -> (B, 2C + 1, H//f, W//f).
    """
    masked = image * (1.0 - mask)
    z_masked = encode(masked)
    mask_lat = downsample_mask(mask, f)
    return torch.cat([z_masked, mask_lat, z_t], dim=1)


def zero_conv(channels):
    """
    ControlNet "zero convolution": nn.Conv2d(channels, channels, kernel_size=1) whose weight AND bias are
    initialized to exactly zero (they remain trainable Parameters). Returns the module.
    """
    conv = nn.Conv2d(channels, channels, kernel_size=1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class TinyBlock(nn.Module):
    """Given: a tiny UNet-style block, Conv3x3-SiLU-Conv3x3 with a residual connection (C -> C)."""

    def __init__(self, C, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Conv2d(C, C, 3, padding=1)
        self.c2 = nn.Conv2d(C, C, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.c1(x)))


class ControlledBlock(nn.Module):
    """
    ControlNet wrapper (given __init__): base is the frozen pretrained block, control_branch a trainable copy /
    encoder of the control signal, and self.zero = zero_conv(C) the zero-initialized 1x1 conv that adds the
    branch's features to the base's output.
    """

    def __init__(self, base, control_branch, C):
        super().__init__()
        self.base = base
        self.control_branch = control_branch
        self.zero = zero_conv(C)

    def forward(self, x, control):
        """
        x : (B, C, h, w) the block's normal input; control : (B, C, h, w) control features (e.g. an encoded edge
        map). Returns base(x) + zero(control_branch(control)) : (B, C, h, w). At init this equals base(x) exactly,
        and one gradient step on the zero conv makes the output depend on control.
        """
        return self.base(x) + self.zero(self.control_branch(control))


def make_alphas_cumprod(T=1000, beta_start=1e-4, beta_end=0.02):
    """Given: DDPM linear schedule, returns abar (T,) = cumprod(1 - beta_t)."""
    betas = torch.linspace(beta_start, beta_end, T)
    return torch.cumprod(1.0 - betas, dim=0)


def q_sample(x0, t, noise, alphas_cumprod):
    """Given: forward diffusion x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) noise. t : (B,) long, broadcast per sample."""
    ab = alphas_cumprod[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def repaint_step(z_pred, z0_known, mask_lat, t, noise, alphas_cumprod):
    """
    RePaint known-region replacement at step t.
    z_pred   : (B, C, h, w) the sampler's latent for step t (its prediction for the unknown region);
    z0_known : (B, C, h, w) the clean latent of the known image (values under the hole are ignored);
    mask_lat : (B, 1, h, w) in {0, 1}, 1 = hole; t : (B,) long; noise : (B, C, h, w) ~ N(0, I).
    Returns  mask_lat * z_pred + (1 - mask_lat) * q_sample(z0_known, t, noise, alphas_cumprod):
    the hole keeps the sampler's latent untouched, the known region is set to the forward-noised known latent at t.
    """
    z_known_t = q_sample(z0_known, t, noise, alphas_cumprod)
    return mask_lat * z_pred + (1.0 - mask_lat) * z_known_t


class TinyClassifier(nn.Module):
    """Given: noise-aware classifier on x_t: global-average-pool -> concat t/1000 -> MLP -> logits (B, n_classes)."""

    def __init__(self, C, n_classes, hidden=32, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(C, hidden, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, n_classes))

    def forward(self, x_t, t):
        h = F.silu(self.conv(x_t)).mean(dim=(2, 3))
        return self.fc(torch.cat([h, t.float().view(-1, 1) / 1000.0], dim=-1))


def classifier_log_prob_grad(classifier, x_t, y, t):
    """
    classifier(x_t, t) -> logits (B, n_classes). x_t : (B, C, h, w) (any requires_grad state; do not modify it in
    place); y : (B,) long labels; t : (B,) long. Returns grad : (B, C, h, w) = d/dx_t [log softmax(logits)[y]],
    computed with torch.autograd.grad on the SUM over the batch of the per-sample log-probs (each sample's gradient
    only depends on its own x_t). The returned tensor is detached.
    """
    x = x_t.detach().requires_grad_(True)
    logits = classifier(x, t)
    logp = F.log_softmax(logits, dim=-1).gather(1, y.view(-1, 1)).sum()
    (grad,) = torch.autograd.grad(logp, x)
    return grad.detach()


def classifier_guided_eps(eps, grad, t, alphas_cumprod, scale):
    """
    eps : (B, C, h, w) the model's noise prediction at x_t; grad : (B, C, h, w) = d log p(y | x_t) / d x_t;
    t : (B,) long; scale : float guidance scale s. Returns
        eps_hat = eps - sqrt(1 - abar_t) * s * grad     (abar_t broadcast per sample to (B, 1, 1, 1)).
    (Since eps = -sqrt(1 - abar_t) * score, this is the score of p(x_t) p(y | x_t)^s written as an eps.)
    """
    ab = alphas_cumprod[t].view(-1, 1, 1, 1)
    return eps - (1 - ab).sqrt() * scale * grad
