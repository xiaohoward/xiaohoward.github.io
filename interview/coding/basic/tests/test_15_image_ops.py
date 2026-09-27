import torch
import torch.nn.functional as F


def _ref_resize(x, h, w):
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=False)


def check_resize_matches_torch(mod):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 12, 17)
    cases = [(24, 34), (36, 51), (6, 9), (5, 7), (12, 17), (19, 5), (1, 1), (1, 40), (13, 13)]
    for h, w in cases:
        out = mod.bilinear_resize(x, h, w)
        ref = _ref_resize(x, h, w)
        assert out.shape == (2, 3, h, w), f"shape {tuple(out.shape)} != {(2, 3, h, w)}"
        assert torch.allclose(out, ref, atol=1e-5), \
            f"resize to ({h},{w}) mismatch vs F.interpolate, max err {(out - ref).abs().max():.2e}"
    # identity when size unchanged
    assert torch.allclose(mod.bilinear_resize(x, 12, 17), x, atol=1e-6), "same-size resize must be identity"
    # edge clamping: constant image stays constant under any scale
    c = torch.full((1, 1, 5, 5), 3.0)
    assert torch.allclose(mod.bilinear_resize(c, 20, 3), torch.full((1, 1, 20, 3), 3.0), atol=1e-6)


def check_resize_differentiable(mod):
    torch.manual_seed(1)
    x = torch.randn(1, 2, 8, 8, requires_grad=True)
    out = mod.bilinear_resize(x, 11, 5)
    out.sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    _ref_resize(x2, 11, 5).sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, x2.grad, atol=1e-5), "gradient wrt input must match torch"


def check_gaussian_kernel_and_blur(mod):
    for sigma in (0.5, 1.0, 2.3):
        k = mod.gaussian_kernel1d(sigma)
        assert k.ndim == 1 and k.numel() % 2 == 1, "kernel must be 1D with odd length"
        assert abs(k.sum().item() - 1.0) < 1e-6, "kernel must sum to 1"
        assert torch.allclose(k, k.flip(0)), "kernel must be symmetric"
        assert k.argmax().item() == k.numel() // 2, "peak must be at the centre"
    k = mod.gaussian_kernel1d(1.0, radius=2)
    assert k.numel() == 5
    expected = torch.exp(-torch.tensor([4.0, 1.0, 0.0, 1.0, 4.0]) / 2)
    assert torch.allclose(k, expected / expected.sum(), atol=1e-6), "kernel values wrong"

    torch.manual_seed(2)
    x = torch.randn(2, 3, 16, 20)
    for sigma, radius in ((1.0, None), (1.5, 3), (0.8, 2)):
        out = mod.gaussian_blur(x, sigma, radius) if radius is not None else mod.gaussian_blur(x, sigma)
        assert out.shape == x.shape
        k1 = mod.gaussian_kernel1d(sigma, radius) if radius is not None else mod.gaussian_kernel1d(sigma)
        r = (k1.numel() - 1) // 2
        k2 = torch.outer(k1, k1).view(1, 1, k1.numel(), k1.numel()).expand(3, 1, -1, -1)
        ref = F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), k2, groups=3)
        assert torch.allclose(out, ref, atol=1e-5), \
            f"blur sigma={sigma} mismatch vs 2D conv reference, max err {(out - ref).abs().max():.2e}"
    # constant image is unchanged (kernel normalised + reflect padding)
    c = torch.full((1, 1, 9, 9), 2.5)
    assert torch.allclose(mod.gaussian_blur(c, 1.0), c, atol=1e-5), "blur of a constant image must be constant"
    # blur reduces high-frequency energy
    noise = torch.randn(1, 1, 32, 32)
    assert mod.gaussian_blur(noise, 1.0).var() < 0.5 * noise.var(), "blur must reduce variance of white noise"


def check_pyramid(mod):
    torch.manual_seed(3)
    x = torch.randn(2, 3, 33, 48)
    pyr = mod.gaussian_pyramid(x, 4, sigma=1.0)
    assert isinstance(pyr, (list, tuple)) and len(pyr) == 4
    assert torch.equal(pyr[0], x), "level 0 must be the input"
    shapes = [tuple(p.shape) for p in pyr]
    assert shapes == [(2, 3, 33, 48), (2, 3, 17, 24), (2, 3, 9, 12), (2, 3, 5, 6)], f"pyramid shapes {shapes}"
    assert len(mod.gaussian_pyramid(x, 1)) == 1
    # each level is blur-then-downsample of the previous
    ref1 = mod.bilinear_resize(mod.gaussian_blur(x, 1.0), 17, 24)
    assert torch.allclose(pyr[1], ref1, atol=1e-5), "level 1 must be resize(blur(level 0))"
    # means are (approximately) preserved down the pyramid
    for p in pyr[1:]:
        assert abs(p.mean().item() - x.mean().item()) < 0.05


def run(mod):
    check_resize_matches_torch(mod);   print("  ok  bilinear resize matches F.interpolate (up/down/non-integer/size-1)")
    check_resize_differentiable(mod);  print("  ok  resize gradient matches torch")
    check_gaussian_kernel_and_blur(mod); print("  ok  gaussian kernel normalised + separable blur matches conv2d")
    check_pyramid(mod);                print("  ok  gaussian pyramid shapes / composition")
