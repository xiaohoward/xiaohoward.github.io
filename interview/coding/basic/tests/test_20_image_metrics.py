import math
import numpy as np
import torch


def _ssim_reference(x, y, data_range=1.0, ws=11, sigma=1.5):
    """Straightforward numpy SSIM: explicit loop over valid windows, per channel. x, y: (C, H, W) float64."""
    c = (ws - 1) / 2
    ax = np.arange(ws) - c
    g1 = np.exp(-ax ** 2 / (2 * sigma ** 2))
    g1 /= g1.sum()
    w = np.outer(g1, g1)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    C, H, W = x.shape
    vals = []
    for ch in range(C):
        for i in range(H - ws + 1):
            for j in range(W - ws + 1):
                px = x[ch, i:i + ws, j:j + ws]
                py = y[ch, i:i + ws, j:j + ws]
                mx, my = (w * px).sum(), (w * py).sum()
                vx = (w * px * px).sum() - mx ** 2
                vy = (w * py * py).sum() - my ** 2
                cxy = (w * px * py).sum() - mx * my
                vals.append(((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2)))
    return float(np.mean(vals))


def check_mse_mae_psnr(mod):
    torch.manual_seed(0)
    B, C, H, W = 3, 3, 16, 20
    x = torch.rand(B, C, H, W)
    delta = torch.tensor([0.05, 0.1, 0.2]).view(B, 1, 1, 1)
    y = x + delta                                  # constant offset -> MSE = delta^2 exactly
    m = mod.mse(x, y)
    assert m.shape == (B,), f"mse must return (B,), got {tuple(m.shape)}"
    assert torch.allclose(m, delta.view(-1) ** 2, atol=1e-6), "mse of constant offset must be delta^2"
    assert torch.allclose(mod.mae(x, y), delta.view(-1), atol=1e-6), "mae of constant offset must be |delta|"
    p = mod.psnr(x, y, data_range=1.0, reduce=False)
    assert p.shape == (B,), f"psnr(reduce=False) must return (B,), got {tuple(p.shape)}"
    expect = torch.tensor([20 * math.log10(1.0 / d) for d in (0.05, 0.1, 0.2)])
    assert torch.allclose(p, expect, atol=1e-4), f"psnr closed form mismatch: {p} vs {expect}"
    p255 = mod.psnr(x * 255, y * 255, data_range=255.0, reduce=False)
    assert torch.allclose(p255, expect, atol=1e-3), "psnr must be invariant to rescaling x, y and data_range together"
    pm = mod.psnr(x, y, reduce=True)
    assert pm.dim() == 0 and abs(pm.item() - expect.mean().item()) < 1e-4, "reduce=True must return the batch mean"
    same = mod.psnr(x, x.clone(), reduce=False)
    assert torch.isinf(same).all() and (same > 0).all(), "identical images must give +inf PSNR"


def check_ssim_identity_and_shapes(mod):
    torch.manual_seed(1)
    x = torch.rand(2, 3, 24, 30)
    s = mod.ssim(x, x.clone(), reduce=False)
    assert s.shape == (2,), f"ssim(reduce=False) must return (B,), got {tuple(s.shape)}"
    assert torch.allclose(s, torch.ones(2), atol=1e-5), f"SSIM(x, x) must be 1, got {s}"
    sr = mod.ssim(x, x.clone())
    assert sr.dim() == 0 and abs(sr.item() - 1) < 1e-5
    w = mod.gaussian_window(11, 1.5)
    assert w.shape == (11, 11) and abs(w.sum().item() - 1) < 1e-6, "window must be 11x11 and sum to 1"
    assert torch.allclose(w, w.T) and w[5, 5] == w.max(), "window must be symmetric and peak at the centre"
    # single-channel and different batch sizes
    assert mod.ssim(torch.rand(4, 1, 11, 11), torch.rand(4, 1, 11, 11), reduce=False).shape == (4,)


def check_ssim_monotone_in_noise(mod):
    torch.manual_seed(2)
    # smooth image: low-frequency cosines (so noise clearly degrades structure)
    H, W = 32, 32
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    base = 0.5 + 0.25 * torch.cos(2 * math.pi * xx / 16) * torch.cos(2 * math.pi * yy / 12)
    x = base.expand(2, 3, H, W).clone()
    vals = []
    noise = torch.randn_like(x)
    for sig in (0.02, 0.1, 0.3):
        vals.append(mod.ssim(x, x + sig * noise).item())
    assert vals[0] > vals[1] > vals[2], f"SSIM must decrease with noise level: {vals}"
    assert vals[0] > 0.9 and vals[2] < 0.5, f"unexpected SSIM range: {vals}"


def check_ssim_matches_reference(mod):
    torch.manual_seed(3)
    x = torch.rand(2, 2, 18, 22)
    y = (x + 0.15 * torch.randn_like(x)).clamp(0, 1)
    mine = mod.ssim(x, y, reduce=False)
    for b in range(2):
        ref = _ssim_reference(x[b].double().numpy(), y[b].double().numpy())
        assert abs(mine[b].item() - ref) < 1e-4, f"SSIM mismatch vs numpy reference (image {b}): {mine[b].item():.6f} vs {ref:.6f}"
    # data_range scaling: metric must be invariant if x, y and data_range are scaled together
    m255 = mod.ssim(x * 255, y * 255, data_range=255.0, reduce=False)
    assert torch.allclose(m255, mine, atol=1e-4), "SSIM must be invariant to joint rescaling with data_range"


def run(mod):
    check_mse_mae_psnr(mod);               print("  ok  mse / mae / psnr closed form, inf for identical")
    check_ssim_identity_and_shapes(mod);   print("  ok  ssim(x,x)=1, window, shapes")
    check_ssim_monotone_in_noise(mod);     print("  ok  ssim decreases with noise")
    check_ssim_matches_reference(mod);     print("  ok  ssim matches numpy reference (1e-4)")
