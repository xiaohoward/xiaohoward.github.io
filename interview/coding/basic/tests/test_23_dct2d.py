import math
import numpy as np
import torch

try:
    from scipy.fft import dctn as _scipy_dctn
except Exception:  # scipy is optional; fall back to the naive reference
    _scipy_dctn = None


def _naive_dct2(img):
    """Naive O(N^4) orthonormal 2-D DCT-II of a (H, W) float64 numpy array."""
    H, W = img.shape
    out = np.zeros((H, W))
    for k in range(H):
        ck = math.sqrt(1 / H) if k == 0 else math.sqrt(2 / H)
        for l in range(W):
            cl = math.sqrt(1 / W) if l == 0 else math.sqrt(2 / W)
            s = 0.0
            for m in range(H):
                for n in range(W):
                    s += img[m, n] * math.cos(math.pi * (2 * m + 1) * k / (2 * H)) * math.cos(math.pi * (2 * n + 1) * l / (2 * W))
            out[k, l] = ck * cl * s
    return out


def _reference_dct2(img):
    if _scipy_dctn is not None:
        return _scipy_dctn(img, type=2, norm="ortho")
    return _naive_dct2(img)


def check_matrix_orthogonal(mod):
    for N in (1, 4, 8, 33, 64):
        D = mod.dct_matrix(N)
        assert D.shape == (N, N) and D.dtype == torch.float32
        assert torch.allclose(D @ D.T, torch.eye(N), atol=1e-6), f"D D^T != I for N={N}"
        assert torch.allclose(D[0], torch.full((N,), math.sqrt(1 / N)), atol=1e-6), "row 0 must be the constant sqrt(1/N)"
        if N > 1:
            assert abs(D[1, 0].item() - math.sqrt(2 / N) * math.cos(math.pi / (2 * N))) < 1e-6, "D[1,0] wrong"
    D64 = mod.dct_matrix(16, dtype=torch.float64)
    assert D64.dtype == torch.float64 and torch.allclose(D64 @ D64.T, torch.eye(16, dtype=torch.float64), atol=1e-12)


def check_roundtrip_and_reference(mod):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 12, 16)
    X = mod.dct2d(x)
    assert X.shape == x.shape
    assert torch.allclose(mod.idct2d(X), x, atol=1e-5), "idct2d(dct2d(x)) must be the identity"
    # DC coefficient = mean * sqrt(H W)
    assert torch.allclose(X[..., 0, 0], x.mean(dim=(-2, -1)) * math.sqrt(12 * 16), atol=1e-4), "DC coefficient wrong"
    # reference (scipy if available, else naive) on an 8x8 and a non-square image
    rng = np.random.default_rng(0)
    for shape in ((8, 8), (6, 10)):
        img = rng.standard_normal(shape)
        if _scipy_dctn is None and shape != (8, 8):
            continue
        ref = _reference_dct2(img)
        mine = mod.dct2d(torch.from_numpy(img).float()[None, None])[0, 0].double().numpy()
        assert np.allclose(mine, ref, atol=1e-4), f"dct2d mismatch vs reference on {shape}: max {np.abs(mine-ref).max():.2e}"
    # always run the naive 8x8 check too (independent of scipy)
    img = rng.standard_normal((8, 8))
    mine = mod.dct2d(torch.from_numpy(img).float()[None, None])[0, 0].double().numpy()
    assert np.allclose(mine, _naive_dct2(img), atol=1e-4), "dct2d mismatch vs naive double-loop reference"


def check_parseval(mod):
    torch.manual_seed(1)
    x = torch.randn(3, 2, 20, 24)
    X = mod.dct2d(x)
    e_x = (x ** 2).flatten(1).sum(1)
    e_X = (X ** 2).flatten(1).sum(1)
    assert torch.allclose(e_x, e_X, rtol=1e-5), "Parseval: DCT must preserve energy (orthonormal)"


def check_lowpass(mod):
    torch.manual_seed(2)
    H, W = 32, 32
    f = mod.radial_freq_grid(H, W)
    assert f.shape == (H, W) and f[0, 0] == 0
    assert abs(f[0, 8].item() - 0.25) < 1e-6 and abs(f[8, 0].item() - 0.25) < 1e-6, "normalised frequency: k/N"
    assert abs(f[3, 4].item() - math.sqrt((3 / H) ** 2 + (4 / W) ** 2)) < 1e-6
    fr = mod.radial_freq_grid(H, W, normalize=False)
    assert fr[3, 4].item() == 5.0, "raw grid must be sqrt(kx^2 + ky^2) in index units"
    # masks
    mb = mod.lowpass_mask(H, W, 0.5, mode="box")
    assert mb.dtype == torch.bool and mb.shape == (H, W)
    assert mb[:17, :17].all() and not mb[17:].any() and not mb[:, 17:].any(), "box mask must keep max(kx,ky) <= cutoff"
    mr = mod.lowpass_mask(H, W, 0.5, mode="radial")
    assert mr[0, 16] and not mr[0, 17] and mr[11, 11] and not mr[12, 12], "radial mask boundary wrong"
    assert mod.lowpass_mask(H, W, 0.0).sum() == 1, "cutoff 0 keeps only DC"
    assert mod.lowpass_mask(H, W, 2.0).all(), "large cutoff keeps everything"
    try:
        mod.lowpass_mask(H, W, 0.5, mode="banana"); raise AssertionError("unknown mode must raise ValueError")
    except ValueError:
        pass
    # smooth image: sum of low-frequency cosines -> low-pass error tiny
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    smooth = (torch.cos(math.pi * (2 * xx + 1) * 2 / (2 * W)) * torch.cos(math.pi * (2 * yy + 1) * 3 / (2 * H))
              + 0.5 * torch.cos(math.pi * (2 * xx + 1) * 5 / (2 * W)))[None, None]
    for mode in ("radial", "box"):
        rec = mod.lowpass_reconstruct(smooth, 0.3, mode=mode)
        assert rec.shape == smooth.shape
        assert (rec - smooth).abs().max() < 1e-4, f"low-pass of a smooth image must be near exact ({mode})"
    # white noise: kept energy fraction == fraction of kept coefficients (in expectation; 32x32x4 samples)
    noise = torch.randn(4, 1, H, W)
    for mode, cutoff in (("radial", 0.5), ("box", 0.5)):
        frac_coef = mod.lowpass_mask(H, W, cutoff, mode).float().mean().item()
        rec = mod.lowpass_reconstruct(noise, cutoff, mode=mode)
        frac_energy = ((rec ** 2).sum() / (noise ** 2).sum()).item()
        assert abs(frac_energy - frac_coef) < 0.05, f"white-noise kept energy {frac_energy:.3f} vs coef fraction {frac_coef:.3f} ({mode})"
    # radial and box differ
    assert not torch.allclose(mod.lowpass_reconstruct(noise, 0.5, "radial"), mod.lowpass_reconstruct(noise, 0.5, "box"))


def check_energy_compaction(mod):
    torch.manual_seed(3)
    N = 64
    rng = np.random.default_rng(3)
    # "cartoon" natural-like image: piecewise-constant rectangles + a ramp (1/f amplitude spectrum), DC removed
    img = np.zeros((N, N))
    for _ in range(12):
        r0, c0 = rng.integers(0, N - 8, size=2)
        h, w = rng.integers(8, 32, size=2)
        img[r0:r0 + h, c0:c0 + w] += rng.uniform(-1, 1)
    img += np.linspace(0, 1, N)[None, :]
    img -= img.mean()
    x = torch.from_numpy(img).float()[None, None]
    ec = mod.energy_compaction(x, N // 2)
    assert ec.shape == (1,), "energy_compaction must return (B,)"
    assert ec.item() > 0.9, f"natural-like image should have > 90% energy in the top-left quarter, got {ec.item():.3f}"
    assert abs(mod.energy_compaction(x, N).item() - 1) < 1e-6, "k = N must give 1"
    # white noise: ~ fraction of coefficients (0.25 for the quarter block)
    noise = torch.randn(4, 3, N, N)
    ecn = mod.energy_compaction(noise, N // 2)
    assert ecn.shape == (4,) and (ecn - 0.25).abs().max() < 0.03, f"white noise compaction should be ~0.25, got {ecn}"
    assert (ec > ecn).all(), "structured image must compact more than noise"


def run(mod):
    check_matrix_orthogonal(mod);        print("  ok  DCT matrix orthogonal, correct normalisation")
    check_roundtrip_and_reference(mod);  print("  ok  round trip exact, matches " + ("scipy.fft.dctn" if _scipy_dctn else "naive reference"))
    check_parseval(mod);                 print("  ok  Parseval")
    check_lowpass(mod);                  print("  ok  radial grid, masks, low-pass (smooth vs white noise)")
    check_energy_compaction(mod);        print("  ok  energy compaction")
