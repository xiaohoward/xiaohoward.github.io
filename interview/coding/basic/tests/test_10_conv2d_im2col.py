import torch
import torch.nn.functional as F


def check_im2col_matches_unfold(mod):
    torch.manual_seed(0)
    for (N, C, H, W, k, s, p) in [(2, 3, 8, 8, 3, 1, 1), (1, 2, 7, 9, (3, 2), (2, 1), (1, 0)), (3, 1, 5, 5, 5, 1, 0), (2, 4, 10, 6, 2, 2, 0)]:
        x = torch.randn(N, C, H, W)
        cols = mod.im2col(x, k, s, p)
        ref = F.unfold(x, k, dilation=1, padding=p, stride=s)
        assert cols.shape == ref.shape, f"im2col shape {tuple(cols.shape)} != unfold {tuple(ref.shape)} for {(N,C,H,W,k,s,p)}"
        assert torch.allclose(cols, ref, atol=1e-6), f"im2col values/layout mismatch for {(N,C,H,W,k,s,p)}"


def check_conv_matches_torch(mod):
    torch.manual_seed(1)
    cases = [
        dict(N=2, C=3, H=8, W=8, O=4, k=3, s=1, p=1),
        dict(N=1, C=5, H=11, W=7, O=2, k=(3, 2), s=(2, 1), p=(1, 0)),
        dict(N=3, C=2, H=6, W=6, O=6, k=1, s=1, p=0),
        dict(N=2, C=3, H=16, W=16, O=8, k=4, s=4, p=0),   # patchify
        dict(N=1, C=1, H=5, W=5, O=1, k=5, s=1, p=2),
    ]
    for c in cases:
        x = torch.randn(c["N"], c["C"], c["H"], c["W"])
        kh, kw = (c["k"], c["k"]) if isinstance(c["k"], int) else c["k"]
        w = torch.randn(c["O"], c["C"], kh, kw) / (c["C"] * kh * kw) ** 0.5
        b = torch.randn(c["O"])
        y = mod.conv2d_im2col(x, w, b, c["s"], c["p"])
        ref = F.conv2d(x, w, b, stride=c["s"], padding=c["p"])
        assert y.shape == ref.shape, f"conv shape {tuple(y.shape)} != {tuple(ref.shape)} for {c}"
        assert torch.allclose(y, ref, atol=1e-4), f"conv values mismatch {(y-ref).abs().max():.2e} for {c}"
        y0 = mod.conv2d_im2col(x, w, None, c["s"], c["p"])
        assert torch.allclose(y0, F.conv2d(x, w, None, stride=c["s"], padding=c["p"]), atol=1e-4), "no-bias mismatch"


def check_edge_cases(mod):
    x = torch.randn(1, 1, 3, 3)
    try:
        mod.im2col(x, 5, 1, 0)
        raise AssertionError("kernel larger than input without padding should raise ValueError")
    except ValueError:
        pass
    # identity kernel reproduces the input
    w = torch.zeros(1, 1, 3, 3); w[0, 0, 1, 1] = 1.0
    assert torch.allclose(mod.conv2d_im2col(x, w, None, 1, 1), x, atol=1e-6), "centre-tap kernel with pad 1 must be identity"


def run(mod):
    check_im2col_matches_unfold(mod); print("  ok  im2col matches F.unfold (layout, stride, padding)")
    check_conv_matches_torch(mod);    print("  ok  conv2d_im2col matches F.conv2d")
    check_edge_cases(mod);            print("  ok  edge cases (kernel too large, identity kernel)")
