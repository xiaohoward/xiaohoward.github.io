import torch


def check_eval_identity(mod):
    torch.manual_seed(0)
    x = torch.randn(5, 7)
    y, _ = mod.dropout(x, 0.5, training=False)
    assert torch.equal(y, x), "eval mode must be the exact identity (no scaling)"
    y0, m0 = mod.dropout(x, 0.0, training=True)
    assert torch.equal(y0, x) and m0.all(), "p=0 must be identity"
    y1, m1 = mod.dropout(x, 1.0, training=True)
    assert (y1 == 0).all() and (~m1).all() and torch.isfinite(y1).all(), "p=1 must give zeros, no inf/nan"


def check_values_and_mask(mod):
    g = torch.Generator().manual_seed(123)
    x = torch.randn(64, 64) + 3.0
    p = 0.3
    y, mask = mod.dropout(x, p, training=True, generator=g)
    assert y.shape == x.shape and mask.shape == x.shape and mask.dtype == torch.bool
    assert (y[~mask] == 0).all(), "dropped elements must be exactly 0"
    assert torch.allclose(y[mask], x[mask] / (1 - p)), "kept elements must be x/(1-p)"
    # reproducibility with the same generator seed
    g2 = torch.Generator().manual_seed(123)
    y2, mask2 = mod.dropout(x, p, training=True, generator=g2)
    assert torch.equal(mask, mask2) and torch.equal(y, y2), "same generator seed must reproduce the mask"


def check_expectation_preserved(mod):
    g = torch.Generator().manual_seed(7)
    x = torch.full((2000, 100), 2.0)
    p = 0.4
    y, mask = mod.dropout(x, p, training=True, generator=g)
    keep_rate = mask.float().mean().item()
    assert abs(keep_rate - (1 - p)) < 0.01, f"keep fraction {keep_rate:.4f} should be ~{1-p}"
    # per-column mean over 2000 samples: std of the mean = 2*sqrt(p/(1-p))/sqrt(2000) ~ 0.037
    col_mean = y.mean(0)
    assert (col_mean - 2.0).abs().max() < 0.2, f"E[y] should equal x; worst column mean {col_mean}"
    assert abs(y.mean().item() - 2.0) < 0.01, f"global mean {y.mean().item():.4f} should be ~2.0"
    # variance matches x^2 * p/(1-p)
    var = y.var().item()
    assert abs(var - 4.0 * p / (1 - p)) < 0.1, f"Var[y] {var:.3f} should be ~{4*p/(1-p):.3f}"


def check_invalid_p(mod):
    x = torch.zeros(3)
    for bad in (-0.1, 1.5):
        try:
            mod.dropout(x, bad); raise AssertionError(f"p={bad} should raise ValueError")
        except ValueError:
            pass


def run(mod):
    check_eval_identity(mod);          print("  ok  eval is identity; p=0 / p=1 edge cases")
    check_values_and_mask(mod);        print("  ok  kept = x/(1-p), dropped = 0, reproducible with generator")
    check_expectation_preserved(mod);  print("  ok  E[y] = x and Var[y] = x^2 p/(1-p) statistically")
    check_invalid_p(mod);              print("  ok  invalid p raises")
