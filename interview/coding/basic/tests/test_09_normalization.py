import torch


def check_layernorm_matches_torch(mod):
    torch.manual_seed(0)
    x = torch.randn(4, 7, 32) * 3 + 1.5
    ln = torch.nn.LayerNorm(32)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(32)); ln.bias.copy_(torch.randn(32))
    y = mod.layer_norm(x, 32, ln.weight.detach(), ln.bias.detach(), eps=ln.eps)
    ref = ln(x).detach()
    assert y.shape == ref.shape
    assert torch.allclose(y, ref, atol=1e-5), f"layer_norm(D) mismatch {(y-ref).abs().max():.2e}"
    # no affine, int normalized_shape
    y2 = mod.layer_norm(x, 32)
    ref2 = torch.nn.functional.layer_norm(x, (32,))
    assert torch.allclose(y2, ref2, atol=1e-5), "layer_norm without affine mismatch"
    # multi-dim normalized_shape
    x3 = torch.randn(3, 5, 6, 8)
    ln3 = torch.nn.LayerNorm((6, 8))
    y3 = mod.layer_norm(x3, (6, 8), ln3.weight.detach(), ln3.bias.detach())
    assert torch.allclose(y3, ln3(x3).detach(), atol=1e-5), "layer_norm over (H,W) mismatch"
    # per-sample stats: each row has mean 0, var 1
    m = y2.mean(-1); v = y2.var(-1, unbiased=False)
    assert m.abs().max() < 1e-5 and (v - 1).abs().max() < 1e-3, "normalised rows should be zero-mean unit-var"


def check_batchnorm_train_and_running_stats(mod):
    torch.manual_seed(1)
    C = 5
    ref = torch.nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(C)); ref.bias.copy_(torch.randn(C))
    mine = mod.BatchNorm2d(C, eps=1e-5, momentum=0.1)
    mine.weight = ref.weight.detach().clone(); mine.bias = ref.bias.detach().clone()
    ref.train()
    for step in range(4):
        x = torch.randn(6, C, 9, 11) * (step + 1) + step
        y_ref = ref(x).detach()
        y = mine.forward(x, training=True)
        assert torch.allclose(y, y_ref, atol=1e-5), f"train output mismatch at step {step}"
        assert torch.allclose(mine.running_mean, ref.running_mean, atol=1e-5), f"running_mean mismatch step {step}"
        assert torch.allclose(mine.running_var, ref.running_var, atol=1e-5), f"running_var mismatch step {step} (unbiased var + momentum)"


def check_batchnorm_eval(mod):
    torch.manual_seed(2)
    C = 3
    ref = torch.nn.BatchNorm2d(C, momentum=0.3)
    mine = mod.BatchNorm2d(C, momentum=0.3)
    ref.train()
    for _ in range(3):
        x = torch.randn(4, C, 5, 5) * 2 + 0.7
        ref(x); mine.forward(x, training=True)
    ref.eval()
    x = torch.randn(2, C, 5, 5)
    rm, rv = mine.running_mean.clone(), mine.running_var.clone()
    y = mine.forward(x, training=False)
    assert torch.allclose(y, ref(x).detach(), atol=1e-5), "eval output mismatch"
    assert torch.equal(rm, mine.running_mean) and torch.equal(rv, mine.running_var), "eval must not touch running stats"
    # eval output != train output on the same batch (they use different stats)
    y_tr = mine.forward(x, training=True)
    assert not torch.allclose(y, y_tr, atol=1e-3), "train and eval should differ when running stats != batch stats"


def run(mod):
    check_layernorm_matches_torch(mod);          print("  ok  layer_norm matches nn.LayerNorm")
    check_batchnorm_train_and_running_stats(mod); print("  ok  batchnorm train mode + running stats (momentum, unbiased var)")
    check_batchnorm_eval(mod);                    print("  ok  batchnorm eval mode uses running stats")
