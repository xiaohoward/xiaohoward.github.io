import torch


def _randomize(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def check_shapes(mod):
    torch.manual_seed(0)
    t_dim = 16
    for (H, W) in ((8, 8), (7, 9), (5, 6)):
        x = torch.randn(2, 8, H, W)
        t = torch.randn(2, t_dim)
        rb = mod.ResBlock(8, 8, t_dim)
        assert rb(x, t).shape == (2, 8, H, W), f"ResBlock same-ch shape at {(H, W)}"
        rb2 = mod.ResBlock(8, 12, t_dim)
        assert rb2(x, t).shape == (2, 12, H, W), f"ResBlock ch-change shape at {(H, W)}"
        d = mod.Downsample(8)(x)
        assert d.shape == (2, 8, (H + 1) // 2, (W + 1) // 2), f"Downsample must give ceil(H/2): got {tuple(d.shape)} for {(H, W)}"
        u = mod.Upsample(8)(d)
        assert u.shape == (2, 8, 2 * ((H + 1) // 2), 2 * ((W + 1) // 2)), f"Upsample default must be 2x: {tuple(u.shape)}"
        u2 = mod.Upsample(8)(d, out_hw=(H, W))
        assert u2.shape == (2, 8, H, W), f"Upsample(out_hw) must hit the requested size: {tuple(u2.shape)}"


def check_init_is_skip(mod):
    torch.manual_seed(1)
    t_dim = 12
    x = torch.randn(2, 8, 6, 6)
    t1, t2 = torch.randn(2, t_dim), torch.randn(2, t_dim)
    rb = mod.ResBlock(8, 8, t_dim)
    out = rb(x, t1)
    assert torch.allclose(out, x, atol=1e-6), "at init (zero-init last conv) the same-channel ResBlock must be the identity"
    assert torch.allclose(rb(x, t2), out, atol=1e-6), "at init the ResBlock must not depend on t"
    # in_ch != out_ch: output must be a per-pixel (1x1) function of x only
    rb2 = mod.ResBlock(8, 16, t_dim)
    o1 = rb2(x, t1)
    assert torch.allclose(rb2(x, t2), o1, atol=1e-6), "at init the ResBlock must not depend on t (ch-change)"
    x2 = x.clone()
    x2[:, :, 2, 3] += 1.0
    diff = (rb2(x2, t1) - o1).abs().sum(dim=1)     # (B, H, W)
    mask = torch.zeros(6, 6, dtype=torch.bool)
    mask[2, 3] = True
    assert (diff[:, mask] > 1e-4).all() and (diff[:, ~mask] < 1e-6).all(), \
        "at init the ch-change ResBlock must be a pure 1x1 skip conv (a pixel change affects only that pixel)"


def check_depends_on_t(mod):
    torch.manual_seed(2)
    t_dim = 12
    rb = mod.ResBlock(8, 8, t_dim)
    _randomize(rb, 10)
    x = torch.randn(2, 8, 6, 6)
    t1, t2 = torch.randn(2, t_dim), torch.randn(2, t_dim)
    o1, o2 = rb(x, t1), rb(x, t2)
    assert (o1 - o2).abs().max() > 1e-3, "with non-zero weights the ResBlock output must depend on t"
    # scale-shift semantics: per-sample conditioning must not leak across the batch
    t_mix = torch.stack([t1[0], t2[1]])
    o_mix = rb(x, t_mix)
    assert torch.allclose(o_mix[0], o1[0], atol=1e-5) and torch.allclose(o_mix[1], o2[1], atol=1e-5), \
        "time conditioning must be applied per sample"
    # the t path is scale-shift after norm2: with t -> t' giving scale=shift=0 vs not, outputs differ (sanity)
    assert not torch.allclose(rb(x, torch.zeros(2, t_dim)), rb(x, 5 * torch.ones(2, t_dim)), atol=1e-4)


def check_unet(mod):
    torch.manual_seed(3)
    t_dim = 16
    for (H, W) in ((8, 8), (7, 9), (6, 5)):
        net = mod.TinyUNet(3, 3, 8, t_dim)
        x = torch.randn(2, 3, H, W)
        t = torch.randn(2, t_dim)
        out = net(x, t)
        assert out.shape == (2, 3, H, W), f"UNet output shape must equal input, got {tuple(out.shape)} for {(H, W)}"
    # gradient flows to every parameter (after randomising so the zero-init convs don't block the signal)
    net = mod.TinyUNet(3, 4, 8, t_dim)
    _randomize(net, 11)
    x = torch.randn(2, 3, 7, 9)
    t = torch.randn(2, t_dim)
    out = net(x, t)
    assert out.shape == (2, 4, 7, 9)
    (out ** 2).mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"
        assert p.grad.abs().sum() > 0, f"zero gradient at {name} (is it used in forward?)"
    # output depends on t through the UNet too
    net.zero_grad()
    o1 = net(x, t)
    o2 = net(x, t + 1.0)
    assert (o1 - o2).abs().max() > 1e-3, "UNet output must depend on t"


def run(mod):
    check_shapes(mod);        print("  ok  shapes for even/odd sizes (ResBlock, Down, Up)")
    check_init_is_skip(mod);  print("  ok  ResBlock at init == skip path, independent of t")
    check_depends_on_t(mod);  print("  ok  output depends on t (per-sample scale-shift)")
    check_unet(mod);          print("  ok  TinyUNet shape == input, gradients reach all params")
