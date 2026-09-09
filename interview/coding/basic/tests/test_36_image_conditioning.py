import math
import torch
import torch.nn.functional as F
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_inpaint_input(mod):
    # hand example: 4x4 pixel mask, f=2 -> 2x2 latent mask; a single hole pixel in a cell marks the whole cell
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, 0, 1] = 1          # top-left cell has one hole pixel
    mask[0, 0, 2:4, 2:4] = 1      # bottom-right cell fully hole
    ml = mod.downsample_mask(mask, 2)
    assert ml.shape == (1, 1, 2, 2), f"latent mask shape {tuple(ml.shape)}"
    assert torch.equal(ml, torch.tensor([[[[1., 0.], [0., 1.]]]])), f"max-pooled mask {ml.tolist()}"
    # avg pooling would have given 0.25 for the first cell
    assert not torch.equal(ml, F.avg_pool2d(mask, 2))
    g = torch.Generator().manual_seed(0)
    B, C, H, W, f = 2, 3, 8, 8, 4
    image = torch.randn(B, C, H, W, generator=g)
    mask = (torch.rand(B, 1, H, W, generator=g) < 0.3).float()
    z_t = torch.randn(B, C, H // f, W // f, generator=g)
    enc = lambda x: mod.toy_encode(x, f)
    inp = mod.build_inpaint_input(image, mask, z_t, enc, f)
    assert inp.shape == (B, 2 * C + 1, H // f, W // f), f"inpaint input shape {tuple(inp.shape)}"
    assert torch.equal(inp[:, :C], enc(image * (1 - mask))), "channels [0, C): encode(image with holes zeroed)"
    assert torch.equal(inp[:, C:C + 1], F.max_pool2d(mask, f)), "channel C: max-pooled mask"
    assert torch.equal(inp[:, C + 1:], z_t), "channels (C, 2C]: the noisy latent"
    # holes must be zeroed BEFORE encoding: a fully-hole image gives an all-zero masked latent
    inp2 = mod.build_inpaint_input(image, torch.ones(B, 1, H, W), z_t, enc, f)
    assert torch.equal(inp2[:, :C], torch.zeros(B, C, H // f, W // f)) and torch.equal(inp2[:, C], torch.ones(B, H // f, W // f))


def check_zero_conv(mod):
    C = 4
    zc = mod.zero_conv(C)
    assert isinstance(zc, torch.nn.Conv2d) and zc.kernel_size == (1, 1) and zc.in_channels == C and zc.out_channels == C
    assert torch.all(zc.weight == 0) and zc.bias is not None and torch.all(zc.bias == 0), "zero conv must start at exactly zero"
    assert zc.weight.requires_grad and zc.bias.requires_grad
    torch.manual_seed(0)
    base = mod.TinyBlock(C, seed=1)
    branch = mod.TinyBlock(C, seed=2)
    blk = mod.ControlledBlock(base, branch, C)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, C, 6, 6, generator=g)
    ctrl = torch.randn(2, C, 6, 6, generator=g)
    out = blk(x, ctrl)
    assert torch.equal(out, base(x)), "at init the controlled block must equal the base block exactly"
    assert torch.equal(blk(x, torch.zeros_like(ctrl)), blk(x, 5 * ctrl)), "at init the output must not depend on control"
    # gradient: the zero conv gets a non-zero gradient, the control branch gets zero gradient at init
    loss = (out * torch.randn(out.shape, generator=g)).sum()
    loss.backward()
    assert blk.zero.weight.grad is not None and blk.zero.weight.grad.abs().max() > 1e-4, "zero conv weight grad must be non-zero"
    assert blk.zero.bias.grad.abs().max() > 1e-4
    assert all(p.grad is None or torch.all(p.grad == 0) for p in branch.parameters()), "control branch gets zero grad at init (blocked by zero weights)"
    # one step later the output depends on the control
    opt = torch.optim.SGD(blk.zero.parameters(), lr=0.1)
    opt.step()
    assert blk.zero.weight.abs().max() > 0
    out2 = blk(x, ctrl)
    assert not torch.allclose(out2, base(x)), "after one step the branch must contribute"
    assert not torch.allclose(blk(x, ctrl), blk(x, torch.zeros_like(ctrl)))


def check_repaint(mod):
    g = torch.Generator().manual_seed(0)
    ab = mod.make_alphas_cumprod(1000)
    B, C, h, w = 2, 4, 5, 5
    z_pred = torch.randn(B, C, h, w, generator=g)
    z0 = torch.randn(B, C, h, w, generator=g)
    noise = torch.randn(B, C, h, w, generator=g)
    mask = (torch.rand(B, 1, h, w, generator=g) < 0.4).float()
    t = torch.tensor([10, 700])
    out = mod.repaint_step(z_pred, z0, mask, t, noise, ab)
    assert out.shape == (B, C, h, w)
    hole = mask.expand(B, C, h, w).bool()
    assert torch.equal(out[hole], z_pred[hole]), "hole region must be left exactly as the sampler predicted"
    known_ref = mod.q_sample(z0, t, noise, ab)
    assert torch.equal(out[~hole], known_ref[~hole]), "known region must equal the forward-noised known latent at t"
    # explicit formula for one known pixel
    b, c, i, j = 1, 2, 0, 0
    if mask[b, 0, i, j] == 0:
        ref = math.sqrt(ab[700]) * z0[b, c, i, j] + math.sqrt(1 - ab[700]) * noise[b, c, i, j]
        assert torch.allclose(out[b, c, i, j], ref, atol=1e-6)
    # t = 0 with the DDPM schedule: known region ~ z0 (abar_0 = 1 - 1e-4)
    out0 = mod.repaint_step(z_pred, z0, mask, torch.zeros(B, dtype=torch.long), noise, ab)
    assert torch.allclose(out0[~hole], z0[~hole], atol=0.05)
    # all-hole mask: output is z_pred; no-hole mask: output is the noised known
    assert torch.equal(mod.repaint_step(z_pred, z0, torch.ones(B, 1, h, w), t, noise, ab), z_pred)
    assert torch.equal(mod.repaint_step(z_pred, z0, torch.zeros(B, 1, h, w), t, noise, ab), known_ref)


def check_classifier_guidance(mod):
    ab = mod.make_alphas_cumprod(1000)
    g = torch.Generator().manual_seed(0)
    B, C, h, w = 3, 2, 4, 4
    x_t = torch.randn(B, C, h, w, generator=g)
    y = torch.tensor([0, 2, 1])
    t = torch.tensor([5, 500, 900])
    clf = mod.TinyClassifier(C, 3, hidden=16, seed=0)
    grad = mod.classifier_log_prob_grad(clf, x_t, y, t)
    assert grad.shape == x_t.shape and not grad.requires_grad
    assert not x_t.requires_grad, "input must not be modified in place"
    # finite differences in float64
    clf64 = clf.double()
    x64 = x_t.double()
    def logp(x):
        return F.log_softmax(clf64(x, t), dim=-1).gather(1, y.view(-1, 1)).squeeze(1)
    fd = torch.zeros_like(x64)
    eps = 1e-5
    for idx in [(0, 0, 0, 0), (0, 1, 2, 3), (1, 0, 3, 1), (2, 1, 1, 2), (1, 1, 0, 0), (2, 0, 3, 3)]:
        e = torch.zeros_like(x64); e[idx] = eps
        fd[idx] = ((logp(x64 + e) - logp(x64 - e)) / (2 * eps))[idx[0]]
        assert abs(fd[idx].item() - grad[idx].item()) < 1e-4, f"grad vs finite diff at {idx}: {grad[idx].item():.6f} vs {fd[idx].item():.6f}"
    grad64 = mod.classifier_log_prob_grad(clf64, x64, y, t)
    assert torch.allclose(grad64.float(), grad, atol=1e-5)
    # analytic: quadratic log-likelihood classifier, logit_c = -0.5 ||x - mu_c||^2
    mu = torch.randn(3, C, h, w, generator=g, dtype=torch.float64)

    class Quad(torch.nn.Module):
        def forward(self, x, t):
            return -0.5 * ((x[:, None] - mu[None]) ** 2).sum(dim=(2, 3, 4))
    gq = mod.classifier_log_prob_grad(Quad(), x64, y, t)
    p = F.softmax(Quad()(x64, t), dim=-1)                                    # (B, 3)
    ref = -(x64 - mu[y]) + (p[:, :, None, None, None] * (x64[:, None] - mu[None])).sum(1)
    assert torch.allclose(gq, ref, atol=1e-10), f"analytic quadratic gradient mismatch, max err {(gq - ref).abs().max():.2e}"
    # guided eps
    epsn = torch.randn(B, C, h, w, generator=g)
    s = 3.0
    ge = mod.classifier_guided_eps(epsn, grad, t, ab, s)
    ref = epsn - (1 - ab[t]).sqrt().view(B, 1, 1, 1) * s * grad
    assert torch.allclose(ge, ref, atol=1e-6), "eps_hat = eps - sqrt(1 - abar_t) * s * grad"
    assert torch.equal(mod.classifier_guided_eps(epsn, grad, t, ab, 0.0), epsn), "scale 0 is the unguided eps"
    # per-sample abar: sample 0 at t=5 has ~zero noise scale, sample 2 at t=900 nearly full
    d = (ge - epsn).abs().flatten(1).max(1).values / (s * grad.abs().flatten(1).max(1).values)
    assert d[0] < 0.05 and d[2] > 0.9, f"guidance strength must follow sqrt(1 - abar_t) per sample: {d.tolist()}"


def run(mod):
    check_inpaint_input(mod);         print("  ok  inpainting input: [z_masked | max-pooled mask | z_t]")
    check_zero_conv(mod);             print("  ok  zero conv: identity at init, non-zero grad, contributes after a step")
    check_repaint(mod);               print("  ok  RePaint step: hole untouched, known = q_sample(z0, t)")
    check_classifier_guidance(mod);   print("  ok  classifier guidance grad vs finite diff / analytic; guided eps")
