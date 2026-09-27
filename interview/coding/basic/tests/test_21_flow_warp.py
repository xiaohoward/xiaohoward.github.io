import torch
import torch.nn.functional as F


def check_matches_grid_sample(mod):
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 13, 17
    img = torch.randn(B, C, H, W)
    for scale in (0.5, 3.0, 12.0):                # sub-pixel, moderate, and mostly-out-of-frame flows
        flow = torch.randn(B, 2, H, W) * scale
        grid = mod.flow_to_grid(flow)
        assert grid.shape == (B, H, W, 2), f"grid shape must be (B,H,W,2), got {tuple(grid.shape)}"
        ref = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        out = mod.backward_warp(img, flow)
        assert out.shape == img.shape
        assert torch.allclose(out, ref, atol=1e-5), f"warp mismatch vs grid_sample (scale {scale}): max {(out-ref).abs().max():.2e}"
    # exact-integer and half-integer coordinates (floor edge cases)
    flow = torch.zeros(B, 2, H, W)
    flow[:, 0] = 1.0
    flow[:, 1] = -0.5
    ref = F.grid_sample(img, mod.flow_to_grid(flow), mode="bilinear", padding_mode="zeros", align_corners=False)
    assert torch.allclose(mod.backward_warp(img, flow), ref, atol=1e-5), "integer / half-integer flow mismatch"
    # grid_sample's own normalisation with zero flow must give the identity
    g0 = mod.flow_to_grid(torch.zeros(B, 2, H, W))
    assert torch.allclose(F.grid_sample(img, g0, align_corners=False), img, atol=1e-5), "flow_to_grid(0) must be identity for grid_sample"


def check_zero_flow_identity(mod):
    torch.manual_seed(1)
    img = torch.randn(1, 2, 8, 9)
    out = mod.backward_warp(img, torch.zeros(1, 2, 8, 9))
    assert torch.allclose(out, img, atol=1e-6), "zero flow must be the identity"


def check_integer_translation_is_roll(mod):
    torch.manual_seed(2)
    B, C, H, W = 1, 2, 10, 12
    img = torch.randn(B, C, H, W)
    flow = torch.zeros(B, 2, H, W)
    flow[:, 0] = 2.0     # out[y, x] = img[y - 1, x + 2]
    flow[:, 1] = -1.0
    ref = torch.roll(img, shifts=(1, -2), dims=(2, 3))
    ref[:, :, 0, :] = 0
    ref[:, :, :, W - 2:] = 0
    out = mod.backward_warp(img, flow)
    assert torch.allclose(out, ref, atol=1e-6), "integer translation must equal torch.roll with zero fill"


def check_occlusion_mask(mod):
    B, H, W = 1, 12, 16
    fwd = torch.zeros(B, 2, H, W)
    fwd[:, 0] = 3.0                       # everything moves 3 px right
    bwd = -fwd.clone()                    # perfectly consistent backward flow
    m = mod.fb_consistency_mask(fwd, bwd, alpha=1.0)
    assert m.shape == (B, H, W) and m.dtype == torch.bool
    assert not m[:, :, :W - 3].any(), "consistent flows must produce no occlusion in the interior"
    assert m[:, :, W - 3:].all(), "pixels whose forward flow leaves the frame must be flagged (zero padding)"
    # inconsistent: backward flow is zero -> |f + 0| = 3 > alpha everywhere
    m2 = mod.fb_consistency_mask(fwd, torch.zeros_like(bwd), alpha=1.0)
    assert m2.all(), "zero backward flow with 3px forward flow must be flagged everywhere"
    # half the image is inconsistent
    bwd3 = bwd.clone()
    bwd3[:, :, :H // 2] = 0
    m3 = mod.fb_consistency_mask(fwd, bwd3, alpha=1.0)
    assert m3[:, :H // 2, :W - 3].all() and not m3[:, H // 2:, :W - 3].any(), "occlusion mask must be spatially selective"
    # threshold semantics: alpha above the error -> not flagged
    assert not mod.fb_consistency_mask(fwd, torch.zeros_like(bwd), alpha=3.5).any(), "alpha above the error must not flag"


def run(mod):
    check_matches_grid_sample(mod);        print("  ok  backward_warp matches F.grid_sample (align_corners=False, zeros)")
    check_zero_flow_identity(mod);         print("  ok  zero flow is identity")
    check_integer_translation_is_roll(mod);print("  ok  integer translation == torch.roll with zero fill")
    check_occlusion_mask(mod);             print("  ok  forward-backward occlusion mask")
