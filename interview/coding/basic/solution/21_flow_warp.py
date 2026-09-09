"""
21 — Optical-flow backward warping, grid conversion, forward-backward occlusion mask

Given an image and a dense pixel flow field, produce the warped image by BACKWARD warping: every output pixel
(x, y) samples the input at (x + dx, y + dy) with bilinear interpolation and zeros outside the image. Then write
the helper that converts a pixel flow into the normalised sampling grid F.grid_sample expects, and the classic
forward-backward consistency check used to build occlusion masks (Sundaram et al. 2010, used in RAFT/FlowNet-style
unsupervised losses and in video diffusion temporal-consistency metrics).

Signatures:
    def backward_warp(img, flow) -> Tensor (B, C, H, W)        # by hand, no F.grid_sample
    def flow_to_grid(flow) -> Tensor (B, H, W, 2)             # for F.grid_sample(..., align_corners=False)
    def fb_consistency_mask(flow_fwd, flow_bwd, alpha=1.0) -> BoolTensor (B, H, W)   # True = occluded/inconsistent

Constraints: torch (CPU) only. The solution must NOT call F.grid_sample (the test compares against it: match
F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False) to 1e-5). Flow is in pixels,
channel 0 = dx (along W), channel 1 = dy (along H). Pixel centres are at integer coordinates (pixel i covers
[i-0.5, i+0.5)); a sample exactly on an integer returns that pixel.

Interview budget: 25 min

Discussion follow-ups:
  * Backward vs forward warping (splatting): why is backward warping the default, and when do you need forward
    (softmax splatting, e.g. frame interpolation)?
  * Gradients: bilinear sampling is differentiable w.r.t. the flow (piecewise linear) — where are the gradients
    zero / discontinuous, and why does that hurt flow optimisation with large displacements (coarse-to-fine)?
  * align_corners=True vs False: write the two pixel <-> normalised coordinate maps. Which one is resolution
    consistent?
  * The occlusion mask misfires on fast motion at image borders; how do you tell "left the frame" apart from
    "occluded"?
"""
import torch

__implement__ = ["backward_warp", "flow_to_grid", "fb_consistency_mask"]


def backward_warp(img, flow):
    """
    Backward-warp `img` by `flow` with bilinear interpolation and zero padding, implemented by hand.

    img  : (B, C, H, W) float tensor.
    flow : (B, 2, H, W) float tensor in PIXELS; flow[:, 0] = dx (x / width axis), flow[:, 1] = dy (y / height axis).
    Returns out (B, C, H, W) with out[b, :, y, x] = bilinear sample of img[b] at (x + dx, y + dy).

    Bilinear: x0 = floor(xs), x1 = x0 + 1 (same for y); weights (1 - fx)(1 - fy), fx (1 - fy), ...; every corner
    that falls outside [0, W-1] x [0, H-1] contributes ZERO (that is what padding_mode="zeros" does, so a sample
    at x = -0.5 returns half the value of pixel 0). Do NOT call F.grid_sample. Any dtype of img is fine as long as
    the output has the same dtype.
    """
    B, C, H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=flow.dtype), torch.arange(W, dtype=flow.dtype), indexing="ij")
    sx = xs.unsqueeze(0) + flow[:, 0]            # (B, H, W) sample x coords
    sy = ys.unsqueeze(0) + flow[:, 1]
    x0 = torch.floor(sx)
    y0 = torch.floor(sy)
    fx = (sx - x0)
    fy = (sy - y0)
    x0 = x0.long()
    y0 = y0.long()
    flat = img.reshape(B, C, H * W)
    out = torch.zeros_like(img)
    for dx_i, dy_i, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                            (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + dx_i
        yi = y0 + dy_i
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).reshape(B, 1, H * W).expand(B, C, H * W)
        vals = torch.gather(flat, 2, idx).reshape(B, C, H, W)
        out = out + vals * (wgt * valid.to(img.dtype)).unsqueeze(1)
    return out


def flow_to_grid(flow):
    """
    Convert a pixel flow (B, 2, H, W) [dx, dy] into the sampling grid (B, H, W, 2) that makes
    F.grid_sample(img, grid, align_corners=False) equal backward_warp(img, flow).

    With align_corners=False, normalised coordinate u in [-1, 1] maps to pixel x = ((u + 1) * W - 1) / 2, i.e.
    u = (2 x + 1) / W - 1  (pixel centres at -1 + 1/W, ..., 1 - 1/W). grid[..., 0] is the x (width) coordinate,
    grid[..., 1] the y (height) coordinate.
    """
    B, _, H, W = flow.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=flow.dtype), torch.arange(W, dtype=flow.dtype), indexing="ij")
    sx = xs.unsqueeze(0) + flow[:, 0]
    sy = ys.unsqueeze(0) + flow[:, 1]
    gx = (2 * sx + 1) / W - 1
    gy = (2 * sy + 1) / H - 1
    return torch.stack([gx, gy], dim=-1)


def fb_consistency_mask(flow_fwd, flow_bwd, alpha=1.0):
    """
    Forward-backward consistency check.

    flow_fwd : (B, 2, H, W) flow from frame 1 to frame 2 (in frame-1 pixel coordinates).
    flow_bwd : (B, 2, H, W) flow from frame 2 to frame 1 (in frame-2 pixel coordinates).
    alpha    : threshold in pixels.

    Returns a bool tensor (B, H, W), True where the pixel is OCCLUDED / inconsistent:
        || flow_fwd(x) + flow_bwd(x + flow_fwd(x)) ||_2 > alpha,
    where flow_bwd is sampled at x + flow_fwd(x) with backward_warp (bilinear, zeros outside). Consequence of the
    zero padding: pixels whose forward flow leaves the frame see flow_bwd = 0 and are flagged whenever
    ||flow_fwd|| > alpha (treated as occluded), which is the usual convention.
    """
    bwd_at_fwd = backward_warp(flow_bwd, flow_fwd)
    err = torch.linalg.norm(flow_fwd + bwd_at_fwd, dim=1)
    return err > alpha
