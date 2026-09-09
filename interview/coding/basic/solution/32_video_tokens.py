"""
32 — Video tokens: 3D patchify, factorized 3D sin-cos / RoPE position codes, block-causal frame masks, VAE frame count

The input side of a video DiT (WAN 2.1, CogVideoX, HunyuanVideo). A latent video (B, C, T, H, W) is cut into
(pt, ph, pw) patches in temporal-major order, each token gets a position code built per axis (t, h, w) and then
concatenated along the channel/head dim, attention may be restricted so frames only look backwards (autoregressive /
streaming video, world models), and the causal 3D VAE maps T pixel frames to (T-1)/4 + 1 latent frames.

Signatures:
    def patchify_3d(x, patch) -> Tensor                           (B, C, T, H, W) -> (B, N, pt*ph*pw*C)
    def unpatchify_3d(tokens, patch, C, T, H, W) -> Tensor        exact inverse
    def get_3d_sincos_pos_embed(dims, grid_t, grid_h, grid_w) -> Tensor      (T*H*W, dt+dh+dw)
    def rope_3d_freqs(dims, grid_t, grid_h, grid_w, theta=10000.0) -> (cos, sin)   each (N, dt+dh+dw)
    def block_causal_mask(T, tokens_per_frame, window=None) -> BoolTensor    (N, N), N = T*tokens_per_frame
    def vae_latent_frames(T, stride=4) -> int ;  def vae_pixel_frames(T_lat, stride=4) -> int

Constraints: torch (CPU) only, float32. Token n = t*(h*w) + i*w + j for latent patch (t, i, j) with
(h, w) = (H/ph, W/pw): time is the slowest axis, then row, then column (flatten of (T', H', W')). Inside a token
the layout is (pt, ph, pw, C) flattened. Position codes use the same token order. Sizes divisible by the patch.
get_1d_sincos_pos_embed (problem 16) and apply_rope_interleaved are given.

Interview budget: 25 min

Discussion follow-ups:
  * WAN uses 3D RoPE with head-dim split 16/24/24 (of 64... or 44/42/42 of 128); why split the head dim by axis
    rather than sum three full-dim rotations? What breaks when you generate more frames / higher resolution than
    trained (extrapolation) and how do NTK / YaRN-style rescalings or per-axis frequency scaling help?
  * Block-causal attention makes a bidirectional video DiT into a streaming / autoregressive model (Self-Forcing,
    CausVid, world models). What has to change in the KV cache and in the noise schedule (per-frame timesteps)?
  * Temporal 4x + spatial 8x compression: why is the first frame encoded alone (causal 3D VAE, T = 4k + 1), and
    what does that do to image-video joint training?
  * Token count = T'*H'*W' with attention cost ~ N^2. Where does foveated / mixed-resolution tokenization (or
    a sliding temporal window) cut this, and what does the position code need to remain consistent?
"""
import torch

__implement__ = ["patchify_3d", "unpatchify_3d", "get_3d_sincos_pos_embed", "rope_3d_freqs",
                 "block_causal_mask", "vae_latent_frames", "vae_pixel_frames"]


def patchify_3d(x, patch):
    """
    x : (B, C, T, H, W); patch = (pt, ph, pw) with T % pt == H % ph == W % pw == 0.
    Returns (B, N, pt*ph*pw*C) with N = (T/pt)*(H/ph)*(W/pw) and token order temporal-major:
        n = t*(h*w) + i*w + j   for patch (t, i, j), where (h, w) = (H/ph, W/pw)
    Inside a token the layout is (a, b, c, ch) flattened:
        tokens[:, n, ((a*ph + b)*pw + c)*C + ch] = x[:, ch, t*pt + a, i*ph + b, j*pw + c]
    (WAN's unpatchify einsum 'fhwpqrc->cfphqwr' read backwards). view/permute/reshape only, no loops.
    """
    B, C, T, H, W = x.shape
    pt, ph, pw = patch
    t, h, w = T // pt, H // ph, W // pw
    x = x.view(B, C, t, pt, h, ph, w, pw)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)            # (B, t, h, w, pt, ph, pw, C)
    return x.reshape(B, t * h * w, pt * ph * pw * C)


def unpatchify_3d(tokens, patch, C, T, H, W):
    """
    Inverse of patchify_3d. tokens : (B, N, pt*ph*pw*C). Returns (B, C, T, H, W) exactly.
    """
    B = tokens.shape[0]
    pt, ph, pw = patch
    t, h, w = T // pt, H // ph, W // pw
    x = tokens.view(B, t, h, w, pt, ph, pw, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)            # (B, C, t, pt, h, ph, w, pw)
    return x.reshape(B, C, T, H, W)


def get_1d_sincos_pos_embed(embed_dim, pos):
    """Given helper (problem 16): (M,) positions -> (M, D) = [sin(pos*omega), cos(pos*omega)], omega_k = 10000^(-2k/D)."""
    half = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float64) / half))
    out = pos.to(torch.float64).reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def get_3d_sincos_pos_embed(dims, grid_t, grid_h, grid_w):
    """
    dims = (dt, dh, dw), each even. Returns (grid_t*grid_h*grid_w, dt+dh+dw) float32 in token order
    n = t*(grid_h*grid_w) + i*grid_w + j, equal to
        concat[ get_1d_sincos_pos_embed(dt, t_n), get_1d_sincos_pos_embed(dh, i_n), get_1d_sincos_pos_embed(dw, j_n) ]
    i.e. the first dt features encode the frame index, the next dh the row, the last dw the column (CogVideoX /
    Latte-style factorized 3D sin-cos).
    """
    dt, dh, dw = dims
    tt, ii, jj = torch.meshgrid(torch.arange(grid_t, dtype=torch.float32), torch.arange(grid_h, dtype=torch.float32),
                                torch.arange(grid_w, dtype=torch.float32), indexing="ij")
    return torch.cat([get_1d_sincos_pos_embed(dt, tt.reshape(-1)),
                      get_1d_sincos_pos_embed(dh, ii.reshape(-1)),
                      get_1d_sincos_pos_embed(dw, jj.reshape(-1))], dim=1)


def rope_3d_freqs(dims, grid_t, grid_h, grid_w, theta=10000.0):
    """
    Axis-split 3D RoPE tables (WAN 2.1 style; head_dim = dt + dh + dw, e.g. (16, 24, 24) for head_dim 64).
    dims = (dt, dh, dw), each even. For axis a with D_a dims and token coordinate p_a (frame / row / column index):
        freq_k = theta ** (-2k / D_a),  k = 0 .. D_a/2 - 1
        angle[n, 2k] = angle[n, 2k+1] = p_a(n) * freq_k         (INTERLEAVED pairs: dims (2k, 2k+1) rotate together)
    The per-axis (N, D_a) angle blocks are concatenated in the order t, h, w to give (N, dt+dh+dw).
    Returns (cos, sin), each (N, dt+dh+dw) float32, token order n = t*(grid_h*grid_w) + i*grid_w + j. These feed
    apply_rope_interleaved(x, cos, sin) (given) so that q.k after rotation depends only on the per-axis offsets.
    """
    tt, ii, jj = torch.meshgrid(torch.arange(grid_t, dtype=torch.float32), torch.arange(grid_h, dtype=torch.float32),
                                torch.arange(grid_w, dtype=torch.float32), indexing="ij")
    parts = []
    for D_a, pos in zip(dims, (tt.reshape(-1), ii.reshape(-1), jj.reshape(-1))):
        freq = theta ** (-torch.arange(0, D_a, 2, dtype=torch.float32) / D_a)      # (D_a/2,)
        ang = pos[:, None] * freq[None, :]                                          # (N, D_a/2)
        parts.append(ang.repeat_interleave(2, dim=1))                               # (N, D_a)
    ang = torch.cat(parts, dim=1)
    return torch.cos(ang), torch.sin(ang)


def apply_rope_interleaved(x, cos, sin):
    """Given helper: x (..., N, D) with interleaved pairs (2k, 2k+1); cos/sin (N, D). Rotates each pair by its angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rot * sin


def block_causal_mask(T, tokens_per_frame, window=None):
    """
    T latent frames, each holding tokens_per_frame tokens laid out contiguously (token n belongs to frame
    n // tokens_per_frame — the temporal-major order of patchify_3d). Returns allowed : (N, N) bool, N = T*tokens_per_frame,
    True where query n may attend key m:
        frame(m) <= frame(n)                          (full attention inside a frame, none to later frames)
        and, if window is not None: frame(n) - frame(m) < window   (own frame plus the window-1 previous frames)
    window=1 is per-frame attention only; window=None or window >= T is plain block-causal. Vectorized.
    """
    f = torch.arange(T).repeat_interleave(tokens_per_frame)          # frame index per token
    d = f[:, None] - f[None, :]                                        # frame(n) - frame(m)
    allowed = d >= 0
    if window is not None:
        allowed = allowed & (d < window)
    return allowed


def vae_latent_frames(T, stride=4):
    """
    Causal 3D VAE (WAN / CogVideoX / HunyuanVideo) frame count: the first frame is encoded alone, then every
    `stride` frames give one latent frame: T_lat = (T - 1) // stride + 1. Valid for any T >= 1 (81 -> 21, 1 -> 1, 49 -> 13).
    """
    return (T - 1) // stride + 1


def vae_pixel_frames(T_lat, stride=4):
    """
    Inverse: T = (T_lat - 1) * stride + 1, the pixel frame count decoded from T_lat latent frames (21 -> 81). This is
    the smallest T with vae_latent_frames(T) == T_lat; sample frame counts should be of the form stride*k + 1.
    """
    return (T_lat - 1) * stride + 1
