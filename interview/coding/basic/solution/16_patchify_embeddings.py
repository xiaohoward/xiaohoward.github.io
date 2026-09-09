"""
16 — Patchify / unpatchify, 2D sin-cos positional embedding, sinusoidal timestep embedding

The input side of a DiT / MAE / ViT: cut an image into non-overlapping patches (tokens), give each token a fixed 2D
sin-cos position code, and embed the diffusion timestep with sinusoidal features + a small MLP.

Signatures:
    def patchify(x, p) -> Tensor                    (B, C, H, W) -> (B, N, p*p*C),  N = (H/p)*(W/p)
    def unpatchify(tokens, p, C, H, W) -> Tensor    exact inverse
    def get_1d_sincos_pos_embed(embed_dim, pos) -> Tensor   (M,) positions -> (M, embed_dim)
    def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w) -> Tensor   (grid_h*grid_w, embed_dim)
    def timestep_embedding(t, dim, max_period=10000) -> Tensor   (B,) -> (B, dim)
    def embed_timestep(t, dim, mlp) -> Tensor        mlp(timestep_embedding(t, dim))

Constraints: torch (CPU) only, no torchvision/timm. Patch order is row-major over the patch grid (patch (i, j) is
token i*(W/p)+j); inside a token the layout is (p, p, C) flattened, i.e. token[..., (ph*p + pw)*C + c] =
x[b, c, i*p+ph, j*p+pw] — exactly the DiT/MAE convention (einsum 'nchpwq->nhwpqc'). H, W divisible by p.
The 2D pos embed is the MAE/DiT one: half the dims encode one grid axis, half the other (see the docstring for
which is first); each half is [sin(pos*omega), cos(pos*omega)] with omega_k = 1 / 10000^(2k/D_half), k < D_half/2.
Timestep embedding is the ADM/DiT one: freqs_i = exp(-ln(max_period) * i / half), i < half = dim/2, and the
output is concat[cos(t*freqs), sin(t*freqs)] (cos FIRST), zero-padded if dim is odd. Use float32.

Interview budget: 20 min

Discussion follow-ups:
  * Why do DiTs use a fixed sin-cos position embedding rather than learned? What breaks when you generate at a
    resolution different from training (and how do RoPE / interpolation / extrapolation schemes address it)?
  * The timestep MLP output modulates every block via adaLN. Why sinusoidal features first rather than feeding the
    scalar t (or log-SNR) directly to an MLP?
  * Patch size p trades token count (attention cost ~ (HW/p^2)^2) against detail. Where does patch size 2 in latent
    space come from for DiT, and what does a mixed-resolution (foveated) token grid need from the pos embed?
"""
import math
import torch

__implement__ = ["patchify", "unpatchify", "get_1d_sincos_pos_embed", "get_2d_sincos_pos_embed",
                 "timestep_embedding", "embed_timestep"]


def patchify(x, p):
    """
    x : (B, C, H, W), H % p == 0 and W % p == 0.
    Returns (B, N, p*p*C) with N = (H//p)*(W//p). Token n = i*(W//p) + j holds patch (row i, col j); inside a
    token the flattened order is (ph, pw, c): token[n, (ph*p + pw)*C + c] = x[:, c, i*p+ph, j*p+pw].
    Use view/permute/reshape only (no loops, no F.unfold).
    """
    B, C, H, W = x.shape
    h, w = H // p, W // p
    x = x.view(B, C, h, p, w, p)                # (B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)             # (B, h, w, p, p, C)
    return x.reshape(B, h * w, p * p * C)


def unpatchify(tokens, p, C, H, W):
    """
    Inverse of patchify. tokens : (B, N, p*p*C) with N = (H//p)*(W//p). Returns (B, C, H, W) exactly.
    """
    B = tokens.shape[0]
    h, w = H // p, W // p
    x = tokens.view(B, h, w, p, p, C)
    x = x.permute(0, 5, 1, 3, 2, 4)             # (B, C, h, p, w, p)
    return x.reshape(B, C, H, W)


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim : even int D. pos : float tensor (M,) of positions.
    Returns (M, D) float32: out[m, k] = sin(pos[m] * omega_k), out[m, D/2 + k] = cos(pos[m] * omega_k) for
    k < D/2, with omega_k = 1 / 10000 ** (2k / D). (MAE get_1d_sincos_pos_embed_from_grid.)
    """
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float64) / half))       # (D/2,)
    out = pos.to(torch.float64).reshape(-1, 1) * omega.reshape(1, -1)                # (M, D/2)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """
    embed_dim : int divisible by 4. Returns (grid_h*grid_w, embed_dim) float32 in row-major grid order
    (token n = i*grid_w + j is at row i, col j).
    Exactly the MAE/DiT get_2d_sincos_pos_embed: grid = np.meshgrid(arange(grid_w), arange(grid_h)) (so grid[0] is
    the COLUMN index j and grid[1] is the ROW index i), and
        out = concat[get_1d_sincos_pos_embed(D/2, grid[0].flatten()), get_1d_sincos_pos_embed(D/2, grid[1].flatten())]
    i.e. the first D/2 features encode the column (x) coordinate, the last D/2 the row (y) coordinate. (MAE names
    these emb_h / emb_w, which is misleading — say so.)
    """
    assert embed_dim % 4 == 0
    ii, jj = torch.meshgrid(torch.arange(grid_h, dtype=torch.float32),
                            torch.arange(grid_w, dtype=torch.float32), indexing="ij")
    emb_x = get_1d_sincos_pos_embed(embed_dim // 2, jj.reshape(-1))   # grid[0] in MAE: column index
    emb_y = get_1d_sincos_pos_embed(embed_dim // 2, ii.reshape(-1))   # grid[1] in MAE: row index
    return torch.cat([emb_x, emb_y], dim=1)


def timestep_embedding(t, dim, max_period=10000):
    """
    t : (B,) float or int tensor of timesteps (any real values, e.g. 0..999 or 0..1 scaled).
    dim : output width. Returns (B, dim) float32:
        half = dim // 2;  freqs_i = exp(-ln(max_period) * i / half), i = 0..half-1
        out = concat[cos(t * freqs), sin(t * freqs)]  (cos first), then a zero column appended if dim is odd.
    (ADM / DiT TimestepEmbedder.timestep_embedding.)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def embed_timestep(t, dim, mlp):
    """
    t : (B,) timesteps; dim : sinusoidal feature width; mlp : an nn.Module mapping (B, dim) -> (B, hidden).
    Returns mlp(timestep_embedding(t, dim)) : (B, hidden). This is the DiT TimestepEmbedder forward.
    """
    return mlp(timestep_embedding(t, dim))


def make_timestep_mlp(dim, hidden, seed=0):
    """Given helper: the 2-layer SiLU MLP of DiT's TimestepEmbedder (Linear -> SiLU -> Linear)."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden))
