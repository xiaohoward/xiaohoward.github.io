"""
29 — Vector quantization layer (VQ-VAE / VQGAN): nearest code, straight-through, losses, EMA, dead-code reset

The discrete bottleneck of VQ-VAE / VQGAN / MAGVIT-style tokenizers. Encoder outputs z (N, D) are snapped to the
nearest of K codebook vectors e_k; the decoder sees z_q but gradients flow back to z with a straight-through
estimator; the codebook is trained either with a codebook loss or an EMA update; usage/perplexity diagnostics and a
dead-code reset keep the codebook from collapsing.

Signatures:
    def nearest_code(z, codebook) -> LongTensor               (N, D), (K, D) -> (N,) argmin_k ||z - e_k||^2
    def quantize(z, codebook) -> (z_q, idx)                    STE: z_q = z + (e[idx] - z).detach()
    def vq_losses(z, codebook, idx, beta=0.25) -> (codebook_loss, commitment_loss)
    def ema_update(codebook, cluster_size, embed_avg, z, idx, decay=0.99, eps=1e-5) -> (codebook', N', m')
    def codebook_usage(idx, K) -> (counts, perplexity)
    def reset_dead_codes(codebook, counts, z, generator) -> (codebook', dead_mask)

Constraints: torch (CPU) only. nearest_code must be vectorized (expand ||z||^2 - 2 z.e + ||e||^2 or use cdist),
no Python loop over N or K. Losses are MEANS over all N*D elements (F.mse_loss convention). float32 everywhere.

Interview budget: 25 min

Discussion follow-ups:
  * Why does VQ-VAE need the straight-through estimator at all, and what does it get wrong (the decoder's gradient
    is applied at z, not at z_q)? How do FSQ / LFQ avoid the codebook entirely?
  * EMA vs loss-based codebook training: which one is invariant to the learning rate / optimizer, and why does
    EMA behave like online k-means? What does Laplace smoothing protect against?
  * Codebook collapse: how do you detect it (perplexity << K), and compare dead-code reset, lower codebook dim +
    l2-normalized codes (ViT-VQGAN), and a larger commitment beta.
  * Scaling to video tokenizers (MAGVIT-v2): why does LFQ with K = 2^18 work while a learned 2^18 codebook does not?
"""
import torch

__implement__ = ["nearest_code", "quantize", "vq_losses", "ema_update", "codebook_usage", "reset_dead_codes"]


def nearest_code(z, codebook):
    """
    z : (N, D) float, codebook : (K, D) float.
    Returns idx : (N,) int64 with idx[n] = argmin_k ||z[n] - codebook[k]||^2 (ties -> smallest k, as torch.argmin).
    Vectorized: build the (N, K) squared-distance matrix ||z||^2 - 2 z @ e^T + ||e||^2 (or torch.cdist); no loops.
    """
    d = (z * z).sum(1, keepdim=True) - 2.0 * z @ codebook.t() + (codebook * codebook).sum(1)[None, :]   # (N, K)
    return d.argmin(dim=1)


def quantize(z, codebook):
    """
    z : (N, D) (may require grad), codebook : (K, D) (may require grad).
    Returns (z_q, idx): idx = nearest_code(z, codebook); z_q = z + (codebook[idx] - z).detach().
    Forward value of z_q is exactly codebook[idx]; backward is the straight-through estimator: d z_q / d z = I,
    and NO gradient reaches codebook through z_q.
    """
    idx = nearest_code(z, codebook)
    e = codebook[idx]
    z_q = z + (e - z).detach()
    return z_q, idx


def vq_losses(z, codebook, idx, beta=0.25):
    """
    z : (N, D), codebook : (K, D), idx : (N,) assignment from nearest_code, beta : commitment weight.
    Returns (codebook_loss, commitment_loss), both 0-dim float tensors:
        codebook_loss   = mean( (sg[z] - codebook[idx])^2 )       -> gradient only w.r.t. codebook
        commitment_loss = beta * mean( (z - sg[codebook[idx]])^2 ) -> gradient only w.r.t. z
    mean is over all N*D elements. (VQ-VAE eq. 3; the total VQ loss is their sum.)
    """
    e = codebook[idx]
    codebook_loss = ((z.detach() - e) ** 2).mean()
    commitment_loss = beta * ((z - e.detach()) ** 2).mean()
    return codebook_loss, commitment_loss


def ema_update(codebook, cluster_size, embed_avg, z, idx, decay=0.99, eps=1e-5):
    """
    EMA codebook update (VQ-VAE appendix A.1 / Sonnet VectorQuantizerEMA), one step, no autograd needed.
    codebook : (K, D) current codes; cluster_size : (K,) running EMA of counts N_k; embed_avg : (K,) x D running EMA
    of summed assigned vectors m_k; z : (N, D) this batch's encoder outputs (detach them); idx : (N,) assignments.
    With one-hot A (N, K):  n_k = sum_n A[n, k],  s_k = sum_n A[n, k] z[n]
        N_k <- decay * N_k + (1 - decay) * n_k
        m_k <- decay * m_k + (1 - decay) * s_k
        Laplace smoothing:  N~_k = (N_k + eps) / (sum_j N_j + K * eps) * sum_j N_j
        e_k <- m_k / N~_k
    Returns (new_codebook (K, D), new_cluster_size (K,), new_embed_avg (K, D)) as NEW tensors (inputs untouched).
    """
    K = codebook.shape[0]
    z = z.detach()
    onehot = torch.nn.functional.one_hot(idx, K).to(z.dtype)        # (N, K)
    n = onehot.sum(0)                                                # (K,)
    s = onehot.t() @ z                                               # (K, D)
    new_N = decay * cluster_size + (1.0 - decay) * n
    new_m = decay * embed_avg + (1.0 - decay) * s
    total = new_N.sum()
    N_smooth = (new_N + eps) / (total + K * eps) * total
    new_codebook = new_m / N_smooth[:, None]
    return new_codebook, new_N, new_m


def codebook_usage(idx, K):
    """
    idx : (N,) int64 assignments, K : codebook size.
    Returns (counts, perplexity): counts (K,) int64 histogram of assignments; perplexity = exp(-sum_k p_k log p_k)
    with p = counts / N (0 log 0 = 0), as a python float. Uniform use -> K; a single code used -> 1.
    """
    counts = torch.bincount(idx, minlength=K)
    p = counts.double() / idx.numel()
    ent = -(p[p > 0] * p[p > 0].log()).sum()
    return counts, float(torch.exp(ent))


def reset_dead_codes(codebook, counts, z, generator):
    """
    codebook : (K, D); counts : (K,) usage histogram (e.g. accumulated over an epoch); z : (N, D) a pool of recent
    encoder outputs with N >= number of dead codes; generator : torch.Generator for determinism.
    Dead codes are those with counts == 0. Returns (new_codebook, dead_mask): new_codebook is a NEW (K, D) tensor
    equal to codebook except that dead rows are replaced by rows of z sampled WITHOUT replacement via
    torch.randperm(N, generator=generator)[:n_dead]; dead_mask is the (K,) bool mask. Live codes are untouched.
    """
    dead = counts == 0
    n_dead = int(dead.sum())
    new_codebook = codebook.detach().clone()
    if n_dead > 0:
        pick = torch.randperm(z.shape[0], generator=generator)[:n_dead]
        new_codebook[dead] = z.detach()[pick]
    return new_codebook, dead


def make_clustered_data(K=8, N=256, D=4, seed=0):
    """Given helper: K well-separated Gaussian clusters (N points) and a codebook near the cluster centers."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(K, D, generator=g) * 5.0
    lab = torch.randint(0, K, (N,), generator=g)
    z = centers[lab] + 0.3 * torch.randn(N, D, generator=g)
    codebook = centers + 0.1 * torch.randn(K, D, generator=g)
    return z, codebook, lab
