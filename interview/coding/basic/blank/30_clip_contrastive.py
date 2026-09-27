"""
30 — CLIP contrastive loss, learnable temperature, retrieval recall@k, SigLIP sigmoid loss

The alignment objective behind CLIP / OpenCLIP / SigLIP text encoders used by every T2I / T2V model. Given a batch
of B paired (image, text) embeddings, L2-normalize, scale cosine similarities by a learnable temperature, and train
with the symmetric InfoNCE loss whose positives are the diagonal. Evaluate with retrieval recall@k. Then the SigLIP
variant: every (i, j) pair is an independent binary classification, so no softmax over the batch.

Signatures:
    def l2_normalize(x, eps=1e-8) -> Tensor                       (..., D) -> unit-norm rows
    def clip_logits(img, txt, logit_scale) -> Tensor              (B, B) logits_per_image = exp(logit_scale) * i @ t^T
    def clip_loss(img, txt, logit_scale) -> Tensor                symmetric InfoNCE, 0-dim
    def recall_at_k(img, txt, k) -> (r_i2t, r_t2i)                python floats in [0, 1]
    def siglip_loss(img, txt, logit_scale, logit_bias) -> Tensor  0-dim

Constraints: torch (CPU) only, float32. logit_scale is a 0-dim tensor holding the LOG of the scale (CLIP
initializes it to log(1/0.07)); the scale is exp(logit_scale). Pair i is (img[i], txt[i]); ground-truth labels are
arange(B). Use F.cross_entropy / F.logsigmoid for numerical stability (no manual exp of logits).

Interview budget: 20 min

Discussion follow-ups:
  * Why does CLIP learn log-temperature instead of temperature, and why clamp it (CLIP clamps scale <= 100)? What
    happens to the gradient of the loss as the scale -> infinity on a perfectly separable batch?
  * InfoNCE needs large batches (32k in CLIP) because the negatives are the batch. How does SigLIP's pairwise loss
    change memory / communication in multi-GPU training (chunked all-gather-free implementation)?
  * The diagonal-labels assumption is wrong when a batch has duplicate captions / near-duplicate images (false
    negatives). What does that do to the loss and how would you mitigate it?
  * Where do these embeddings show up in a diffusion model (CLIP pooled vector vs T5 token sequence in SDXL / FLUX,
    conditioning through cross-attention vs adaLN), and why do modern T2I/T2V models drop CLIP for T5/LLM encoders?
"""
import torch
import torch.nn.functional as F

__implement__ = ["l2_normalize", "clip_logits", "clip_loss", "recall_at_k", "siglip_loss"]


def l2_normalize(x, eps=1e-8):
    """
    x : (..., D). Returns x / max(||x||_2, eps) along the last dim, same shape & dtype. Differentiable.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def clip_logits(img, txt, logit_scale):
    """
    img : (B, D) image embeddings (NOT yet normalized), txt : (B, D) text embeddings, logit_scale : 0-dim tensor,
    the log of the scale (temperature = 1 / exp(logit_scale)).
    Returns logits_per_image : (B, B) with [i, j] = exp(logit_scale) * <img_i / ||img_i||, txt_j / ||txt_j||>.
    logits_per_text is its transpose. Normalization happens inside this function.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def clip_loss(img, txt, logit_scale):
    """
    Symmetric InfoNCE (CLIP): with L = clip_logits(img, txt, logit_scale) and labels y = arange(B),
        loss = 0.5 * ( CE(L, y) + CE(L^T, y) )
    where CE is mean cross-entropy over the batch (image->text retrieval over the row, text->image over the column).
    Returns a 0-dim tensor. Minimum 0 is approached when each img_i matches txt_i and is orthogonal to the other
    txt_j and the scale -> infinity.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def recall_at_k(img, txt, k):
    """
    img, txt : (B, D) (un-normalized). Ground truth: pair i. Rank by cosine similarity.
    r_i2t = fraction of images i whose true text i is among the top-k most similar texts (row-wise top-k of the
            (B, B) cosine matrix); r_t2i is the same for texts querying images (column-wise). Returns python floats.
    Assume no exact ties.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def siglip_loss(img, txt, logit_scale, logit_bias):
    """
    SigLIP pairwise sigmoid loss (Zhai et al. 2023, eq. 2). With z_ij = exp(logit_scale) * cos(img_i, txt_j) + logit_bias
    and labels y_ij = +1 if i == j else -1:
        loss = -(1/B) * sum_i sum_j log sigmoid( y_ij * z_ij )
    i.e. SUM over the B*B pairs, divided by B (mean over images, sum over candidates). logit_bias is a 0-dim tensor
    (SigLIP initializes it to -10 so the B-1 negatives per row start near the correct answer). Returns a 0-dim tensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def make_paired_embeddings(B=16, D=32, noise=0.3, seed=0):
    """Given helper: B paired embeddings; txt = img + noise * gaussian, so pair i is the true match (usually)."""
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(B, D, generator=g)
    txt = img + noise * torch.randn(B, D, generator=g)
    return img, txt
