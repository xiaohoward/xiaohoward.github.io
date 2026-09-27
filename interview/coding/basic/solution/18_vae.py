"""
18 — VAE: reparameterisation, closed-form KL, ELBO loss, training step

A tiny MLP VAE is given (encoder -> (mu, logvar), decoder -> Bernoulli logits over pixels). Implement the
probabilistic core: the reparameterisation trick, the closed-form KL(q(z|x) || N(0, I)), the (beta-)ELBO loss
with a binary-cross-entropy reconstruction term, and one optimiser step.

Signatures:
    def reparameterize(mu, logvar, generator=None) -> Tensor        z = mu + exp(0.5*logvar) * eps
    def kl_standard_normal(mu, logvar) -> Tensor                    (B,) per-sample KL
    def vae_loss(logits, x, mu, logvar, beta=1.0) -> dict           {"loss", "recon", "kl"} (scalars)
    def train_step(model, x, opt, beta=1.0, generator=None) -> dict  one forward/backward/step, returns loss parts

Constraints: torch (CPU) only. Do not use torch.distributions inside the solution (the test uses it as reference).
Reconstruction = BCE-with-logits SUMMED over pixels, then MEAN over the batch (so recon and KL live on the same
per-sample scale — say why that matters for beta). KL uses the closed form
    KL = 0.5 * sum_d ( mu_d^2 + exp(logvar_d) - 1 - logvar_d ).
Everything must be differentiable end to end; gradients must flow to mu AND logvar through the sample.

Interview budget: 20 min

Discussion follow-ups:
  * Why is reparameterisation needed at all (score-function / REINFORCE estimator variance)? What is the
    Gumbel-softmax analogue for discrete latents (VQ-VAE uses straight-through instead)?
  * Posterior collapse: what is it, why does a strong decoder cause it, KL annealing / free bits as fixes.
  * beta-VAE: what beta > 1 trades; how the KL term becomes the "KL penalty" in Stable-Diffusion's latent VAE
    (tiny weight ~1e-6) and why that VAE then behaves almost like an autoencoder with a regularised latent scale.
  * Why BCE (Bernoulli) here vs MSE (Gaussian with fixed variance) — what does the choice of likelihood imply about
    the implicit weighting between reconstruction and KL?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__implement__ = ["reparameterize", "kl_standard_normal", "vae_loss", "train_step"]


class TinyVAE(nn.Module):
    """Given scaffold. encode: (B, D) -> (mu, logvar) each (B, Z); decode: (B, Z) -> logits (B, D)."""

    def __init__(self, d_in=16, hidden=64, z_dim=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, 2 * z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.Tanh(), nn.Linear(hidden, d_in))
        self.z_dim = z_dim

    def encode(self, x):
        h = self.enc(x)
        return h[:, : self.z_dim], h[:, self.z_dim:]

    def decode(self, z):
        return self.dec(z)


def make_binary_data(n=512, d_in=16, seed=0):
    """Given helper: synthetic binary data with a 2D latent structure: x = Bernoulli(sigmoid(W z + b)), z ~ N(0, I)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(2, d_in, generator=g) * 3.0
    b = torch.randn(d_in, generator=g)
    z = torch.randn(n, 2, generator=g)
    p = torch.sigmoid(z @ W + b)
    return (torch.rand(n, d_in, generator=g) < p).float()


def reparameterize(mu, logvar, generator=None):
    """
    mu, logvar : (B, Z) tensors (logvar = log sigma^2). generator : optional torch.Generator for the noise.
    Returns z = mu + exp(0.5 * logvar) * eps with eps ~ N(0, I) drawn via torch.randn(..., generator=generator),
    shape (B, Z). Must be differentiable w.r.t. both mu and logvar (do not detach; do not sample from a
    Normal(mu, sigma) directly).
    """
    eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
    return mu + torch.exp(0.5 * logvar) * eps


def kl_standard_normal(mu, logvar):
    """
    Closed-form KL( N(mu, diag(exp(logvar))) || N(0, I) ) per sample.
    mu, logvar : (B, Z). Returns (B,) = 0.5 * sum_d ( mu^2 + exp(logvar) - 1 - logvar ). Non-negative, 0 at
    mu = 0, logvar = 0.
    """
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)


def vae_loss(logits, x, mu, logvar, beta=1.0):
    """
    logits : (B, D) decoder outputs (Bernoulli logits), x : (B, D) targets in {0, 1} (float),
    mu, logvar : (B, Z), beta : float weight on the KL.
    recon = mean over batch of [ sum over D of BCE-with-logits(logits, x) ]   (use F.binary_cross_entropy_with_logits)
    kl    = mean over batch of kl_standard_normal(mu, logvar)
    loss  = recon + beta * kl
    Returns {"loss": loss, "recon": recon, "kl": kl}, all 0-dim tensors attached to the graph. With beta=0 the KL
    must not influence the loss (loss == recon exactly, zero gradient to mu/logvar).
    """
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=-1).mean()
    kl = kl_standard_normal(mu, logvar).mean()
    loss = recon + beta * kl
    return {"loss": loss, "recon": recon, "kl": kl}


def train_step(model, x, opt, beta=1.0, generator=None):
    """
    One VAE training step: mu, logvar = model.encode(x); z = reparameterize(...); logits = model.decode(z);
    parts = vae_loss(...); opt.zero_grad(); parts["loss"].backward(); opt.step().
    x : (B, D) float in {0, 1}. Returns the dict from vae_loss with the values converted to Python floats.
    """
    mu, logvar = model.encode(x)
    z = reparameterize(mu, logvar, generator)
    logits = model.decode(z)
    parts = vae_loss(logits, x, mu, logvar, beta)
    opt.zero_grad()
    parts["loss"].backward()
    opt.step()
    return {k: v.item() for k, v in parts.items()}
