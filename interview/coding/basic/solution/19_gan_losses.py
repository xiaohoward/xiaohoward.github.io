"""
19 — GAN losses (non-saturating, hinge), R1 gradient penalty, alternating D/G step

A tiny MLP generator / discriminator pair and a fixed 2D Gaussian data distribution are given. Implement the
standard GAN objectives from logits, the R1 regulariser used by StyleGAN, and one alternating training step.

Signatures:
    def d_loss_nonsat(real_logits, fake_logits) -> Tensor      softplus(-D(x)) + softplus(D(G(z))), means
    def g_loss_nonsat(fake_logits) -> Tensor                    softplus(-D(G(z))), mean
    def d_loss_hinge(real_logits, fake_logits) -> Tensor        relu(1 - D(x)) + relu(1 + D(G(z))), means
    def g_loss_hinge(fake_logits) -> Tensor                     -D(G(z)), mean
    def r1_penalty(D, x_real, gamma) -> Tensor                  gamma/2 * mean_b ||grad_x D(x)||^2
    def gan_step(G, D, x_real, opt_g, opt_d, z_dim, loss="nonsat", r1_gamma=0.0, generator=None) -> dict

Constraints: torch (CPU) only. All losses take raw logits (no sigmoid in D) and must be numerically stable — use
F.softplus / F.binary_cross_entropy_with_logits rather than log(sigmoid(.)). R1 must be computed with
torch.autograd.grad(..., create_graph=True) so it can be back-propagated through into D's parameters. In gan_step,
the D update must not touch G's parameters and vice versa (detach the fake batch for the D step; only step the
right optimiser; clear stale grads).

Interview budget: 25 min

Discussion follow-ups:
  * Why "non-saturating" (-log D(G(z))) instead of the minimax log(1 - D(G(z)))? Sketch the gradient magnitude when
    D confidently rejects fakes.
  * R1 vs the WGAN-GP two-sided penalty: what does penalising the gradient only on REAL data do to the optimal D?
    Why does StyleGAN2 apply R1 "lazily" every 16 steps and scale gamma by 16?
  * Hinge loss (SAGAN / BigGAN): what does the margin do; why is G's hinge loss just -D(G(z))?
  * Mode collapse and why alternating updates + two time-scale learning rates (TTUR) matter; what does a GAN loss
    add on top of an L2/LPIPS term in a latent-diffusion VAE decoder?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__implement__ = ["d_loss_nonsat", "g_loss_nonsat", "d_loss_hinge", "g_loss_hinge", "r1_penalty", "gan_step"]


def make_models(z_dim=4, hidden=32, seed=0):
    """Given: tiny MLP generator (z_dim -> 2) and discriminator (2 -> 1 logit). Returns (G, D)."""
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 2))
    D = nn.Sequential(nn.Linear(2, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))
    return G, D


def sample_real(n, generator=None):
    """Given: fixed 2D Gaussian data, mean (2, -1), std (0.5, 0.3). Returns (n, 2)."""
    return torch.randn(n, 2, generator=generator) * torch.tensor([0.5, 0.3]) + torch.tensor([2.0, -1.0])


def d_loss_nonsat(real_logits, fake_logits):
    """
    Non-saturating (standard BCE) discriminator loss from raw logits.
    real_logits : (B,) or (B,1) D(x_real);  fake_logits : (B,) or (B,1) D(G(z)).
    Returns a 0-dim tensor: mean_b softplus(-real) + mean_b softplus(fake)
      (= BCE-with-logits(real, 1) + BCE-with-logits(fake, 0)). Must be stable for |logits| ~ 1e3 (no exp overflow).
    """
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def g_loss_nonsat(fake_logits):
    """
    Non-saturating generator loss: mean_b softplus(-fake) = BCE-with-logits(fake, 1) = -log sigmoid(D(G(z))).
    fake_logits : (B,) or (B,1). Returns 0-dim tensor.
    """
    return F.softplus(-fake_logits).mean()


def d_loss_hinge(real_logits, fake_logits):
    """
    Hinge discriminator loss: mean_b relu(1 - real) + mean_b relu(1 + fake). Returns 0-dim tensor.
    """
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_loss_hinge(fake_logits):
    """
    Hinge generator loss: -mean_b fake. Returns 0-dim tensor.
    """
    return -fake_logits.mean()


def r1_penalty(D, x_real, gamma):
    """
    R1 regulariser (Mescheder et al. 2018; StyleGAN): gamma/2 * mean_b || d D(x)/dx ||_2^2 evaluated on real data.
    D : module (B, d) -> (B, 1) or (B,);  x_real : (B, d) (may or may not require grad — make a copy that does);
    gamma : float. Returns a 0-dim tensor that is still attached to D's parameter graph (compute the gradient with
    torch.autograd.grad(outputs=D(x).sum(), inputs=x, create_graph=True)); backward() through it must give
    non-zero gradients on D's parameters. gamma == 0 -> returns 0 (tensor).
    """
    x = x_real.detach().requires_grad_(True)
    out = D(x).sum()
    (grad,) = torch.autograd.grad(out, x, create_graph=True)
    return 0.5 * gamma * grad.pow(2).sum(dim=-1).mean()


def gan_step(G, D, x_real, opt_g, opt_d, z_dim, loss="nonsat", r1_gamma=0.0, generator=None):
    """
    One alternating GAN update. loss in {"nonsat", "hinge"} selects the (d_loss_*, g_loss_*) pair.
      D step: z ~ N(0, I) (B, z_dim) via torch.randn(..., generator=generator); fake = G(z).detach();
              loss_d = d_loss(D(x_real), D(fake)) + r1_penalty(D, x_real, r1_gamma) (if r1_gamma > 0);
              opt_d.zero_grad(); loss_d.backward(); opt_d.step().
      G step: fresh z; fake = G(z); loss_g = g_loss(D(fake)); opt_g.zero_grad(); loss_g.backward(); opt_g.step().
    The D step must not change G's parameters and must not leave gradients in G; the G step must not change D's
    parameters (D's grads populated during the G step are harmless but must be cleared before D's next backward,
    i.e. the zero_grad above suffices). x_real : (B, 2).
    Returns {"loss_d": float, "loss_g": float, "r1": float}.
    """
    d_fn, g_fn = (d_loss_nonsat, g_loss_nonsat) if loss == "nonsat" else (d_loss_hinge, g_loss_hinge)
    B = x_real.shape[0]
    # --- D step ---
    z = torch.randn(B, z_dim, generator=generator)
    fake = G(z).detach()
    loss_d = d_fn(D(x_real).squeeze(-1), D(fake).squeeze(-1))
    r1 = r1_penalty(D, x_real, r1_gamma) if r1_gamma > 0 else torch.zeros(())
    opt_d.zero_grad()
    (loss_d + r1).backward()
    opt_d.step()
    # --- G step ---
    z = torch.randn(B, z_dim, generator=generator)
    loss_g = g_fn(D(G(z)).squeeze(-1))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return {"loss_d": loss_d.item(), "loss_g": loss_g.item(), "r1": r1.item()}
