"""
27 — Classifier-free guidance: condition dropout, CFG combine, CFG rescale, guidance interval, batched forward

The inference trick every text-to-image / video model relies on, plus the training-side dropout that makes it possible.
Implement per-sample condition dropout (replace the condition with a null embedding with probability p), the CFG
combination, the Lin et al. 2023 CFG-rescale, a guidance interval (Kynkäänniemi et al. 2024: guide only for a range
of noise levels), and the standard "one forward pass on a doubled batch" evaluation.

Signatures:
    def condition_dropout(cond, null, p, generator) -> (Tensor, BoolTensor)   (B, *cdims) -> same shape, mask (B,)
    def cfg_combine(uncond, cond, scale) -> Tensor        uncond + scale * (cond - uncond); scale may be (B,)
    def cfg_rescale(guided, cond, phi) -> Tensor          Lin et al. eq. 15-16, per-sample std over non-batch dims
    def cfg_with_interval(uncond, cond, scale, t, lo, hi) -> Tensor   scale applies only where lo <= t <= hi
    def batched_cfg_forward(model, x, t, cond, null, scale) -> Tensor  ONE model call on cat([x, x]) then combine

Constraints: torch (CPU) only. cond : (B, *cdims) (e.g. (B, L, D) text tokens); null : (*cdims) — the single learned
/ zero "empty prompt" embedding, broadcast across the batch. The dropout must draw ONE Bernoulli per sample from the
given torch.Generator (torch.rand(B, generator=generator) < p), not per element. model(x, t, c) takes x : (B, *dims),
t : (B,), c : (B, *cdims) and returns (B, *dims). In batched_cfg_forward the UNCONDITIONAL half goes FIRST
(diffusers convention: `noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)`).

Interview budget: 20 min

Discussion follow-ups:
  * Derive CFG from Bayes: ∇ log p(x|c) + (w-1)(∇ log p(x|c) - ∇ log p(x)) samples p(x|c) p(c|x)^(w-1) — why
    is this NOT a valid density in general, and what does that imply for high w (oversaturation, mode collapse)?
    Why does CFG rescale (matching the std of the cond prediction) help, and what does it not fix?
  * Why is p = 0.1 the usual dropout rate? What goes wrong at p = 0 (no unconditional model) and p = 0.5?
    With multiple conditions (text + image), how do you get independent guidance scales (drop each independently
    and combine 3 forwards; InstructPix2Pix)?
  * The doubled batch doubles compute per step. Name three ways to cut that: guidance interval, distillation of
    the guided model (guidance distillation as in FLUX-dev), or autoguidance / a smaller unconditional model —
    and what each trades off.
"""
import torch

__implement__ = ["condition_dropout", "cfg_combine", "cfg_rescale", "cfg_with_interval", "batched_cfg_forward"]


class TinyCondModel(torch.nn.Module):
    """Given helper: a deterministic linear 'denoiser' out = Wx x + Wc c + wt t (x : (B, D), c : (B, Dc), t : (B,))."""

    def __init__(self, D, Dc, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.Wx = torch.nn.Parameter(torch.randn(D, D, generator=g) / D ** 0.5)
        self.Wc = torch.nn.Parameter(torch.randn(Dc, D, generator=g) / Dc ** 0.5)
        self.wt = torch.nn.Parameter(torch.randn(D, generator=g))
        self.calls = 0

    def forward(self, x, t, c):
        self.calls += 1
        return x @ self.Wx + c @ self.Wc + t.reshape(-1, 1) * self.wt


def condition_dropout(cond, null, p, generator):
    """
    Training-time condition dropout for CFG.
    cond : (B, *cdims); null : (*cdims) null-condition embedding (learned or zeros); p : drop probability in [0, 1];
    generator : torch.Generator used for the Bernoulli draws (torch.rand(B, generator=generator) < p).
    Returns (out, mask): out : (B, *cdims) equal to cond where mask is False and to null (broadcast) where mask is
    True; mask : (B,) bool. Must not modify cond in place. p = 0 returns cond unchanged, p = 1 returns all nulls.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def cfg_combine(uncond, cond, scale):
    """
    uncond, cond : (B, *dims) model outputs with the null and the real condition.
    scale : float or (B,) tensor of guidance scales (w). Returns uncond + w * (cond - uncond), shape (B, *dims).
    w = 1 gives cond exactly, w = 0 gives uncond exactly.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def cfg_rescale(guided, cond, phi):
    """
    Lin et al. 2023 "CFG rescale" (their eq. 15-16), fixes over-exposure at high guidance scales:
        s_b = std(cond_b) / std(guided_b)        (std over ALL non-batch dims of each sample, unbiased=False)
        rescaled = guided * s_b
        out = phi * rescaled + (1 - phi) * guided
    guided, cond : (B, *dims); phi : float in [0, 1] (0.7 in the paper). Returns (B, *dims).
    phi = 0 returns guided unchanged; phi = 1 gives an output whose per-sample std equals std(cond_b).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def cfg_with_interval(uncond, cond, scale, t, lo, hi):
    """
    Guidance interval (Kynkäänniemi et al. 2024): apply guidance scale `scale` only for samples whose noise level
    t (B,) satisfies lo <= t <= hi (inclusive); elsewhere use scale 1 (i.e. the plain conditional output).
    uncond, cond : (B, *dims); scale : float. Returns (B, *dims). Implement via cfg_combine with a per-sample scale.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def batched_cfg_forward(model, x, t, cond, null, scale):
    """
    Standard CFG evaluation with a single model call on a doubled batch.
    model : callable (x, t, c) -> (B, *dims); x : (B, *dims); t : (B,); cond : (B, *cdims); null : (*cdims);
    scale : float. Build x_in = cat([x, x]), t_in = cat([t, t]), c_in = cat([null expanded to (B, *cdims), cond])
    (UNCONDITIONAL FIRST), call model once, split the output in halves, and return cfg_combine(uncond, cond, scale).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
