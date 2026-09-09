# ML Fundamentals
<!-- weight: 3 -->

## Q: Walk me through the bias-variance decomposition. Where do modern overparameterized networks sit on that curve, and does the classic picture still hold? {diff=1 tags=fundamentals,generalization}
- Follow-up: What's double descent and why does it happen?
### A
**Expected test MSE decomposes as bias$^2$ + variance + irreducible noise**: $\mathbb{E}[(y - \hat f(x))^2] = (f^{\ast}(x) - \mathbb{E}[\hat f(x)])^2 + \mathrm{Var}[\hat f(x)] + \sigma^2$. Bias is how far the *average* model (over training sets) is from the truth; variance is how much the fit wobbles across training sets.

- **High bias** = underfitting: train and val error both high, close together. Fix: bigger model, more features, train longer, less regularization.
- **High variance** = overfitting: train error low, val error much higher. Fix: more data, regularization, augmentation, ensembling, early stopping.

The derivation is just adding and subtracting $\mathbb{E}[\hat f]$ and noting the cross term vanishes because $\mathbb{E}[\hat f - \mathbb{E}\hat f] = 0$.

**Modern caveat (double descent)**: the classic U-shaped test curve holds up to the interpolation threshold (params ≈ samples), where variance spikes because the model is forced to fit noise with a nearly unique interpolant. Past that, with more parameters (or implicit regularization from SGD / min-norm solutions), variance *drops again* and test error keeps improving. So "bigger model overfits more" is not a safe assumption for deep nets; scaling-law regimes are on the right side of the peak. Same story in the data axis: adding data near the threshold can temporarily hurt.

Trap: bias-variance is about a fixed training-set size and an estimator; it says nothing about distribution shift, which is a separate failure mode.

## Q: Show me why minimizing cross-entropy is the same as maximum likelihood, and how KL divergence fits in. {diff=1 tags=fundamentals,probability,loss}
- Follow-up: Which direction of KL is it, and why does that matter?
### A
**Cross-entropy minimization is MLE under the model's predictive distribution, and equals minimizing forward KL from data to model up to a constant.**

For i.i.d. data $(x_i, y_i)$ and a model $p_\theta(y \mid x)$, the negative log-likelihood is
$$
\mathrm{NLL}(\theta) = -\sum_i \log p_\theta(y_i \mid x_i).
$$
For classification with one-hot targets $y$, cross-entropy per sample is $-\sum_k y_k \log p_\theta(k \mid x) = -\log p_\theta(y \mid x)$. So **sum of CE = NLL**: minimizing CE is exactly MLE.

In expectation over the true data distribution $q$:
$$
\mathbb{E}_q[-\log p_\theta] = H(q) + \mathrm{KL}(q \,\|\, p_\theta).
$$
$H(q)$ is constant in $\theta$, so minimizing expected CE ⇔ minimizing **$\mathrm{KL}(q \,\|\, p_\theta)$** — the *forward* KL (data first). Forward KL is mean-seeking / mass-covering: $p_\theta$ is penalized heavily ($\log 0$) wherever $q$ has mass and $p_\theta$ doesn't, so MLE models tend to over-cover modes rather than drop them. Reverse KL, $\mathrm{KL}(p_\theta \,\|\, q)$, is mode-seeking and is what you get in variational inference or when training with samples from the model (e.g., some GAN/RL-style objectives).

Same logic for regression: Gaussian likelihood with fixed $\sigma$ gives MSE; Laplace gives L1. So the choice of loss *is* a choice of noise model.

Follow-up interviewers like: diffusion's simple $\epsilon$-prediction loss is a reweighted ELBO, hence an upper bound on NLL — same MLE/KL lineage.

## Q: Why is the gradient of softmax + cross-entropy with respect to the logits just $(p - y)$? Derive it. {diff=1 tags=fundamentals,backprop,loss}
### A
**Because the log-sum-exp derivative reproduces the softmax, and the target term is linear in the logits.**

Let $z$ be the logits, $p_k = \exp(z_k) / \sum_j \exp(z_j)$, and $y$ one-hot. Loss:
$$
L = -\sum_k y_k \log p_k = -\sum_k y_k z_k + \log \sum_j \exp(z_j) \quad (\text{using } \textstyle\sum_k y_k = 1)
$$

Differentiate wrt $z_i$:
- first term: $-y_i$
- second term: $\exp(z_i) / \sum_j \exp(z_j) = p_i$

So **$\partial L/\partial z_i = p_i - y_i$**. In vector form $\nabla_z L = p - y$.

Why it matters:
- Gradient magnitude is bounded in $[-1, 1]$ per logit and is exactly the prediction error — no saturating derivative like sigmoid+MSE (which has an extra $p(1-p)$ factor that kills gradients when confidently wrong).
- Implementation: always fuse `log_softmax` + NLL (`F.cross_entropy`) so the log-sum-exp is computed with max-subtraction; never `log(softmax(z))` in fp16/bf16.
- Generalizes to soft targets (label smoothing, distillation): gradient is still $p - y$ with $y$ the soft distribution.

Trap: the Jacobian of softmax alone is $\mathrm{diag}(p) - p p^\top$, which is a full matrix; it's the chain rule with the log that collapses it to $p - y$.

## Q: Is logistic regression convex? Prove it, and tell me what that buys you in practice. {diff=2 tags=fundamentals,optimization,convexity}
- Follow-up: Does it have a unique minimizer?
### A
**Yes — the negative log-likelihood is convex in the weights because its Hessian is a weighted sum of positive semidefinite outer products.**

Model: $p(y=1 \mid x) = \sigma(w^\top x)$, $\sigma(a) = 1/(1+e^{-a})$. Per-sample NLL for label $y \in \{0,1\}$:
$$
\ell(w) = -y \log \sigma(w^\top x) - (1-y) \log(1 - \sigma(w^\top x)) = \log(1 + \exp(w^\top x)) - y\, w^\top x.
$$

Gradient: $\nabla \ell = (\sigma(w^\top x) - y)\, x$. Hessian: $\nabla^2 \ell = \sigma(1-\sigma) \cdot x x^\top$, with $\sigma(1-\sigma) \in (0, 1/4]$. Each $x x^\top$ is PSD, positive scalar times PSD is PSD, sums of PSD are PSD ⇒ **Hessian $\succeq 0$ ⇒ convex**. Adding $\lambda \|w\|^2$ makes it $\lambda$-strongly convex.

Alternatively: log-sum-exp is convex, composed with an affine map stays convex, plus a linear term.

What convexity buys:
- Every local min is global; GD/Newton converge to it; no dependence on init.
- Newton's method is IRLS, converges in a handful of iterations for small feature counts.

Uniqueness: **not guaranteed without regularization.** If the data is linearly separable, the NLL has no minimizer — $\|w\| \to \infty$ drives the loss to 0 (GD then converges in *direction* to the max-margin separator). Also if $X$ is rank-deficient the Hessian is only PSD, so the minimizer is a flat set. L2 fixes both. Multiclass softmax regression is convex the same way (Hessian is $x x^\top \otimes (\mathrm{diag}(p) - p p^\top)$, which is PSD).

Trap: a one-hidden-layer network is *not* convex in its weights even with a convex loss — composition with a nonlinearity breaks it.

## Q: What's the difference between L2 regularization and weight decay, and why does it matter for Adam? What does AdamW actually change? {diff=3 tags=optimization,regularization,adam}
- Follow-up: How would you set weight decay for a 1B-param transformer?
### A
**They coincide for plain SGD but diverge for adaptive optimizers; AdamW decouples the decay so it isn't rescaled by the second-moment normalizer.**

- L2 regularization adds $\tfrac{\lambda}{2} \|\theta\|^2$ to the loss, so the gradient becomes $g + \lambda\theta$.
- Weight decay is a direct update $\theta \leftarrow \theta - \eta \lambda \theta$ (shrink toward zero) applied *outside* the gradient.

For SGD, $\theta \leftarrow \theta - \eta(g + \lambda\theta) = (1 - \eta\lambda)\theta - \eta g$ — identical. For Adam, L2 puts $\lambda\theta$ *inside* the gradient that feeds $m$ and $v$:
$$
\theta \leftarrow \theta - \eta \cdot \frac{\hat m}{\sqrt{\hat v} + \epsilon}, \quad \text{where } \hat m, \hat v \text{ now include } \lambda\theta.
$$
Parameters with large historical gradients get large $\hat v$, so their regularization term is *divided down* — weights that move a lot get regularized least, the opposite of what you want. Also the effective decay becomes coupled to the gradient scale, so $\lambda$ can't be tuned independently of lr.

**AdamW** (Loshchilov & Hutter): keep Adam's step on the pure gradient, then apply $\theta \leftarrow \theta - \eta \lambda \theta$ separately. The decay is then a true multiplicative shrinkage, uniform across parameters, and $\lambda$ is tunable roughly independently of $\eta$. This is why AdamW is the transformer default.

Practical numbers a strong candidate states:
- $\lambda \approx 0.1$ for LLMs/DiTs (with lr ~1e-4 to 3e-4), 0.01–0.05 for ViTs from scratch; effective per-step shrink is $\eta\lambda \approx 10^{-5}$.
- Don't decay biases, LayerNorm/RMSNorm gains, or usually embeddings — decaying norm gains distorts scale.
- Since the shrink is $\eta\lambda$, it follows the lr schedule (goes to zero with cosine), which is intended.

Trap: with L2 in PyTorch's `torch.optim.Adam(weight_decay=...)` you are getting the coupled version; `AdamW` is the decoupled one.

## Q: Write down the update rules for SGD, SGD with momentum, and Adam. When does Adam hurt compared to SGD, and why? {diff=2 tags=optimization,adam,sgd}
- Follow-up: What is Adam's $\epsilon$ for, and what happens if it's too small in bf16?
### A
**Rules**, with gradient $g_t$, lr $\eta$:

- SGD: $\theta \leftarrow \theta - \eta g_t$
- Momentum (heavy ball): $v \leftarrow \beta v + g_t$; $\theta \leftarrow \theta - \eta v$. Nesterov evaluates the gradient at $\theta - \eta\beta v$ first. $\beta \approx 0.9$ gives an effective step $\approx \eta/(1-\beta) = 10\eta$ along consistent directions and averages out noise in oscillating ones.
- Adam:
$$
\begin{aligned}
m &\leftarrow \beta_1 m + (1-\beta_1)\, g & (\beta_1 = 0.9) \\
v &\leftarrow \beta_2 v + (1-\beta_2)\, g^2 & (\beta_2 = 0.999,\ \text{or } 0.95 \text{ for LLMs}) \\
\hat m &= m / (1-\beta_1^t), \quad \hat v = v / (1-\beta_2^t) & (\text{bias correction}) \\
\theta &\leftarrow \theta - \eta \cdot \hat m / (\sqrt{\hat v} + \epsilon)
\end{aligned}
$$
Per-coordinate step is $\approx \eta \cdot \operatorname{sign}\text{-ish}(g)$ with magnitude ~$\eta$ regardless of gradient scale — so Adam is roughly **scale-invariant per parameter** and the lr means "how far each weight moves per step," which is why it's robust across layers with wildly different gradient magnitudes (embeddings, LayerNorm gains, attention logits).

**When Adam hurts:**
- Generalization on vision CNNs (ResNet on ImageNet): SGD+momentum typically wins by ~1% top-1. The per-coordinate rescaling changes the implicit bias — SGD prefers flatter / lower-norm solutions; Adam's whitening doesn't.
- Early-training instability: $v$ is tiny at start, steps are huge; needs warmup and bias correction. In LLM training, $\beta_2 = 0.999$ reacts too slowly to gradient-scale spikes ⇒ loss spikes; people use $\beta_2 = 0.95$.
- Memory: 2 extra fp32 states per param (3× the weight memory) — for a 10B model that's 80 GB of optimizer state, which drives ZeRO/FSDP sharding.
- Sparse or heavy-tailed gradients in bf16: $\epsilon = 10^{-8}$ is below bf16 resolution when $\sqrt{\hat v}$ is tiny, so the denominator effectively becomes 0 and steps blow up. Use $\epsilon = 10^{-6}$ to $10^{-5}$ or keep optimizer state in fp32.

Trap: Adam is *not* invariant to loss scaling only because of $\epsilon$; without $\epsilon$ it would be exactly invariant.

## Q: Why do transformers need learning-rate warmup? And why is cosine decay the default afterwards? {diff=2 tags=optimization,schedule,transformer}
- Follow-up: What's the relationship between warmup and pre-norm vs post-norm?
### A
**Warmup keeps the early steps small while Adam's second-moment estimate and the network's activation scales are still unreliable; cosine gives a smooth anneal that lands at a low-noise solution.**

Why warmup specifically for transformers:
- At init, attention logits and residual streams have poorly calibrated scale; a few large updates can push softmax into saturation (attention entropy collapse) or blow up post-norm residuals, from which training doesn't recover.
- Adam's $v$ is bias-corrected but still high-variance over the first $\sim 1/(1-\beta_2) \approx 1000$ steps; without warmup the effective step is far larger than intended in low-gradient coordinates.
- Post-norm transformers (original Vaswani) have gradients that scale badly with depth at init and *cannot* train without warmup; pre-norm is far more tolerant (Xiong et al. 2020), which is why deep models moved to pre-norm and why warmup got shorter but never zero.
- Typical: linear warmup over 1–2% of steps (2k–10k steps for LLMs; 1k–5k for DiTs), peak lr ~1e-4 for 1B-scale diffusion transformers, 3e-4 to 6e-4 for small LLMs, decreasing with model size (roughly $\propto 1/\sqrt{\text{width}}$ under standard param, constant under $\mu$P).

Cosine decay: $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$. Smooth, no tuning of step milestones, and the slow tail lets the iterate settle into a lower-loss region as gradient noise shrinks (the annealing is essentially reducing SGD temperature $\propto \eta/\text{batch}$). Downsides: you must know $T$ in advance; changing the budget mid-run means re-scheduling — which is why WSD (warmup–stable–decay) and "constant + short cooldown" schedules are popular now: they give equivalent final loss with a decay phase of ~10–20% of steps and allow branching checkpoints.

Trap: warmup length should scale with batch size (larger batch → larger lr → longer warmup), not with dataset size.

## Q: Compare BatchNorm, LayerNorm, and RMSNorm. Why do transformers use LayerNorm or RMSNorm rather than BatchNorm? {diff=2 tags=normalization,transformer,architecture}
- Follow-up: What goes wrong with BatchNorm under DDP or with batch size 2?
### A
**BatchNorm normalizes each channel across the batch (and spatial dims); LayerNorm normalizes each token across its feature dim; RMSNorm is LayerNorm without mean-centering.** Transformers use per-token norms because statistics must not depend on other sequences or on batch composition.

Formulas, for $x \in \mathbb{R}^{B \times N \times D}$ (batch, tokens, features):
- BN: per feature $d$, $\mu_d, \sigma_d$ over $(B, N)$; $y = \gamma\, (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$. Uses running averages at test time.
- LN: per token $(b, n)$, $\mu, \sigma$ over $D$; $y = \gamma \odot (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$.
- RMSNorm: $y = \gamma \odot x / \sqrt{\operatorname{mean}(x^2) + \epsilon}$. No mean subtraction, no $\beta$; ~10–15% cheaper than LN and empirically equal in quality (LLaMA, T5, most DiTs).

Why not BN for transformers:
- **Variable-length sequences and padding** corrupt batch statistics.
- **Train/test mismatch**: running stats vs. batch stats differ, worse with the heavy-tailed activations of attention.
- **Autoregressive decoding** processes one token at a time — no batch to normalize over.
- LN is a per-token operation, so it's trivially parallel across devices and doesn't couple samples.

BatchNorm failure modes:
- Small batch (≤8 per GPU): noisy stats, sharp accuracy drop; GroupNorm was invented for this (detection with batch 2).
- DDP: each rank computes stats on its local shard, so effective normalization batch = per-GPU batch, and different ranks see different statistics. SyncBatchNorm fixes this with an all-reduce of $(\mu, \sigma)$ per layer — extra comm.
- Leaks information across samples (contrastive learning, GAN discriminators, RL) — cheating or instability.
- Running stats become stale with EMA weights or after fine-tuning on a shifted domain.

In diffusion transformers the norms are additionally *modulated* by the timestep (adaLN), which only makes sense with per-token normalization; the $\gamma, \beta$ become functions of $t$.

## Q: Explain Xavier and He initialization. Why does depth require careful init, and how do residual networks change the story? {diff=2 tags=initialization,architecture,depth}
- Follow-up: Why do GPT-2 / DiT scale the output projections by $1/\sqrt{2L}$ or zero them?
### A
**Init aims to keep the variance of activations and of backpropagated gradients constant across layers, so signals neither vanish nor explode with depth.**

For $y = Wx$ with $\text{fan}_{\text{in}}$ inputs, $\mathrm{Var}(y) = \text{fan}_{\text{in}} \cdot \mathrm{Var}(w) \cdot \mathrm{Var}(x)$ (i.i.d., zero-mean). To preserve variance forward you need $\mathrm{Var}(w) = 1/\text{fan}_{\text{in}}$; for backward, $1/\text{fan}_{\text{out}}$.
- **Xavier/Glorot**: $\mathrm{Var}(w) = 2/(\text{fan}_{\text{in}} + \text{fan}_{\text{out}})$ — a compromise, derived for tanh/linear units.
- **He/Kaiming**: $\mathrm{Var}(w) = 2/\text{fan}_{\text{in}}$ — ReLU zeros half the pre-activations, halving the second moment, so you double the variance. (Uniform vs normal just changes the bound: $U(\pm\sqrt{6/\text{fan}_{\text{in}}})$.)

Why depth needs it: a stack of $L$ layers multiplies variances, so a per-layer gain of $a \neq 1$ gives $a^L$ — with $L = 50$ even $a = 1.1$ gives 117× growth, $a = 0.9$ gives 0.005×. Same for the gradient via the Jacobian product. Without careful init deep plain nets are untrainable (this is why pre-2015 nets were < 20 layers).

**Residual networks change it**: $x_{l+1} = x_l + f(x_l)$. The identity path gives a per-block variance growth of $\mathrm{Var}(x_l) + \mathrm{Var}(f)$ — additive, not multiplicative, so signal grows $\propto L$ rather than exponentially, and gradient flows straight through. But the residual stream's variance still grows with $L$, so:
- Scale the last layer of each residual branch by $1/\sqrt{2L}$ (GPT-2; $2L$ = number of residual adds in $L$ blocks each with attention + MLP) so the total stream variance stays $O(1)$.
- Or **zero-init** the branch's output projection (Fixup, "zero-$\gamma$" in ResNets, DiT's adaLN-Zero which zeroes the gate $\alpha$) — every block starts as an identity, training begins from a shallow effective network and grows depth gradually. Empirically this is what makes very deep DiTs train stably without warmup tricks.
- Pre-norm makes the stream variance grow like $\sum 1/l \sim \log L$ instead, another reason it tolerates depth.

Trap: with Adam the init scale matters less for the *step* (per-coordinate normalized) but still fully matters for the forward signal and attention logit scale.

## Q: What causes vanishing and exploding gradients, and how do you diagnose and fix them? What does gradient clipping actually do? {diff=2 tags=optimization,stability,training}
- Follow-up: Clip by norm or by value? What threshold, and what do you monitor?
### A
**Gradients are products of per-layer Jacobians along depth (or time); if their spectral norms are consistently below or above 1, the product vanishes or explodes exponentially.**

Causes:
- Saturating nonlinearities ($\sigma' \le 0.25$, tanh) — vanishing.
- Bad init scale (see Xavier/He) — either direction.
- Recurrence: the same $W$ is multiplied $T$ times; eigenvalues of $W$ beyond 1 explode. LSTMs' gated additive cell state was the fix; residual connections are the feed-forward analog.
- Attention: logits growing large ⇒ softmax saturates ⇒ gradient through attention ~0 (attention entropy collapse), while the residual stream may still explode — QK-norm addresses this.
- Loss scaling / precision: fp16 underflow of small gradients (needs dynamic loss scaling; bf16 avoids it with an 8-bit exponent).

Diagnosis: log per-layer gradient norms and activation RMS; a healthy transformer has global grad norm roughly stable (e.g., 0.2–2) and per-layer norms within a decade of each other. Exploding shows up as sudden spikes in grad norm *before* the loss spike; vanishing as early layers with ~1e-6 norms and dead ReLUs.

**Gradient clipping by global norm**: $g \leftarrow g \cdot \min(1, c / \|g\|_2)$ with $c \approx 1.0$ (LLMs, DiTs). It doesn't change direction, only caps the step size on rare heavy-tailed batches — it's a trust region, not a fix for systematically wrong scale. Clip-by-value changes direction and is rarely used now. Monitor how often clipping triggers: if > 10–20% of steps, the lr or $\beta_2$ is wrong, not the clip threshold.

Fixes in priority order: correct init and normalization, residual connections, lower lr / longer warmup, clipping, then architectural (QK-norm, zero-init gates, $\mu$P-style parametrization for width transfer).

## Q: You're training a model and val loss starts rising while train loss keeps dropping. Walk me through your diagnostics and regularization options, and when each one actually helps. {diff=1 tags=generalization,regularization,training}
- Follow-up: Does dropout help transformers / diffusion models?
### A
**That's classic overfitting: the model is memorizing training data; first confirm it's real (not a data or eval bug), then attack it in the order data > augmentation > early stopping > explicit regularization.**

Diagnostics:
- Check train/val are from the same distribution and there's no leak in the *other* direction (val too easy would show the opposite). Confirm the gap is beyond seed noise by running 2–3 seeds.
- Plot the gap vs. training set size: if doubling data noticeably shrinks it, you're variance-limited — get data or synthesize it.
- Look at *which* examples: overfitting to duplicates / near-duplicates in train (dedup!).

Options and when they work:
- **More / cleaner data, dedup** — always the strongest lever.
- **Augmentation** — encodes invariances; for images: crops, flips, color jitter, RandAugment, Mixup/CutMix. Effective when the invariance is real (don't flip text; careful with flips for pose/handedness).
- **Early stopping** — free, equivalent to an L2-like constraint on how far you walk from init; keep the best-val checkpoint; use a patience window, not the first uptick.
- **Weight decay** — mild, always on (AdamW $\lambda$ 0.01–0.1).
- **Dropout** — $p = 0.1$ in attention/MLP helps small-data transformers (ViT on ImageNet-1k, BERT fine-tuning); at LLM / large-scale diffusion pretraining scale it's usually 0 because those models are data-limited, not variance-limited, and dropout slows convergence. Stochastic depth (drop-path) is the better regularizer for deep ViTs.
- **Smaller model / bottleneck** — only if data is truly tiny.
- **Label smoothing, ensembling, EMA of weights** — EMA in particular is standard in diffusion training and tightens the train/val gap.

Trap: rising val *loss* with flat or improving val *accuracy* is often just overconfidence (calibration), not real overfitting; check the metric you care about.

## Q: What evaluation pitfalls have you seen bite people — data leakage, val vs test, metric choice? Give concrete examples in generative modeling. {diff=3 tags=evaluation,methodology,generative}
- Follow-up: How do you compare two image generators fairly?
### A
**The three big ones: information leaking from test into training, tuning on the test set until it's a validation set, and optimizing a metric that doesn't measure the thing you care about.**

Leakage:
- Duplicate or near-duplicate images across train/test (LAION vs. COCO, ImageNet-train vs. -val has ~1% dupes). Dedup by perceptual hash or CLIP-embedding similarity.
- Temporal leakage: random splits of video frames put adjacent frames (nearly identical) on both sides — split by *clip/scene*, not by frame. Same for robot episodes: split by episode, and ideally by scene/object, not by timestep.
- Preprocessing fit on the whole dataset (normalization stats, PCA, tokenizer/VAE trained on data that includes test images).
- Pretrained backbones (CLIP, DINO) that have seen the test set — inflates zero-shot numbers and also corrupts FID/CLIP-score judges.

Val vs. test: every hyperparameter decision, early-stopping choice, or prompt tweak that used the test set consumes it. Keep a sealed test set and touch it once; report seed variance (mean ± std over ≥3 seeds) or at least paired comparisons.

Metric choice in generative models:
- FID is biased by sample count (needs ≥10k, ideally 50k samples; comparing FID at 5k vs 50k is meaningless), depends on the Inception feature space (poor for non-natural images, high-res), and rewards matching the reference distribution, not fidelity to prompts. FID between two sets of *generated* images (e.g., fast sampler vs. full sampler) measures drift, not quality.
- CLIP score measures prompt alignment but saturates and can be gamed; ImageReward / human preference (Elo / win rate) are closer to what users want.
- For speedups: report *wall-clock* on a fixed GPU and batch, not just NFE; and hold the seed and prompt set fixed, paired per prompt, because per-prompt variance dominates.
- For video: FVD has the same caveats as FID plus strong sensitivity to frame count/resolution; also report temporal consistency and human study.

Fair comparison recipe: same prompts, same seeds, same resolution, same CFG and step count, ≥10k samples, report mean ± CI, and at least one human preference study for the headline claim.

## Q: What's the relationship between PCA and SVD? Give me the low-rank intuition and connect it to why LoRA works. {diff=2 tags=linear-algebra,pca,lora}
- Follow-up: What's the optimal rank-k approximation and how do you pick the rank?
### A
**PCA is the SVD of the centered data matrix; the principal components are the right singular vectors and the variances are the squared singular values divided by $n - 1$.**

For centered $X \in \mathbb{R}^{n \times d}$, the covariance is $C = X^\top X/(n-1)$. If $X = U \Sigma V^\top$, then $C = V (\Sigma^2/(n-1)) V^\top$, so the eigenvectors of $C$ are the columns of $V$ and eigenvalues are $\sigma_i^2/(n-1)$. Projecting onto the top-$k$ components gives $X V_k = U_k \Sigma_k$ — the scores. Computing via SVD avoids forming $X^\top X$ (squares the condition number) and is what every library does.

**Eckart–Young–Mirsky**: the best rank-$k$ approximation of any matrix in Frobenius or spectral norm is the truncated SVD, $X_k = U_k \Sigma_k V_k^\top$, with error $\|X - X_k\|_F^2 = \sum_{i>k} \sigma_i^2$. So the singular value spectrum tells you how compressible a matrix is: a fast-decaying spectrum ⇒ a few directions capture most of the energy. Pick $k$ by explained-variance threshold (e.g., 95%) or by a knee in the scree plot.

**Connection to LoRA**: LoRA freezes $W_0$ and learns $\Delta W = B A$, with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll d$ ($r$ = 4–64 for a 4096-wide layer, ~0.1–1% of the params). The hypothesis (Aghajanyan et al.; Hu et al.) is that fine-tuning updates have low *intrinsic rank* — the singular values of the full-fine-tuning $\Delta W$ decay fast, so a rank-$r$ factor captures most of it. Empirically $\Delta W$ from full fine-tuning on a task has effective rank in the single digits to tens even in 4k-wide matrices. It's the same statement as "top-$k$ SVD captures the data" but applied to the *update*.

Practical notes: $B$ is zero-initialized ($\Delta W = 0$ at start), $A$ random (Kaiming); scale by $\alpha/r$; merge into $W_0$ at inference (no latency cost). LoRA on attention $q, v$ projections is the classic choice; on DiTs for SPEED/foveated diffusion, all attention + MLP projections at rank 16–64 is typical.

Trap: PCA on un-centered data finds the mean direction as component 1; and PCA is rotation of coordinates, not a nonlinear embedding — it can't uncover a curved manifold.

## Q: Quickly: what is EM, and why is k-means a special case of it? {diff=3 tags=probability,em,clustering}
- Follow-up: Why does EM never decrease the likelihood? Where is EM used in diffusion / generative modeling today?
### A
**EM maximizes a likelihood with latent variables by alternating a posterior over latents (E) and a weighted MLE given that posterior (M); k-means is EM on a Gaussian mixture with equal spherical covariances in the $\sigma^2 \to 0$ limit, where the posterior becomes a hard assignment.**

EM for observed $x$, latent $z$, parameters $\theta$:
- E-step: $q(z) = p(z \mid x; \theta_{\text{old}})$.
- M-step: $\theta_{\text{new}} = \arg\max_\theta \mathbb{E}_q[\log p(x, z; \theta)]$.

Why it's monotone: $\log p(x;\theta) = \mathrm{ELBO}(q, \theta) + \mathrm{KL}(q \,\|\, p(z \mid x;\theta))$. The E-step makes the KL zero, so $\mathrm{ELBO} = \log p(x; \theta_{\text{old}})$; the M-step raises the ELBO; the log-likelihood is $\ge$ ELBO, hence $\ge$ its old value. It's coordinate ascent on the ELBO — the same objective VAEs optimize with amortized $q$.

Gaussian mixture EM:
- E: responsibilities $r_{ik} \propto \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$.
- M: $\mu_k = \sum_i r_{ik} x_i / \sum_i r_{ik}$, $\Sigma_k$ likewise, $\pi_k$ = mean of $r_{ik}$.

k-means: set $\Sigma_k = \sigma^2 I$, equal $\pi_k$, take $\sigma^2 \to 0$. Then $r_{ik} \to 1$ for the nearest center and 0 otherwise (the softmax over $-\|x - \mu_k\|^2/(2\sigma^2)$ becomes an argmax); the M-step is the mean of assigned points. So k-means = hard-EM with spherical Gaussians. Consequences: k-means finds equal-size spherical blobs, is sensitive to scale (normalize features), and can be improved with k-means++ init ($D^2$ sampling) or by running GMM for soft assignments.

Where it shows up now: mixture-of-experts routing as soft EM, learning vector quantization codebooks (VQ-VAE codebook update *is* the k-means M-step, often with EMA), Gaussian splatting density control, and several distillation / mode-seeking objectives. Both EM and k-means are local: run multiple inits.

Trap: EM converges to a stationary point, which may be a saddle or a local optimum; and the number of components $k$ is not learned — needs BIC / held-out likelihood.

<!-- rapid-fire -->

## Q: What's the difference between a generative and a discriminative model? {diff=1 tags=fundamentals,generative,rapid-fire}
### A
**A discriminative model learns $p(y \mid x)$ — the decision boundary; a generative model learns $p(x)$ or the joint $p(x, y)$, so it can sample new $x$.**

Classifiers, depth and flow estimators are discriminative: they only need to separate classes and spend no capacity modeling the input. VAEs, diffusion models, and LLMs are generative: they model the data distribution, so they can sample, score likelihoods, and act as priors (e.g. a diffusion prior for inverse problems).

Trap: a generative model can be used discriminatively via Bayes, $p(y \mid x) \propto p(x \mid y)\, p(y)$, but with enough labels it usually loses to a direct discriminative model — it solves a harder problem than the task asks. The reverse is impossible: a discriminative model can't generate.

## Q: What's the difference between a likelihood and a probability? {diff=1 tags=fundamentals,probability,rapid-fire}
### A
**Same function, opposite argument: $p(x \mid \theta)$ is the probability of data $x$ with $\theta$ fixed, and the likelihood $L(\theta; x)$ of parameters $\theta$ with $x$ fixed.**

Probability integrates to 1 over $x$; the likelihood does *not* integrate to 1 over $\theta$ — it isn't a distribution over parameters. Maximum likelihood picks the $\theta$ under which the observed data are most probable. To get a real distribution over $\theta$ you need a prior: $p(\theta \mid x) \propto p(x \mid \theta)\, p(\theta)$.

Trap: for continuous data the likelihood is a density, so it can exceed 1 and its log can be positive — bits-per-dim of an image model has an arbitrary scale, and comparing likelihoods across different data discretizations or scalings is meaningless.

## Q: Entropy, cross-entropy, and KL divergence — define all three in one breath. {diff=1 tags=fundamentals,information-theory,rapid-fire}
### A
**Entropy $H(p) = -\mathbb{E}_p[\log p]$ is the average surprise under the true distribution; cross-entropy $H(p, q) = -\mathbb{E}_p[\log q]$ is the average surprise when you code $p$'s samples with $q$; $\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$ is the excess.**

So KL is the price in nats of using the wrong model: $\ge 0$ by Jensen, zero iff $p = q$. Minimizing cross-entropy over $q$ equals minimizing $\mathrm{KL}(p \,\|\, q)$ because $H(p)$ is constant — that's why the classification loss is called cross-entropy and equals MLE.

Trap: KL is asymmetric. $\mathrm{KL}(\text{data} \,\|\, \text{model})$, the MLE direction, is mass-covering; $\mathrm{KL}(\text{model} \,\|\, \text{data})$, the variational direction, is mode-seeking. And KL is infinite wherever $q = 0$ but $p > 0$.

## Q: What is the reparameterization trick and why is it needed? {diff=1 tags=fundamentals,vae,gradients,rapid-fire}
### A
**Write a sample as a deterministic function of the parameters plus parameter-free noise, $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, so gradients flow through the sampling step into $\mu$ and $\sigma$.**

It's needed because $\nabla_\theta \mathbb{E}_{z \sim q_\theta}[f(z)]$ can't be obtained by sampling $z$ and differentiating $f$ — the sampling itself depends on $\theta$. Reparameterization moves $\theta$ out of the distribution and into the integrand, $\nabla_\theta \mathbb{E}_\epsilon[f(\mu + \sigma \epsilon)]$, a low-variance pathwise estimator. The alternative, REINFORCE, works for discrete $z$ but has far higher variance.

Trap: it needs a differentiable sampling map (Gaussian, location-scale families); for categorical latents you use Gumbel-softmax or straight-through — which is why VQ-VAE uses a straight-through estimator.

## Q: What is the curse of dimensionality? {diff=1 tags=fundamentals,geometry,rapid-fire}
### A
**In high dimensions volume grows exponentially, so data become sparse, distances concentrate, and any "nearby points" method needs exponentially many samples.**

Numbers: in $d$ dimensions almost all of a ball's volume sits in a thin shell at its surface; nearest and farthest neighbor distances become nearly equal, so k-NN and kernel methods lose signal; covering $[0,1]^d$ at resolution 0.1 needs $10^d$ points. A Gaussian sample sits at radius $\approx \sqrt{d}$, not near the mean — which is why $x_T$ in diffusion lives on a sphere of radius $\sqrt{d}$, not "near zero".

Why deep learning escapes it: real data lie near a low-dimensional manifold and networks exploit compositional structure, so the effective dimension is far below the ambient one.

## Q: What is a convex function, and why does convexity matter for optimization? {diff=1 tags=fundamentals,optimization,convexity,rapid-fire}
### A
**A function is convex if every chord lies above the graph, $f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y)$; equivalently the Hessian is positive semidefinite.**

Why it matters: every local minimum is global, gradient descent with a suitable step provably converges, and the set of minimizers is convex — no worries about initialization or getting stuck. Linear regression, logistic regression, and SVMs are convex in their parameters; that's the source of their guarantees.

Trap: neural networks are non-convex in the weights (permutation symmetry alone makes minima non-unique), so none of the guarantees hold — SGD works because the landscape is benign in practice. Convexity of the loss in the *output* (cross-entropy in the logits) still helps: it's why the softmax + CE gradient is well-behaved.

## Q: What's the difference between bagging and boosting? {diff=1 tags=fundamentals,ensembles,rapid-fire}
### A
**Bagging trains independent models on bootstrap resamples and averages them to reduce variance; boosting trains models sequentially, each fitting the residual errors of the ensemble so far, to reduce bias.**

Random forests are bagging plus random feature subsets: each tree overfits, but averaging decorrelated overfits cancels their variance. Gradient boosting (XGBoost, LightGBM) adds shallow trees along the negative gradient of the loss — functional gradient descent — turning weak, high-bias learners into a strong one.

Rule: bagging is embarrassingly parallel and hard to overfit; boosting is sequential and *can* overfit with too many rounds, so it needs shrinkage and early stopping. Deep-learning analogues: ensembles and EMA weight averaging are bagging-like; there's no common boosting analogue because a large net already has low bias.

## Q: Define precision, recall, and F1 from a confusion matrix, and tell me when accuracy lies. {diff=1 tags=fundamentals,evaluation,metrics,rapid-fire}
### A
**The confusion matrix counts TP, FP, FN, TN. Precision $= \mathrm{TP} / (\mathrm{TP} + \mathrm{FP})$: of what I flagged, how much was right. Recall $= \mathrm{TP} / (\mathrm{TP} + \mathrm{FN})$: of what was there, how much I caught. $\mathrm{F1} = 2 P R / (P + R)$, the harmonic mean, punishing whichever is low.**

Accuracy $= (\mathrm{TP} + \mathrm{TN}) / \text{total}$ lies under class imbalance: with 1% positives, always predicting "negative" scores 99% with zero recall. It also hides *which* error you make, and errors have different costs (missed tumor vs false alarm). Use PR curves or AUROC to see the threshold trade-off; report per-class metrics.

Trap: F1 ignores true negatives and depends on a threshold; average precision (area under PR) is threshold-free and standard for detection (mAP).

## Q: What's the difference between parametric and non-parametric models? {diff=1 tags=fundamentals,models,rapid-fire}
### A
**A parametric model has a fixed number of parameters independent of dataset size; a non-parametric model's capacity grows with the data.**

Linear/logistic regression and a fixed-size neural network are parametric: once trained, the data can be discarded. k-NN, kernel density estimation, Gaussian processes, and fully grown trees are non-parametric: the training set (or something scaling with it) *is* the model, so inference cost grows with $N$ and they can fit any function given enough data — at the price of the curse of dimensionality.

Trap: "non-parametric" doesn't mean "no parameters" — a GP has kernel hyperparameters. And an LLM is parametric, but retrieval-augmented generation bolts a non-parametric memory onto it — exactly why RAG adds new facts without retraining.

## Q: What is the manifold hypothesis? {diff=1 tags=fundamentals,geometry,generative,rapid-fire}
### A
**Natural high-dimensional data — images, audio, video — concentrate near a low-dimensional manifold embedded in pixel space; almost all of $\mathbb{R}^d$ is noise no one would call an image.**

Evidence: interpolating in a good model's latent space gives valid images while pixel interpolation ghosts; the spectrum of image patches decays fast; VAEs compress 512×512×3 to 64×64×16 with little loss. It's why generative modeling is possible — learn the manifold and a density on it — and why the score is meaningful: it points from off-manifold noise back toward data.

Trap: the manifold has varying intrinsic dimension and awkward topology, which is why a single-Gaussian latent struggles and diffusion, which assumes no global chart, beats a plain VAE. It also explains adversarial examples: tiny off-manifold moves are unlike anything seen in training.
