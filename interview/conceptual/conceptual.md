# Interview prep — conceptual question bank


## ML Fundamentals


### [01-01] Walk me through the bias-variance decomposition. Where do modern overparameterized networks sit on that curve, and does the classic picture still hold?  (difficulty 1; tags: fundamentals, generalization)

- Follow-up: What's double descent and why does it happen?


<details><summary>Answer</summary>

**Expected test MSE decomposes as bias$^2$ + variance + irreducible noise**: $\mathbb{E}[(y - \hat f(x))^2] = (f^{\ast}(x) - \mathbb{E}[\hat f(x)])^2 + \mathrm{Var}[\hat f(x)] + \sigma^2$. Bias is how far the *average* model (over training sets) is from the truth; variance is how much the fit wobbles across training sets.

- **High bias** = underfitting: train and val error both high, close together. Fix: bigger model, more features, train longer, less regularization.
- **High variance** = overfitting: train error low, val error much higher. Fix: more data, regularization, augmentation, ensembling, early stopping.

The derivation is just adding and subtracting $\mathbb{E}[\hat f]$ and noting the cross term vanishes because $\mathbb{E}[\hat f - \mathbb{E}\hat f] = 0$.

**Modern caveat (double descent)**: the classic U-shaped test curve holds up to the interpolation threshold (params ≈ samples), where variance spikes because the model is forced to fit noise with a nearly unique interpolant. Past that, with more parameters (or implicit regularization from SGD / min-norm solutions), variance *drops again* and test error keeps improving. So "bigger model overfits more" is not a safe assumption for deep nets; scaling-law regimes are on the right side of the peak. Same story in the data axis: adding data near the threshold can temporarily hurt.

Trap: bias-variance is about a fixed training-set size and an estimator; it says nothing about distribution shift, which is a separate failure mode.

</details>


### [01-02] Show me why minimizing cross-entropy is the same as maximum likelihood, and how KL divergence fits in.  (difficulty 1; tags: fundamentals, probability, loss)

- Follow-up: Which direction of KL is it, and why does that matter?


<details><summary>Answer</summary>

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

</details>


### [01-03] Why is the gradient of softmax + cross-entropy with respect to the logits just $(p - y)$? Derive it.  (difficulty 1; tags: fundamentals, backprop, loss)


<details><summary>Answer</summary>

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

</details>


### [01-04] Is logistic regression convex? Prove it, and tell me what that buys you in practice.  (difficulty 2; tags: fundamentals, optimization, convexity)

- Follow-up: Does it have a unique minimizer?


<details><summary>Answer</summary>

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

</details>


### [01-05] What's the difference between L2 regularization and weight decay, and why does it matter for Adam? What does AdamW actually change?  (difficulty 3; tags: optimization, regularization, adam)

- Follow-up: How would you set weight decay for a 1B-param transformer?


<details><summary>Answer</summary>

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

</details>


### [01-06] Write down the update rules for SGD, SGD with momentum, and Adam. When does Adam hurt compared to SGD, and why?  (difficulty 2; tags: optimization, adam, sgd)

- Follow-up: What is Adam's $\epsilon$ for, and what happens if it's too small in bf16?


<details><summary>Answer</summary>

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

</details>


### [01-07] Why do transformers need learning-rate warmup? And why is cosine decay the default afterwards?  (difficulty 2; tags: optimization, schedule, transformer)

- Follow-up: What's the relationship between warmup and pre-norm vs post-norm?


<details><summary>Answer</summary>

**Warmup keeps the early steps small while Adam's second-moment estimate and the network's activation scales are still unreliable; cosine gives a smooth anneal that lands at a low-noise solution.**

Why warmup specifically for transformers:
- At init, attention logits and residual streams have poorly calibrated scale; a few large updates can push softmax into saturation (attention entropy collapse) or blow up post-norm residuals, from which training doesn't recover.
- Adam's $v$ is bias-corrected but still high-variance over the first $\sim 1/(1-\beta_2) \approx 1000$ steps; without warmup the effective step is far larger than intended in low-gradient coordinates.
- Post-norm transformers (original Vaswani) have gradients that scale badly with depth at init and *cannot* train without warmup; pre-norm is far more tolerant (Xiong et al. 2020), which is why deep models moved to pre-norm and why warmup got shorter but never zero.
- Typical: linear warmup over 1–2% of steps (2k–10k steps for LLMs; 1k–5k for DiTs), peak lr ~1e-4 for 1B-scale diffusion transformers, 3e-4 to 6e-4 for small LLMs, decreasing with model size (roughly $\propto 1/\sqrt{\text{width}}$ under standard param, constant under $\mu$P).

Cosine decay: $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$. Smooth, no tuning of step milestones, and the slow tail lets the iterate settle into a lower-loss region as gradient noise shrinks (the annealing is essentially reducing SGD temperature $\propto \eta/\text{batch}$). Downsides: you must know $T$ in advance; changing the budget mid-run means re-scheduling — which is why WSD (warmup–stable–decay) and "constant + short cooldown" schedules are popular now: they give equivalent final loss with a decay phase of ~10–20% of steps and allow branching checkpoints.

Trap: warmup length should scale with batch size (larger batch → larger lr → longer warmup), not with dataset size.

</details>


### [01-08] Compare BatchNorm, LayerNorm, and RMSNorm. Why do transformers use LayerNorm or RMSNorm rather than BatchNorm?  (difficulty 2; tags: normalization, transformer, architecture)

- Follow-up: What goes wrong with BatchNorm under DDP or with batch size 2?


<details><summary>Answer</summary>

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

</details>


### [01-09] Explain Xavier and He initialization. Why does depth require careful init, and how do residual networks change the story?  (difficulty 2; tags: initialization, architecture, depth)

- Follow-up: Why do GPT-2 / DiT scale the output projections by $1/\sqrt{2L}$ or zero them?


<details><summary>Answer</summary>

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

</details>


### [01-10] What causes vanishing and exploding gradients, and how do you diagnose and fix them? What does gradient clipping actually do?  (difficulty 2; tags: optimization, stability, training)

- Follow-up: Clip by norm or by value? What threshold, and what do you monitor?


<details><summary>Answer</summary>

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

</details>


### [01-11] You're training a model and val loss starts rising while train loss keeps dropping. Walk me through your diagnostics and regularization options, and when each one actually helps.  (difficulty 1; tags: generalization, regularization, training)

- Follow-up: Does dropout help transformers / diffusion models?


<details><summary>Answer</summary>

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

</details>


### [01-12] What evaluation pitfalls have you seen bite people — data leakage, val vs test, metric choice? Give concrete examples in generative modeling.  (difficulty 3; tags: evaluation, methodology, generative)

- Follow-up: How do you compare two image generators fairly?


<details><summary>Answer</summary>

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

</details>


### [01-13] What's the relationship between PCA and SVD? Give me the low-rank intuition and connect it to why LoRA works.  (difficulty 2; tags: linear-algebra, pca, lora)

- Follow-up: What's the optimal rank-k approximation and how do you pick the rank?


<details><summary>Answer</summary>

**PCA is the SVD of the centered data matrix; the principal components are the right singular vectors and the variances are the squared singular values divided by $n - 1$.**

For centered $X \in \mathbb{R}^{n \times d}$, the covariance is $C = X^\top X/(n-1)$. If $X = U \Sigma V^\top$, then $C = V (\Sigma^2/(n-1)) V^\top$, so the eigenvectors of $C$ are the columns of $V$ and eigenvalues are $\sigma_i^2/(n-1)$. Projecting onto the top-$k$ components gives $X V_k = U_k \Sigma_k$ — the scores. Computing via SVD avoids forming $X^\top X$ (squares the condition number) and is what every library does.

**Eckart–Young–Mirsky**: the best rank-$k$ approximation of any matrix in Frobenius or spectral norm is the truncated SVD, $X_k = U_k \Sigma_k V_k^\top$, with error $\|X - X_k\|_F^2 = \sum_{i>k} \sigma_i^2$. So the singular value spectrum tells you how compressible a matrix is: a fast-decaying spectrum ⇒ a few directions capture most of the energy. Pick $k$ by explained-variance threshold (e.g., 95%) or by a knee in the scree plot.

**Connection to LoRA**: LoRA freezes $W_0$ and learns $\Delta W = B A$, with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll d$ ($r$ = 4–64 for a 4096-wide layer, ~0.1–1% of the params). The hypothesis (Aghajanyan et al.; Hu et al.) is that fine-tuning updates have low *intrinsic rank* — the singular values of the full-fine-tuning $\Delta W$ decay fast, so a rank-$r$ factor captures most of it. Empirically $\Delta W$ from full fine-tuning on a task has effective rank in the single digits to tens even in 4k-wide matrices. It's the same statement as "top-$k$ SVD captures the data" but applied to the *update*.

Practical notes: $B$ is zero-initialized ($\Delta W = 0$ at start), $A$ random (Kaiming); scale by $\alpha/r$; merge into $W_0$ at inference (no latency cost). LoRA on attention $q, v$ projections is the classic choice; on DiTs for SPEED/foveated diffusion, all attention + MLP projections at rank 16–64 is typical.

Trap: PCA on un-centered data finds the mean direction as component 1; and PCA is rotation of coordinates, not a nonlinear embedding — it can't uncover a curved manifold.

</details>


### [01-14] Quickly: what is EM, and why is k-means a special case of it?  (difficulty 3; tags: probability, em, clustering)

- Follow-up: Why does EM never decrease the likelihood? Where is EM used in diffusion / generative modeling today?


<details><summary>Answer</summary>

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

</details>


### [01-15] What's the difference between a generative and a discriminative model?  (difficulty 1; tags: fundamentals, generative, rapid-fire)


<details><summary>Answer</summary>

**A discriminative model learns $p(y \mid x)$ — the decision boundary; a generative model learns $p(x)$ or the joint $p(x, y)$, so it can sample new $x$.**

Classifiers, depth and flow estimators are discriminative: they only need to separate classes and spend no capacity modeling the input. VAEs, diffusion models, and LLMs are generative: they model the data distribution, so they can sample, score likelihoods, and act as priors (e.g. a diffusion prior for inverse problems).

Trap: a generative model can be used discriminatively via Bayes, $p(y \mid x) \propto p(x \mid y)\, p(y)$, but with enough labels it usually loses to a direct discriminative model — it solves a harder problem than the task asks. The reverse is impossible: a discriminative model can't generate.

</details>


### [01-16] What's the difference between a likelihood and a probability?  (difficulty 1; tags: fundamentals, probability, rapid-fire)


<details><summary>Answer</summary>

**Same function, opposite argument: $p(x \mid \theta)$ is the probability of data $x$ with $\theta$ fixed, and the likelihood $L(\theta; x)$ of parameters $\theta$ with $x$ fixed.**

Probability integrates to 1 over $x$; the likelihood does *not* integrate to 1 over $\theta$ — it isn't a distribution over parameters. Maximum likelihood picks the $\theta$ under which the observed data are most probable. To get a real distribution over $\theta$ you need a prior: $p(\theta \mid x) \propto p(x \mid \theta)\, p(\theta)$.

Trap: for continuous data the likelihood is a density, so it can exceed 1 and its log can be positive — bits-per-dim of an image model has an arbitrary scale, and comparing likelihoods across different data discretizations or scalings is meaningless.

</details>


### [01-17] Entropy, cross-entropy, and KL divergence — define all three in one breath.  (difficulty 1; tags: fundamentals, information-theory, rapid-fire)


<details><summary>Answer</summary>

**Entropy $H(p) = -\mathbb{E}_p[\log p]$ is the average surprise under the true distribution; cross-entropy $H(p, q) = -\mathbb{E}_p[\log q]$ is the average surprise when you code $p$'s samples with $q$; $\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$ is the excess.**

So KL is the price in nats of using the wrong model: $\ge 0$ by Jensen, zero iff $p = q$. Minimizing cross-entropy over $q$ equals minimizing $\mathrm{KL}(p \,\|\, q)$ because $H(p)$ is constant — that's why the classification loss is called cross-entropy and equals MLE.

Trap: KL is asymmetric. $\mathrm{KL}(\text{data} \,\|\, \text{model})$, the MLE direction, is mass-covering; $\mathrm{KL}(\text{model} \,\|\, \text{data})$, the variational direction, is mode-seeking. And KL is infinite wherever $q = 0$ but $p > 0$.

</details>


### [01-18] What is the reparameterization trick and why is it needed?  (difficulty 1; tags: fundamentals, vae, gradients, rapid-fire)


<details><summary>Answer</summary>

**Write a sample as a deterministic function of the parameters plus parameter-free noise, $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, so gradients flow through the sampling step into $\mu$ and $\sigma$.**

It's needed because $\nabla_\theta \mathbb{E}_{z \sim q_\theta}[f(z)]$ can't be obtained by sampling $z$ and differentiating $f$ — the sampling itself depends on $\theta$. Reparameterization moves $\theta$ out of the distribution and into the integrand, $\nabla_\theta \mathbb{E}_\epsilon[f(\mu + \sigma \epsilon)]$, a low-variance pathwise estimator. The alternative, REINFORCE, works for discrete $z$ but has far higher variance.

Trap: it needs a differentiable sampling map (Gaussian, location-scale families); for categorical latents you use Gumbel-softmax or straight-through — which is why VQ-VAE uses a straight-through estimator.

</details>


### [01-19] What is the curse of dimensionality?  (difficulty 1; tags: fundamentals, geometry, rapid-fire)


<details><summary>Answer</summary>

**In high dimensions volume grows exponentially, so data become sparse, distances concentrate, and any "nearby points" method needs exponentially many samples.**

Numbers: in $d$ dimensions almost all of a ball's volume sits in a thin shell at its surface; nearest and farthest neighbor distances become nearly equal, so k-NN and kernel methods lose signal; covering $[0,1]^d$ at resolution 0.1 needs $10^d$ points. A Gaussian sample sits at radius $\approx \sqrt{d}$, not near the mean — which is why $x_T$ in diffusion lives on a sphere of radius $\sqrt{d}$, not "near zero".

Why deep learning escapes it: real data lie near a low-dimensional manifold and networks exploit compositional structure, so the effective dimension is far below the ambient one.

</details>


### [01-20] What is a convex function, and why does convexity matter for optimization?  (difficulty 1; tags: fundamentals, optimization, convexity, rapid-fire)


<details><summary>Answer</summary>

**A function is convex if every chord lies above the graph, $f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y)$; equivalently the Hessian is positive semidefinite.**

Why it matters: every local minimum is global, gradient descent with a suitable step provably converges, and the set of minimizers is convex — no worries about initialization or getting stuck. Linear regression, logistic regression, and SVMs are convex in their parameters; that's the source of their guarantees.

Trap: neural networks are non-convex in the weights (permutation symmetry alone makes minima non-unique), so none of the guarantees hold — SGD works because the landscape is benign in practice. Convexity of the loss in the *output* (cross-entropy in the logits) still helps: it's why the softmax + CE gradient is well-behaved.

</details>


### [01-21] What's the difference between bagging and boosting?  (difficulty 1; tags: fundamentals, ensembles, rapid-fire)


<details><summary>Answer</summary>

**Bagging trains independent models on bootstrap resamples and averages them to reduce variance; boosting trains models sequentially, each fitting the residual errors of the ensemble so far, to reduce bias.**

Random forests are bagging plus random feature subsets: each tree overfits, but averaging decorrelated overfits cancels their variance. Gradient boosting (XGBoost, LightGBM) adds shallow trees along the negative gradient of the loss — functional gradient descent — turning weak, high-bias learners into a strong one.

Rule: bagging is embarrassingly parallel and hard to overfit; boosting is sequential and *can* overfit with too many rounds, so it needs shrinkage and early stopping. Deep-learning analogues: ensembles and EMA weight averaging are bagging-like; there's no common boosting analogue because a large net already has low bias.

</details>


### [01-22] Define precision, recall, and F1 from a confusion matrix, and tell me when accuracy lies.  (difficulty 1; tags: fundamentals, evaluation, metrics, rapid-fire)


<details><summary>Answer</summary>

**The confusion matrix counts TP, FP, FN, TN. Precision $= \mathrm{TP} / (\mathrm{TP} + \mathrm{FP})$: of what I flagged, how much was right. Recall $= \mathrm{TP} / (\mathrm{TP} + \mathrm{FN})$: of what was there, how much I caught. $\mathrm{F1} = 2 P R / (P + R)$, the harmonic mean, punishing whichever is low.**

Accuracy $= (\mathrm{TP} + \mathrm{TN}) / \text{total}$ lies under class imbalance: with 1% positives, always predicting "negative" scores 99% with zero recall. It also hides *which* error you make, and errors have different costs (missed tumor vs false alarm). Use PR curves or AUROC to see the threshold trade-off; report per-class metrics.

Trap: F1 ignores true negatives and depends on a threshold; average precision (area under PR) is threshold-free and standard for detection (mAP).

</details>


### [01-23] What's the difference between parametric and non-parametric models?  (difficulty 1; tags: fundamentals, models, rapid-fire)


<details><summary>Answer</summary>

**A parametric model has a fixed number of parameters independent of dataset size; a non-parametric model's capacity grows with the data.**

Linear/logistic regression and a fixed-size neural network are parametric: once trained, the data can be discarded. k-NN, kernel density estimation, Gaussian processes, and fully grown trees are non-parametric: the training set (or something scaling with it) *is* the model, so inference cost grows with $N$ and they can fit any function given enough data — at the price of the curse of dimensionality.

Trap: "non-parametric" doesn't mean "no parameters" — a GP has kernel hyperparameters. And an LLM is parametric, but retrieval-augmented generation bolts a non-parametric memory onto it — exactly why RAG adds new facts without retraining.

</details>


### [01-24] What is the manifold hypothesis?  (difficulty 1; tags: fundamentals, geometry, generative, rapid-fire)


<details><summary>Answer</summary>

**Natural high-dimensional data — images, audio, video — concentrate near a low-dimensional manifold embedded in pixel space; almost all of $\mathbb{R}^d$ is noise no one would call an image.**

Evidence: interpolating in a good model's latent space gives valid images while pixel interpolation ghosts; the spectrum of image patches decays fast; VAEs compress 512×512×3 to 64×64×16 with little loss. It's why generative modeling is possible — learn the manifold and a density on it — and why the score is meaningful: it points from off-manifold noise back toward data.

Trap: the manifold has varying intrinsic dimension and awkward topology, which is why a single-Gaussian latent struggles and diffusion, which assumes no global chart, beats a plain VAE. It also explains adversarial examples: tiny off-manifold moves are unlike anything seen in training.

</details>


## Deep Learning Architectures


### [02-01] Derive a transformer block for me from scratch: the attention formula, why the $1/\sqrt{d}$ scaling, and what multi-head buys you.  (difficulty 1; tags: transformer, attention, architecture)

- Follow-up: What are the shapes at every step for $B=1$, $N=1024$ tokens, $D=1024$, 16 heads?


<details><summary>Answer</summary>

**A block is $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$; $x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x))$; attention is a data-dependent weighted average of value vectors, with weights from a softmax over query-key dot products scaled by $1/\sqrt{d_k}$.**

For $X \in \mathbb{R}^{N \times D}$:
$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V & (\text{each } D \times d_k,\ d_k = D/H) \\
A &= \operatorname{softmax}\!\left(Q K^\top / \sqrt{d_k}\right) V & (N \times N \text{ scores},\ N \times d_k \text{ output})
\end{aligned}
$$
Multi-head: do this $H$ times in parallel with separate projections, concat along features ($N \times D$), apply $W_O \in \mathbb{R}^{D \times D}$.

**Why $1/\sqrt{d_k}$**: if $q, k$ have i.i.d. unit-variance entries, $q^\top k$ has variance $d_k$ (sum of $d_k$ products), so logits have std $\sqrt{d_k}$. With $d_k = 64$ that's std 8 — the softmax saturates to near one-hot, gradients through it vanish ($\partial\,\mathrm{softmax} = p(1-p)$-like), and training stalls. Dividing by $\sqrt{d_k}$ restores unit-variance logits. Related: QK-norm (RMSNorm on $q$ and $k$ before the dot product) bounds logits by a learned scale and prevents attention-logit growth in large models.

**Why multi-head**: each head is a low-rank ($d_k$) attention pattern; $H$ heads let the layer attend to $H$ different relational structures at once (positional, syntactic, copying) for the same total FLOPs as one wide head — a single head with $d_k = D$ would still produce one $N \times N$ attention matrix. Empirically 8–16 heads with $d_k$ = 64–128 is the sweet spot; more, tinier heads lose expressivity.

MLP: $D \to 4D \to D$ with GELU (or SwiGLU with $\tfrac{8}{3} D$ hidden). Per-layer params $\approx 4D^2$ (attention) $+ 8D^2$ (MLP) $= 12D^2$.

Shapes for the follow-up: $X$ 1024×1024; $Q, K, V$ each 16 heads × 1024 × 64; scores 16 × 1024 × 1024 (16M floats); output back to 1024×1024.

</details>


### [02-02] Self-attention vs cross-attention — what's the difference, and where does cross-attention appear in text-to-image models? Where has it disappeared?  (difficulty 1; tags: attention, text-to-image, diffusion)

- Follow-up: Why did FLUX/SD3 drop cross-attention for joint attention?


<details><summary>Answer</summary>

**Self-attention lets tokens in one sequence attend to each other ($Q, K, V$ all from $X$); cross-attention lets one sequence query another ($Q$ from $X$, $K$ and $V$ from a context $C$).**

Cross-attn: $A = \operatorname{softmax}\!\left(X W_Q (C W_K)^\top / \sqrt{d}\right) (C W_V)$. Its cost is $O(N \cdot M)$ for $N$ image tokens and $M$ context tokens — with $M = 77$ CLIP tokens that's negligible next to $O(N^2)$ self-attention.

Where it appears in T2I:
- **U-Net models (SD 1.x / 2.x / SDXL)**: every transformer block in the U-Net has self-attn over image tokens *then* cross-attn to the CLIP text embeddings (77×768 or 77×2048). The text is a fixed, non-updated context; only image features change through depth. Prompt-to-prompt / attention-map editing works by manipulating these cross-attn maps.
- **PixArt-α, early DiT-based T2I**: DiT blocks with self-attn + an added cross-attn to T5 text embeddings, timestep via adaLN.
- Class-conditional DiT has no cross-attn at all — class and timestep go in through adaLN.

Where it disappeared: **MMDiT (SD3, FLUX)** concatenates text and image tokens into one sequence and runs *joint self-attention* — each modality has its own Q/K/V/MLP weights (dual-stream) but they share one attention matrix. Text tokens are now updated layer by layer and can attend back to the image, which improved text rendering and compositional binding. FLUX uses 19 dual-stream blocks followed by 38 single-stream blocks where text and image share all weights. WAN 2.1 goes the other way: T5 text enters via cross-attention in every block (cheaper for 32k video tokens, and text doesn't need image feedback).

Practical consequence for efficiency work: with cross-attn, text cost is fixed; with joint attention, text tokens (512 T5 tokens in FLUX) add to the quadratic term — but at 4k image tokens it's a minor (~25%) overhead.

</details>


### [02-03] Compare positional encodings: sinusoidal, learned, and RoPE. How exactly does RoPE encode relative position, and how do you extend it to 2D images and 3D video?  (difficulty 2; tags: positional-encoding, rope, transformer, video)

- Follow-up: What breaks when you generate at a resolution larger than training?


<details><summary>Answer</summary>

**Sinusoidal and learned encodings add an absolute position vector to the input; RoPE rotates $q$ and $k$ by position-dependent angles so the dot product depends only on the relative offset, and it composes per axis for images and video.**

- Sinusoidal (Vaswani): $\mathrm{PE}(\text{pos}, 2i) = \sin\!\left(\text{pos} / 10000^{2i/D}\right)$, cos for odd dims. Fixed, extrapolates in principle, but the model never learns to use unseen positions well.
- Learned absolute (BERT, ViT, original DiT): a table of $N \times D$ parameters; no extrapolation at all — ViT interpolates the 2D grid when changing resolution.
- **RoPE** (Su et al.): split the head dim into $d/2$ pairs; for pair $i$ with frequency $\theta_i = 10000^{-2i/d}$, rotate $(q_{2i}, q_{2i+1})$ by angle $m \theta_i$ for token at position $m$:
  $q'_m = R(m\theta)\, q$, $k'_n = R(n\theta)\, k$.
  Because rotations compose, $q'^{\top}_m k'_n = q^\top R((n - m)\theta)\, k$ — **the score depends only on $n - m$**. Absolute position is applied, relative position is what attention sees. It's applied to $q$ and $k$ only (not $v$), inside every layer, and costs nothing in parameters. Low-frequency pairs give long-range smooth positional signal; high-frequency pairs resolve adjacent tokens.

2D / 3D extension (FLUX, WAN, most video DiTs): partition the head dim into axis groups — e.g., for 3D, split $d = 128$ into (t: 32, h: 48, w: 48) or FLUX's (idx: 16, h: 56, w: 56) — and apply 1D RoPE per group with that axis's coordinate. The score then decomposes as a sum of per-axis relative terms. Text tokens in MMDiT get position $(0,0,0)$ or their own 1D axis. Video latents from WAN at 480p × 81 frames: grid (21, 30, 52), each token gets $(t, h, w)$ integer coordinates.

What breaks beyond training resolution: unseen positions produce rotation angles the model never saw in the *low-frequency* pairs (the high-frequency ones wrap around and are fine). Attention gets diffuse or repeats structure (duplicated subjects). Fixes: NTK-aware / YaRN scaling of $\theta$ base (interpolate the low frequencies), position interpolation (scale coordinates back into the training range), or — the SPEED insight — spend most steps at native resolution and only touch high-res in the final stage.

</details>


### [02-04] How does ViT patchify work, and why does patch size matter so much? Give me the token counts.  (difficulty 1; tags: vit, tokenization, efficiency)

- Follow-up: What's the patch size in latent diffusion transformers and why is it 2?


<details><summary>Answer</summary>

**Patchify is a strided conv (or reshape + linear) that turns an $H \times W \times C$ image into $(H/p)(W/p)$ tokens of dimension $D$; token count scales as $1/p^2$, so patch size sets both cost and the finest spatial detail the model can address.**

Mechanics: `nn.Conv2d(C, D, kernel=p, stride=p)` — each $p \times p \times C$ patch (e.g., 16×16×3 = 768 values) is linearly projected to $D$. Add positional embedding, optionally a [CLS] token. Un-patchify at the output for dense prediction / diffusion.

Token counts:
- ViT-B/16 at 224²: 14 × 14 = 196 tokens. ViT-L/14 at 224²: 256 tokens; at 336²: 576.
- Attention cost per layer $\propto N^2 D$ and MLP $\propto N D^2$, so halving $p$ quadruples $N$ and gives **16× attention FLOPs** and 4× MLP FLOPs. ViT-B/8 is far better than /16 at same resolution but ~4–16× the compute.
- Latent DiTs: SD-VAE compresses 8× spatially into 4 (or 16 for FLUX/SD3) channels; DiT uses **$p = 2$** on the latent. 256² image → 32² latent → 16² = 256 tokens; 1024² → 128² latent → **4096 tokens** (FLUX). WAN 2.1's video VAE is 4× temporal, 8× spatial, 16 channels; with its (1,2,2) patchify, 480p×81 frames → 21×30×52 ≈ **32,760 tokens**.

Why patch size matters beyond cost:
- Sets the effective output resolution of the transformer — $p = 2$ on latents means each token owns a 16×16 pixel region; going to $p = 1$ would give 4× tokens for marginal quality gain (DiT paper: FID improves steadily as $p$ shrinks, at proportional Gflops).
- Larger patches = stronger compression of local detail into one vector at the very first layer, lost forever — hence "patchify stem" ideas (early conv stages) and, in this candidate's work, *mixed* patch sizes: foveated diffusion uses fine patches in the fovea and coarse ones in the periphery to cut token count 2–4× while keeping local detail where it matters.
- Scaling laws in ViT are cleaner in terms of *tokens* than pixels.

</details>


### [02-05] DiT vs U-Net for diffusion: how does a DiT condition on timestep, class, and text? Explain adaLN-Zero and the MMDiT dual-stream design.  (difficulty 2; tags: diffusion, dit, architecture, conditioning)

- Follow-up: Why zero-init the gate, and why is adaLN better than cross-attention or token concatenation for the timestep?


<details><summary>Answer</summary>

**A DiT is a plain ViT over latent patches whose LayerNorms are modulated by the conditioning vector (adaLN); U-Nets condition by adding a timestep embedding into every residual block and cross-attending to text. MMDiT extends DiT by treating text as a second token stream with its own weights and joint attention.**

adaLN-Zero (Peebles & Xie):
$$
\begin{aligned}
c &= \mathrm{MLP}(t_{\text{emb}} + y_{\text{emb}}) & (\text{conditioning vector, shape } D) \\
(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) &= \mathrm{Linear}(\mathrm{SiLU}(c)) & (6D \text{ outputs per block}) \\
h &= x + \alpha_1 \cdot \mathrm{Attn}\big(\mathrm{LN}(x) \cdot (1 + \gamma_1) + \beta_1\big) \\
x &= h + \alpha_2 \cdot \mathrm{MLP}\big(\mathrm{LN}(h) \cdot (1 + \gamma_2) + \beta_2\big)
\end{aligned}
$$
- $\gamma, \beta$ replace LayerNorm's fixed affine with a *conditioning-dependent* scale and shift — like FiLM / StyleGAN's AdaIN. The whole network's feature statistics change with $t$.
- **$\alpha$ is the residual gate, initialized to zero** (the final Linear is zeroed): every block is the identity at init, matching the residual-scaling story — gives markedly better FID than adaLN alone or cross-attention/in-context conditioning at the same FLOPs.
- Cost: ~$6D^2$ extra params per block for the modulation linear (a noticeable fraction; PixArt shares one adaLN across blocks with per-block learned offsets to cut it).

Why adaLN for $t$ rather than a token or cross-attn: the timestep is a *global*, low-dimensional signal that should affect every token identically — a per-token multiplicative modulation is the cheapest and most direct route; a token would have to be read through attention at every layer. Text, by contrast, is a *set* of tokens with spatial correspondence, so it goes through attention.

**MMDiT (SD3, FLUX)**: text tokens (T5 + CLIP pooled) and image tokens get *separate* Q/K/V, output, and MLP weights, but their Q/K/V are concatenated and one joint attention is computed. Both streams are updated, and both are adaLN-modulated by ($t$, pooled text). Rationale: the two modalities have very different statistics, so sharing weights hurt; but joint attention lets text refine based on the image. FLUX: 19 dual-stream blocks + 38 single-stream (shared weights, parallel attn+MLP) blocks, $D = 3072$, 24 heads, ~12B params. Guidance-distilled FLUX.1-dev also feeds the CFG scale through the timestep embedding.

U-Net contrast: convolutional inductive bias, multi-resolution skip connections, fewer tokens at the bottleneck; scales worse with compute (DiT paper shows Gflops → FID is cleaner for transformers) and has no clean path to video/3D joint attention. Modern video (Sora, WAN, CogVideoX, HunyuanVideo) are all DiT-family.

</details>


### [02-06] Why do residual connections work, and what's the difference between pre-norm and post-norm? Which would you pick for a 40-layer DiT and why?  (difficulty 2; tags: architecture, normalization, depth)

- Follow-up: What is the downside of pre-norm at scale, and what's "sandwich norm" / peri-LN?


<details><summary>Answer</summary>

**Residuals turn a deep composition into a sum of shallow paths, keeping an identity gradient path so depth doesn't multiply Jacobians; pre-norm puts the normalization inside the branch, keeping that identity path clean, so it's the choice for deep models.**

Residuals: $x_{l+1} = x_l + f_l(x_l)$. Gradient $\partial L/\partial x_l = \partial L/\partial x_{l+1} \cdot (I + \partial f/\partial x)$ — the "$I$" term guarantees gradient reaches every layer at least unattenuated. Unrolled, the network is an ensemble of $2^L$ paths dominated by short ones (Veit et al.), which is why removing a single block barely hurts a trained ResNet. Also enables the "each block is a small update to a shared stream" view that makes zero-init and layer-drop work.

- **Post-norm** (original transformer): $x \leftarrow \mathrm{LN}(x + f(x))$. Normalization is *on* the residual path; gradient must pass through every LN, whose Jacobian scales like $1/\|x\|$ — at init this makes gradients in early layers large and the model needs warmup + small lr. Better final performance in some shallow settings (BERT-base) because the stream is re-normalized.
- **Pre-norm**: $x \leftarrow x + f(\mathrm{LN}(x))$. Identity path untouched; trains stably without warmup, allows larger lr; standard in GPT-2+, ViT, DiT, LLaMA, FLUX.

Pre-norm downsides at scale: the residual stream's norm grows with depth ($\propto \sqrt{L}$ or more), so later blocks' contributions $f(\mathrm{LN}(x))$ become relatively small — deep layers are under-utilized ("representation collapse" / effectively shallower net). Remedies: **sandwich norm** (norm both input and output of the branch — CogView, Gemma), peri-LN, DeepNorm (scale residual by $\alpha > 1$, init branch by $\beta < 1$), QK-norm to stop attention logit growth that pre-norm leaves unchecked.

For a 40-layer DiT: **pre-norm with adaLN-Zero gating, plus QK-norm and RMSNorm** — that's what FLUX, SD3, and WAN do; zero-gated branches make depth trainable from step 0, and QK-norm prevents the attention-logit divergence that shows up around 1–10B params in bf16. Post-norm at this depth without warmup would not train.

</details>


### [02-07] Give me the basics of mixture-of-experts: routing, why it's efficient, and how load balancing works. What breaks without it?  (difficulty 2; tags: moe, architecture, scaling)

- Follow-up: How would you apply MoE to a diffusion transformer?


<details><summary>Answer</summary>

**MoE replaces the dense MLP with $E$ parallel expert MLPs and a router that sends each token to the top-$k$ experts; parameters scale with $E$ but per-token FLOPs scale with $k$, decoupling capacity from compute.**

Mechanics (Switch / Mixtral style):
$$
\begin{aligned}
g &= \operatorname{softmax}(x W_r) & (\text{router logits over } E \text{ experts}) \\
\text{idx} &= \operatorname{topk}(g, k) & (k = 1 \text{ or } 2) \\
y &= \sum_{e \in \text{idx}} g_e \cdot \mathrm{Expert}_e(x)
\end{aligned}
$$
Mixtral 8×7B: $E = 8$, $k = 2$, ~47B total params but ~13B active per token — dense-13B compute with much more knowledge capacity. DeepSeek-V3: 256 fine-grained experts, $k = 8$, plus a shared always-on expert.

Why efficient: MLP is ~2/3 of transformer FLOPs; with MoE, each expert sees $N k/E$ tokens, so with a fixed compute budget you can grow total parameters 5–10× for the same training FLOPs, and scaling laws show substantially lower loss at iso-FLOPs. Cost: memory (all experts resident), all-to-all communication for expert parallelism, and complexity.

**Load balancing**: the router has a rich-get-richer feedback loop — a slightly favored expert gets more gradient, gets better, gets picked more — until a few experts take all tokens and the rest are dead; also hardware needs roughly equal tokens per expert per device or the busiest expert becomes the straggler. Remedies:
- Auxiliary balance loss (Switch): $L_{\text{aux}} = \alpha \cdot E \cdot \sum_e f_e \, P_e$, where $f_e$ = fraction of tokens routed to $e$, $P_e$ = mean router probability for $e$; minimized when uniform; $\alpha \approx 0.01$.
- Capacity factor: each expert accepts at most $C \cdot N/E$ tokens per batch ($C \approx 1.25$); overflow tokens are dropped (pass through residual only) or re-routed.
- Router z-loss to keep logits small; noisy top-$k$ gating during training; auxiliary-loss-free balancing with per-expert bias terms (DeepSeek-V3).

What breaks without it: expert collapse (effective model = dense small model), wasted parameters, and pathological all-to-all imbalance where one GPU's expert is 4× busier than others, throttling the whole step.

Applying to DiTs: route per image/video token in the MLP (DiT-MoE), or route per *timestep range* — different denoising phases want different experts (eDiff-I's ensemble of denoisers is the coarse-grained version). Trade: latency-critical inference loves fewer active params, but memory footprint grows.

</details>


### [02-08] What inductive biases does a convolution have that attention lacks, and when does that matter? Are ViTs "worse" at small data?  (difficulty 2; tags: cnn, attention, inductive-bias)

- Follow-up: How does a ViT recover locality, and what does DINO show about learned attention?


<details><summary>Answer</summary>

**Convolutions bake in locality, translation equivariance, and weight sharing across positions; attention assumes none of these and must learn them from data — which costs samples but removes a ceiling.**

Conv inductive biases:
- **Locality**: each output depends on a $k \times k$ neighborhood; receptive field grows linearly with depth (or via dilations/pooling).
- **Translation equivariance**: shift the input, output shifts; the same detector runs everywhere (weight sharing ⇒ far fewer params, e.g., $3 \times 3 \times C \times C$ per layer independent of image size).
- Hierarchical multi-scale structure via pooling/striding — good for objects at many scales, natural for U-Nets.

Attention: global receptive field in one layer, content-dependent (dynamic) weights, permutation-equivariant with position given only through embeddings. It can express a conv (a head can learn to attend to a fixed relative offset — Cordonnier et al.) but has to learn it.

When it matters:
- **Small data** (ImageNet-1k from scratch, 1.3M images): ViT-B underperforms ResNet-50 without heavy augmentation/regularization (DeiT recipe closes the gap); with 300M images (JFT) ViT wins decisively. The crossover is roughly 10–100M images, and pretraining (MAE, DINO, CLIP) moves it much lower.
- **Dense prediction & high-res**: conv/hybrid stems, Swin's windowed attention, and ConvNeXt show that with matched training recipes the two families are close; the "transformer advantage" is mostly scaling behavior and multimodal flexibility (joint attention over text + image + video tokens), not raw image accuracy.
- **Generative**: U-Nets' locality is great for texture; DiTs win on global coherence, text, and compute scaling. Many video DiTs keep a small conv in the VAE and patch embedding.

Recovering locality in ViTs: 2D RoPE / relative position bias (Swin), early-conv stems ("early convolutions help transformers see better"), or simply learning it — trained ViTs' early heads attend locally, later heads globally. DINO shows self-supervised ViT attention maps segment objects with no labels — the learned bias can be *better* than the hard-coded one.

</details>


### [02-09] Attention is $O(n^2)$. Spell out what that means for a 480p, 81-frame WAN 2.1 generation — token count, attention FLOPs, memory — and what you'd do about it.  (difficulty 3; tags: attention, efficiency, video, scaling)

- Follow-up: Does FlashAttention change the asymptotics? What about sparse/linear attention for video?


<details><summary>Answer</summary>

**About 32k tokens per sample, so the attention score matrix is $\sim 10^9$ entries per head per layer; attention FLOPs dominate the MLP by 5–10× and the *only* way to speed it up substantially is to reduce $N$ or the fraction of pairs computed.**

Token count: WAN 2.1 VAE compresses 4× in time, 8× in space, 16 channels; 81 frames → 21 latent frames, 480×832 → 60×104 latent; patchify (1,2,2) → **21 × 30 × 52 = 32,760 tokens**. (720p×81f ≈ 21×45×80 = 75,600.)

Per layer, $D = 5120$ (14B model), FLOPs for attention scores + weighted sum $\approx 4 N^2 D = 4 \times (3.3 \times 10^4)^2 \times 5120 \approx$ **$2.2 \times 10^{13}$ = 22 TFLOPs**; projections + MLP $\approx 24 N D^2 \approx 24 \times 3.3 \times 10^4 \times 2.6 \times 10^7 \approx 2.1 \times 10^{13}$. So already comparable at 480p; at 720p attention is 5× the MLP cost. Over 40 layers × 50 steps × 2 (CFG) that's $\sim 10^{17}$ FLOPs — minutes on one H100 at ~50% MFU, which matches reality.

Memory: naive attention would materialize $N^2 \times \text{heads} = 1.07 \times 10^9 \times 40 \text{ heads} \approx 43$ G entries — 86 GB in bf16 per layer. **FlashAttention** tiles Q/K/V through SRAM and never writes the $N \times N$ matrix, so memory is $O(N)$ and it's ~2–4× faster from IO savings — but it does *not* change the $O(N^2)$ FLOP count. It makes 32k tokens feasible, not cheap.

What to do about it:
- **Reduce $N$**: lower resolution / fewer frames for most of the trajectory (SPEED — spatial-first 240p→480p, 2.5× wall-clock on WAN), mixed-resolution tokens (foveated diffusion — 4× on video), token merging/pruning, larger VAE compression (8×16×16 or 4×32×32 like DC-AE/Wan 2.2's more aggressive VAE).
- **Reduce pairs**: sparse attention with spatiotemporal windows (Swin-style, STA / sliding tile attention), block-sparse from learned masks (Sparse VideoGen, radial attention exploiting the fact that attention energy decays with spatiotemporal distance), sequence-parallel attention (Ulysses / ring attention) to spread $N$ over GPUs.
- **Linear attention** (Mamba, gated linear attention, Lightning) — $O(N)$ but weaker global coherence for video; hybrids with a few full-attention layers are the current compromise.
- **Fewer NFE**: distillation (LCM, DMD, CausVid) — orthogonal and multiplicative with the above.

Trap: 2× resolution = 4× tokens = 16× attention FLOPs; a "1080p" model is not 2× a 480p model.

</details>


### [02-10] Explain grouped-query attention and multi-query attention. Why do they exist, and what's the actual bottleneck they fix?  (difficulty 2; tags: attention, inference, kv-cache, efficiency)

- Follow-up: Numbers for LLaMA-2 70B: KV-cache size per token with MHA vs GQA?


<details><summary>Answer</summary>

**MQA shares one K/V head across all query heads; GQA shares K/V across groups of query heads. They cut the KV cache — the memory-bandwidth bottleneck of autoregressive decoding — by $H/G$ with almost no quality loss.**

The bottleneck: during decoding each new token attends to all cached $K, V$. Per token the cache is $2$ (K and V) $\times L$ layers $\times H_{kv}$ heads $\times d_{\text{head}} \times$ bytes. Decode is *bandwidth-bound* — every step reads the whole cache from HBM to produce one token, so latency $\propto$ cache size, and batch size is capped by cache memory.

Numbers for LLaMA-2 70B ($L = 80$, $d_{\text{head}} = 128$, bf16):
- MHA, $H = 64$ kv heads: $2 \times 80 \times 64 \times 128 \times 2\,\text{B}$ = **2.6 MB per token** — a 4k context is 10.7 GB per sequence.
- GQA, 8 kv groups: $2 \times 80 \times 8 \times 128 \times 2\,\text{B}$ = **0.33 MB per token** — 8× smaller. That's why LLaMA-2 70B and all LLaMA-3 models use GQA-8.
- MQA (1 kv head): 41 KB/token — 64× smaller, used by PaLM, Falcon; slight quality drop and less parallel-friendly under tensor parallelism (you'd replicate the single K/V head on every rank, wasting compute), which is exactly why GQA with $G$ = TP degree is the sweet spot.

Quality: GQA-8 matches MHA in perplexity within noise on the LLaMA scaling; MQA loses a little. Both can be *uptrained* from an MHA checkpoint by mean-pooling the K/V heads and fine-tuning for ~5% of pretraining compute.

Relevance beyond LLMs: in autoregressive world / video-action models (streaming rollouts with a KV cache over past latents), the cache is the dominant memory — the same lever applies, and it composes with token-level KV eviction (keeping only surprising or attended tokens, ~40–50% reduction in the candidate's lingbot-VA work). For bidirectional diffusion transformers there's no cache, so GQA only saves projection params and a bit of bandwidth; FLUX/WAN still use full MHA.

MLA (DeepSeek-V2/V3) is the next step: cache a low-rank latent (512-d) per token and up-project $K, V$ on the fly — ~93% smaller cache than MHA with *better* quality than GQA.

</details>


### [02-11] Compare ReLU, GELU, SiLU, and SwiGLU. Why do modern transformers use gated units, and what does that do to the MLP dimensions?  (difficulty 1; tags: activation, mlp, architecture)

- Follow-up: What's the dead-ReLU problem and does GELU fix it?


<details><summary>Answer</summary>

**ReLU is $\max(0, x)$; GELU and SiLU are smooth, non-monotone approximations that let small negative values pass; SwiGLU replaces the MLP's single nonlinearity with a gated product, which gives better loss per parameter at the cost of a third weight matrix.**

- $\mathrm{ReLU}(x) = \max(0, x)$. Cheap, sparse; gradient exactly 0 for $x < 0$ ⇒ **dead units** that never recover if their pre-activations go negative for the whole dataset (common with high lr / bad init). Doesn't scale as well in transformers.
- $\mathrm{GELU}(x) = x\, \Phi(x)$ (Gaussian CDF); $\approx x\, \sigma(1.702\, x)$. Used in BERT, GPT-2, ViT, DiT. Smooth, has a small negative dip (min $\approx -0.17$ at $x \approx -0.75$), non-zero gradient for moderately negative inputs so units can recover — mostly fixes dead-ReLU.
- $\mathrm{SiLU}/\mathrm{Swish}(x) = x\, \sigma(x)$. Nearly identical to GELU, slightly cheaper; used in EfficientNet, LLaMA, and as the nonlinearity inside adaLN modulation MLPs.
- **SwiGLU** (Shazeer, "GLU Variants Improve Transformer"):
  $\mathrm{FFN}(x) = (\mathrm{SiLU}(x W_1) \odot x W_3)\, W_2$.
  Three matrices instead of two; to keep parameter count equal to the $4D$ standard MLP, the hidden size is set to $\tfrac{8}{3} D$ (LLaMA rounds to a multiple of 256: 4096 → 11008; FLUX single-stream blocks and most 2024+ DiTs use gated MLPs too). Gives ~1–2% lower loss at equal params/FLOPs; the multiplicative gate gives the MLP a data-dependent "attention over features," which is the intuition for why it helps (same reason gating helps LSTMs and MoE routers).

Why not sigmoid/tanh: saturate, gradients $\le 0.25$, cause vanishing gradients in deep nets — the original reason ReLU won in 2012.

Trap: the choice of activation matters far less than normalization, init, and lr schedule; don't spend hyperparameter budget here. But do note that GELU's exact `erf` form vs. the tanh approximation matters for reproducing checkpoints (GPT-2 uses tanh-approx; most HF ViTs use exact).

</details>


### [02-12] Estimate the parameter count and training FLOPs for a transformer layer and a full model. Where does the $6ND$ rule come from, and what are the FLOPs per token at inference?  (difficulty 3; tags: scaling, flops, transformer, systems)

- Follow-up: Apply it to training a 10B-param video DiT on 1T tokens on 512 H100s — how long?


<details><summary>Answer</summary>

**Per layer $\approx 12 D^2$ params ($4 D^2$ attention + $8 D^2$ MLP); training cost $\approx 6$ FLOPs per parameter per token (2 forward, 4 backward); inference $\approx 2$ FLOPs per parameter per token, plus the attention term $4 N D$ per token per layer that the rule ignores.**

Parameter count, width $D$, $L$ layers, vocabulary $V$ (LLM) or patch dim (DiT):
- Attention: $W_Q, W_K, W_V, W_O$ each $D \times D$ → $4 D^2$.
- MLP: $D \times 4D$ and $4D \times D$ → $8 D^2$ (SwiGLU with $\tfrac{8}{3} D$ hidden also → $8 D^2$).
- Total $\approx$ **$12 D^2 L$** + embeddings ($V D$) + adaLN modulation ($6 D^2$ per block for DiT — non-trivial: ~1/3 extra).
- Check: GPT-3, $D = 12288$, $L = 96$: $12 \times 1.5 \times 10^8 \times 96 = 1.74 \times 10^{11} \approx$ 175B. FLUX: $D = 3072$, 19 dual blocks (×2 streams) + 38 single $\approx 12 \times 9.4 \times 10^6 \times (38 + 38) \approx$ 8.6B plus modulation/text ≈ 12B.

**$6ND$ rule** (Kaplan / Chinchilla): a matmul with weight $W \in \mathbb{R}^{a \times b}$ on one token costs $2ab$ FLOPs (multiply + add). Forward = 2 FLOPs per param per token. Backward computes gradients wrt both the input (2 per param) and the weights (2 per param) = 4. Total **6 FLOPs/param/token**; training cost $C \approx 6 \cdot N_{\text{params}} \cdot D_{\text{tokens}}$. Ignores attention's $N^2$-term (fine when context $\ll 12 D$, wrong for long-context video) and activation recomputation (adds ~1 forward ⇒ $8ND$).

Attention term per token per layer: scores $Q K^\top$ ($2ND$) + $AV$ ($2ND$) = **$4ND$** forward; relative to the $24 D^2$ of the weights it's $N/(6D)$. For an LLM at $N$ = 4k, $D$ = 4k: ~17% overhead. For a video DiT at $N$ = 32k, $D$ = 5k: attention ≈ MLP; at 75k tokens it dominates.

Inference: $\approx 2 N_{\text{params}}$ FLOPs per token, plus $4NDL$ for attention (grows with context). For diffusion: multiply by NFE × CFG passes × tokens — a 12B FLUX at 4096 tokens: $2 \times 1.2 \times 10^{10} \times 4096 \approx 10^{14}$ FLOPs per forward, $\sim 10^{16}$ for 50 steps × 2 — about 10–20 s on one H100 at ~50% MFU, matching practice.

Follow-up: 10B DiT, 1T tokens: $C = 6 \times 10^{10} \times 10^{12} = 6 \times 10^{22}$ FLOPs. Add ~1.5× for the attention term at 32k-token videos and recompute ⇒ $\sim 10^{23}$. 512 H100s at ~400 TFLOP/s bf16 achieved (40% MFU of 989 dense) = $2 \times 10^{17}$ FLOP/s ⇒ **$\sim 5 \times 10^5$ s ≈ 6 days.** Memory: 10B params in mixed precision with Adam = 16 bytes/param = 160 GB of states ⇒ needs ZeRO-2/3 or FSDP across the 512 GPUs, plus sequence parallelism for the 32k-token activations.

</details>


### [02-13] What is an embedding?  (difficulty 1; tags: architecture, embedding, representation, rapid-fire)


<details><summary>Answer</summary>

**A learned map from a discrete or structured input — token, patch, timestep, class — to a dense vector in $\mathbb{R}^d$, such that geometry in that space (dot product, distance) reflects semantic similarity.**

It exists because networks compute on continuous vectors and one-hot inputs are huge, sparse, and carry no similarity: every pair of words is equally far apart. A lookup table $E \in \mathbb{R}^{V \times d}$ is just a linear layer on a one-hot vector, trained end-to-end. The same idea covers ViT patch embeddings (linear projection of pixels), diffusion timestep embeddings (sinusoidal features → MLP), and CLIP/T5 text embeddings as conditioning.

Trap: the word is overloaded — input lookup table vs. an encoder's output representation. And embeddings are only meaningful relative to their trained head: cosine similarity across two different models means nothing.

</details>


### [02-14] What is layer normalization? Give the formula.  (difficulty 1; tags: normalization, architecture, rapid-fire)


<details><summary>Answer</summary>

**LayerNorm normalizes each token's feature vector to zero mean and unit variance across the feature dimension, then rescales with learned $\gamma, \beta$: $y = \gamma \odot (x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta$, where $\mu, \sigma^2$ are computed over the $d$ features of that single token.**

It keeps activations at a fixed scale layer after layer so gradients neither vanish nor explode and the learning rate stays meaningful; unlike BatchNorm it has no batch dependence, so it works at batch size 1, with variable sequence lengths, and identically at train and test. RMSNorm drops the mean subtraction, $y = \gamma \odot x / \mathrm{RMS}(x)$, and is cheaper with no quality loss; LLaMA and most DiTs use it.

Trap: in DiTs $\gamma$ and $\beta$ are not static — adaLN generates them from the timestep/class embedding, so the norm *is* the conditioning mechanism.

</details>


### [02-15] What is a residual connection?  (difficulty 1; tags: architecture, residual, depth, rapid-fire)


<details><summary>Answer</summary>

**A residual connection adds a layer's input to its output, $y = x + F(x)$, so the layer learns a correction to the identity instead of a whole new mapping.**

Why: deep plain networks got *worse* with depth even on training loss — an optimization problem, not overfitting. With residuals the identity is the default, so adding layers never hurts at init, and the gradient has a direct path, $\partial y/\partial x = I + \partial F/\partial x$, that never vanishes through the stack. Each block makes a small update to a shared residual stream — the mental model behind transformers.

Trap: the residual stream's variance grows with depth as blocks add to it, which is why pre-norm and zero-init of each block's last layer (adaLN-Zero) matter for very deep DiTs.

</details>


### [02-16] What is the receptive field?  (difficulty 1; tags: cnn, architecture, receptive-field, rapid-fire)


<details><summary>Answer</summary>

**The receptive field of an output unit is the region of the input that can influence it; for stacked convolutions it grows with depth and kernel size, $\mathrm{RF}_l = \mathrm{RF}_{l-1} + (k - 1) \cdot (\text{product of strides so far})$.**

A conv layer only sees a $k \times k$ window, so relating distant pixels needs depth, dilation, striding, or attention; two 3×3 convs match one 5×5 with fewer parameters (the VGG argument). In a ViT or DiT every token attends to every token, so the receptive field is global from layer one — that's the inductive-bias trade: convs get locality for free and must earn global context; attention gets global context and must learn locality.

Trap: the *effective* receptive field is much smaller than the formula — gradient contributions are Gaussian-shaped around the center — so deep CNNs "see" less than claimed.

</details>


### [02-17] What's the difference between a U-Net skip connection and a residual connection?  (difficulty 2; tags: unet, architecture, skip-connection, rapid-fire)


<details><summary>Answer</summary>

**A residual connection adds a block's input to its output at the *same* depth, $y = x + F(x)$; a U-Net skip carries encoder features across the whole network to the decoder at the *same spatial resolution*, usually by channel concatenation.**

Different problems: residuals fix optimization depth. U-Net skips fix information loss — downsampling to a bottleneck destroys high-frequency detail, and the skip hands the decoder the fine encoder features so it can localize edges, essential for dense prediction and denoising. Concatenation lets the decoder learn how to weight the two sources.

Trap: in diffusion U-Nets the skips carry so much high-frequency content that the decoder can bypass the bottleneck; FreeU shows down-weighting them at inference improves samples. DiTs have no U-Net skips — the full-resolution residual stream does the job, which is why DiT quality leans so heavily on the VAE.

</details>


### [02-18] Define attention in one sentence, and give me the intuition for Q, K, and V.  (difficulty 1; tags: attention, transformer, rapid-fire)


<details><summary>Answer</summary>

**Attention is a data-dependent weighted average: each token builds a query, compares it against every token's key to get softmax weights, and sums those tokens' values with those weights — $\mathrm{Attn} = \operatorname{softmax}(Q K^\top / \sqrt{d})\, V$.**

Intuition: a soft dictionary lookup. Query is "what am I looking for", key is "what I advertise", value is "what I hand back if chosen". Separating K from V lets a token be found by one property and contribute another. Softmax makes a convex combination, so the output stays in the span of the values, and $\sqrt{d}$ keeps logits $O(1)$ so softmax doesn't saturate at init.

Trap: attention is a set operation with no notion of position — RoPE is what makes it a sequence model — and it's quadratic in tokens, the entire reason for efficient-attention and token-reduction work.

</details>


### [02-19] What is a causal mask?  (difficulty 1; tags: attention, autoregressive, masking, rapid-fire)


<details><summary>Answer</summary>

**A causal mask sets attention logits to $-\infty$ for every key at a later position than the query, so token $i$ attends only to tokens $\le i$; after softmax those entries are exactly zero.**

It exists so one forward pass over a whole sequence trains next-token prediction at every position in parallel while guaranteeing no position sees its own target. At inference it's what makes the KV cache valid: past keys and values never change when new tokens arrive, so each token is computed once.

Trap: you choose the granularity. In video models, full per-token causality is slow and unnecessary; block-causal per frame (bidirectional within a frame, causal across frames) is what CausVid / Self-Forcing use. And mask with $-\infty$ *before* softmax, not by zeroing after, or rows won't sum to one.

</details>


### [02-20] What is teacher forcing?  (difficulty 1; tags: autoregressive, training, exposure-bias, rapid-fire)


<details><summary>Answer</summary>

**Teacher forcing trains a sequence model by feeding it the ground-truth previous tokens as context at every step, rather than its own predictions, so the whole sequence trains in one parallel pass.**

It exists because sampling your own outputs during training would be sequential and slow and would route gradients through a stochastic decoder. With teacher forcing plus a causal mask, next-token loss at every position comes from a single forward pass — why transformers train efficiently.

The trap it creates is **exposure bias**: at inference the model conditions on its *own* imperfect outputs, a distribution it never saw in training, so errors compound over long rollouts — acute in autoregressive video / world models, where a small frame error drifts into garbage within seconds. Fixes: scheduled sampling, noise-augmented context, DAgger-style rollouts, Self-Forcing.

</details>


## Diffusion & Flow Fundamentals


### [03-01] Write down the DDPM forward process and derive the closed form of $q(x_t \mid x_0)$. Why does that closed form matter for training?  (difficulty 1; tags: diffusion, forward-process, derivation)

- Follow-up: what is $\bar\alpha_T$ for the original linear schedule, and is it exactly zero?


<details><summary>Answer</summary>

**The forward process is a fixed Markov chain of Gaussian noising steps, and because Gaussians compose, you can jump to any $t$ in one shot.**

Per step: $q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\big)$. Write $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s \le t} \alpha_s$.

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, z_t \\
    &= \sqrt{\alpha_t} \left(\sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_{t-1}}\, z_{t-1}\right) + \sqrt{1-\alpha_t}\, z_t
\end{aligned}
$$
The two noise terms are independent zero-mean Gaussians, so they merge into one with variance $\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t \alpha_{t-1}$. By induction:

**$q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I\big)$**, i.e. $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$.

It is variance-preserving: $\text{signal}^2 + \text{noise}^2 = 1$, so $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$.

Why it matters: training is **simulation-free** — sample $(x_0, t, \epsilon)$, form $x_t$ directly, regress. No need to run the chain. It also gives the tractable posterior $q(x_{t-1} \mid x_t, x_0)$, which is what the ELBO compares against.

Trap: with DDPM's linear $\beta$ ($10^{-4} \to 0.02$, $T=1000$), $\bar\alpha_T \approx 4 \times 10^{-5}$, not 0. SNR at $T$ is small but nonzero, so the model never sees pure noise yet you sample from pure noise — the "zero terminal SNR" mismatch (mean/brightness leak).

</details>


### [03-02] Derive the simplified $\epsilon$-prediction loss from the ELBO. Why is dropping the per-timestep weight okay, and how does $\epsilon$-prediction relate to score matching?  (difficulty 2; tags: diffusion, elbo, score-matching, derivation)

- Follow-up: write the score of $q(x_t \mid x_0)$ in terms of $\epsilon$.


<details><summary>Answer</summary>

**The ELBO reduces to a sum of Gaussian KLs; each KL is a weighted MSE between the true posterior mean and the model mean; reparameterizing the mean in terms of $\epsilon$ and dropping the weight gives $\|\epsilon - \epsilon_\theta\|^2$.**

Sketch:
$$
-\log p(x_0) \le L_T + \sum_{t>1} \mathrm{KL}\big( q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t) \big) + L_0
$$
Posterior $q(x_{t-1} \mid x_t, x_0)$ is Gaussian with mean $\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon\right)$. If $p_\theta$ uses the same variance $\sigma_t^2$, the KL is $\frac{1}{2\sigma_t^2} \|\tilde\mu_t - \mu_\theta\|^2$. Parameterize $\mu_\theta$ identically with $\epsilon_\theta$, and the difference collapses to

$$
L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2\, \alpha_t (1-\bar\alpha_t)} \cdot \|\epsilon - \epsilon_\theta(x_t, t)\|^2.
$$

Ho et al. drop the bracket: **$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \|\epsilon - \epsilon_\theta\|^2$**. Two reasons it works: (1) the true weight is large at small $t$ (fine details that are perceptually cheap) and tiny at large $t$; uniform weighting **up-weights the high-noise, structure-defining timesteps**, which improves FID even though it is no longer the exact likelihood bound. (2) Kingma & Gao (2023) show any monotone weighting in log-SNR is still a valid ELBO for a noise-perturbed data distribution — so it is a weighted ELBO, not something ad hoc.

Score matching link: $\nabla_{x_t} \log q(x_t \mid x_0) = -(x_t - \sqrt{\bar\alpha_t}\, x_0)/(1-\bar\alpha_t) =$ **$-\epsilon/\sigma_t$** with $\sigma_t = \sqrt{1-\bar\alpha_t}$. Vincent's denoising score matching says regressing onto the conditional score has the same minimizer as regressing onto the marginal score $\nabla \log q(x_t)$. So the $\epsilon$-loss is $\sigma_t^2$-weighted denoising score matching, and $s_\theta = -\epsilon_\theta/\sigma_t$. Tweedie gives the denoiser: $\hat x_0 = (x_t + \sigma_t^2 s_\theta)/\sqrt{\bar\alpha_t}$.

</details>


### [03-03] Give the SDE view of diffusion and derive the probability-flow ODE. Why does the ODE give deterministic sampling, and when would you still prefer the SDE?  (difficulty 3; tags: diffusion, sde, ode, derivation)

- Follow-up: what does "same marginals" mean here, and how do you show it?


<details><summary>Answer</summary>

**Forward diffusion is an Itô SDE; its reverse is another SDE driven by the score; and there is a deterministic ODE that transports exactly the same marginals $p_t$, so you can integrate it backward from noise without any randomness.**

Forward: $\mathrm{d}x = f(x,t)\,\mathrm{d}t + g(t)\,\mathrm{d}w$. For VP, $f = -\tfrac{1}{2}\beta(t)\, x$, $g = \sqrt{\beta(t)}$.
Reverse (Anderson 1982): $\mathrm{d}x = [f - g^2 \nabla \log p_t(x)]\,\mathrm{d}t + g\, \mathrm{d}\bar w$, run with $t$ decreasing.

Derivation of the PF-ODE via Fokker–Planck:
$$
\begin{aligned}
\frac{\partial p}{\partial t} &= -\nabla \cdot (f p) + \tfrac{1}{2} g^2 \Delta p \\
      &= -\nabla \cdot (f p) + \nabla \cdot \left(\tfrac{1}{2} g^2\, p\, \nabla \log p\right) & \text{since } \Delta p = \nabla \cdot (p\, \nabla \log p) \\
      &= -\nabla \cdot \left( \left[f - \tfrac{1}{2} g^2 \nabla \log p\right] p \right)
\end{aligned}
$$
That last line is a continuity equation for a **deterministic** velocity field $v = f - \tfrac{1}{2} g^2 \nabla \log p_t$. So $\mathrm{d}x/\mathrm{d}t = f(x,t) - \tfrac{1}{2} g(t)^2 \nabla \log p_t(x)$ has the same $p_t$ as the SDE at every $t$, with zero diffusion term.

Why deterministic: no $\mathrm{d}w$ — given $x_T$ the whole trajectory is fixed. Consequences: **invertible encoding** (DDIM inversion, editing), latent interpolation is meaningful, exact likelihood via instantaneous change of variables, and smooth trajectories that high-order solvers (Heun, DPM-Solver) exploit, so 20–50 NFE suffice.

When SDE still wins: the reverse SDE = ODE drift + a Langevin correction term ($\tfrac{1}{2} g^2 \nabla \log p$ plus matching noise). That Langevin part **pulls samples back toward the model's $p_t$**, correcting accumulated model error. With many steps (hundreds), stochastic samplers often give better FID (EDM's "churn"); with few steps the injected noise is not integrated well and the ODE wins.

Note the $\tfrac{1}{2}$: the SDE uses the full $g^2 \nabla \log p$, the ODE half — a common whiteboard slip.

</details>


### [03-04] Show that DDIM is a discretization of the probability-flow ODE. Why is it better than plain Euler on the raw ODE?  (difficulty 2; tags: diffusion, ddim, ode, samplers)

- Follow-up: what does $\eta$ do, and what happens to inversion when $\eta > 0$?


<details><summary>Answer</summary>

**DDIM's deterministic update is exactly one Euler step of the PF-ODE written in signal-normalized coordinates — equivalently an exponential integrator that handles the linear part exactly.**

DDIM update ($\eta = 0$):
$$
x_s = \sqrt{\bar\alpha_s}\, \hat x_0 + \sqrt{1-\bar\alpha_s}\, \epsilon_\theta, \quad \text{with } \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta}{\sqrt{\bar\alpha_t}}.
$$

Change variables: $\bar x = x/\sqrt{\bar\alpha}$, $\bar\sigma = \sqrt{1-\bar\alpha}/\sqrt{\bar\alpha}$ (the "noise-to-signal" ratio). Substituting:
$$
\bar x_s = \bar x_t + (\bar\sigma_s - \bar\sigma_t) \cdot \epsilon_\theta(x_t, t)
$$
That is Euler's method for $\mathrm{d}\bar x/\mathrm{d}\bar\sigma = \epsilon_\theta$, which is the VP probability-flow ODE after removing the linear scaling term. Song et al. show the PF-ODE in these coordinates is exactly this.

Why better than naive Euler on $\mathrm{d}x/\mathrm{d}t = f - \tfrac{1}{2} g^2 s$: the ODE is **semilinear** — a linear drift $-\tfrac{1}{2}\beta x$ plus a nonlinear score term. Euler discretizes both; DDIM (= DPM-Solver-1) integrates the linear part **exactly** and only approximates the nonlinear part as constant over the step. Same NFE, much smaller error at large steps; this is why DDIM gives usable samples at 20–50 steps where ancestral DDPM needs ~1000.

$\eta$: interpolates to a stochastic sampler; $\sigma_t^2 = \eta \cdot \tilde\beta_t$ gives the DDPM ancestral sampler at $\eta = 1$. With $\eta > 0$ the map is no longer invertible, so DDIM inversion for editing requires $\eta = 0$, and inversion error still grows with CFG scale because the ODE is being integrated with a different (guided) velocity than the one that generated the image.

</details>


### [03-05] Derive the flow-matching / rectified-flow objective for $x_t = (1-t)\,x_0 + t\,\epsilon$. How does it relate to v-prediction and to diffusion, and why do "straight paths" help few-step sampling?  (difficulty 2; tags: flow-matching, rectified-flow, derivation)

- Follow-up: is the learned marginal velocity field straight?


<details><summary>Answer</summary>

**Regress a network onto the conditional velocity of a linear interpolant; the conditional target has the same gradient as the intractable marginal one, and the resulting ODE is a diffusion PF-ODE with a particular (non-variance-preserving) schedule.**

Interpolant (SD3/FLUX convention, $t=0$ data, $t=1$ noise): $x_t = (1-t)\,x_0 + t\,\epsilon$. Differentiate along a fixed pair: $\mathrm{d}x_t/\mathrm{d}t =$ **$\epsilon - x_0$**. Loss:
$$
L = \mathbb{E}_{t, x_0, \epsilon} \|v_\theta(x_t, t) - (\epsilon - x_0)\|^2.
$$
Lipman et al.: the marginal velocity $u_t(x) = \mathbb{E}[\epsilon - x_0 \mid x_t = x]$ is what generates $p_t$, and conditional FM has the same gradient in $\theta$ as regressing on $u_t$. So sampling is $\mathrm{d}x/\mathrm{d}t = v_\theta$ from $x_1 = \epsilon$ to $t=0$.

Conversions: $\hat x_0 = x_t - t\,v$, $\hat\epsilon = x_t + (1-t)\,v$, $\text{score} = -\hat\epsilon/t$.

Relation to diffusion: it is a Gaussian path with $\alpha_t = 1-t$, $\sigma_t = t$. Not VP ($\alpha^2 + \sigma^2 \neq 1$). log-SNR $\lambda(t) = 2 \log\big((1-t)/t\big)$, a logistic schedule; the deterministic sampler is the PF-ODE of that schedule. Every diffusion trick (CFG, DPM-Solver, weighting) carries over.

Relation to v-prediction: DDPM $v = \alpha_t \epsilon - \sigma_t x_0$ on a VP/cosine (angular) schedule. Same "difference of endpoints" flavor, bounded target at both ends, but a different path — and only the linear interpolant makes the conditional paths straight lines.

Why straight helps: if trajectories were straight, **one Euler step is exact**. Conditional paths are straight, but the marginal field is not (paths cross, so the marginal velocity is an average). Still, the linear interpolant has lower curvature than VP paths, and **reflow** (re-pairing noise with generated samples and retraining) straightens it further — that is what makes 1–4 step rectified-flow models work. Trap: FM is not "faster than diffusion" in general — same NFE at 50 steps; the gains show up under aggressive step reduction and distillation.

</details>


### [03-06] Compare the $\epsilon$, $x_0$, $v$ and flow-velocity parameterizations. Give the conversions, and tell me which is numerically stable at which SNR and why.  (difficulty 2; tags: diffusion, parameterization, numerics)

- Follow-up: what does EDM's preconditioning do differently?


<details><summary>Answer</summary>

**They are all the same denoiser under a linear change of output; the differences are which SNR extremes blow up the effective error and which implicit loss weighting you get.**

With $x_t = \alpha x_0 + \sigma \epsilon$ (VP: $\alpha^2 + \sigma^2 = 1$), $v = \alpha \epsilon - \sigma x_0$:
$$
\begin{aligned}
x_0 &= (x_t - \sigma \epsilon)/\alpha, & \epsilon &= (x_t - \alpha x_0)/\sigma \\
x_0 &= \alpha x_t - \sigma v, & \epsilon &= \sigma x_t + \alpha v \\
\text{FM } (\alpha = 1-t,\ \sigma = t):\quad x_0 &= x_t - t\, v_{\text{fm}}, & \epsilon &= x_t + (1-t)\, v_{\text{fm}}
\end{aligned}
$$
Stability:
- **$\epsilon$-prediction** at low SNR ($\sigma \to 1$, $\alpha \to 0$): $\hat x_0 = (x_t - \sigma \hat\epsilon)/\alpha$ divides by $\alpha$ — a tiny $\epsilon$ error becomes a huge $x_0$ error. At exactly zero terminal SNR it is degenerate: $x_t = \epsilon$, so predicting $\epsilon$ is trivial and says nothing about $x_0$. That is why zero-terminal-SNR models switch to $v$.
- **$x_0$-prediction** at high SNR ($\sigma \to 0$): $\hat\epsilon = (x_t - \alpha \hat x_0)/\sigma$ divides by $\sigma$; the network must reproduce $x_t$ almost exactly, and any error becomes an enormous noise estimate — fine-detail steps get wrecked. $x_0$-pred at low SNR is easy and well-behaved.
- **$v$-prediction** is bounded at both ends: at high SNR $v \approx \epsilon$, at low SNR $v \approx -x_0$. That is why progressive distillation and Imagen Video use it.
- **FM velocity** $\epsilon - x_0$ is likewise bounded at both ends; its conversion to $x_0$ has no division at all.

Loss-weighting view: at fixed $x_0$ loss, $\epsilon$-loss $= \mathrm{SNR} \cdot L_{x_0}$, $v$-loss $= (\mathrm{SNR}+1) \cdot L_{x_0}$. Picking a parameterization silently picks a weighting.

EDM makes this explicit: the network predicts $F$, and $D(x;\sigma) = c_{\text{skip}}(\sigma)\, x + c_{\text{out}}(\sigma)\, F(c_{\text{in}}(\sigma)\, x)$, with $c_{\text{skip}} = \sigma_{\text{data}}^2/(\sigma^2 + \sigma_{\text{data}}^2)$ etc., chosen so the network's input and **target both have unit variance at every $\sigma$** and the skip smoothly interpolates between $x_0$-pred (low $\sigma$) and $\epsilon$-like (high $\sigma$).

</details>


### [03-07] Compare the linear and cosine noise schedules. What is the log-SNR view, and what is the "zero terminal SNR" problem?  (difficulty 1; tags: diffusion, noise-schedule, snr)

- Follow-up: does the training schedule have to match the sampling schedule?


<details><summary>Answer</summary>

**A schedule is just a monotone map from time to log-SNR $\lambda = \log(\alpha^2/\sigma^2)$; linear wastes steps near pure noise and doesn't reach SNR=0, cosine spends time more evenly, and both facts matter for sampling.**

- Linear $\beta$ (DDPM): $\beta$ from $10^{-4}$ to $0.02$. $\bar\alpha_t$ decays fast; the last ~20% of timesteps are visually pure noise — wasted capacity — yet $\bar\alpha_T \approx 4 \times 10^{-5} \neq 0$.
- Cosine (Nichol & Dhariwal): $\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$, $s = 0.008$. $\bar\alpha$ decreases nearly linearly in the middle, more steps at mid-SNR where structure is decided; better at low resolution.
- **Log-SNR view** (Kingma; EDM): what matters is the distribution of $\lambda$ over training timesteps and the weighting $w(\lambda)$. Two schedules with the same $p(\lambda)\, w(\lambda)$ train the same model; the schedule in $t$ is an implementation detail. FM-linear: $\lambda = 2 \log\big((1-t)/t\big)$, logistic; VP-cosine: $\lambda = -2 \log \tan(\pi t/2)$.

Zero terminal SNR (Lin et al. 2024): at inference you start from $\mathcal{N}(0, I)$ (SNR = 0) but the model was trained at $\mathrm{SNR}_T > 0$, where $x_T$ still carries mean/low-frequency information about $x_0$ (in latent space the DC component is huge). Result: the model assumes a nonzero mean is present, so it can't produce very dark or very bright images and everything drifts to medium brightness. Fix: rescale $\sqrt{\bar\alpha}$ so $\bar\alpha_T = 0$ exactly, switch to **$v$-prediction** ($\epsilon$ is degenerate at SNR 0), sample starting from the last timestep, and rescale CFG to avoid over-exposure.

Follow-up: no — sampling steps are chosen separately (Karras $\rho$-schedule, "shift", trailing spacing). Training $p(\lambda)$ affects what the model learned; sampling $\sigma_i$ only affects discretization error.

</details>


### [03-08] Why does a higher-resolution (or video) model need "more noise", and how do you derive the timestep shift used by SD3/FLUX/WAN? Tie it to the power spectrum.  (difficulty 3; tags: diffusion, noise-schedule, resolution, spectral, shift)

- Follow-up: FLUX uses a "dynamic shift" that depends on token count — what is it doing?


<details><summary>Answer</summary>

**Per-pixel SNR is the wrong invariant. What the model actually has to resolve at a given step is the SNR of low-frequency content, and that grows with pixel count — so at higher resolution the same $t$ is "easier", and you must shift $t$ toward more noise to keep the effective difficulty constant.**

Derivation (SD3 style). Take a constant patch of value $c$ over $n$ pixels with $x_t = (1-t)\,x_0 + t\,\epsilon$. The patch mean has signal $(1-t)\,c$ and noise std $t/\sqrt{n}$. So the effective log-SNR is
$$
\lambda_n(t) = 2 \log\big((1-t)/t\big) + \log n.
$$
Demand equal $\lambda$ at resolution $m$: $(1-t_m)/t_m = \sqrt{n/m} \cdot (1-t_n)/t_n$. With shift $s = \sqrt{m/n}$:
$$
t_m = \frac{s \cdot t_n}{1 + (s - 1) \cdot t_n}
$$
That is exactly the diffusers `shift` on $\sigma$: $\sigma' = s\sigma/(1 + (s-1)\sigma)$. SD3 fixed $s = 3$ empirically for 1024²; WAN uses shift 3–5; higher res or more frames → larger shift. It moves timesteps toward $t=1$, i.e. **more steps at high noise**, where the global layout is decided.

Spectral statement of the same fact: images have $P(f) \propto f^{-\beta}$, $\beta \approx 2$ (FLUX latents ≈ 1.9, WAN ≈ 2.4). With an orthonormal DCT/FFT, $\epsilon$ has unit power in every bin, so noise power is $t^2$ per bin regardless of resolution, while the **lowest bins' energy scales with the number of pixels** (DC of an $N$-pixel image has energy $\propto N$). Upsampling adds new high-frequency bins and pushes the existing content down in normalized frequency where power is higher. Therefore low frequencies stay above the noise floor at higher $t$: their activation time $t^{\ast}(f)$ (where $(1-t)^2 P(f) = t^2$) moves later in noise. A schedule tuned at 256² spends too few steps in the regime where a 1024² image's coarse structure is still forming.

Dynamic shift (FLUX): $\mu$ is a linear function of sequence length (256 tokens → 0.5, 4096 → 1.15), $s = \exp(\mu)$, so the shift automatically scales with token count — same argument, parameterized by $n$.

Follow-up trap: for video, the temporal axis is much more energetic than spatial at equal normalized frequency ($\beta_t \approx 1.6$ vs $\beta_s \approx 2.7$ in WAN), so frame count shifts the schedule even more than pixel count.

</details>


### [03-09] Uniform timestep sampling versus logit-normal, min-SNR-$\gamma$, P2 — what problem is each fixing, and how do they fit into one framework?  (difficulty 2; tags: diffusion, loss-weighting, training)

- Follow-up: why is uniform-$t$ $\epsilon$-loss "secretly" weighted toward high SNR?


<details><summary>Answer</summary>

**All of them are choosing the effective weight $w(\lambda)\, p(\lambda)$ over log-SNR. Uniform $t$ with $\epsilon$-loss puts most of the gradient budget on high-SNR, perceptually cheap timesteps; the alternatives move it to the mid-SNR region where structure is learned and gradients across timesteps conflict least.**

- **Uniform $t$, $\epsilon$-loss**: the implicit weight relative to an $x_0$ loss is $\mathrm{SNR}(t)$. With linear/cosine schedules, most of the timesteps sit at high SNR, so training is dominated by "remove a little noise" — fine texture — while the hard mid-SNR steps get little signal.
- **Logit-normal (SD3, FLUX)**: $t = \operatorname{sigmoid}(u)$, $u \sim \mathcal{N}(m, s)$. For velocity prediction the target $\epsilon - x_0$ is easy at both ends (at $t \approx 1$ you only need $\mathbb{E}[x_0] \approx 0$; at $t \approx 0$ $\epsilon$ is unpredictable noise, irreducible loss) and hard in the middle, so sampling concentrates there. SD3 found $m=0$, $s=1$ best; shifting $m$ with resolution is another way to express the timestep shift.
- **Min-SNR-$\gamma$** (Hang et al.): weight the $\epsilon$-loss by $\min(\mathrm{SNR}, \gamma)/\mathrm{SNR}$, $\gamma = 5$. Views timesteps as a multi-task problem with conflicting gradients; clipping stops the high-SNR tasks from dominating; 3× faster convergence.
- **P2** (Choi et al.): weight $1/(k + \mathrm{SNR})^\gamma$, explicitly down-weighting high SNR ("imperceptible details") in favor of the "content" and "coarse" stages.
- **EDM**: sample $\ln \sigma \sim \mathcal{N}(-1.2, 1.2^2)$ and weight by $(\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\, \sigma_{\text{data}})^2$ so the loss is roughly uniform in the unit-variance target — log-normal in $\sigma$ is the same idea as logit-normal in $t$.

Unified view (Kingma & Gao): the objective is $\int w(\lambda) \cdot \mathbb{E}\|x_0 - \hat x_0\|^2\, p(\lambda)\, \mathrm{d}\lambda$; schedule, parameterization, and sampling density are three ways to set the same curve. Practical follow-up: when you fine-tune at a new resolution, re-check the $\lambda$ distribution — the schedule that trained the base may put almost no mass where the new resolution's structure is decided.

</details>


### [03-10] Derive classifier-free guidance. What distribution does it sample from, and why does high guidance oversaturate and lose diversity?  (difficulty 2; tags: diffusion, cfg, guidance, derivation)

- Follow-up: what does the model actually compute per sampling step with CFG?


<details><summary>Answer</summary>

**CFG replaces a classifier gradient with the difference between conditional and unconditional scores, extrapolating past the conditional prediction; the result sharpens $p(c \mid x)$ like a temperature, which is exactly what removes diversity and pushes values out of range.**

Derivation: Bayes gives $\nabla \log p(x_t \mid c) = \nabla \log p(x_t) + \nabla \log p(c \mid x_t)$. Classifier guidance scales the second term: $\nabla \log p(x_t) + w \cdot \nabla \log p(c \mid x_t)$. Without a classifier, use $\nabla \log p(c \mid x_t) = \nabla \log p(x_t \mid c) - \nabla \log p(x_t)$:
$$
\tilde s = s_u + w\,(s_c - s_u) \quad \Leftrightarrow \quad \tilde\epsilon = \epsilon_u + w\,(\epsilon_c - \epsilon_u) = (1-w)\,\epsilon_u + w\,\epsilon_c
$$
The unconditional model is the same network with the condition dropped ($\varnothing$ token) 10–20% of the time in training. $w=1$ is pure conditional, $w>1$ extrapolates.

What it samples: the guided score corresponds to $\tilde p(x \mid c) \propto p(x \mid c)^w \cdot p(x)^{1-w} = p(x) \cdot p(c \mid x)^w$ — a **tempered posterior over $c$**. Caveat: the diffused version of this sharpened density is not what you get by sharpening the diffused scores at each $t$, so the guided trajectory is not an exact sampler of anything; it is a heuristic that happens to work.

Why oversaturation: $(\epsilon_c - \epsilon_u)$ at high noise is mostly a low-frequency "move toward the class mode" vector; multiplying it by $w$ = 7–10 gives predicted $\hat x_0$ values that leave the data range $[-1, 1]$ (or unit-variance latent range). Each step compounds this — colors clip, contrast blows up, textures become "crispy". Diversity: $p(c \mid x)^w$ with large $w$ is a near-delta on the most prototypical $x$ for $c$; samples collapse toward the mode (the mode-seeking direction of the extrapolation), which is why FID rises while CLIP score rises.

Per step: two forward passes (conditional + unconditional, usually batched as $2 \times B$), so **CFG doubles NFE**; with a negative prompt or an extra condition (image + text, InstructPix2Pix) it is three.

</details>


### [03-11] How do you get the benefits of strong CFG without the saturation? Compare guidance interval, dynamic thresholding, APG, autoguidance, and guidance distillation — and what does distillation cost you?  (difficulty 3; tags: diffusion, cfg, guidance, distillation)

- Follow-up: why can't you stack CFG on top of a guidance-distilled model like FLUX.1-dev?


<details><summary>Answer</summary>

**Each fix attacks a different symptom: where in the trajectory guidance is applied, whether the guided $\hat x_0$ stays in range, which component of the guidance vector you keep, or what "bad" model you subtract. Distillation removes the two-pass cost but freezes one guidance behavior into the weights.**

- **Guidance interval** (Kynkäänniemi et al. 2024): apply CFG only for mid noise levels. At high $\sigma$ the guided direction chooses a mode and kills diversity; at low $\sigma$ it is unnecessary and only adds saturation. Turning it off at both ends improves FID and lets you raise $w$.
- **Dynamic thresholding** (Imagen): at each step clip $\hat x_0$ to the $s$-th percentile of $|\hat x_0|$ and divide by $s$ ($s>1$). Keeps the prediction in range; pixel-space only, meaningless in latents.
- **CFG rescale** (Lin et al.): rescale the guided output's per-channel std back to the conditional's std, blended by $\phi \approx 0.7$. Cheap, standard with zero-terminal-SNR.
- **APG (adaptive projected guidance)**: decompose $\Delta = \epsilon_c - \epsilon_u$ into components parallel and orthogonal to $\epsilon_c$. The parallel part mostly scales the prediction (saturation); the orthogonal part carries the "which mode" quality signal. Keep orthogonal, down-weight parallel ($\eta \approx 0$), add momentum and a norm cap. Lets you use $w \approx 15$ without burn.
- **Autoguidance** (Karras 2024): guide with a smaller/less-trained version of the same model: $\tilde\epsilon = \epsilon_{\text{bad}} + w(\epsilon_{\text{good}} - \epsilon_{\text{bad}})$. Decouples "sharpen $p(c \mid x)$" from "remove the errors a weak model makes"; improves quality without the diversity loss because you are not tempering the class posterior.
- **Guidance distillation** (Meng et al.; FLUX.1-dev, SD3-turbo lineage): train a student $\epsilon_\theta(x, t, c, w)$ to regress the two-pass guided teacher output with $w$ embedded. Halves inference cost.

Costs of distillation: the model's distribution is the guided one — slightly mode-collapsed and saturated; you lose the unconditional branch, so **negative prompts and true CFG are gone**. Stacking CFG on FLUX.1-dev "double-guides" a model that already has guidance baked in; you get burnt images. That is why the candidate's Qwen-Image experiments use true two-pass CFG while FLUX-dev runs single-pass. Follow-up on cost: true CFG doubles NFE and, batched, doubles activation memory; sequential halves memory but doubles latency.

</details>


### [03-12] Compare Euler, Heun, DPM-Solver++ and UniPC. What history does each keep, and what breaks if you change the state mid-trajectory — say you change resolution — without resetting the solver?  (difficulty 3; tags: diffusion, samplers, ode-solvers, multistep)

- Follow-up: in diffusers, what exactly do you reset?


<details><summary>Answer</summary>

**Euler and Heun are memoryless single-step methods; DPM-Solver++(2M) and UniPC are multistep — they build higher-order accuracy from a buffer of previous model outputs. That buffer assumes one continuous trajectory in one state space, so any discontinuity in $x$ makes the stored derivative estimates wrong.**

Work in EDM form: $\mathrm{d}x/\mathrm{d}\sigma = (x - \hat x_0(x, \sigma))/\sigma$.
- **Euler**: $x_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) \cdot d_i$. First order, 1 NFE/step, no history.
- **Heun**: Euler predictor, then average slope at both ends; second order, 2 NFE/step. EDM found it better than Euler at equal NFE; still memoryless.
- **DPM-Solver++**: exploits the semilinear structure — integrates the linear part exactly (exponential integrator in $\lambda$ = log-SNR) and Taylor-expands $\hat x_0(\lambda)$. The 2M (multistep) variant estimates the first derivative of $\hat x_0$ from the **previous step's model output**; 1 NFE/step, second order. Keeps a buffer of 1 (order 2) or 2 (order 3) past $\hat x_0$'s plus their $\lambda$'s.
- **UniPC**: unified predictor–corrector; the corrector reuses the *new* evaluation to refine the previous step, so it is one NFE for higher effective order. Keeps $k$ past outputs (`model_outputs`), their timesteps, and an order counter that ramps 1→2→3 at the start (`lower_order_nums`).

What breaks: the finite-difference derivative $(\hat x_0^{(i)} - \hat x_0^{(i-1)})/(\lambda_i - \lambda_{i-1})$ assumes both outputs are samples of the same smooth function along one trajectory. If you upsample the latent, re-noise, swap conditioning or change the prompt mid-way, the buffered outputs are (a) the wrong shape — crash, or (b) the right shape after resizing but from a different trajectory and a different SNR — the extrapolation is garbage, you get ringing, ghosted structure, or blown contrast that shows up one or two steps after the transition and looks like a model bug. Euler has no such problem, which is why staged-resolution pipelines either use Euler across a transition or **reset the solver**: clear the output buffer, set the order counter to 0 so the next steps are first-order again, and set the step index so the $\sigma$ schedule is picked up at the right place. In diffusers that is `scheduler.model_outputs = [None]*k`, `lower_order_nums = 0`, `_step_index` / `set_begin_index`.

Independent of solver: the $\sigma$ grid (Karras $\rho=7$, shift, trailing spacing) is a separate choice; a good grid matters as much as solver order at ≤20 steps.

</details>


### [03-13] Going from 50 steps to 10, what degrades and why? Separate discretization error from model error, and explain exposure bias and how autoguidance or noise augmentation address it.  (difficulty 2; tags: diffusion, samplers, exposure-bias, error-analysis)

- Follow-up: why does quality plateau at ~30–50 steps and sometimes get *worse* with more deterministic steps?


<details><summary>Answer</summary>

**Three error sources: discretization error (shrinks with more steps and solver order), model error (independent of steps — the network is not the true score), and exposure bias (the model is evaluated on its own imperfect $x_t$, which is off the training distribution). Cutting steps mostly raises the first; the plateau is set by the other two.**

- **Discretization**: the ODE is integrated with steps of size $h$; local error $O(h^{p+1})$, global $O(h^p)$ for a $p$-th order solver. The trajectory is curved because the marginal velocity changes along the path — at large steps the model gives an *instantaneous* velocity where you need the *average* one. Symptoms at 10 Euler steps: blur, missing high-frequency detail, oversmoothed textures; the coarse layout is fine because it is decided by the first few steps anyway.
- **Model error**: $\epsilon_\theta \neq$ true $\epsilon$. Does not shrink with steps; at 50+ steps the deterministic sampler is essentially converged and the remaining gap is this. Stochastic samplers can *reduce* it: the Langevin term re-projects onto $p_t$, which is why EDM churn / SDE samplers beat ODE samplers at high NFE.
- **Exposure bias**: training uses $q(x_t \mid x_0)$ — exact noising of real data — but at inference $x_t$ is the sampler's own output, carrying the accumulated errors. The network extrapolates; errors compound over steps. This is also why more deterministic steps can get slightly worse: you take more opportunities to drift.

Fixes:
- **Noise augmentation / input perturbation** in training: add extra Gaussian noise to $x_t$ (or to the conditioning signal in cascades — Imagen, SVD, SDXL refiner) so the model sees inputs that look like imperfect predictions, and condition on the augmentation level.
- **Autoguidance**: $\tilde\epsilon = \epsilon_{\text{bad}} + w(\epsilon_{\text{good}} - \epsilon_{\text{bad}})$. The weak model's errors are amplified versions of the strong model's, so the difference cancels systematic bias — including the drift from being off-distribution.
- **$\epsilon$-scaling** at inference or SDE "churn" to re-noise and let the model correct.
- **Distillation** is the real few-step answer: teach the student the average velocity over a big step so discretization error disappears by construction.

</details>


### [03-14] Compare progressive distillation, consistency models, DMD, and adversarial distillation. What do you give up when you go to one step?  (difficulty 3; tags: diffusion, distillation, few-step)

- Follow-up: why is DMD mode-seeking, and is that good or bad?


<details><summary>Answer</summary>

**All of them teach a student to jump along the teacher's trajectory with one or a few evaluations; they differ in whether they supervise the trajectory (progressive, consistency) or the distribution (DMD, GAN), and that determines diversity, training cost, and what downstream features survive.**

- **Progressive distillation** (Salimans & Ho): student learns one step = two teacher DDIM steps; halve the step count each round. Needs $v$-parameterization (bounded targets at low step counts). Error compounds over rounds; $\log_2(T)$ training phases; keeps the deterministic map so it inherits teacher diversity.
- **Consistency models** (Song et al.): learn $f(x_t, t) = x_0$ for all $t$ on the same ODE trajectory with boundary condition $f(x, \epsilon) = x$ enforced by a skip parameterization. Train by consistency distillation (adjacent points from one teacher ODE step) or from scratch. One-step by construction; multistep by re-noising. LCM applies this to latents with CFG scale as input. Weakness: matching at adjacent points accumulates error along the trajectory; blurry at one step.
- **DMD** (Yin et al.): distribution matching. A one-step generator $G$ is trained so the score of its output distribution matches the teacher's: $\nabla_\theta \approx \mathbb{E}\big[(s_{\text{real}}(x_t) - s_{\text{fake}}(x_t)) \cdot \partial G/\partial \theta\big]$, where $s_{\text{fake}}$ is an auxiliary diffusion model kept trained on $G$'s outputs (same idea as VSD). DMD2 drops the regression loss, uses two-timescale updates and adds a GAN loss. It minimizes reverse KL → **mode-seeking**: excellent fidelity, reduced diversity. Three networks in memory.
- **Adversarial** (ADD/SDXL-Turbo, LADD): discriminator on the student's outputs (in pixel space with a DINO backbone, or in latent space using teacher features) plus a score-distillation term. Fast, sharp, but GAN instability and mode dropping.

What one step costs: diversity and mode coverage (reverse-KL or GAN objectives); no ability to trade steps for quality; CFG is baked in (you cannot change $w$, use negative prompts, or apply guidance interval); inversion and editing tricks break because there is no trajectory; ControlNet/adapter compatibility often degrades; artifacts in fine text and hands. The practical sweet spot is 4–8 steps, where a distilled model keeps most of the teacher's controllability. Follow-up: reflow (rectified flow) is the FM-native alternative — straighten first, then a one-step student is almost exact.

</details>


### [03-15] Why does EMA of the weights matter so much for diffusion models, and what can go wrong with it?  (difficulty 1; tags: diffusion, training, ema)

- Follow-up: how should the EMA half-life scale with training length?


<details><summary>Answer</summary>

**The diffusion loss is intrinsically noisy — the regression target is a random $\epsilon$ that cannot be predicted, so the minimum loss is nonzero and every minibatch gradient carries large target variance. Raw weights jitter around the basin; EMA averages that jitter out and gives a lower-variance estimate of the denoiser, which can be worth several FID points, sometimes 2×.**

$\theta_{\text{ema}} \leftarrow \beta\, \theta_{\text{ema}} + (1-\beta)\, \theta$, $\beta \approx 0.9999$, i.e. an average over roughly $1/(1-\beta) \approx 10\text{k}$ steps. This is Polyak averaging; for a noisy quadratic it converges to the basin center while the iterate oscillates. Diffusion is worse than classification here because the target noise never decreases with training, and because sample quality is very sensitive to small weight perturbations at high-SNR timesteps.

Things that go wrong:
- **No warm-up**: with $\beta = 0.9999$ from step 0 the EMA is dominated by random init for thousands of steps; use a $\beta$ ramp or start EMA later.
- **Wrong length for the run**: too short and you keep the noise; too long and you average over pre-convergence weights (a common reason a short fine-tune "does nothing" — the EMA barely moved). Karras et al. 2024 show the optimal EMA length scales with training length and batch size, and propose power-function EMA plus **post-hoc EMA** (store a few snapshots, reconstruct any EMA length after the fact).
- Precision: keep the EMA copy in fp32; bf16 EMA loses the small updates.
- Sampling with the raw weights by mistake during eval — one of the first things to check when "the loss is fine but samples are bad".
- With LoRA or DMD-style distillation the EMA copy should be of the trained parameters only.

</details>


### [03-16] Latent versus pixel diffusion — what is the VAE's role, why is the KL weight so small, and what is the scaling factor for?  (difficulty 1; tags: diffusion, latent, vae)

- Follow-up: why do 16- or 32-channel VAEs make the diffusion model's job harder?


<details><summary>Answer</summary>

**The VAE is a perceptual compressor that gives the diffusion model an 8× (spatially) smaller, roughly unit-variance, smooth space; the KL term is a light regularizer, not a generative prior; the scaling factor makes the latents match the noise schedule's assumption that data has unit variance.**

Cost motivation: a DiT's attention is quadratic in tokens. 1024² pixels → 128² latents (8×) → 64² tokens with 2×2 patchify = 4096 tokens. Video VAEs (WAN: 4× temporal, 8× spatial, 16 channels) matter even more.

VAE training: encoder gives $\mu, \sigma$; reconstruction with L1 + LPIPS + a patch GAN loss for sharpness; the KL to $\mathcal{N}(0, I)$ has weight ~1e-6. Reason: a real VAE with a strong prior gets posterior collapse / blurry reconstructions; here we only want the latent space **smooth and centered without holes**, so the diffusion model doesn't have to learn a jagged manifold. It is essentially a regularized autoencoder.

Scaling factor: latents come out with std ≈ 5.5 for SD1.x, so `0.18215 = 1/std` scales them to unit variance. Diffusion's SNR definition assumes unit-variance data; without the scaling the whole schedule shifts (too much signal at every $t$). FLUX/SD3 use a per-model shift and scale (e.g. shift 0.1159, scale 0.3611). Forgetting to apply — or double-applying — this factor is a classic "trains but samples garbage" bug.

Trade-offs: latent diffusion has a quality ceiling set by the decoder (text, faces, thin structures) and the 8× compression; pixel diffusion has none but must use cascades or heavy patching (PixelGen in the candidate's evaluation). More channels (16, 32 in DC-AE) improve reconstruction but the latent distribution gets harder to model — more information per token, less spatially redundant — so you need more diffusion compute or representation alignment (VA-VAE, REPA-style). Latents still show a power-law spectrum, just with a different $\beta$ than pixels.

</details>


### [03-17] Derive why diffusion generates coarse-to-fine. Under $x_t = (1-t)\,x_0 + t\,\epsilon$ with data power spectrum $P(f)$, at what time does frequency $f$ become resolvable, and what does that imply for compute?  (difficulty 3; tags: diffusion, spectral, coarse-to-fine, derivation, speed)

- Follow-up: what does the optimal denoiser do to frequencies below the noise floor?


<details><summary>Answer</summary>

**Noise is white — equal power at every frequency — while natural images are red, $P(f) \propto f^{-\beta}$ with $\beta \approx 2$. So low frequencies are the first to exceed the noise floor as $t$ decreases, and the denoiser cannot say anything about a frequency until its signal power beats $t^2$. That gives an explicit activation time per frequency.**

Derivation. Apply an orthonormal DCT/FFT. Because $\epsilon$ is i.i.d. unit-variance and the transform is orthonormal, every coefficient of $\epsilon$ is still unit variance: **noise power $= t^2$ in every bin**. The signal coefficient at radial frequency $f$ has power $(1-t)^2 P(f)$. Define activation as SNR = 1:
$$
\begin{aligned}
(1-t)^2 P(f) &= t^2 \quad \Rightarrow \quad \frac{1-t}{t} = P(f)^{-1/2} \\
t^{\ast}(f) &= \frac{1}{1 + P(f)^{-1/2}}
\end{aligned}
$$
With $P(f) = A f^{-\beta}$: $t^{\ast}(f) = 1/\big(1 + f^{\beta/2}/\sqrt{A}\big)$. $P$ decreasing in $f$ ⇒ $t^{\ast}$ decreasing in $f$: DC and low frequencies activate at large $t$ (early in the reverse process), high frequencies last. Inverting, the highest resolvable frequency at time $t$ is
$$
f_{\text{act}}(t) = \left(\sqrt{A} \cdot \frac{1-t}{t}\right)^{2/\beta},
$$
which for $\beta=2$ is just $\propto (1-t)/t$ — the noise-to-signal ratio.

The optimal (MMSE) denoiser makes this concrete: for a Gaussian model, per frequency $\mathbb{E}[x_0 \mid x_t]$ is a Wiener filter, $\hat x_0(f) = \frac{P(f)}{P(f) + t^2/(1-t)^2} \cdot \frac{x_t(f)}{1-t}$. Frequencies with $P(f) \ll$ noise are shrunk to ~0 — the network output is low-pass filtered early on, and the cutoff moves up as $t \to 0$. A trained denoiser behaves the same way; that is the whole coarse-to-fine phenomenon, not an architectural choice.

Compute implication: before $t^{\ast}(f_{\text{Nyquist}}(r))$ nothing above the Nyquist frequency of a lower resolution $r$ is resolvable, so those steps can be run at resolution $r$ with no information loss, and stepped up when the next band activates (the SPEED schedule; measured $\beta$: FLUX 1.92, WAN 2.42). In DDPM variables the same condition is $\mathrm{SNR}(t) \cdot P(f) = 1$.

Caveats to volunteer: assumes stationarity and ignores phase; activation is a soft threshold so a margin is needed; and this is per-frequency SNR — the per-pixel SNR is a poor proxy, which is the same lesson as the resolution shift.

</details>


### [03-18] Your diffusion model trains with a healthy loss curve but samples garbage. Walk me through your debugging checklist.  (difficulty 2; tags: diffusion, debugging, training, practical)

- Follow-up: what single test localizes a train/inference mismatch fastest?


<details><summary>Answer</summary>

**Bisect train versus inference: the loss says the network computes something sensible on $q(x_t \mid x_0)$, so the bug is almost always a convention mismatch between how you noised and how you sample. Start with the one-step denoising test, then go down the list.**

The localizing test: take a real training image, encode, noise to a chosen $t$ with the *training* code, run one forward pass, convert the output to $\hat x_0$ with the *sampler's* conversion, decode and compare to the original (PSNR vs $t$). If $\hat x_0$ is good at every $t$, the network and parameterization are fine and the bug is in the solver loop; if it is bad only at high $t$, look at the schedule ends and conditioning; bad everywhere → conversion.

Checklist:
- **Parameterization ↔ scheduler** (`prediction_type`): $\epsilon$ vs $v$ vs $x_0$ vs FM velocity; sign of the velocity ($\epsilon - x_0$ vs $x_0 - \epsilon$); whether $t=0$ is data or noise.
- **Timestep convention**: discrete index 0–999 vs continuous $[0,1]$; timestep shift applied at train but not inference (or vice versa); `trailing` vs `leading` spacing.
- **VAE scaling / shift factor** applied once, in the same direction, both sides; decode with the right VAE.
- **Initial noise**: $x_T \sim \mathcal{N}(0, I)$ but does the schedule reach SNR 0? For EDM-style, $x_T = \sigma_{\max} \epsilon$, not $\epsilon$. Zero-terminal-SNR mismatch gives grey, low-contrast images.
- **EMA**: are you sampling the EMA weights? Is the EMA long enough to have moved?
- **CFG**: was condition dropout on during training? Is the null condition the same tensor at inference? $w$ too high → burnt; $w=1$ to isolate.
- **Conditioning plumbing**: text encoder in eval mode, attention masks, padding tokens, dtype of embeddings.
- **Numerics**: schedule math ($\bar\alpha, \sigma, \lambda$) in fp32 even if the model is bf16; NaN/inf guards off so you see failures; check `torch.compile` graph breaks changing dtype.
- **Data normalization**: $[-1, 1]$ vs $[0, 1]$ mismatch shows up as washed-out or clipped samples.
- **Per-timestep loss histogram**: flat loss in some $\lambda$ range means those steps are not being learned (bad $p(t)$).
- **Overfit one batch**: if a 16-image subset can't be memorized and reproduced from its own noise, the pipeline is broken, not the model.
- **Solver sanity**: replace fancy multistep with 200-step Euler/DDIM; if that fixes it, the solver state (order buffer, step index) is stale or the $\sigma$ grid is wrong.

</details>


### [03-19] What is a diffusion model, in one sentence?  (difficulty 1; tags: diffusion, definition, rapid-fire)


<details><summary>Answer</summary>

**A diffusion model learns to reverse a fixed process of gradually adding Gaussian noise to data, by training a network to denoise at every noise level, then generates by starting from pure noise and denoising step by step.**

Why it works: instead of learning the whole distribution in one shot (GAN, VAE), it breaks generation into many small, easy conditional steps — each a regression problem with a plain MSE loss, stable to train and scalable. The network implicitly learns the score $\nabla_x \log p_t(x)$ at every noise level, and sampling is numerical integration of an ODE or SDE driven by it.

Trap: the cost moves to inference — tens of network evaluations per sample instead of one — which is why samplers, distillation, and coarse-to-fine tricks like SPEED are a research area.

</details>


### [03-20] What is the noise schedule?  (difficulty 1; tags: diffusion, noise-schedule, rapid-fire)


<details><summary>Answer</summary>

**The noise schedule says how much signal and noise are mixed at each timestep: $x_t = \alpha_t\, x_0 + \sigma_t\, \epsilon$, from clean data at $t = 0$ to pure noise at $t = T$.**

Common choices: DDPM's linear $\beta$ schedule (variance-preserving, $\alpha_t^2 + \sigma_t^2 = 1$), cosine, EDM's log-normal over $\sigma$, and the flow-matching interpolant $\alpha_t = 1-t$, $\sigma_t = t$. It defines *which* denoising problems the network trains on and how densely — a distribution over difficulty. The invariant is log-SNR $\lambda_t = \log(\alpha_t^2/\sigma_t^2)$; schedules with the same $\lambda$ range differ only in step spacing.

Trap: it must reach SNR ≈ 0 at $T$ or the model never sees pure noise and produces low-contrast images (zero-terminal-SNR), and it must be shifted at higher resolution because per-pixel noise averages out over bigger images.

</details>


### [03-21] What is the SNR at a timestep?  (difficulty 1; tags: diffusion, snr, noise-schedule, rapid-fire)


<details><summary>Answer</summary>

**$\mathrm{SNR}(t) = \alpha_t^2 / \sigma_t^2$, the ratio of signal to noise power in $x_t = \alpha_t\, x_0 + \sigma_t\, \epsilon$; log-SNR $\lambda_t$ is the natural coordinate for everything in diffusion.**

It matters because the schedule's index ($\beta$, $t \in [0,1]$, $\sigma$) is arbitrary but SNR is not: the denoising task at a given SNR is the same task however you index it. Loss weightings (min-SNR, P2), the resolution timestep shift, parameterization choice ($\epsilon$ at high SNR, $x_0$ at low), and step spacing are all cleanly stated in $\lambda$. Under flow matching, $\mathrm{SNR} = \big((1-t)/t\big)^2$.

Trap: SNR is per-pixel, but perception is per-frequency: low frequencies carry far more power, so at a given $t$ coarse structure is already resolved while fine detail is still buried — the coarse-to-fine behavior, and the reason high-res models need a shifted schedule.

</details>


### [03-22] What is the score function?  (difficulty 1; tags: diffusion, score, rapid-fire)


<details><summary>Answer</summary>

**The score is the gradient of the log density with respect to the data, $\nabla_x \log p(x)$: a vector field pointing toward higher-probability regions.**

It exists because it sidesteps the normalizing constant — $\nabla \log p$ doesn't depend on $Z$ — so you can learn it without ever computing a likelihood. For the noised marginal $p_t$ the score is exactly what a denoiser gives you: $\nabla_x \log p_t(x_t) = -\hat\epsilon / \sigma_t$ (Tweedie), so $\epsilon$-prediction *is* score estimation up to scale. Sampling then follows the score: Langevin dynamics, the reverse SDE, or the probability-flow ODE.

Trap: the raw data score is ill-defined off the data manifold (zero density), which is precisely why we noise the data — $p_t$ is smooth with full support for $\sigma > 0$, so the score exists everywhere and points back toward the manifold.

</details>


### [03-23] What is Tweedie's formula?  (difficulty 2; tags: diffusion, tweedie, denoising, rapid-fire)


<details><summary>Answer</summary>

**Tweedie: the posterior mean of the clean signal given a Gaussian-corrupted observation is the observation plus noise variance times the score, $\mathbb{E}[x_0 \mid x_t] = \big(x_t + \sigma_t^2\, \nabla \log p_t(x_t)\big) / \alpha_t$.**

It links the two views of diffusion in one line: the optimal MSE denoiser and the score are the same object. From it, $\hat\epsilon = -\sigma_t \nabla \log p_t$, $\hat x_0 = (x_t - \sigma_t \hat\epsilon)/\alpha_t$, and every parameterization conversion follows. It's also the tool behind training-free guidance and inverse problems (DPS): you get an $x_0$ estimate at any step to compute a measurement gradient against.

Trap: $\hat x_0$ is a *posterior mean*, so at high noise it's a blurry average of all plausible images — a one-step $\hat x_0$ at $t = T$ is grey mush. Sampling re-noises or integrates the ODE precisely so you land on a mode, not the mean.

</details>


### [03-24] What is v-prediction?  (difficulty 1; tags: diffusion, parameterization, v-prediction, rapid-fire)


<details><summary>Answer</summary>

**$v$-prediction has the network output $v = \alpha_t\, \epsilon - \sigma_t\, x_0$, the velocity of $x_t$ along the trajectory; for a variance-preserving schedule, $\hat\epsilon = \alpha_t\, \hat v + \sigma_t\, x_t$ and $\hat x_0 = \alpha_t\, x_t - \sigma_t\, \hat v$.**

It exists because $\epsilon$-prediction blows up at low SNR ($x_0 = (x_t - \sigma_t \hat\epsilon)/\alpha_t$ divides by a tiny $\alpha_t$ and amplifies error) and $x_0$-prediction is useless at high SNR (the target is trivially $x_t$). $v$ mixes both, so the target has unit variance and the implied loss weight is well-behaved at *every* SNR — what made progressive distillation and zero-terminal-SNR sampling work.

Trap: for the flow-matching interpolant $x_t = (1-t)\,x_0 + t\,\epsilon$ the velocity is $\epsilon - x_0$ — same idea, different sign — so pairing a $v$-pred checkpoint with an FM scheduler silently flips the sign: the classic "healthy loss, garbage samples" bug.

</details>


### [03-25] Explain classifier-free guidance in one minute.  (difficulty 1; tags: diffusion, cfg, guidance, rapid-fire)


<details><summary>Answer</summary>

**CFG trains one network for conditional and unconditional denoising by randomly dropping the condition (~10%), then at sampling extrapolates: $\tilde\epsilon = \epsilon(\varnothing) + w \cdot (\epsilon(c) - \epsilon(\varnothing))$ with $w > 1$.**

Why: unguided conditional samples follow the prompt weakly. The difference $\epsilon(c) - \epsilon(\varnothing)$ is proportional to $\nabla \log p(c \mid x)$, an implicit classifier gradient, so amplifying it sharpens toward the condition — sampling from $p(x \mid c) \cdot p(c \mid x)^{w-1}$, a lower-temperature distribution in prompt-relevant directions. It replaced classifier guidance because it needs no separate noise-aware classifier.

Traps: two forward passes per step (batch them); too-high $w$ oversaturates and collapses diversity because the extrapolated $\epsilon$ leaves the manifold — hence guidance intervals, rescaling, and distillation into one pass. The null condition must be the same tensor at train and inference.

</details>


### [03-26] DDPM versus DDIM sampling — in one breath.  (difficulty 1; tags: diffusion, ddim, ddpm, samplers, rapid-fire)


<details><summary>Answer</summary>

**DDPM reverses the Markov chain one step at a time with fresh noise injected each step — stochastic, ~1000 steps; DDIM uses the same trained network and marginals but a deterministic, non-Markovian update that lets you skip steps — same model, ~20–50 steps.**

The DDIM step: predict $\hat x_0$ from $\hat\epsilon$, then jump to the next timestep, $x_{t'} = \alpha_{t'}\, \hat x_0 + \sigma_{t'}\, \hat\epsilon$ — Euler on the probability-flow ODE in a good coordinate. Determinism means a fixed seed gives a fixed image, enabling inversion (encode a real image to its noise) and smooth latent interpolation.

Trap: DDIM's $\eta$ knob interpolates back toward DDPM stochasticity; a little noise corrects accumulated error at many steps, but at few steps deterministic wins. Neither retrains anything — sampling is a property of the solver, not the model.

</details>


### [03-27] What is a consistency model?  (difficulty 2; tags: diffusion, consistency, distillation, few-step, rapid-fire)


<details><summary>Answer</summary>

**A consistency model learns $f(x_t, t)$ that maps any point on a probability-flow ODE trajectory directly to that trajectory's endpoint $x_0$, with the constraint that all points on one trajectory give the same output — self-consistency.**

Why: a diffusion model reaches $x_0$ only after integrating the ODE over many steps; if $f$ is consistent, one evaluation from pure noise is a sample, and you trade quality for steps by re-noising and calling $f$ again. Training enforces $f(x_t, t) \approx f(x_{t'}, t')$ for adjacent points on a trajectory, via a teacher solver (consistency distillation) or a one-sample trajectory estimate (consistency training). The boundary condition $f(x_0, 0) = x_0$ is baked into the parameterization.

Trap: multi-step consistency sampling doesn't converge to the teacher — errors don't cancel — and training is unstable (step-count curriculum, EMA target). LCM is the latent version behind 4-step SDXL.

</details>


### [03-28] Flow matching versus diffusion — what's the difference, in one minute?  (difficulty 1; tags: flow-matching, diffusion, comparison, rapid-fire)


<details><summary>Answer</summary>

**Both learn a vector field transporting noise to data along a prescribed path; diffusion uses a Gaussian noising process (curved variance-preserving path, $\epsilon$ / score target), flow matching uses the straight interpolant $x_t = (1-t)\,x_0 + t\,\epsilon$ and regresses the constant velocity $\epsilon - x_0$.**

What actually differs: (1) the path — straight lines make the ODE nearly linear, so few Euler steps suffice; (2) the target — unit variance at all $t$, no SNR-dependent blow-up; (3) the framing — no SDE or ELBO, just "regress a velocity along a coupling", which extends trivially to non-Gaussian sources and paired couplings (reflow, OT couplings).

Trap: it's a reparameterization, not a new model class — FM with the linear schedule is diffusion with $\alpha_t = 1-t$, $\sigma_t = t$ and a $v$-like target; its score is recoverable via Tweedie. The wins in SD3/FLUX/WAN come from the straight path and timestep shift, not new theory.

</details>


## Image & Video Generation


### [04-01] Walk me through the VAE in a latent diffusion model. Why did the field move from 4-channel to 16-channel latents, and what does the "scaling factor" do?  (difficulty 1; tags: vae, latent-diffusion, architecture)

- Follow-up: KL-regularized vs VQ — which would you pick for a diffusion backbone and why?
- Follow-up: What happens if you skip the scaling factor?


<details><summary>Answer</summary>

**1. Why use a VAE?**

VAE is short for variational auto encoder, it's main goal is to reduce the computational and memory complexity of the diffusion models by allowing them to generate in a compact latent space encoded by the VAE. Also, by encoding into a compact representation, this surpresses pixel space information such as sensor noise, extreme high-frequency variations that are otherwise harder for the diffusion model to learn to generate. 

**Why VAE? **AE is auto encoder and is fully deterministic, it maps any RGB space input into a fixed latent representation $z$. VAE instead maps the input into a distribution, typically normal distribution with mean $\mu$ and std $\sigma$, and $z$ is then sampled from this posterior distribution $p(z \mid x)$. This allows the KL divergence loss to be applied to ensure that the encoded latents are well-behaved, typically KL is applied against standard normal. VAE tries to make the latent space smoother and densely organized. 

**2. What is the architecture of VAE?**

For image gen models, the VAE encoder is typically convolutional with conv layers, ResNet blocks with Group Norm, Conv 3x3, and SiLU, and potentially attention blocks at the bottleneck with the lowest spatial resolution. After a few blocks the resolution is typically reduced by 2x. Typically it's 8x spatial downsampling, $C$ channels, the output is $2C$ dimensional, $C$ for $\log \mu^2$ and $C$ for $\sigma$. The decoder is kind of the inverse of the architecture, with upsampling oeprations and ResNet blocks that uses Group Norm, SiLU, Conv3x3. There could also be attention blocks at the beginning/bottleneck.

For video gen models, the VAE encoder is still convolutional, typically 3D but causal convolutional, i..e the current frame cannot use information from any future frame. This can be useful as the video can be encoded in separate chunks and encode long video's causal structure. 

Aside: what is GroupNorm, why not BatchNorm? How is it different from LayerNorm?

Group Norm divides the channels into separate groups and normalizes per group. BatchNorm normalizes according to an entire batch. Typically the diffusion model training uses very small batches so BatchNorm doesn't stablize things very well, and there's training-inference mismatch. 

Layer Norm normalizes one large collection of features per sample (one G). So completed unrelated feature channels do not mix together but channels within a related subset also share normalization statistics. 

Aside 2: What is SiLU?

It's short for sigmoid linear unit, $x\,\sigma(x)$. The sigmoid function is $1/(1 + e^{-x})$, so SiLU is $x/(1 + e^{-x})$. For large positive $x$ it is roughly $x$, for very negative $x$ it is almost 0. Unlike ReLU, it doesn't abruptly chop all negative values to 0. SiLU is smooth so tend to work well. 
**3. How is the VAE trained?**

**The VAE is a lossy compressor: 8× spatial downsampling, $C$ channels, trained with reconstruction + perceptual (LPIPS) + adversarial loss and a tiny KL weight (~1e-6) so the latent is roughly Gaussian-scaled but essentially deterministic.** Diffusion then runs on a $64 \times 64 \times C$ tensor instead of 512×512×3 — 48× fewer elements at $C=4$.
 
 
- **KL vs VQ (Vector Quantization):** KL-VAE gives a continuous latent, which is what a Gaussian/flow diffusion process needs. VQ gives discrete codes in a look up table, which suits autoregressive/masked-token models (VQGAN, MaskGIT). For a diffusion backbone, continuous KL is the standard choice; VQ throws away information at the quantizer and the codebook collapse problem is a headache. 
- **4 → 16 channels (SD3, FLUX, SDXL's successor VAEs):** the 4-ch VAE at 8× compression is the reconstruction bottleneck — text, faces, fine texture come out wrong *before* diffusion even runs. SD3 showed rFID drops sharply with $C$ (4→16) while the diffusion model's FID-vs-compute curve stays favorable *if the model is big enough*; small models actually fit 4-ch latents faster because the latent is "easier". The trade: more channels = higher-entropy latent = harder to model, so it pays off only with scale. 16 channels also keeps compression rate (8× spatial) — you're paying with channels, not resolution. 
- **Scaling factor** (0.18215 for SD1.x, ~0.36 for SD3/FLUX with a per-channel shift): multiplies latents so their std ≈ 1, matching the unit-variance noise. Skip it and the SNR schedule is silently wrong — at $t=0.5$ the "signal" is ~5× weaker than intended, so the effective schedule shifts toward noisier and the model trains badly. It's the same reason SPEED normalizes the DCT so noise has unit power per bin: $\mathrm{SNR}=1$ must mean $\mathrm{SNR}=1$. 
 

Trap: people say "VAE" but the KL weight is so small it's effectively an autoencoder; the KL term just keeps the latent scale bounded.

</details>


### [04-02] Compare cross-attention text conditioning (SD1/SDXL) with MMDiT-style joint attention (SD3/FLUX). Why did the field move to joint attention, and what's the cost?  (difficulty 2; tags: conditioning, architecture, dit, text-to-image)

- Follow-up: Why keep both CLIP and T5 encoders?
- Follow-up: What are "in-context" conditioning tokens and when are they preferable?


<details><summary>Answer</summary>

**Cross-attention injects a fixed text-encoder output as keys/values in every block; MMDiT concatenates text tokens and image tokens into one sequence with separate weights per modality and lets both streams update each other.**
 
 
- **Cross-attn:** image queries, text K/V. Strength: Text representation is frozen after the encoder; cheap (text has ~77-256 tokens, so cost is $O(N_{\text{img}} \times N_{\text{txt}})$). Weakness: text never sees the image, so compositional/spatial binding ("red cube left of blue sphere") is weak. 
- **MMDiT joint attention:** sequence = [text tokens; image tokens], one attention over both, with modality-specific QKV/MLP weights (two "streams") and shared attention. Text tokens are refined conditioned on the current image estimate — that's what gives SD3/FLUX their prompt-following jump. FLUX uses ~19 double-stream blocks then ~38 single-stream blocks (shared weights) to save parameters. 
- **Cost:** attention is now $O((N_{\text{img}} + N_{\text{txt}})^2)$ instead of $O(N_{\text{img}}^2) + O(N_{\text{img}} \cdot N_{\text{txt}})$. At 4096 image tokens + 512 text tokens that's ~27% more attention FLOPs. Also text tokens now sit in the RoPE grid (FLUX gives them position 0 on the image axes ($(0, 0, 0)$). 
- **Why keep CLIP + T5:** CLIP-L/G give a pooled global vector (aligned with images, good for style/global semantics) that feeds the timestep-modulation (adaLN) path; T5-XXL gives ~4.7B-param sequence tokens with real language understanding for long, compositional prompts. In T5, every token gets its own contextual vector embedding. Clip instead gives one global vector. SD3 ablation: dropping T5 hurts text rendering and complex prompts, dropping CLIP hurts aesthetics slightly. Newer models (Qwen-Image, Z-Image) use a VLM/LLM encoder as a single strong text tower.
- Aside: what is the adaLN path: for FLUX tells every transformer block what is the current timestep $t$, and $e_t = \mathrm{TimeEmbed}(t)$. then the embedding adds the clip vector and adds the guidance embedding. So adaLN first applies regular LN, $\hat{h} = \mathrm{LN}(h)$, then do $h_{\text{mod}} = (1 + \text{scale})\,\hat{h} + \text{shift}$. Then the gate controls how strongly the block's output is added back through the residual connection, $h_{\text{out}} = h + \text{gate} \cdot F(h_{\text{mod}})$. so adaLN is a small neural layer that convert the conditioning vector into per-feature modulation params.  
- **In-context tokens:** conditioning images (reference, edit source, ControlNet-like maps) are patchified and concatenated as extra tokens with their own position ids, rather than added as channels. Preferable when the condition isn't pixel-aligned with the output (reference subject, multi-image editing) — the model learns correspondence via attention. FLUX Kontext / OmniGen do this. Cost is token count, so it's exactly where token-reduction ideas matter.

</details>


### [04-03] What actually scales in a DiT? Depth, width, or token count — and how do Gflops relate to FID?  (difficulty 1; tags: dit, scaling, architecture)

- Follow-up: Is patch size 2 vs 4 a "free" 4× compute saving?


<details><summary>Answer</summary>

**Peebles & Xie's finding: FID is a near-monotone function of forward-pass Gflops, almost regardless of whether you get those Gflops from depth, width, or more tokens (smaller patch).** DiT-XL/2 at ~119 Gflops/forward hits FID 2.27 on ImageNet-256 with CFG; the S/B/L/XL × patch-8/4/2 grid lines up on one Gflops-vs-FID curve.
 
 
- **Depth/width:** parameter count scales as $\approx 12 L d^2$ for the transformer; per-token FLOPs $\approx 2 \times$ params. Both work; width is more hardware-efficient (bigger matmuls), depth gives more "sequential computation". 
- **Tokens (patch size):** halving patch size from 4 to 2 gives 4× tokens, ~4× MLP FLOPs and 16× attention FLOPs, with *zero* new parameters — and it improves FID more than adding parameters at equal Gflops. So no, patch 4 is not free: you lose fine detail because each token must explain 16× more pixels. This is the key insight behind token-count-driven methods (SPEED, Foveated Diffusion): tokens are where compute goes, and where quality comes from — so spend them where and when they matter. 
- **Trends at scale:** loss follows a power law in compute; SD3 showed validation loss correlates with human preference across 0.8B-8B. FID saturates and stops discriminating at large scale (see the metrics question). 
- Interviewer's real question: "given 4× more compute, what do you do?" Answer: mostly more tokens / higher res and more data, some width; depth beyond ~40-60 layers has diminishing returns at these scales, and you need the VAE to keep up. 
Once you already have a reasonably large DiT, extra compute is often better used to let the model process a richer image representation—more spatial tokens/higher resolution—while also increasing width (channel size) and training data, rather than simply stacking many more identical Transformer layers. And if you increase what the DiT can model, the autoencoder must preserve enough detail for those gains to survive into pixels.

</details>


### [04-04] Describe the architecture of a modern video diffusion model like WAN 2.1 or CogVideoX: the 3D VAE, the attention pattern, and the position encoding.  (difficulty 2; tags: video, architecture, vae, attention, rope)

- Follow-up: Why a *causal* 3D VAE?
- Follow-up: Full spatiotemporal attention vs factorized — which wins, and why did the field move to full?


<details><summary>Answer</summary>

**Three pieces: a 3D causal VAE compressing 4× in time and 8× in space to 16 channels, a DiT with full spatiotemporal self-attention over all latent tokens, and 3D RoPE with separate frequency bands for $(t, h, w)$.**
 
 
- **3D causal VAE:** 81 frames at 832×480 → $(1 + 80/4) = 21$ latent frames × 60×104 spatial × 16 ch. Causal convolutions (pad only on the past) mean the first frame is encoded alone — so an image is just a 1-frame video, which unifies T2I/I2V training and lets you encode arbitrarily long videos chunk-wise with a cache. CogVideoX also uses causal 3D conv; WAN adds a feature cache for chunked decode to bound memory.
- Aside: what is chunked decoding, why do we want to use it? For a long video, decoding it could consume too much memory, but chunking creates a problem that we are throwing away decoded context. So feature cache essentially saves only the tail of each decoded chunk's feature as cache for next chunk's decoding.  
- **Tokens:** patchify (1, 2, 2) → 21 × 30 × 52 ≈ 32.8k tokens for 480p/81f. At 720p that's ~75k. This is why attention dominates cost — see the inference-cost question. 
- **Full 3D attention vs factorized:** early models (VDM, Make-A-Video, AnimateDiff) inserted temporal-attention layers into a 2D UNet — cheap ($O(T^2)$ per pixel + $O(HW^2)$ per frame), but motion is modeled only along a pixel's own column, so large motion, camera moves, and object permanence suffer. Full attention (Sora, CogVideoX, WAN, HunyuanVideo) lets any token attend to any $(t, x, y)$; cost is $O((THW)^2)$ but quality is decisively better and it scales with data. Middle ground: windowed/sparse attention (STA, sliding-tile) for speed. 
- **3D RoPE:** head dimension split into three chunks (e.g., 128 → 44 + 42 + 42 for t/h/w in WAN, 16+56+56 in FLUX for the [txt, h, w] axes), each getting 1D RoPE over its coordinate. Gives relative-position awareness in all three axes and some extrapolation ability. Text tokens in WAN go through cross-attention (not joint) so they need no position. 
- **Conditioning:** timestep via adaLN (WAN uses a single shared adaLN modulation across blocks with per-block learned biases to save params); text via cross-attention from umT5; I2V via concatenated latent of the first frame + mask channels. you also make a mask telling the network:

>  

“This location comes from the provided conditioning image.”
 
 

versus:
 

>  

“This location is something you need to generate.”

 

Then all three tensors are concatenated along the **channel dimension**:
$$
x_{\text{model}} = \operatorname{concat}_{C} \left[ z_t,\, z_{\text{cond}},\, m \right].
$$

So with a 16-channel video latent and a one-channel mask, you'd conceptually get
$$
16 + 16 + 1 = 33
$$

input channels:

</details>


### [04-05] Why does a diffusion model degrade when you sample above its training resolution, and what are the fixes?  (difficulty 3; tags: resolution, rope, extrapolation, video)

- Follow-up: Explain NTK-aware / YaRN-style RoPE scaling in one breath.
- Follow-up: Why does "generate at native res for most of the trajectory, then upscale late" work so well?


<details><summary>Answer</summary>

**Two failure modes: (1) positional extrapolation — RoPE has never seen positions beyond the training grid, so attention patterns become unreliable and you get repeated objects / duplicated limbs; (2) noise-schedule mismatch — at higher res the per-pixel SNR at a given $t$ is effectively higher (more redundant pixels average out noise), so the model's timestep semantics shift.**

- **RoPE extrapolation:** RoPE rotates q/k by angle $\text{pos} \times \theta_i$ with $\theta_i = \text{base}^{-2i/d}$. Low-frequency dims have periods longer than the training length and have only seen a fraction of a cycle; at 2× length they hit unseen phases. Fixes: **position interpolation** (scale positions by $1/s$ — keeps in-distribution but compresses fine detail), **NTK-aware** (raise the base so high-freq dims are untouched and low-freq dims are interpolated — "change base so the longest wavelength stretches by $s$"), **YaRN** (per-dimension: interpolate dims whose wavelength exceeds context, leave high-freq dims alone, blend in between, plus a temperature on attention logits). In video/image: FLUX/WAN users apply NTK scaling to the h/w axes; models like FiT/Lumina train with variable grids so RoPE sees many scales.
- **Schedule mismatch:** SD3 shifts the timestep schedule with resolution — shift factor $\propto \sqrt{N_{\text{tokens}} / N_{\text{base}}}$ — so at 4× tokens you spend more steps at high noise. Same logic as SPEED's SNR correction: embedding a signal into a bigger grid spreads energy, so the noise level that "means" a given structure level moves.
- **Why native-then-upscale works:** the coarse structure (layout, composition, low frequencies) is decided in the early, high-noise steps; that's exactly where the model must be in-distribution, and at native res it is. Late steps only add high frequencies, which are local — a few steps of RoPE-extrapolated high-res denoising with well-established low frequencies is easy (it's essentially guided super-resolution). SPEED's 960p WAN result: 3.8× faster *and* better than full-res, because ~80% of the trajectory never enters the extrapolation regime. The general principle: **the trajectory's low-frequency content is the extrapolation-sensitive part; keep it in-distribution.**

</details>


### [04-06] Compare progressive / multi-resolution generation strategies: cascaded diffusion, SDXL refiner, Matryoshka Diffusion, pixel-space models like PixelGen, your SPEED, and Foveated Diffusion. What's the axis each one cuts along?  (difficulty 3; tags: progressive, resolution, speed, foveated, cascades)

- Follow-up: Why does continuous per-step frequency masking blur, but staged reveal doesn't?
- Follow-up: Could SPEED and Foveated Diffusion be combined?


<details><summary>Answer</summary>

**They all exploit the same fact — coarse structure is cheap and determined early; detail is expensive and local — but cut along different axes: separate models per resolution (cascades), separate phase of one model (refiner), shared multi-scale weights (Matryoshka), *time* (SPEED), or *space* (Foveated).**

- **Cascaded diffusion (Imagen, DALL·E 2, Stable Cascade):** base 64² model + 1-2 super-res diffusion models conditioned on the low-res image, with noise-augmentation on the conditioning to bridge train/test gap. Pros: each stage is cheap and specialized. Cons: multiple models to train/serve, errors compound, SR stages hallucinate detail inconsistent with the base.
- **SDXL refiner:** same latent space, a second model specialized on the last ~20% of noise levels (img2img). It's a "denoising-expert" split in *time* — but no resolution change, so no speedup, only quality.
- **Matryoshka Diffusion:** one model jointly denoises a pyramid of resolutions in pixel space with nested U-Net features; progressive training schedule. Elegant but the multi-res target is still computed at full cost.
- **PixelGen / pixel-space DiTs:** skip the VAE; work directly on pixels with large patches or a pixel-pyramid. Removes VAE artifacts and decode cost but token counts explode — which is exactly why resolution scheduling is even more valuable there (SPEED evaluates on it).
- **SPEED (mine):** one pretrained model, no architecture change; the *resolution grows along the denoising trajectory*, with stage transitions placed where the spectrum says the next frequency band becomes signal-dominated ($(1-t)^2 P(f) = t^2$ with $P(f) = A f^{-\beta}$). Transitions are DCT embedding + noise fill for the new band + an SNR correction ($1/r_{\text{eff}}$ attenuation). Training-free or light LoRA. Speedup comes purely from fewer tokens per step early on: 7.09× on FLUX, 2.54× on WAN.
- **Foveated Diffusion (mine, co-author):** *spatial* axis — one forward pass mixes high-res tokens in a fovea and coarse tokens in the periphery; LoRA teaches the model mixed-res token statistics and position embeddings at multiple patch scales; 2× image / 4× video.

**Why continuous masking blurs but staged reveal doesn't:** with a brick-wall radial mask growing every step, the highest frequencies get revealed in the last 1-2 steps — they're step-starved — and per-step hard cutoffs create ringing that the model treats as signal. A staged reveal exposes an entire band at once at the point the spectrum says it's $\mathrm{SNR} \ge 1$, then gives it all remaining steps. Also: a low-pass at full res gives *no* speedup; only real token reduction does. **Combining:** yes — SPEED decides *when* tokens exist, Foveated decides *where*; the product of the two speedups is plausible since attention cost is quadratic in total tokens, and both are LoRA-compatible on the same backbone.

</details>


### [04-07] How do you build the training data for a text-to-image or text-to-video model? Talk about captions, aspect ratios, and resolution curriculum.  (difficulty 2; tags: data, captioning, training, text-to-image, video)

- Follow-up: What goes wrong if you train only on synthetic captions?


<details><summary>Answer</summary>

**Three levers: (1) dense synthetic captions from a VLM, (2) aspect-ratio bucketing so you never crop/distort, (3) a resolution and duration curriculum from low to high.**

- **Captions:** raw alt-text is short, noisy, and often unrelated. DALL·E 3's recipe: train a captioner to produce long, descriptive captions, then train the generator on ~95% synthetic / 5% original. Synthetic captions give far better prompt following and text rendering. Trap: train on 100% synthetic and you overfit to the captioner's phrasing — users write short prompts, so at inference you either get a prompt-style mismatch or you need an LLM prompt "upsampler". Mix caption lengths and styles; for video, caption motion and camera explicitly (VLMs describe frames, not dynamics — WAN/HunyuanVideo train dedicated video captioners).
- **Aspect-ratio bucketing (NovelAI / SDXL):** group images into buckets of equal token count but varying (h, w) — e.g., 1024², 832×1216, 1216×832 — so every batch has a uniform shape and no center-cropping ("headless people" problem). SDXL additionally conditions on original size and crop coordinates as micro-conditioning so the model learns not to reproduce low-res/cropped statistics.
- **Resolution / duration curriculum:** pretrain at 256², then 512², then 1024², with a small final high-quality/aesthetic stage; for video: images → low-res short clips → high-res long clips (WAN: 256p → 480p → 720p; HunyuanVideo similar), with joint image-video batches throughout so the model doesn't forget spatial quality. Low-res stages are where the compute goes — each 2× res step is 4× tokens and up to 16× attention FLOPs — which is the same intuition SPEED applies at inference time.
- **Filtering:** aesthetic score, watermark/OCR filters, CLIP similarity, dedup, motion-magnitude filters for video (drop static clips and hard cuts using scene detection).
- Data flywheel: the model itself + human preference data (see LoRA/RL post-training) is the last mile.

</details>


### [04-08] What's wrong with FID, and what would you use instead to evaluate a high-resolution T2I or a T2V model?  (difficulty 2; tags: evaluation, metrics, fid, fvd, vbench)

- Follow-up: Why is FID especially misleading at high resolution or for a speedup paper?


<details><summary>Answer</summary>

**FID measures Fréchet distance between Gaussians fit to Inception pool-3 features of 299×299 images — so it's (a) blind to anything Inception downsampling destroys, (b) biased toward ImageNet-like content, (c) assumes Gaussian features, (d) needs ~10-50k samples and is biased with fewer.**

- **Why bad at high res:** every image is resized to 299², so a 1024² image loses ~90% of its pixels; sharpness, text, and fine texture — the things high-res models are for — are invisible. Two models differing only in high-frequency detail can have identical FID. For a *speedup* paper this is dangerous: a method that blurs slightly (e.g., continuous frequency masking) can have *better* FID than the baseline because blur reduces feature variance. That's why SPEED reports multiple metrics and human/side-by-side comparisons and inspects the spectrum.
- **Alternatives for images:** CLIP score (prompt alignment, but saturates and prefers "literal" images), ImageReward / PickScore / HPSv2 (learned from human preferences, correlate best with humans), GenEval / T2I-CompBench / DPG-Bench (object count, color, spatial relations via detectors — tests prompt following structurally), DINOv2- or CLIP-based FD (FD_DINO is much more sensitive than FID), and OCR accuracy for text rendering. Always report at native resolution where possible.
- **Video:** FVD (I3D features, 16 frames at 224²) — same problems plus insensitivity to temporal coherence beyond 16 frames and a preference for static videos. VBench decomposes into ~16 dimensions (subject consistency, motion smoothness, dynamic degree, aesthetic, imaging quality, text alignment) — better, but gameable: "motion smoothness" rewards near-static video, so pair it with "dynamic degree". Human preference (side-by-side Elo, as in the Artificial Analysis / VideoArena style) is the gold standard.
- **For a speed method specifically:** paired comparison with the same seed/prompt against the full-cost baseline: PSNR/SSIM/LPIPS vs baseline output *plus* a reference-free quality score, plus wall-clock on identical hardware. PSNR-to-baseline alone rewards being a blurry copy.

</details>


### [04-09] You LoRA-fine-tune a pretrained DiT to change its input statistics — new resolution, mixed-resolution tokens. Which layers, what rank, why does LoRA suffice, and what LR/steps would you use?  (difficulty 2; tags: lora, fine-tuning, dit, foveated, speed)

- Follow-up: Why not just full fine-tune?
- Follow-up: When does LoRA *not* suffice?


<details><summary>Answer</summary>

**LoRA on the attention QKV/output projections (and usually the MLP) at rank 16-64 is enough because changing input statistics is a low-rank adjustment — the model already knows how to denoise; it just needs to recalibrate how it reads positions and token scales.**

- **Mechanism:** $W' = W + (\alpha/r)\, B A$ with $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $B$ initialized to zero. Trainable params for FLUX (12B) at rank 32 on all linear layers ≈ 100-300M (~1-2%).
- **Which layers:** attention Q/K/V/O first — resolution and mixed-res tokens change *where* attention should look (RoPE frequencies interact with q/k), so q/k adaptation matters most. Add MLP up/down for changes in token content statistics (new patch scale = new pixel-to-token mapping). The patch embedder and final unpatchify layer: train fully (they're tiny) if the patchify scale changes, as in Foveated Diffusion's multi-scale patchify. Modulation (adaLN) layers: often left alone.
- **Rank:** 16-64; ablations typically show rank 8 vs 128 differ little for distribution-shift tasks. Alpha = rank or 2×rank; rsLoRA scaling ($\alpha/\sqrt{r}$) helps at high rank.
- **LR / steps:** 1e-4 (LoRA tolerates ~10× higher LR than full FT since it's a scaled low-rank delta), AdamW, bf16, batch of ~32-64 at the target resolution, 2k-10k steps — a few hours on 8 GPUs. Warmup 100-500 steps; cosine or constant. Monitor with the validation loss at a fixed set of timesteps and visual samples every 500 steps.
- **Why not full FT:** memory (12B params × 16 bytes for Adam states ≈ 200 GB per replica → needs FSDP), catastrophic forgetting of prompt following, and no need. LoRA also ships as a 200 MB adapter that composes with other LoRAs.
- **When LoRA isn't enough:** a genuinely new modality/channel count (e.g., 16-ch → 32-ch VAE), new attention pattern requiring different inductive bias, or large distribution shifts where you want to change *what* the model generates, not *how it reads input*. Then: full FT of the affected blocks, or train new embedder + LoRA on the rest.

</details>


### [04-10] Break down where wall-clock time goes when sampling a 480p, 81-frame video with a 14B DiT like WAN 2.1. Attention vs MLP FLOPs, VAE decode, CFG.  (difficulty 3; tags: inference, cost, video, attention, flops)

- Follow-up: At what token count does attention overtake the MLP?
- Follow-up: What would you optimize first?


<details><summary>Answer</summary>

**Roughly: 50 steps × 2 CFG passes × 40 blocks of a 32k-token transformer ≈ 95% of the time; VAE decode ≈ 3-5%; text encoding negligible. Within a block at 32k tokens, attention FLOPs ≈ 2× the MLP FLOPs, so attention is ~60-65% of the transformer time.**

Numbers (WAN 2.1 14B: 40 blocks, $d = 5120$, heads 40, MLP 13824, patch (1,2,2)):
- Tokens: latent 21 × 60 × 104 → 21 × 30 × 52 = **32,760 tokens**.
- Per block, per token: QKVO projections $4 \cdot 2 \cdot d^2 \approx 210$ MFLOP; MLP $2 \cdot 2 \cdot d \cdot d_{\text{ff}} \approx 283$ MFLOP; cross-attn K/V is tiny (512 text tokens). So linear layers ≈ 0.5 GFLOP/token → ×32.8k tokens ≈ **16 TFLOP per block**.
- Self-attention: $QK^\top$ and $AV$ $= 2 \times 2 \times N^2 \times d = 4 \cdot (32.8\text{k})^2 \cdot 5120 \approx$ **22 TFLOP per block**. So attention:linear ≈ 1.4:1 at 480p; at 720p (~75k tokens) it's ~3:1.
- Crossover: attention = linear when $4 N d \approx 4 \cdot 2 \cdot d^2 + 4 \cdot d \cdot d_{\text{ff}}$ → $N \approx 2d + d_{\text{ff}} \approx$ **24k tokens**. Below that (images: FLUX at 1024² is 4096 tokens) the MLP dominates and cost is ~linear in tokens; above it, quadratic.
- Whole forward: ~40 × 38 ≈ 1.5 PFLOP. × 100 passes (50 steps, CFG) ≈ 150 PFLOP. At ~600 TFLOP/s effective bf16 on a B200 with FlashAttention → ~4 min; in practice ~5-8 min single-GPU on H100 because attention runs at lower MFU than matmul.
- VAE decode: a 3D conv decoder at 832×480×81 — ~5-10 s, plus memory pressure (chunked decode).

**What to optimize first:** (1) tokens — quadratic term — hence SPEED (fewer tokens for most of the trajectory) and Foveated (fewer tokens everywhere); (2) number of forward passes — distillation, CFG-free / guidance distillation, step caching (TeaCache); (3) attention kernel — FA3 / sparse tile attention; (4) parallelism — sequence parallel (Ulysses/ring) across 8 GPUs for latency. SPEED's 2.54× on WAN is consistent with spending most steps at 1/4 tokens: linear layers drop 4×, attention 16×.

</details>


### [04-11] What causes temporal flicker and inconsistency in generated video, and how do you fix it?  (difficulty 2; tags: video, temporal-consistency, flicker)

- Follow-up: How would you diagnose whether flicker comes from the VAE or the diffusion model?


<details><summary>Answer</summary>

**Flicker = high-frequency content that isn't temporally correlated: either the model denoises frames with insufficient cross-frame coupling, or the VAE decodes each frame with independent high-frequency errors, or the sampler injects uncorrelated noise per frame.**

Causes and fixes:
- **Architecture:** per-frame 2D models with weak temporal layers (AnimateDiff-style) — temporal attention only along each pixel's column can't track moving content. Fix: full 3D attention, 3D VAE with temporal compression (the latent itself is temporally smooth), 3D RoPE.
- **Noise correlation:** i.i.d. noise per frame at initialization biases toward uncorrelated detail. Fixes: shared/mixed noise (PYoCo's progressive noise: $\epsilon_t = \text{shared} \oplus \text{per-frame}$), FreeNoise for long video, or simply the 3D VAE, which makes "one latent frame = 4 pixel frames" so noise is shared.
- **Training data:** clips with cuts, camera shake, or compression artifacts teach flicker; filter by scene detection and motion statistics.
- **VAE:** 2D VAE decoding a video frame-by-frame produces flicker on textures; 3D causal VAE decoders with temporal receptive field fix it. Diagnosis: encode-decode a *real* video and measure per-frame PSNR/LPIPS and temporal LPIPS — if the reconstruction flickers, it's the VAE.
- **Sampler-level:** progressive methods must keep new noise consistent with the existing trajectory — in SPEED's video version, the 3D DCT expansion fills new spatiotemporal frequencies with $t\,\epsilon$ in the joint spectrum so temporal detail is revealed as a band, not per frame; and resetting the multistep solver's history at transitions matters because stale UniPC history across a resolution change produces exactly this artifact.
- **Long video / autoregressive:** error accumulation and drift. Fixes: overlapping-chunk or diffusion-forcing training (per-frame noise levels), KV-cached causal models (Self-Forcing, CausVid) trained with rollout so they see their own errors, anchor-frame conditioning.
- **Metric:** VBench temporal flickering / subject consistency, warped-frame error with optical flow, or the temporal power spectrum — flicker shows as excess high temporal frequency energy.

</details>


### [04-12] How do you do image editing or inpainting with a diffusion model? Cover RePaint, conditioning-channel inpainting, and ControlNet, and say when you'd use each.  (difficulty 1; tags: editing, inpainting, controlnet, conditioning)

- Follow-up: Why does naive RePaint produce seams, and what's the resampling trick?


<details><summary>Answer</summary>

**Three families: training-free masking of the sampling trajectory (RePaint), training with the mask and masked image as extra input channels (SD-inpainting), and a trainable side-network injecting spatial conditions (ControlNet).**

- **RePaint (training-free):** at every step, replace the known region with the forward-noised ground truth $x_t^{\text{known}} = (1-t)\,x_0 + t\,\epsilon$ and keep the model's prediction in the unknown region. Problem: the unknown region is denoised without ever "seeing" a harmonized known region — it only sees the noised version — so you get seams and semantically inconsistent fills. Fix: **resampling** — go forward a few steps (re-noise) and denoise again, ~10 times per step, so information propagates across the boundary. Cost: 10× slower. Good for quick experiments, no training.
- **Conditioning channels (SD-inpainting, FLUX-Fill):** concatenate [noisy latent (16ch), masked-image latent (16ch), mask (1ch)] → 33 input channels; train (or fine-tune) with random masks. Model learns to inpaint coherently in one pass. Best quality and speed; needs fine-tuning and a modified first layer (zero-init the new channels' weights so training starts from the pretrained behaviour).
- **ControlNet:** clone the encoder blocks, feed a spatial condition (edges, depth, pose) through them, add outputs to the frozen model via zero-initialized convolutions. Frozen backbone preserves generality; zero-init means the model starts identical to the base. In DiTs, the analog is a few copied blocks whose outputs are added to the residual stream. Use it for pixel-aligned structural control; for reference/style (non-aligned), use IP-Adapter-style decoupled cross-attention or in-context tokens.
- **Instruction editing (InstructPix2Pix, FLUX Kontext):** condition on the source image (channels or in-context tokens) + edit text, trained on synthetic edit pairs.
- **Trap:** in inpainting, blend with the original in pixel space at the end (Poisson or feathered) — the VAE round-trip changes unmasked pixels.

</details>


### [04-13] Autoregressive vs diffusion for images and video — VAR, MAR, and hybrids. Where does each win, and where is the field converging?  (difficulty 2; tags: autoregressive, diffusion, var, mar, hybrid, video)

- Follow-up: Why does raster-order next-token AR do poorly on images, and how does VAR fix it?
- Follow-up: How does this relate to causal video generation and world models?


<details><summary>Answer</summary>

**Diffusion wins on per-sample quality and parallelism at fixed length; AR wins on variable length, streaming/causality, and unification with LLMs. The convergence point is "AR over time, diffusion within a chunk" — next-frame (or next-scale) prediction where each step is a small diffusion.**

- **Raster AR (DALL·E 1, Parti, LlamaGen):** VQ tokens, next-token in scan order. Problems: a 1D order on 2D data is arbitrary, quantization loses information, and $N$ tokens = $N$ sequential steps (1024² → 4096 steps). Quality lags diffusion.
- **VAR (next-scale prediction):** predict the whole next resolution level conditioned on all coarser levels — 10 scales instead of 4096 tokens, each step parallel. Matches the coarse-to-fine inductive bias (the same one SPEED exploits in the diffusion setting), with better scaling laws than raster AR. Still VQ.
- **MAR / diffusion-loss AR:** keep AR ordering (random-order masked) but replace the categorical head with a small per-token diffusion MLP so tokens are continuous — no VQ. Bridges to diffusion.
- **Hybrids:** Transfusion / Show-o (one transformer, next-token for text, diffusion for images); for video, **diffusion forcing**, CausVid, Self-Forcing, MAGI-1: a causal transformer over frames/chunks with KV cache, each chunk denoised by a few diffusion steps, distilled from a bidirectional teacher. That's what streaming and interactive world models (Genie 3-style, lingbot-VA) need: the model must be *causal* in time so actions can be injected, and it must run at frame-rate — bidirectional 50-step video diffusion can't.
- **Practical numbers:** a distilled causal 1.3B model produces ~10-16 fps at 480p on one H100; bidirectional WAN 14B takes minutes per 5-s clip.
- **Where I'd bet:** continuous latents + flow matching stay for the "within chunk" denoiser; autoregression over time with KV cache for long horizon; the open problems are drift over long rollouts (exposure bias), KV memory growth (my surprise-gated KV-compression work: keep tokens that are surprising and attended by the action expert, ~44-54% reduction), and unifying training so the model sees its own errors.

</details>


### [04-14] Design a 10B-parameter text-to-video model from scratch. Give me the top-level decisions and the numbers behind them.  (difficulty 3; tags: design, video, architecture, training, scaling)

- Follow-up: What would you change if the target is real-time interactive generation instead of offline quality?


<details><summary>Answer</summary>

**Decisions: (1) 3D causal VAE 4×8×8, 16 ch; (2) single-stream DiT with full 3D attention, 3D RoPE, adaLN timestep, joint image-video training; (3) rectified-flow objective with logit-normal timestep sampling and resolution-dependent shift; (4) LLM/VLM text encoder; (5) curriculum in resolution and duration; (6) post-training: preference tuning + distillation for speed.**

- **VAE:** train it first, on images + video, causal 3D conv, 16 ch, reconstruction + LPIPS + GAN; rFID/PSNR on held-out video decides everything downstream. Consider 8× temporal / 16× spatial with 32-64 ch (DC-AE, Wan 2.2's VAE) for cheaper tokens if the model is big enough to model the denser latent.
- **Backbone (10B):** ~40 layers, $d \approx 4096$, 32 heads of 128, MLP 4× → $12 L d^2 \approx$ 8B + embeddings/modulation. Patch (1,2,2). Text via joint attention (MMDiT for the first blocks) or cross-attention — I'd use cross-attn from a VLM encoder for simplicity and cost at 30k+ tokens. QK-norm for stability at scale, RMSNorm, bf16 with fp32 master weights.
- **Objective:** $x_t = (1-t)\,x_0 + t\,\epsilon$, predict velocity $v = \epsilon - x_0$; logit-normal $t$ sampling; shift $t$ toward noise for high token counts (SD3: $\text{shift} \approx \sqrt{N/N_{\text{base}}}$). CFG dropout 10%.
- **Data:** ~1B images + ~100M video clips, dense captions (video captioner that describes motion/camera), scene-cut removal, motion filters, aesthetic filters, dedup. Aspect bucketing.
- **Curriculum:** images 256² → 512² → video 256p/2 s → 480p/5 s → 720p/5-10 s, joint image-video batches throughout. Most FLOPs go into the low-res stages — cheap tokens.
- **Compute:** 10B × 6 FLOPs/param/token × ~$10^{12}$ tokens $\approx 6 \times 10^{22}$ FLOPs of linear work, plus attention which at 30k tokens is comparable — call it $\sim 10^{23}$ FLOPs ≈ 512 H100s at 40% MFU for ~2 months. Parallelism: FSDP + sequence parallel (Ulysses across 8 GPUs) for long sequences, activation checkpointing, FlashAttention-3.
- **Post-training:** SFT on curated high-quality clips, preference optimization (DPO/GRPO with a video reward model), then distillation (DMD / consistency) to 4-8 steps and guidance distillation to remove the CFG doubling.
- **Efficiency by design:** because attention is quadratic in tokens above ~24k, I'd bake in resolution-progressive sampling (SPEED-style schedules derived from the training data's spectrum, $\beta \approx 2.4$ for video latents) and train with mixed-resolution token batches so foveated inference is native, not a LoRA afterthought.
- **Eval:** VBench + human side-by-side at every stage; watch validation loss per timestep bin.

**If real-time interactive:** causal in time (chunk-wise AR with KV cache), smaller backbone (1-2B) distilled to 1-4 steps per chunk, diffusion-forcing training with rollout so it sees its own errors, action conditioning as tokens, and KV-cache compression for long horizons. Quality target drops from "cinematic" to "consistent and controllable".

</details>


### [04-15] One minute: what is a latent diffusion model?  (difficulty 1; tags: latent-diffusion, vae, rapid-fire)


<details><summary>Answer</summary>

A **latent diffusion model** runs the diffusion/flow process in the compressed latent space of a pretrained autoencoder instead of in pixel space. A VAE encoder maps a 1024×1024×3 image to something like 128×128×16 (8× spatial downsampling); the denoiser is trained on those latents; at sampling time you denoise in latent space and decode once at the end. It exists because pixel-space diffusion spends most of its compute on perceptually irrelevant high-frequency detail; the VAE handles that cheaply, and the denoiser's cost drops by ~64× per step. Trap: the VAE bounds quality — text, faces, and fine texture are limited by reconstruction, which is why the field moved to 16-channel latents. Follow-up: "why not train the VAE and denoiser jointly?" Latent drift makes it unstable; freeze the VAE.

</details>


### [04-16] One minute: what is a VAE, and what is the ELBO in words?  (difficulty 1; tags: vae, elbo, rapid-fire)


<details><summary>Answer</summary>

A **VAE** is an encoder-decoder trained as a latent-variable generative model: the encoder outputs a distribution $q(z \mid x)$ (mean and variance), you sample $z$ with the reparameterization trick, and the decoder reconstructs $x$ from $z$. The **ELBO** is a lower bound on $\log p(x)$ that you maximise instead of the intractable marginal: in words, "reconstruct the input well" (expected log-likelihood under the decoder) **minus** "keep the encoder's posterior close to the prior" ($\mathrm{KL}(q(z \mid x) \,\|\, \mathcal{N}(0, I))$). The KL term is what makes the latent space smooth and sampleable. Trap: in latent-diffusion VAEs the KL weight is tiny (~1e-6) and there is a GAN + LPIPS loss, so it is really a regularised autoencoder, not a generative model in its own right — the diffusion model is the prior.

</details>


### [04-17] One minute: what is a DiT?  (difficulty 1; tags: dit, architecture, rapid-fire)


<details><summary>Answer</summary>

A **DiT (Diffusion Transformer)** replaces the U-Net denoiser with a plain ViT: patchify the (latent) image into tokens (patch size 2 on a 128×128 latent gives 4096 tokens), add position embeddings, run $N$ transformer blocks with full self-attention, unpatchify to predict noise or velocity. Timestep and class/text conditioning enter through **adaLN-Zero**: a small MLP maps $(t, c)$ to per-block scale/shift/gate, with the gate initialised to zero so each block starts as identity. It exists because transformers scale predictably with compute (FID tracks Gflops) and have no convolutional inductive bias to fight at high resolution. Trap: cost is quadratic in tokens, so resolution and video length are the expensive axes — which is what token-reduction methods like SPEED and Foveated Diffusion attack.

</details>


### [04-18] What is a 3D causal VAE, and why "causal"?  (difficulty 2; tags: video, vae, causal, rapid-fire)


<details><summary>Answer</summary>

A **3D causal VAE** is the video autoencoder used by WAN, CogVideoX, and Hunyuan: 3D convolutions compress a video 8× spatially and 4× temporally (81 frames → 21 latent frames), and every temporal convolution is **causal** — padded only on the past, so latent frame $k$ depends on input frames $\le k$. Two reasons: (1) the first frame is encoded independently, so a single image is a valid 1-frame video and image and video training share one latent space; (2) you can encode and decode in chunks, streaming through arbitrary-length videos with constant memory by carrying a small cache of past features. Trap: temporal compression means one latent frame mixes 4 pixel frames, so fast motion blurs at reconstruction — the VAE, not the denoiser, caps temporal fidelity.

</details>


### [04-19] Temporal attention vs spatial attention in a video model — what's the difference, and which do modern models use?  (difficulty 1; tags: video, attention, rapid-fire)


<details><summary>Answer</summary>

**Spatial attention** lets tokens attend within a single frame; **temporal attention** lets each spatial position attend across frames at that same position. Early video models (AnimateDiff, VideoLDM, SVD) factorised attention this way — spatial layers inherited from an image model plus inserted temporal layers — because full attention over $T \times H \times W$ tokens was unaffordable and it let them reuse image weights. The cost is $T (HW)^2 + HW\, T^2$ instead of $(THW)^2$. Modern DiTs (WAN, Sora-style, CogVideoX) use **full 3D attention** over all spatiotemporal tokens with 3D RoPE, because factorised attention cannot model a moving object attending to where it was, which shows up as flicker and drift. Follow-up: "how do you make full 3D attention affordable?" FlashAttention, sequence parallelism, and token reduction.

</details>


### [04-20] One minute: what is LoRA?  (difficulty 1; tags: lora, fine-tuning, rapid-fire)


<details><summary>Answer</summary>

**LoRA (Low-Rank Adaptation)** freezes the pretrained weights $W$ and learns an additive update $\Delta W = B A$ with $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $r \ll d$ (typically 8–128), applied to the attention and sometimes MLP projections; the forward pass becomes $W x + (\alpha/r)\, B A x$. It exists because full fine-tuning of a 12B model needs optimizer state for every parameter and produces a full-size checkpoint per task; LoRA trains <1% of the parameters, fits on one GPU, and the adapter is a few hundred MB that can be merged into $W$ at zero inference cost. Trap: initialise $B$ to zero so training starts at the pretrained function. Follow-up: "why does low rank suffice?" Task adaptation is a small change relative to the pretraining manifold — which is exactly why LoRA sufficed for SPEED's and Foveated Diffusion's input-statistics shifts.

</details>


### [04-21] What is FID, in one minute?  (difficulty 1; tags: fid, evaluation, rapid-fire)


<details><summary>Answer</summary>

**FID (Fréchet Inception Distance)** embeds real and generated images with an Inception-v3 pool3 layer (2048-d), fits a Gaussian to each set, and reports the Fréchet (Wasserstein-2) distance between them: $\|\mu_r - \mu_g\|^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big)$. Lower is better; it captures both fidelity (mean) and diversity (covariance) in one number, which is why it became the default. Traps: it is biased by sample count (always compare at the same $N$, usually 5k–50k), the Gaussian assumption is crude, Inception is trained on ImageNet at 299px so it is blind to high-resolution detail and text rendering, and it is not comparable across resolutions or datasets. Follow-up: "what instead?" CLIP-FID or DINOv2-FD for features, plus human preference and prompt-alignment metrics.

</details>


### [04-22] What is CFG distillation, and what does it mean that FLUX-dev is "guidance-distilled"?  (difficulty 2; tags: cfg, distillation, flux, rapid-fire)


<details><summary>Answer</summary>

**Classifier-free guidance** needs two forward passes per step — conditional and unconditional — then extrapolates: $v = v_{\text{uncond}} + w\,(v_{\text{cond}} - v_{\text{uncond}})$. **CFG distillation** trains a student to output that guided prediction in one pass, taking the guidance scale $w$ as an extra conditioning input (embedded like the timestep), so inference is 2× cheaper for free. FLUX.1-dev is exactly this: the "guidance" argument in diffusers is an input to the network, not a second pass, and there is no negative prompt. Trap: the student only covers the range of $w$ it saw, and you lose the ability to guide on new conditions or negative prompts; that is why Qwen-Image or WAN still run true CFG and why SPEED's reported speedups must say whether the baseline counts one or two passes per step.

</details>


## Efficient Deep Learning & Inference


### [05-01] Explain FlashAttention. Why is standard attention IO-bound, how do tiling and online softmax fix it, and what did FA2 and FA3 add?  (difficulty 3; tags: flash-attention, attention, gpu, memory)

- Follow-up: FlashAttention doesn't reduce FLOPs — so why is it faster?
- Follow-up: What's the memory complexity, forward and backward?


<details><summary>Answer</summary>

**Standard attention materializes the $N \times N$ score matrix in HBM: write $S = QK^\top$, read it back for softmax, write $P$, read it again for $PV$. For $N = 32\text{k}$, fp16, that's 2 GB per head — the kernel is bottlenecked on HBM bandwidth (~3.3 TB/s on H100), not on the tensor cores (~1 PFLOP/s). FlashAttention never writes $S/P$ to HBM.**

- **Tiling:** load a block of $Q$ ($B_r$ rows) into SRAM (shared memory, ~228 KB per SM on H100), then stream blocks of $K, V$ ($B_c$ columns); compute the score tile, apply softmax incrementally, accumulate $O$. Everything stays on-chip; HBM traffic drops from $O(N^2)$ to $O(N^2 d / M)$ where $M$ is SRAM size — in practice ~10× fewer bytes moved.
- **Online softmax:** softmax needs the row max and row sum over *all* columns, but we see columns in blocks. Keep running $(m, \ell)$ per row; when a new block gives max $m'$, rescale: $\ell \leftarrow \ell\, e^{m - m'} + \sum e^{s - m'}$, $O \leftarrow O\, e^{m - m'} + P' V$. Exact, not approximate.
- **Memory:** $O(N)$ extra instead of $O(N^2)$ — store only $O$ and the logsumexp per row. Backward recomputes $S/P$ per tile from $Q, K, V$ and the saved logsumexp (recomputation is cheap because the matmuls are fast; memory reads are the bottleneck). So FLOPs go *up* ~30% and wall-clock goes *down* 2-4×.
- **FA2:** better work partitioning — parallelize over sequence length as well as batch×heads (matters for long sequences with small batch, exactly the video-DiT regime), reduce non-matmul FLOPs, keep $Q$ in registers and loop over $K/V$ within a warp rather than splitting $K/V$ across warps (avoids shared-memory shuffles). ~2× over FA1, ~70% of peak on A100.
- **FA3 (Hopper):** exploit asynchrony — warp specialization with producer warps issuing TMA loads while consumer warps run WGMMA; ping-pong scheduling to overlap softmax (on the slow multi-function units) with matmul; fp8 with block-wise quantization and incoherent processing (random orthogonal transform to spread outliers). ~75% utilization in bf16 (~740 TFLOP/s), ~1.2 PFLOP/s in fp8.
- Trap: "FlashAttention makes attention linear" — no; it's still $O(N^2)$ FLOPs. It removes the memory wall. Token reduction is the only way to change the $N^2$ itself.

</details>


### [05-02] How big is the KV cache for a 70B LLM at 32k context, and why does that make decoding memory-bound?  (difficulty 1; tags: kv-cache, llm, inference, memory)

- Follow-up: What do GQA/MQA change in this arithmetic?


<details><summary>Answer</summary>

**KV cache bytes $= 2\,(\text{K and V}) \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{bytes\_per\_element}$.** For Llama-2-70B (80 layers, 64 heads, $d_{\text{head}} = 128$, MHA) in fp16: $2 \times 80 \times 64 \times 128 \times 2\,\text{B} = 2.6\,\text{MB}$ per token → **32k tokens ≈ 84 GB** — more than one H100. With GQA (Llama-3-70B: 8 KV heads) it's $2 \times 80 \times 8 \times 128 \times 2 = 328$ KB/token → 10.5 GB at 32k, an 8× cut, which is the whole reason GQA exists.

- **Why decode is memory-bound:** each generated token requires reading *all* weights (140 GB for 70B fp16) and the *entire* KV cache once, to do only ~$2 \times 70\text{B}$ FLOPs. Arithmetic intensity ≈ 2 FLOPs/byte for batch 1; H100's ridge point is ~300 FLOPs/byte (990 TFLOP/s ÷ 3.35 TB/s). So the GPU is >99% idle on compute; the token rate $\approx \text{bandwidth} / \text{bytes read} \approx 3.35\,\text{TB/s} \div 140\,\text{GB} \approx 24$ tokens/s for batch 1.
- **Fixes:** batch many requests (weights are read once per step for all of them → intensity scales with batch, until KV reads dominate), quantize weights and KV (int8/fp8 KV halves the bytes), GQA/MQA, paged attention to avoid fragmentation, speculative decoding to get more tokens per weight read.
- **Same reasoning applies to autoregressive video/world models:** a causal video model with a KV cache over frames grows the cache by (tokens per frame × per-token bytes) per frame; at 1.5k tokens/frame and a 1.3B model (30 layers, 12 heads, $d = 128$, bf16 → 184 KB/token) that's ~280 MB/frame, ~17 GB/min at 1 fps of latent frames — hence KV compression (my surprise-gated work: −44-54%).

</details>


### [05-03] Explain int8 / fp8 / W4A16 quantization. Per-channel vs per-tensor, why activation outliers hurt, what SmoothQuant does, and what changes with fp8 on Hopper/Blackwell.  (difficulty 2; tags: quantization, fp8, int8, inference, hardware)

- Follow-up: Why is weight-only 4-bit useful if compute is still done in fp16?


<details><summary>Answer</summary>

**Quantization maps $x \to \operatorname{round}(x/s) \cdot s$ with scale $s$; the question is always where outliers live and at what granularity you pick $s$.**

- **Formats:** int8 (uniform grid, 256 levels, needs a good scale); fp8 E4M3 (3 mantissa bits, range $\pm 448$, for weights/activations) and E5M2 (for gradients — more range); int4/nf4 weight-only. bf16 → int8 halves memory and doubles tensor-core throughput; fp8 on H100 ≈ 2× bf16 FLOPs (~2 PFLOP/s dense); Blackwell adds **fp4 (NVFP4/MXFP4)** with micro-block scaling (block of 16-32 elements shares an fp8 scale), ~2× again.
- **Per-tensor vs per-channel vs per-block:** one scale per tensor is cheapest but a single outlier crushes everyone's resolution. Per-channel (one scale per output channel of $W$, or per token/row for activations) is the standard for weights; per-block (128 elements, as in DeepSeek-V3's fp8 training) is now the norm because it bounds the damage of any outlier and is hardware-friendly.
- **Why activation outliers hurt:** transformer activations (especially after LayerNorm, in specific channels) have values 20-100× the typical magnitude. Per-token scaling doesn't help because the outlier sits in a *channel*, so every token's scale is inflated and the other channels round to zero. Weights are well-behaved; activations aren't — so W8A8 fails on >6B LLMs while W8A16 is fine.
- **SmoothQuant:** migrate the difficulty from activations to weights offline: $Y = (X \operatorname{diag}(s)^{-1})(\operatorname{diag}(s) W)$. Choose $s_j = \max|X_j|^{\alpha} / \max|W_j|^{1-\alpha}$, $\alpha \approx 0.5$. Activation channel $j$ is divided by $s_j$ (outliers shrunk), weight row $j$ multiplied (weights absorb it, they can take it). Mathematically identical, both now int8-friendly. Fold $\operatorname{diag}(s)^{-1}$ into the preceding LayerNorm.
- **W4A16 (GPTQ/AWQ):** weights in 4-bit, dequantized to fp16 on the fly. Useful because decode is memory-bound: 4× fewer weight bytes → up to ~3× faster decoding, even though compute is unchanged. AWQ picks scales to protect the 1% "salient" weight channels identified by activation magnitude.
- **Diffusion specifics:** DiTs tolerate fp8 weights + activations for the linear layers well (SVDQuant does W4A4 on FLUX with a low-rank branch absorbing outliers); timestep-dependent activation ranges mean you calibrate across $t$. FA3 fp8 attention needs the incoherent-processing trick because $q/k$ have outliers too.

</details>


### [05-04] Pruning, structured 2:4 sparsity, distillation — when does each actually help in practice?  (difficulty 2; tags: pruning, sparsity, distillation, efficiency)

- Follow-up: Why does unstructured 90% sparsity rarely speed anything up?


<details><summary>Answer</summary>

**Rule of thumb: distillation gives real speedups reliably; 2:4 sparsity gives a guaranteed but modest ~1.3-1.5× on linear layers; unstructured pruning is a research result that rarely translates to wall-clock.**

- **Unstructured pruning** (magnitude, lottery ticket, SparseGPT/Wanda for LLMs): 50-90% zeros, tiny accuracy loss. But a dense tensor core doesn't care that 90% are zero — you need sparse kernels, which win only above ~95% sparsity or on CPUs. Mostly a memory-compression story.
- **2:4 structured sparsity (Ampere+):** in every block of 4 weights, 2 are zero; stored as 2 values + 2-bit indices; the sparse tensor core skips the zeros for **2× matmul throughput** in theory, ~1.3-1.5× end-to-end since attention, norms, and memory traffic don't shrink. Recipe: prune with magnitude to 2:4, fine-tune briefly (ASP), or Wanda-style one-shot for LLMs. Quality: near-lossless for ViTs/LLMs at 50% because the constraint is mild. Works for weights only (activations are dynamic).
- **Structured pruning** (drop heads, layers, channels): real speedups on any hardware; needs retraining. Minitron/Sheared-LLaMA: prune a 15B to 8B + distill on ~100B tokens beats training the 8B from scratch on far more. Depth pruning hurts less than you'd expect on LLMs (later layers are redundant), width pruning is more hardware-friendly.
- **Distillation:** student trained on teacher's outputs/logits/features. Where it shines: (1) *step* distillation for diffusion — progressive distillation, consistency models, DMD, LCM: 50 steps → 1-4, that's a 10-50× speedup and the single biggest inference win in diffusion; (2) *guidance* distillation to fold CFG into one pass (2× — FLUX-dev is guidance-distilled); (3) *model* distillation to a smaller student, where the teacher's soft targets are a better signal than the data. Cost: training compute and usually a small diversity/quality loss (mode-seeking in DMD).
- **In practice for a video DiT:** step + guidance distillation first (10-100× combined), then token reduction (SPEED/foveated: 2-7×, orthogonal and composable), then fp8 (1.5-2×), then 2:4 (1.2×). Order by payoff / effort.

</details>


### [05-05] Token reduction for transformers — ToMe, token dropping, mixed-resolution tokens like your Foveated Diffusion. Give me the cost model and where each breaks.  (difficulty 2; tags: token-reduction, tome, foveated, attention, cost-model)

- Follow-up: If I halve the tokens, what's the speedup on a 4k-token image model vs a 32k-token video model?


<details><summary>Answer</summary>

**Cost model per transformer block: linear layers $\approx 24 N d^2$ FLOPs (linear in tokens), attention $\approx 4 N^2 d$ FLOPs (quadratic). Halving $N$ cuts the linear part 2× and attention 4×; whether you see ~2× or ~3.5× depends on which dominates, with crossover at $N \approx 6d$ ($\approx$ 24k tokens for $d = 4096$).**

- **4k-token image model (FLUX, $d = 3072$):** MLP-dominated, so halving tokens ≈ 1.9× — roughly linear. **32k-token video (WAN 14B):** attention ≈ 60% of block time, so halving tokens $\approx 0.4/2 + 0.6/4 \to$ ~2.9×. This is why token methods pay off more on video and at higher res, and why Foveated Diffusion gets 2× on images but 4× on video.
- **ToMe (Token Merging):** bipartite soft matching merges the $r$ most similar token pairs per layer (by key cosine similarity), tracking sizes for proportional attention; unmerge at the end. Training-free, ~2× on ViT classification. For diffusion (ToMe-SD) it merges in attention only and unmerges before the residual — gains are smaller (~1.5×) because merging in early/late layers blurs detail and because you must unmerge for the residual stream. Breaks when content has no redundancy (dense texture) and on high-res detail.
- **Token dropping / pruning (EViT, DynamicViT, "register" tricks):** discard low-attention tokens for classification. For generation you can't discard outputs, so drop only in intermediate layers (or route: MoD-style). Breaks: generation needs every output token.
- **Mixed-resolution tokens (Foveated Diffusion):** patchify the periphery at 2× or 4× coarser scale, the fovea at native; all tokens go through one sequence. Needs: position embeddings that are consistent across patch scales (RoPE on continuous coordinates — the center of each patch), patch embedders/unpatchifiers per scale, and a LoRA so the model learns the new token statistics. Blend at boundaries by upsampling periphery predictions and feathering in the latent. Breaks: if the fovea is wrong (task-dependent salience) or if the periphery contains text/faces. This is the spatial dual of SPEED's temporal-in-trajectory token reduction; both are multiplicative with distillation and quantization.
- **Sparse attention (STA, sliding-tile, NSA):** keep all tokens but restrict attention to local 3D windows plus a few global tokens — cuts only the quadratic term, ~1.5-3× on video with FA-compatible tiling. Often combined with token reduction.

</details>


### [05-06] Feature caching across diffusion steps — DeepCache, TeaCache. Why do adjacent steps have similar features, and when does caching fail?  (difficulty 2; tags: caching, diffusion, inference, deepcache, teacache)

- Follow-up: How does a caching method decide *which* steps to skip?


<details><summary>Answer</summary>

**Adjacent denoising steps see inputs that differ by a small $\Delta t$ of the ODE, and the network is smooth in its input, so deep features change slowly — especially in the middle of the trajectory. Caching reuses those features and only recomputes a cheap part.**

- **Why similar:** for the PF-ODE $\mathrm{d}x/\mathrm{d}t = v(x, t)$, consecutive $x_t$ differ by $O(\Delta t)$; the model output $v$ itself varies slowly in $t$ except near the ends ($t \to 1$ where layout forms, $t \to 0$ where fine detail forms). Empirically, cosine similarity of block outputs between adjacent steps is $> 0.95$ for most of the trajectory.
- **DeepCache (UNet):** exploit skip connections — cache the deep, low-res branch features and recompute only the shallowest encoder/decoder blocks for $N-1$ out of $N$ steps. ~2-4× on SD with mild quality loss. Doesn't transfer to DiTs (no skip structure).
- **TeaCache (DiT):** cache the *residual* of the whole transformer, i.e., $\text{output} - \text{input}$, and reuse it when the input change is small. Decide with a cheap proxy: the relative $L_1$ change of the *timestep-modulated input* to the first block (after adaLN), passed through a fitted polynomial that maps it to the expected output change; accumulate and skip while the accumulated estimate is below a threshold. 1.5-2.5× on WAN/HunyuanVideo/FLUX; threshold trades speed vs quality.
- **Other axes:** block-level caching (cache attention in some blocks, recompute MLP — FORA, Δ-DiT), token-level (recompute only tokens that changed most — ToCa), and CFG caching (reuse unconditional branch across steps).
- **When it fails:** few-step distilled models (steps are far apart, features differ a lot — no redundancy left); early steps (layout still forming — skipping there changes composition); high-motion video (temporal features change fast); and it doesn't compose freely with resolution-progressive methods — in SPEED, a cached residual from the low-res stage is meaningless after the DCT expansion, so caches must be flushed at transitions (same bug class as stale UniPC history). Also note the speedup is upper-bounded by the fraction of steps you can skip, unlike token reduction which cuts every step.

</details>


### [05-07] Speculative decoding — why does it work, and what determines the speedup?  (difficulty 1; tags: speculative-decoding, llm, inference)

- Follow-up: Is the output distribution exactly the target model's?
- Follow-up: Does this idea transfer to diffusion or video?


<details><summary>Answer</summary>

**Because decoding is memory-bound, verifying $k$ tokens in one forward pass costs about the same as generating one. A small draft model proposes $k$ tokens autoregressively; the big model scores all $k+1$ positions in parallel; you accept the longest prefix that passes a rejection-sampling test, and get on average several tokens per big-model pass.**

- **Exactness:** yes. Accept draft token $x$ with prob $\min(1, p(x)/q(x))$ where $p$ = target, $q$ = draft; on rejection, sample from the residual $\operatorname{norm}(\max(0, p - q))$. This preserves the target distribution exactly (Leviathan et al., Chen et al. 2023).
- **Speedup:** with per-token acceptance rate $\alpha$ and draft length $k$, expected accepted tokens per verify step $= \frac{1 - \alpha^{k+1}}{1 - \alpha}$. At $\alpha = 0.8$, $k = 5$: ~3.3 tokens per pass. Net speedup $\approx$ that divided by $(1 + kc)$ where $c$ is draft-cost/target-cost; with a 100× smaller draft, $c \approx 0.01$, so ~3×. Acceptance depends on how well draft matches target — code and formulaic text are high (~0.9), creative sampling at high temperature lower.
- **Variants:** Medusa (extra heads on the target model, no separate draft), EAGLE (draft from the target's hidden states — higher $\alpha$), n-gram/lookahead drafts (free), tree-structured verification of multiple candidate branches.
- **Transfer to diffusion/video:** the analog is "cheap proposal, expensive verify". For AR video models with per-frame KV cache, a small draft model could propose frames and the big model verify with one parallel pass — but continuous-valued frames don't have a clean rejection test (proposals are never exactly "accepted"). Closer analogs in diffusion: parallel sampling (ParaDiGMS, Picard iteration — guess the whole trajectory, refine in parallel), and cheap-model-early / expensive-model-late schedules. SPEED is in the same spirit — do most of the trajectory cheaply where it's provably (spectrally) safe.

</details>


### [05-08] What GPU concepts does an ML researcher need to actually know? Memory hierarchy, arithmetic intensity, roofline, kernel fusion, torch.compile, tensor cores, occupancy — conceptually.  (difficulty 2; tags: gpu, cuda, roofline, kernel-fusion, torch-compile, hardware)

- Follow-up: Why is a LayerNorm slow relative to its FLOPs, and what fixes it?


<details><summary>Answer</summary>

**The one mental model: every kernel is either compute-bound or memory-bound, decided by its arithmetic intensity (FLOPs per byte moved from HBM) relative to the hardware's ridge point (peak FLOPs ÷ bandwidth). Almost everything in a network except big matmuls is memory-bound.**

- **Memory hierarchy (H100):** HBM3 80 GB at ~3.35 TB/s; L2 50 MB; shared memory/L1 up to 228 KB per SM (132 SMs); registers 256 KB per SM — bandwidth increases ~10× per level. Data in registers/SMEM is nearly free; every HBM round-trip is the cost. B200: 192 GB HBM3e at ~8 TB/s, ~2.2 PFLOP/s dense bf16 — ridge point ≈ 275 FLOPs/byte.
- **Roofline:** $\text{attainable FLOP/s} = \min(\text{peak}, \text{intensity} \times \text{bandwidth})$. A bf16 matmul of $(M \times K)(K \times N)$ has intensity $\approx \frac{MNK}{(MK + KN + MN) \cdot 2\,\text{B}}$ — large square matmuls are ~1000s of FLOPs/byte (compute-bound); a bias-add or LayerNorm is ~1 FLOP/byte (memory-bound, runs at <1% of peak FLOPs but 100% of bandwidth).
- **Kernel fusion:** chain memory-bound elementwise ops so intermediates stay in registers — LayerNorm + residual + GELU as one kernel reads and writes the tensor once instead of 3-5 times. That's most of what torch.compile (Inductor → Triton) does: graph capture, fusion of pointwise/reduction ops, CUDA graphs to remove launch overhead (~5-10 μs per launch matters when a DiT block launches ~40 kernels). Typical gains on a DiT: 1.2-1.5×; more with CUDA graphs on small batches.
- **Tensor cores & mixed precision:** dedicated matrix units doing $D = AB + C$ on $16 \times 16$-ish tiles; bf16/fp16 in, fp32 accumulate; ~16× the throughput of fp32 CUDA cores. Need dims multiples of 8 (16 for fp8) and contiguous layouts. Mixed precision = bf16 storage/compute, fp32 master weights and optimizer state; bf16's 8 exponent bits avoid the loss-scaling dance fp16 needs.
- **Occupancy:** how many warps are resident per SM — limited by registers and SMEM per thread block. High occupancy hides memory latency; but FlashAttention-style kernels deliberately use *low* occupancy with big tiles in registers because they'd rather have reuse than latency hiding. So occupancy is a tool, not a target.
- **Why LayerNorm is slow:** two passes over the row (mean, var) then a normalize write — 3 reads + 1 write of the activation for ~5 FLOPs/element. Fix: fused single-pass Welford kernel, fuse with the following linear's input (or RMSNorm which needs one statistic), or keep it in fp16 with fp32 accumulation.
- **Communication:** NVLink ~900 GB/s per GPU (H100), ~1.8 TB/s (B200); all-reduce of a 10B-param bf16 gradient = 20 GB → ~$2 \times 20/900 \approx 45$ ms per step within a node; across nodes over InfiniBand (400 Gb/s ≈ 50 GB/s) it's seconds unless overlapped — which is why FSDP/ZeRO overlap comms with backward.

</details>


### [05-09] Efficient ViT attention — linear attention, windowed/Swin attention. Why does linear attention underperform, and where does each still make sense?  (difficulty 2; tags: efficient-vit, linear-attention, swin, attention)

- Follow-up: How would you get long-context video attention without going quadratic?


<details><summary>Answer</summary>

**Softmax attention costs $O(N^2 d)$; linear attention rewrites $\operatorname{softmax}(QK^\top)V$ as $\phi(Q)(\phi(K)^\top V)$ so it's $O(N d^2)$ — but it loses the sharp, input-dependent selectivity of softmax and ends up with a low-rank, "averaging" attention map. Windowed attention keeps softmax and instead limits *who* you attend to.**

- **Linear attention (Performer, Linear Transformer, and modern SSM/RetNet/Mamba/GLA variants):** with kernel feature map $\phi$, output $= \phi(Q) S$ where $S = \sum_i \phi(k_i) v_i^\top$ is a $d \times d$ state. Compute $O(N d^2)$; constant memory in causal form (recurrent). **Why it underperforms:** (1) the effective attention matrix has rank $\le d$, so with $N \gg d$ it can't represent sharp, sparse attention (retrieval, copying, "attend to exactly that token"); (2) no softmax normalization per query → attention "dilution" across many keys; (3) feature maps like elu+1 or random features approximate exp poorly. Fixes that partly help: gating/decay (Mamba-2, GLA), delta-rule updates (DeltaNet), hybrid layers (a few softmax layers among linear ones, as in Jamba/Zamba) — the hybrids are what actually work at scale.
- **Windowed / Swin:** attend within local (shifted) windows, cost $O(N w^2 d)$ with window size $w$; shifting alternates partitions so information propagates across windows over depth. Keeps softmax's precision; loses global receptive field per layer. Works great for dense prediction (detection/segmentation backbones); for generation, windowed-only models show seams and weak global coherence — need some global layers or a hierarchy. In video, 3D windows = Sliding Tile Attention (STA) / NATTEN — 1.5-3× on WAN/Hunyuan with tile-aligned windows that keep FA efficiency.
- **Where each makes sense:** linear/SSM for very long causal streams where quality per token can be lower (audio, long video memory, the "state" in a world model); windowed for high-res dense vision; full attention where quality per token matters most. In a video DiT, the pragmatic answer is full attention with an efficient kernel plus token reduction (fewer tokens is better than worse attention over more tokens — that's the SPEED/foveated bet), and sparse/windowed attention for the long-range temporal axis.
- **Long-context video without quadratic cost:** hierarchical — full attention within a chunk, compressed/summarized tokens (pooled or selected — e.g., surprise-gated KV) across chunks, plus a recurrent state; or sparse attention with learned block selection (NSA/MoBA-style) so every query picks a few blocks globally.

</details>


### [05-10] A diffusion pipeline is slower than you expect. Walk me through how you'd profile it with the PyTorch profiler and what you'd look for.  (difficulty 1; tags: profiling, torch-profiler, inference, debugging)

- Follow-up: The profiler shows 90% of time in "cudaStreamSynchronize" — what does that mean?


<details><summary>Answer</summary>

**Wrap the sampling loop in torch.profiler.profile with CPU + CUDA activities, warm up first, record a few steps, export a Chrome trace, and read it top-down: per-step wall time → per-block → per-kernel; then compare achieved vs roofline for the top kernels.**

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=3),
    record_shapes=True, with_stack=True, profile_memory=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof")) as p:
    for step in range(6):
        with torch.profiler.record_function(f"denoise_step"):
            x = pipe.step(x, t[step]); p.step()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

What I look for, in order:
- **GPU idle gaps in the trace** (CPU-bound launch overhead, Python in the loop, host-device syncs from `.item()`, `.cpu()`, `torch.nonzero`, data-dependent control flow). Fix: CUDA graphs / torch.compile(mode="reduce-overhead"), remove syncs, precompute schedules. Small-batch DiT inference is often 30% launch-bound.
- **Top kernels by CUDA time:** expect FlashAttention and GEMMs. If `aten::copy_`, `cat`, `softmax`, elementwise, or `layer_norm` kernels are near the top → unfused ops or non-contiguous layouts (`.contiguous()` copies from permutes, e.g., RoPE implemented with lots of reshapes). If attention is a *math* (non-flash) kernel → the SDPA backend fell back (wrong dtype, mask, or head dim not a multiple of 8).
- **Tensor-core utilization:** GEMMs with odd shapes (M not a multiple of 8/64) or fp32 fall off the fast path; check dtypes.
- **Memory timeline:** VAE decode spikes; activation memory forcing small batches; fragmentation from varying shapes (resolution-progressive sampling — SPEED — changes shapes at each stage, which can trigger allocator churn and cuDNN autotune; fix with `expandable_segments` and per-stage warmup).
- **Outside the model:** text encoder each call (cache it), VAE decode, CPU preprocessing, tokenization, `torch.manual_seed` per call, and — for CFG — whether cond/uncond are batched or run sequentially.
- Then measure: `torch.cuda.Event` timing per stage with `torch.cuda.synchronize()`, $n \ge 5$ runs, report median. Cross-check with `nsys` if kernel-level detail is needed.

**cudaStreamSynchronize at 90%:** the CPU is waiting for the GPU — i.e., the GPU *is* the bottleneck and the sync is just where the CPU parks; look at the GPU kernel timeline, not the CPU op table. (Conversely, if the GPU timeline has gaps, the CPU is the bottleneck.)

</details>


### [05-11] Batching for diffusion throughput — CFG batching, multi-prompt batching, parallel sampling across GPUs. What are the trade-offs?  (difficulty 1; tags: batching, cfg, throughput, parallelism)

- Follow-up: When does batching *not* help?


<details><summary>Answer</summary>

**Batching raises arithmetic intensity: weights are read once per step for the whole batch, so memory-bound kernels get faster per sample until you become compute-bound; for diffusion the free batch is CFG (cond + uncond as batch 2), then multiple prompts, then splitting one long sequence across GPUs when latency matters.**

- **CFG batching:** run $[x; x]$ with $[c; \varnothing]$ in one forward — same FLOPs as two calls but one weight read and half the launches. For small-token image models (FLUX at 4k tokens is only partly compute-bound at batch 1) it's ~1.6-1.9× faster than sequential; for a 32k-token video model each pass is already compute-bound, so the win is ~1.05-1.1× and memory doubles — sometimes you *can't* batch CFG at 720p on an 80 GB GPU. Alternatives: guidance distillation (no uncond pass), CFG only on some steps / intervals, or adaptive guidance that drops CFG late.
- **Multi-prompt batching:** throughput scales near-linearly until compute-bound; needs same shape (aspect bucketing at inference too) and same number of steps. Latency per sample doesn't improve. For serving: continuous batching across users with steps interleaved.
- **Parallel sampling across GPUs:** (1) data parallel — different prompts/seeds per GPU, best throughput, no comm. (2) **Sequence parallel** (Ulysses: all-to-all so each GPU holds all tokens for a subset of heads during attention; ring attention: pass K/V blocks around) — reduces latency of one sample nearly linearly to 8 GPUs; comm per block ≈ activation size ($32\text{k} \times 5120 \times 2\,\text{B} \approx 330$ MB) over NVLink — a few ms, OK. (3) CFG parallel — cond on GPU 0, uncond on GPU 1, all-reduce the output (2×, trivially). (4) **Pipeline/temporal parallel across steps** (DistriFusion, AsyncDiff): patches on different GPUs use stale activations from the previous step for cross-patch attention — exploits the same adjacent-step similarity as caching.
- **When batching doesn't help:** already compute-bound (large tokens), memory-limited, latency-critical single requests (use sequence parallel instead), or different resolutions per request (padding waste). Also with resolution-progressive sampling, batch shapes change per stage, so batching must be per-stage.
- The best "batching" is often not doing the work: fewer steps (distillation), fewer tokens (SPEED/foveated), fewer passes (no CFG).

</details>


### [05-12] How do you fairly report a speedup for an efficiency method? What's the trap in counting FLOPs?  (difficulty 3; tags: evaluation, benchmarking, speedup, pareto, methodology)

- Follow-up: A reviewer says your 7× is unfair because the baseline uses 50 steps and could use 20. Respond.


<details><summary>Answer</summary>

**Report wall-clock on the same hardware, same software stack, same batch size and precision, with the same number of sampler steps and the same seeds; pair it with a quality metric (ideally several plus human preference) as a Pareto curve, and compare against the baseline's *own* best Pareto points (fewer steps, distilled variants), not just its default.**

- **The FLOP trap:** FLOPs ignore memory traffic and kernel efficiency. Token reduction from 32k to 8k cuts attention FLOPs 16× but FlashAttention at 8k tokens runs at lower utilization than at 32k, and MLP FLOPs only drop 4×; a reported "10× fewer FLOPs" might be 3× wall-clock. Conversely, caching or sparsity can *add* FLOPs and still be faster. MACs vs FLOPs confusion (2×) is another classic. Always give both: theoretical FLOPs (for hardware-independent understanding) and measured latency (what users get). State the hardware (H100 vs B200 changes attention/MLP balance because bandwidth and FLOP ratios differ), dtype, and whether torch.compile / FA3 were on for *both* arms.
- **Quality:** reference-free metrics (ImageReward, VBench) + paired metrics vs the baseline output at the same seed (PSNR/LPIPS — shows you're producing the *same* image, not just a good one) + human side-by-side with win/tie/loss. Never just FID (blur wins FID). Report at native resolution. Show the failure cases.
- **Baseline fairness:** the baseline must be tuned — best scheduler, same step count or a step-count sweep, and the obvious cheap tricks (CFG batching, compile). Compare on the Pareto frontier: plot quality vs latency for baseline at {20, 30, 50} steps and yours at each configuration; you win if you dominate, not if you beat one default point. Include orthogonal methods (TeaCache, distillation) as *composable* rather than as rivals if they truly compose — and show the combination.
- **Statistics:** median of ≥5 runs after warmup, `torch.cuda.synchronize()`, fixed clocks if possible, batch size stated, note variance; for quality, ≥1k prompts and confidence intervals on human preference.
- **Reviewer response to "50 vs 20 steps":** agree, and show the sweep — at 20 steps the baseline degrades (measurably, e.g., ImageReward and VBench drop), and SPEED at 50 steps with the same wall-clock as baseline-at-~8-steps is still better; more importantly, SPEED's speedup is per-step (token count), so it *stacks* with step reduction: SPEED at 20 steps vs baseline at 20 steps gives the same ~5-7× ratio. What would *not* be fair: comparing my 50-step method against a 50-step baseline while the practical baseline is a 4-step distilled model — so I'd also show the number against the distilled model, or apply my method on top of it (in SPEED's case with a caveat that few-step regimes have fewer transitions to exploit).

</details>


### [05-13] One minute: what is a KV cache?  (difficulty 1; tags: kv-cache, inference, rapid-fire)


<details><summary>Answer</summary>

In autoregressive decoding, every new token attends to all previous tokens. Without a cache you would recompute the keys and values of the whole prefix at every step — $O(n^2)$ work per generated token. The **KV cache** stores each layer's K and V for every past token, so decoding one token costs one forward pass over one token plus attention against cached K/V. Size per token $= 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes}$; for Llama-70B in bf16 that is ~320 KB per token, so long contexts and large batches are limited by cache memory, not compute. Follow-up: "how do you shrink it?" GQA/MQA (fewer KV heads), quantised cache, paged attention (vLLM) to avoid fragmentation, and eviction/compression — which is what my surprise-gated KV work on world-action models did.

</details>


### [05-14] One minute: what is quantization?  (difficulty 1; tags: quantization, inference, rapid-fire)


<details><summary>Answer</summary>

**Quantization** stores weights (and optionally activations) in fewer bits than the training precision: a bf16 tensor becomes int8 or int4 plus a scale (and zero-point) per tensor, per channel, or per group of 128. Value $\approx \text{scale} \times q$. It exists because inference is usually memory-bandwidth-bound: halving bytes roughly halves weight-load time for decoding, and int8/fp8 tensor cores also double throughput on compute-bound layers. Two flavours: **post-training quantization** (calibrate scales on a few hundred samples; GPTQ, AWQ, SmoothQuant) and **quantization-aware training** (fake-quantize in the forward pass, straight-through gradients). Trap: activation outliers in a few channels blow up per-tensor scales — that is why per-channel scales and SmoothQuant exist. Follow-up: fp8 is a floating format with its own exponent, so it tolerates outliers better than int8 at the same width.

</details>


### [05-15] One minute: what is knowledge distillation?  (difficulty 1; tags: distillation, rapid-fire)


<details><summary>Answer</summary>

**Knowledge distillation** trains a small student to match a large teacher's outputs rather than only the hard labels: minimise $\mathrm{KL}(p_{\text{teacher}}(\cdot/T) \,\|\, p_{\text{student}}(\cdot/T))$ at a softened temperature $T$, mixed with the ordinary cross-entropy. The soft targets carry "dark knowledge" — which wrong classes are plausible — which is a richer, lower-variance training signal than one-hot labels, so the student generalises better than training from scratch at its size. Variants: feature/hidden-state distillation, sequence-level distillation for LLMs, and for diffusion models **step distillation** (progressive distillation, consistency, DMD) where the "teacher" is the many-step sampler and the student learns to do it in 1–4 steps. Trap: the student is bounded by the teacher's errors and its own capacity; distillation cannot add capabilities the teacher lacks.

</details>


### [05-16] One minute: what is pruning?  (difficulty 1; tags: pruning, sparsity, rapid-fire)


<details><summary>Answer</summary>

**Pruning** removes parameters that contribute little — by magnitude, by a Hessian/Taylor importance score, or by learned masks — then fine-tunes to recover accuracy. **Unstructured** pruning zeroes individual weights: high sparsity is achievable (80–90%) but gives no speedup on dense GPU kernels. **Structured** pruning removes whole channels, heads, or layers, which shrinks the actual matmul shapes and speeds things up but hurts accuracy more. The middle ground is NVIDIA's **2:4 semi-structured sparsity**: two of every four weights are zero, and Ampere+ tensor cores run it at ~2× throughput. Trap: for memory-bound decoding, pruning weights helps like quantization does (fewer bytes); for compute-bound prefill, only structured or 2:4 helps. Follow-up: depth pruning (dropping layers) is often the most latency-effective on LLMs.

</details>


### [05-17] What is kernel fusion, and why does it matter?  (difficulty 1; tags: kernel-fusion, gpu, rapid-fire)


<details><summary>Answer</summary>

**Kernel fusion** combines several consecutive GPU operations into one kernel so intermediate tensors stay in registers or shared memory instead of round-tripping to HBM. Eager PyTorch launches a separate kernel for each op in `x * sigmoid(x)` — each reads and writes the whole tensor — so an element-wise chain that does trivial math is completely bandwidth-bound. Fusing removes the memory traffic and launch overhead. FlashAttention is the canonical example: it fuses $QK^\top$, softmax, and $\cdot V$ so the $n \times n$ attention matrix never exists in HBM. `torch.compile` (Inductor/Triton) does this automatically for element-wise and reduction chains. Trap: fusion only helps ops that are memory-bound; a big matmul is already compute-bound and gains nothing except its epilogue (bias, activation) being fused in.

</details>


### [05-18] What does "memory-bound vs compute-bound" mean?  (difficulty 1; tags: roofline, gpu, rapid-fire)


<details><summary>Answer</summary>

A kernel is **compute-bound** if the time is set by how fast the ALUs can do the FLOPs, and **memory-bound** if it is set by how fast bytes move from HBM. Which one you are is decided by arithmetic intensity (FLOPs per byte) compared with the hardware's ratio: an H100 does ~1000 TFLOP/s bf16 over ~3.35 TB/s, so the ridge point is ~300 FLOP/byte. A large matmul (intensity $\propto$ matrix dimension) is compute-bound; element-wise ops, softmax, layer norm, and batch-1 LLM decoding (each weight is loaded once and used once) are memory-bound. The practical point: for a memory-bound op, more FLOPs are free and the only fix is fewer bytes (fusion, quantization, batching); for a compute-bound op, the fix is fewer FLOPs or better tensor-core utilisation.

</details>


### [05-19] What is arithmetic intensity?  (difficulty 1; tags: roofline, arithmetic-intensity, rapid-fire)


<details><summary>Answer</summary>

**Arithmetic intensity** is FLOPs performed per byte moved from memory, $I = \text{FLOPs} / \text{bytes}$. In the roofline model, $\text{attainable throughput} = \min(\text{peak FLOPs}, I \times \text{bandwidth})$, so a kernel whose intensity is below the hardware ridge point (~300 FLOP/byte for H100 bf16) is memory-bound and cannot reach peak no matter how well it is written. Examples: a matmul of $M \times K$ by $K \times N$ has $I \approx \frac{2MNK}{2(MK + KN + MN)}$ — grows with the smallest dimension, so batch-1 decode ($M = 1$) has intensity ~1 and is hopelessly memory-bound, while batch 256 pushes it past the ridge. Element-wise ops have $I \approx 1/(\text{bytes per element})$. Follow-up: the way to raise intensity is reuse — tiling in shared memory, batching, and fusion — not more FLOPs.

</details>


### [05-20] Speculative decoding in one sentence — and then the one trap.  (difficulty 1; tags: speculative-decoding, llm, rapid-fire)


<details><summary>Answer</summary>

A cheap draft model proposes $k$ tokens, the big model verifies them all in **one** forward pass (decoding is memory-bound, so $k$ tokens cost about the same as one), and a rejection-sampling rule accepts the longest matching prefix so the output distribution is exactly the big model's. The speedup is roughly $(\text{expected accepted tokens} + 1) / (\text{draft cost} + 1\ \text{verify})$, so it lives or dies on the draft's acceptance rate — 70–90% on predictable text, poor on creative or code-heavy outputs. Trap: it is only a win when the target is memory-bound (small batch); at large batch the verify pass becomes compute-bound and the extra $k$-token work is no longer free. Follow-up: Medusa/EAGLE drop the separate draft model and predict multiple tokens from the target's own hidden states.

</details>


## Distributed & Large-Scale Training


### [06-01] Your CV says "distributed and multi-GPU training". Start simple: what does PyTorch DDP actually do during a step, and how does it hide the communication?  (difficulty 1; tags: distributed, ddp, data-parallel)

- Follow-up: what happens if one rank has a parameter that received no gradient this step?
- Follow-up: why is `find_unused_parameters=True` slow?


<details><summary>Answer</summary>

**DDP = data parallelism: every rank holds a full replica, sees a different micro-batch, and the gradients are averaged with an all-reduce before the (identical, replicated) optimizer step.** Because all ranks start from the same weights and apply the same averaged gradient, replicas stay bit-identical without ever communicating parameters.

The trick that makes it fast is **overlap with backward**. DDP registers autograd hooks on every parameter and groups parameters into **buckets** (default 25 MB, in reverse registration order because backward produces gradients for the last layers first). As soon as every gradient in a bucket is ready, DDP launches an asynchronous all-reduce on that bucket on a separate CUDA stream while autograd keeps computing earlier layers. By the time backward finishes, most of the communication is done; only the last bucket is exposed. Gradient averaging is done by all-reducing the sum and dividing by world size.

Traps an interviewer looks for:
- **Unused parameters**: a bucket never becomes "ready", the all-reduce never fires, and the run hangs at the end of backward. `find_unused_parameters=True` fixes it by traversing the autograd graph each iteration to mark those params ready — that traversal is the overhead, and it disables some static-graph optimizations.
- All ranks must execute the **same collectives in the same order**; a data-dependent branch on one rank is a classic hang.
- With gradient accumulation you must wrap the non-final micro-steps in `model.no_sync()` or you all-reduce every micro-batch and waste bandwidth.

</details>


### [06-02] Derive the cost of a ring all-reduce. Why is it built from a reduce-scatter and an all-gather, and what does "all-reduce is bandwidth-bound" mean?  (difficulty 2; tags: distributed, nccl, all-reduce)

- Follow-up: put numbers on it — 10B parameters, bf16 gradients, 8 GPUs on NVLink versus 64 nodes on InfiniBand.
- Follow-up: when is a tree all-reduce better than a ring?


<details><summary>Answer</summary>

**A ring all-reduce over $N$ GPUs on $M$ bytes moves $2\frac{N-1}{N} M$ bytes per GPU, and the time is $\approx 2\frac{N-1}{N} \cdot \frac{M}{B} + 2(N-1)\alpha$**, where $B$ is per-GPU link bandwidth and $\alpha$ is per-step latency. For large $N$ that's $\approx 2M/B$ — independent of $N$, which is why data parallelism scales.

Decomposition:
- **Reduce-scatter**: split the buffer into $N$ chunks. In $N-1$ steps each GPU sends one chunk to its right neighbour and adds the chunk it receives. After $N-1$ steps every GPU owns one fully reduced chunk. Traffic per GPU: $\frac{N-1}{N} M$.
- **All-gather**: another $N-1$ steps circulating the reduced chunks so everyone has all of them. Traffic: $\frac{N-1}{N} M$.
- Total $2\frac{N-1}{N} M$, and the reduce-scatter/all-gather split is exactly what ZeRO/FSDP exploit: they stop halfway (each rank keeps only its shard).

**"Bandwidth-bound"** means for the sizes we care about (tens of GB of gradients) the $M/B$ term dominates the latency term, so the wall-clock is set by the slowest link's bytes/second, not by how many messages are sent. That's why you bucket small tensors (amortize $\alpha$) and why interconnect bandwidth, not GPU FLOPs, decides DP scaling.

**Bandwidths to hold in your head** (per GPU, per direction): NVLink 4 on H100 ≈ 450 GB/s (900 bidirectional; 1.8 TB/s on B200), NDR InfiniBand 400 Gb/s ≈ 50 GB/s per NIC (one NIC per GPU on a DGX), PCIe Gen5 x16 ≈ 64 GB/s, HBM3 ≈ 3.35 TB/s. NCCL builds every collective (all-reduce, reduce-scatter, all-gather, broadcast, all-to-all, send/recv) from rings and trees over that discovered topology, hierarchically: reduce inside the node on NVLink, then across nodes on IB, so IB carries 1/8 of the naïve traffic. `NCCL_DEBUG=INFO` prints the rings/trees and confirms GPUDirect RDMA is on.

Numbers: 10B params in bf16 = 20 GB of gradients. On an 8-GPU H100 node (NVLink ≈ 450 GB/s per direction per GPU): $2 \cdot \frac{7}{8} \cdot 20/450 \approx 78$ ms. Across 64 nodes over NDR InfiniBand at 400 Gb/s ≈ 50 GB/s per GPU: $2 \cdot 20/50 \approx 0.8$ s. A forward+backward step on a 10B model with a decent per-GPU batch is a few seconds, so DDP still works, but the exposed fraction grows fast if you can't overlap. Hierarchical/tree all-reduce wins when $N$ is large and latency matters (small messages); NCCL picks automatically and uses NVLink for the intra-node stage and IB for the inter-node stage.

</details>


### [06-03] Explain ZeRO stages 1, 2, 3 and FSDP. Do the memory arithmetic for a mixed-precision Adam run.  (difficulty 2; tags: distributed, zero, fsdp, memory)

- Follow-up: which one is PyTorch FSDP equivalent to?
- Follow-up: what does ZeRO-3 NOT shard?


<details><summary>Answer</summary>

**Mixed-precision Adam costs 16 bytes per parameter**: $2\,(\text{bf16 weight}) + 2\,(\text{bf16 gradient}) + 4\,(\text{fp32 master weight}) + 4\,(\text{Adam } m) + 4\,(\text{Adam } v)$. A 10B model is 160 GB of state before a single activation — it does not fit on one 80 GB H100, so plain DDP is out.

ZeRO observes that in DDP all of that state is **replicated** on every rank, which is pure waste. With $N$ data-parallel ranks:
- **ZeRO-1** shards the optimizer states (12 of the 16 bytes): per-GPU $4 + 12/N$ bytes/param. Each rank updates only its $1/N$ slice of parameters, then all-gathers the updated weights.
- **ZeRO-2** also shards gradients: $2 + 14/N$. Gradients are reduce-scattered instead of all-reduced, so each rank receives only the slice it will update.
- **ZeRO-3** also shards the bf16 weights: $16/N$. Parameters are all-gathered just-in-time before each layer's forward and backward, then freed.

For 10B on 64 GPUs: DDP 160 GB (impossible), ZeRO-1 ≈ 52 GB, ZeRO-2 ≈ 34 GB, ZeRO-3 ≈ 2.5 GB per GPU for model state. The rest of the memory budget is **activations**, which ZeRO does not touch — that is what tensor/sequence parallelism and activation checkpointing are for.

**PyTorch FSDP is ZeRO-3** (with `SHARD_GRAD_OP` giving ZeRO-2 behaviour and HSDP giving ZeRO-3 within a node and replication across nodes). FSDP wraps modules into "units" (typically one transformer block each); each unit is a flat sharded parameter that is all-gathered on entry and reduce-scattered on backward, with prefetching of the next unit to overlap. ZeRO-3 also does not shard activations, and in its basic form it does not reduce the per-layer temporary: you still need the full unsharded weights of one block resident at a time.

</details>


### [06-04] What is the communication cost of ZeRO-3 relative to plain data parallelism, and when would you pick ZeRO-3/FSDP over tensor parallelism?  (difficulty 3; tags: distributed, zero, fsdp, communication)

- Follow-up: why does ZeRO-3 scale badly across many nodes while ZeRO-1 does not?


<details><summary>Answer</summary>

Count bytes per parameter per step, with $M$ = model size in bf16:
- **DDP**: one all-reduce of gradients = reduce-scatter + all-gather = **$2M$**.
- **ZeRO-1/2**: reduce-scatter of gradients ($M$) + all-gather of updated weights ($M$) = **$2M$ — same as DP**. Free memory savings.
- **ZeRO-3**: all-gather weights for forward ($M$) + all-gather weights again for backward ($M$) + reduce-scatter gradients ($M$) = **$3M$, i.e. 1.5× DP**.

The 1.5× is per-parameter volume; the real problem is granularity and latency: ZeRO-3 issues one all-gather per layer on the critical path, and every layer's compute must wait for its own parameters. Within a node on NVLink that is easily overlapped by prefetching the next block; across InfiniBand the all-gathers become exposed unless the per-GPU batch is large enough that the compute per block ($\approx 6 \cdot \text{params}_{\text{block}} \cdot \text{tokens}$ FLOPs) exceeds all-gather time. That is why ZeRO-1 scales to thousands of GPUs without much thought, while ZeRO-3 is usually confined to a node or a few nodes (HSDP: shard within the node, replicate across nodes).

**When ZeRO-3/FSDP vs tensor parallelism**:
- FSDP is a memory trick that keeps compute local — every GEMM is full-size and efficient; no code changes to the model. Use it when the model fits in aggregate node memory and per-GPU batch is decent.
- TP splits each GEMM so every layer needs two all-reduces on activations (forward) and two more (backward). It reduces **activation** memory as well and shortens per-GPU time per step, but every GEMM gets smaller (worse utilisation), so it only pays inside NVLink.
- Rule of thumb from Megatron/Meta: TP up to 8 inside the node, then FSDP/ZeRO-1 across nodes, add PP only when TP=8 + FSDP still doesn't fit or the all-gather volume dominates (models $\gg$ 70B).

For a 10B video DiT the honest answer is: FSDP within a node plus context parallelism handles it; TP is optional.

</details>


### [06-05] Walk me through Megatron-style tensor parallelism for one transformer layer. How many collectives per layer, and why does TP stay inside the node? Contrast with expert parallelism.  (difficulty 2; tags: distributed, tensor-parallel, megatron, moe)

- Follow-up: what does Megatron "sequence parallelism" add on top of TP?
- Follow-up: why column-then-row and not row-then-column?


<details><summary>Answer</summary>

**Tensor parallelism splits the weight matrices of a layer across $t$ GPUs so that each GPU does $1/t$ of every GEMM, at the cost of one all-reduce on the activations per sub-block.**

MLP: $Y = \operatorname{GeLU}(XA)\,B$.
- **Column-split $A$** into $[A_1 \dots A_t]$: each GPU computes $\operatorname{GeLU}(XA_i)$ independently — the nonlinearity is elementwise so no sync is needed. This is why column comes first; row-splitting $A$ would require a sum before the GeLU.
- **Row-split $B$**: each GPU computes its partial product $Y_i = \operatorname{GeLU}(XA_i)\,B_i$ and the outputs are summed with an **all-reduce**.

Attention: split heads across GPUs ($Q, K, V$ column-parallel — each GPU owns $h/t$ heads, so attention is local), output projection row-parallel → one all-reduce. Layer total: **2 all-reduces in forward, 2 in backward** (the identity/all-reduce pair f/g are conjugate). Each all-reduce is on a tensor of shape $[b, s, h]$ in bf16 — for $s = 32\text{k}$, $h = 4096$, $b = 1$ that is 256 MB per all-reduce, four times per layer, ~50 layers: ~50 GB per step per GPU of activation traffic. Only NVLink (~450 GB/s) makes that cheap; over IB (~50 GB/s) it would take a second per step. Hence **TP $\le 8$, within the NVLink domain**.

Megatron **sequence parallelism** notices that LayerNorm and dropout between the TP regions are replicated on every GPU; it shards those along the sequence and replaces each all-reduce with a reduce-scatter + all-gather (same bytes), cutting activation memory by another factor $t$.

**Expert parallelism** places different MoE experts on different GPUs. The router decides per token which expert it goes to, so the communication is an **all-to-all** (dispatch tokens, compute, all-to-all back) — two all-to-alls per MoE layer. Volume scales with tokens × hidden, like TP, but the pattern is irregular and load-imbalanced (hot experts), which is why capacity factors, token dropping and load-balancing losses exist. EP is usually paired with DP inside the expert groups and can span nodes better than TP because the all-to-all is one-shot rather than per-GEMM.

</details>


### [06-06] Pipeline parallelism: what is the bubble, how do GPipe and 1F1B differ, and what do interleaved schedules buy you?  (difficulty 2; tags: distributed, pipeline-parallel, scheduling)

- Follow-up: why is PP awkward for a DiT with adaLN conditioning or for encoder-decoder models?


<details><summary>Answer</summary>

**Pipeline parallelism puts different layers on different GPUs; the price is the bubble — time when stages idle waiting for the first micro-batches to arrive or the last to drain.**

With $p$ stages and a global batch split into $m$ micro-batches, each stage is busy for $m$ forward + $m$ backward slots but the schedule spans $(m + p - 1)$ of each, so **bubble fraction $= \frac{p-1}{m}$** of the ideal time (equivalently, efficiency $= \frac{m}{m+p-1}$). To keep the bubble under ~5% you need $m \gtrsim 20p$ micro-batches, which means large global batches — often not available with 32k-token video sequences.

- **GPipe**: run all $m$ forwards, then all $m$ backwards. Simple, but every stage must hold activations for all $m$ micro-batches → activation memory $\propto m$.
- **1F1B** (PipeDream-flush, Megatron): after warm-up, each stage alternates one forward and one backward, so at most $p$ micro-batches are in flight per stage → activation memory $\propto p$ instead of $m$. Same bubble fraction, much less memory.
- **Interleaved 1F1B**: give each GPU $v$ non-contiguous chunks of layers (e.g. GPU0 holds layers 0-3 and 16-19). The "virtual" pipeline is $pv$ deep with smaller stages, so the bubble drops to **$\frac{p-1}{vm}$** at the cost of $v\times$ more point-to-point sends per micro-batch. Zero-bubble schedules (ZB-H1/H2) go further by splitting backward into input-gradient and weight-gradient parts and reordering.

Traps: PP needs the model to be a clean chain of stages with equal compute; DiT blocks are uniform (good), but the shared timestep/text conditioning must be broadcast to every stage, and anything with cross-stage skip connections (U-Nets) or unbalanced first/last stages (embedding, VAE, loss) kills balance. PP traffic is tiny (one activation tensor per micro-batch boundary), so it is the parallelism you put across nodes when you must.

</details>


### [06-07] You are training a video DiT on 32k-token sequences. Explain sequence/context parallelism: DeepSpeed-Ulysses versus ring attention. When would you use each, and how do you fix load imbalance with a causal mask?  (difficulty 3; tags: distributed, context-parallel, ring-attention, ulysses, video)

- Follow-up: how does the KV-heads count of GQA constrain Ulysses?
- Follow-up: can you combine Ulysses and ring?


<details><summary>Answer</summary>

**Context parallelism (CP) shards the sequence axis across $P$ GPUs so activations shrink by $P$; the only operation that needs the whole sequence is attention, and the two schemes differ in how they get it.**

**Ulysses (all-to-all on heads)**: every GPU holds $s/P$ tokens of all heads after the MLP. Before attention, an all-to-all re-partitions $Q, K, V$ so each GPU holds all $s$ tokens of $h/P$ heads; attention is then completely local (regular flash-attention). After attention, another all-to-all goes back to sequence-sharded. Four all-to-alls per layer ($Q, K, V, O$), each moving $\frac{P-1}{P}$ of a $[s/P, h]$ tensor per GPU — traffic per GPU $\propto sh/P$, so it *decreases* with $P$. Limitation: **$P \le$ number of heads** (with GQA, $\le$ KV heads unless you replicate KV), and all-to-all is latency-heavy across nodes.

**Ring attention (pass KV blocks)**: each GPU keeps its $s/P$ queries and its own KV block. In $P-1$ steps it sends its KV block to the next GPU in the ring and receives the previous one, computing a partial attention block each step and merging with the running online-softmax (log-sum-exp) state, exactly like flash-attention across GPUs. Communication per step is a $[s/P, h_{\text{kv}}]$ KV block, overlapped with the compute of the current block. No head limit, scales to millions of tokens, but $P-1$ sequential rounds and overlap only works if block compute time $\ge$ KV transfer time — which needs long chunks per GPU.

**Causal imbalance**: with a causal mask and contiguous chunks, GPU 0 does 1 block of work while GPU $P-1$ does $P$ blocks. The fix is **zig-zag assignment**: split into $2P$ chunks and give GPU $i$ chunks $i$ and $2P-1-i$, so every GPU has one early and one late chunk and does the same total work. Video DiTs are usually bidirectional in space and time so this only matters for autoregressive/causal video models.

When to use: Ulysses for $P \le 8$ within a node (cheap, simple, no kernel changes); ring for cross-node or very long context; **2D hybrid** (Ulysses inside the node × ring across nodes) is what USP/Megatron-CP do for 100k+ token video. For our 32k tokens and a ~48-head model, CP = 4–8 with Ulysses is the natural pick.

</details>


### [06-08] Give me a concrete layout: a 10B-parameter DiT, 32k tokens per sample, 512 H100s. Choose DP/TP/PP/CP and justify with memory and communication numbers.  (difficulty 3; tags: distributed, parallelism, planning, dit, video)

- Follow-up: what changes if it's a B200 cluster?
- Follow-up: what if the model were a 10B MoE with 64 experts?


<details><summary>Answer</summary>

Start from the constraints, then place the parallelism.

**Model state**: $10\text{B} \times 16\,\text{B} = 160$ GB → must be sharded across $\ge 3$ GPUs; ZeRO-1/FSDP across the DP group handles this.

**Activations**: for a transformer block with flash attention, $\approx 34\,sbh$ bytes per layer (Megatron's formula minus the $s^2 a$ term). With $h = 4096$, $s = 32768$, $b = 1$: $34 \times 32768 \times 4096 \approx 4.6$ GB per layer; ~48 layers → ≈ 220 GB per sample. Even with full checkpointing (keep only the block input, $2sh = 256$ MB per layer) you still need to hold one block's internals (~4.6 GB) during recompute — fine — but the un-checkpointed case demands sharding the sequence. So **CP (or TP+SP) is mandatory**, and activation memory, not weights, is the binding constraint.

**Compute**: $6N$ per token $= 60$ GFLOP, plus attention $12 L s h$ per token $\approx 12 \times 48 \times 32768 \times 4096 \approx 77$ GFLOP (the $s/(6h) \approx 1.33$ factor — at 32k tokens attention exceeds the GEMMs). $\approx 140$ GFLOP/token. Global batch 512 samples = 16.8M tokens/step → 2.35 EFLOP per step. At 40% MFU, $512 \times 989\,\text{TFLOPS} \times 0.4 \approx 200$ PFLOP/s → ~12 s/step.

**Layout** (64 nodes × 8 GPUs):
- **TP = 1 or 2**: the 4096-wide GEMMs are already small per token; TP would shrink them further. I'd rather spend the NVLink on CP.
- **CP = 8 within the node** (Ulysses on heads; $48\ \text{heads} / 8 = 6$ heads per GPU): activation memory per GPU $\approx 220/8 \approx 28$ GB unchecked, ~10 GB with selective checkpointing. All-to-all traffic per layer ~$4 \times 256\,\text{MB} / 8$ per GPU on NVLink — negligible.
- **PP = 1**: uniform blocks make PP possible, but with $b = 1$ per GPU the micro-batch count is too small to hide the bubble ($\frac{p-1}{m}$ with $m \approx 8$ is 12%+). Not worth it at 10B.
- **DP = 64 across nodes with FSDP/HSDP**: shard weights within the node, replicate across → per-node all-gathers on NVLink; the cross-node traffic is one reduce-scatter of 20 GB of grads per step over $8 \times 400\,\text{Gb/s} \approx 400\,\text{GB/s}$ per node $\approx 0.1$ s — under 1% of the 12 s step and fully overlappable.

Result: (CP 8) × (DP 64), FSDP inside the node, ZeRO-1 optimizer sharding across, selective activation checkpointing, VAE latents pre-encoded. On B200s (192 GB, 1.8 TB/s NVLink, ~2.2 PFLOPS bf16) I'd drop checkpointing first, then reduce CP to 4 to raise per-GPU sequence length. For an MoE variant, add EP = 8 across the same NVLink group and keep the all-to-all inside the node, with DP over experts across nodes.

</details>


### [06-09] Where does activation memory come from in a transformer, and how does activation checkpointing trade it for compute? What is "selective" recomputation?  (difficulty 2; tags: distributed, memory, activation-checkpointing)

- Follow-up: the "33% extra compute" number — where does it come from exactly?


<details><summary>Answer</summary>

Activations are every intermediate autograd needs for backward: attention inputs, Q/K/V, softmax output (if not fused), MLP pre-activations, LayerNorm inputs, dropout masks. Megatron's count for one layer in bf16 is **$sbh\,(34 + 5as/h)$ bytes**: the $34\,sbh$ is the linear-size stuff, the $5 a s^2 b$ term is the attention probabilities — quadratic in sequence length and the first thing that explodes at 32k tokens. **Flash attention removes the $s^2$ term** by recomputing the softmax in backward from stored log-sum-exp statistics, which is itself a form of selective checkpointing.

**Activation checkpointing** stores only the input of each transformer block ($2sbh$ bytes) and re-runs the block's forward during backward to regenerate the intermediates. Memory drops from ~$34\,sbh$ per layer to ~$2sbh$ plus one live block; the cost is one extra forward. Since forward is ~1/3 of the step's FLOPs (forward $2ND$, backward $4ND$), full recomputation adds **≈ 33% compute** — MFU looks fine but HFU rises to $8ND$ per token.

**Selective recomputation** (Megatron "selective") only checkpoints the expensive-to-store but cheap-to-recompute parts — historically the attention softmax/dropout, which is ~70% of the memory and ~few % of FLOPs. With flash attention that part is already gone, so modern selective schemes checkpoint every $k$-th block, or offload the block inputs to CPU (asynchronously over PCIe) instead of recomputing. The practical decision rule: checkpoint just enough that you can raise per-GPU batch or sequence until the GEMMs saturate; beyond that the 33% is wasted.

</details>


### [06-10] bf16 versus fp16 versus fp8: what breaks with each, why do we keep fp32 master weights, and what is loss scaling?  (difficulty 1; tags: mixed-precision, bf16, fp8, numerics)

- Follow-up: what exactly does Transformer Engine do to make fp8 work?


<details><summary>Answer</summary>

**fp16** has 5 exponent bits: max 65504, smallest normal $\approx 6 \times 10^{-5}$. Gradients routinely underflow to zero, so you use **loss scaling**: multiply the loss by $S$ (e.g. $2^{16}$) before backward, unscale the gradients before the optimizer step, and dynamically halve $S$ when you see inf/NaN and double it after a few thousand clean steps. That is `torch.cuda.amp.GradScaler`.

**bf16** keeps fp32's 8 exponent bits and drops mantissa to 7 bits (≈ 3 significant decimal digits). Same dynamic range as fp32 → no loss scaling, no overflow drama, and it is the default on A100/H100. The price is precision: a weight update $\text{lr} \cdot g$ with $\text{lr} \cdot g / w < 2^{-8} \approx 0.4\%$ is **rounded away entirely** if the weight itself is bf16. That's why we keep **fp32 master weights** (and fp32 Adam moments, and usually an fp32 all-reduce or at least fp32 accumulation in the reduction): the update is applied in fp32, then cast to bf16 for the next forward. Also keep sensitive reductions in fp32: softmax, LayerNorm statistics, loss, and the attention logits if they grow large.

**fp8** (H100+): two formats, **E4M3** (more mantissa, range $\pm 448$, for weights and activations in forward) and **E5M2** (more range, for gradients). Both are useless without **per-tensor scaling**: each tensor is multiplied by a scale so its max lands in range, and Transformer Engine uses *delayed scaling* — the scale for this step comes from the amax history of previous steps so the cast can be done without a separate reduction. Only the GEMM inputs are fp8; accumulation is fp32, master weights and optimizer states stay in higher precision, and LayerNorm/softmax stay bf16. Payoff: 2× the peak of bf16 (≈ 1979 vs 989 dense TFLOPS on H100) and half the activation bytes; risk: outlier channels and instability late in training, which is why people often fall back to bf16 for the last few percent.

</details>


### [06-11] Gradient accumulation, effective batch size, and learning-rate scaling: how do they interact, and why warmup?  (difficulty 1; tags: optimization, batch-size, learning-rate)

- Follow-up: does gradient accumulation give the same result as a bigger batch with BatchNorm?


<details><summary>Answer</summary>

**Effective batch = micro-batch × gradient-accumulation steps × data-parallel world size.** Accumulation runs $k$ forward/backward passes summing gradients before one optimizer step; mathematically identical to a $k\times$ larger batch (for losses that are means, divide by $k$), except for anything with batch statistics — BatchNorm sees the micro-batch, which is one reason large models use LayerNorm/RMSNorm. In DDP wrap the first $k-1$ micro-steps in `no_sync()` so you all-reduce once.

**LR scaling** when you grow the batch by $k$:
- **Linear** (Goyal et al., SGD): $\text{lr} \propto k$, justified because $k\times$ batch with $\text{lr} \cdot k \approx k$ small steps of $\text{lr}$ in the small-lr limit. Works until the "critical batch size", beyond which extra samples stop reducing gradient noise and you get diminishing returns (McCandlish gradient-noise scale).
- **Square-root** ($\text{lr} \propto \sqrt{k}$) is the common heuristic for Adam because Adam normalizes by gradient magnitude; the noise standard deviation shrinks as $1/\sqrt{k}$.
- Neither is a law; sweep at small scale and, ideally, use μP so the optimal LR transfers across width.

**Warmup** exists because early on the Adam second-moment estimate $v$ is tiny and noisy (bias-corrected but from few samples), the network's activations are far from their stationary statistics, and a full-size LR step kicks the weights into a bad region you never recover from. Linear warmup over ~1–5% of training (or a few thousand steps) lets $v$ stabilise and the LayerNorm gains settle. Follow with cosine or WSD (warmup-stable-decay) decay. Practical DiT setting: batch 256–1024 videos, AdamW lr $10^{-4}$, $\beta = (0.9, 0.95)$, warmup 1–5k steps, no weight decay on norms/biases, EMA of weights for sampling.

</details>


### [06-12] State the Chinchilla scaling-law result and how you would use scaling laws to plan a video DiT training run.  (difficulty 2; tags: scaling-laws, chinchilla, dit)

- Follow-up: why do practitioners "overtrain" past Chinchilla-optimal?
- Follow-up: what's the y-axis for a diffusion model — loss or FID?


<details><summary>Answer</summary>

**Chinchilla**: for a fixed compute budget $C \approx 6ND$ FLOPs, the loss is minimised when parameters $N$ and training tokens $D$ grow together at roughly **$D \approx 20N$**. Fit form: $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$ with $\alpha \approx 0.34$, $\beta \approx 0.28$. GPT-3 (175B on 300B tokens) was badly under-trained; Chinchilla (70B on 1.4T) beat it with the same compute.

**Overtraining**: the compute-optimal point ignores inference cost. If you will serve the model a lot, a smaller model trained on 5–20× more tokens than optimal is only slightly worse in loss but much cheaper to run (LLaMA-3 8B on 15T tokens is ~100× past Chinchilla). For a research intern project the calculus flips — you want the best model for a fixed training budget.

**Applying it to DiTs**: the same functional form fits (Li et al. 2024 "Scalability of Diffusion Transformers", Liang et al. 2024): loss vs $N$ and $D$ follows a power law and the optimum is also roughly balanced. Caveats specific to diffusion:
- "Tokens" = latent patches × frames; a 32k-token video is one sample. Data $D$ is counted in tokens seen, including repeated epochs, and video datasets are small in unique tokens so repetition curves matter.
- The **training loss is a noisy proxy**: the flow-matching MSE at random $t$ is dominated by high-noise timesteps and correlates only loosely with FID/FVD or human preference. Fit the law on the loss (smooth, cheap), then verify the top few points on FVD/quality.
- Attention makes FLOPs non-linear in $N$ at long sequence (the $s/(6h)$ factor), so use measured FLOPs, not $6ND$.
- Procedure: train 5–8 models from 100M to 2B for several token budgets, fit the frontier, extrapolate to the 10B budget, and derive the target tokens and batch/LR from μP-style transfer. Expect the DiT optimum to be more data-hungry than LLMs because the loss is a regression target with high irreducible variance.

</details>


### [06-13] How do you compute MFU for a training run, and what number would make you happy on H100s? What's the difference between MFU and HFU?  (difficulty 2; tags: mfu, throughput, flops)

- Follow-up: 10B DiT, 16.8M tokens/step, 12 s/step on 512 H100s — what's the MFU?


<details><summary>Answer</summary>

**MFU = model FLOPs actually needed per second ÷ hardware peak FLOPs per second.** The model FLOPs come from a formula, not a profiler: for a dense transformer **$\approx 6N$ per token** ($2N$ forward: one multiply-add per parameter per token; $4N$ backward: gradient w.r.t. inputs and w.r.t. weights) plus the attention term **$12 L s h$ per token** (flash attention's recomputation is *not* counted — that's the point of "model" FLOPs). So

$$
\text{MFU} = \frac{(6N + 12 L s h) \cdot \text{tokens per second}}{n_{\text{GPU}} \cdot \text{peak}}.
$$

Peak: H100 SXM bf16 dense **989 TFLOPS** (the 1979 number is with 2:4 sparsity — never use it), A100 312, B200 ≈ 2.25 PFLOPS dense bf16.

Example: $N = 10\text{B}$, $L = 48$, $s = 32768$, $h = 4096$ → $60 + 77 \approx 137$ GFLOP/token; $16.8\text{M tokens} / 12\,\text{s} = 1.4\text{M}$ tokens/s → $1.92 \times 10^{17}$ FLOP/s; peak $512 \times 989 \times 10^{12} = 5.06 \times 10^{17}$ → **MFU $\approx 38\%$**.

**HFU** (hardware FLOPs utilisation) counts every FLOP the hardware actually executed, including activation recomputation ($+2N$ per token → $8N$ with full checkpointing) and flash-attention recompute. HFU $\ge$ MFU; the gap is your recompute overhead. Report MFU — it is what you'd pay for.

What's good: LLM pretraining on H100 with well-tuned Megatron reaches **40–50%** (PaLM on TPU reported 46%, LLaMA-3 ~38–43%); anything above 35% for a long-sequence video DiT with CP is solid, under 25% means something is exposed (comm, dataloader, small GEMMs from too much TP, python overhead). Roofline: the GEMMs are compute-bound at these shapes; MFU loss comes from non-GEMM ops (norms, activations, RoPE, adaLN modulation — memory-bound, fused kernels help), communication that failed to overlap, and pipeline bubbles.

</details>


### [06-14] Data loading for a 512-GPU video run: how do you shard the dataset, why pre-encode with the VAE, and how do you make resume deterministic?  (difficulty 1; tags: data-loading, webdataset, video, vae)

- Follow-up: how do you know the dataloader is the bottleneck?


<details><summary>Answer</summary>

**Sharded tar archives (WebDataset-style)**: pack samples into ~1 GB shards, assign shards to ranks (rank $r$ takes shards $r, r + W, r + 2W, \dots$, $W$ = number of dataloader workers × world size), stream them sequentially from object storage or a parallel FS. Sequential reads of large files are what storage is good at; millions of small mp4s are what it is terrible at. Shuffle at two levels: shuffle the shard list per epoch, then a per-worker in-memory shuffle buffer of a few thousand samples.

**Pre-encode with the VAE**: decoding mp4 → frames → 3D-VAE encode is far more expensive than the DiT step for a 1.3B–10B model (the WAN VAE compresses 4× temporally, 8×8 spatially; an 81-frame 480p clip is ~$21 \times 60 \times 104 \times 16$ latents). Doing it online wastes GPU time and makes the dataloader the bottleneck. Instead encode once offline (with several random crops/aspect buckets and the text embeddings from T5/UMT5), store latents in bf16 or fp16 plus the text-encoder outputs, and train from tensors. Cost: storage (latents are ~1/100 of pixels, fine) and losing on-the-fly augmentation. Keep aspect-ratio buckets so each batch has uniform token count and no padding.

**Deterministic resume**: the sampler state must be a pure function of (seed, epoch, step). Derive the shard order from `seed + epoch`, record the global sample index in the checkpoint, and on resume fast-forward the iterator (skip $N$ samples, or store per-shard offsets) rather than restarting the epoch — otherwise you retrain on seen data and the loss curve gets a visible kink. Checkpoint the RNG states of every rank too.

Diagnosis: profile with the torch profiler or simply time `next(loader)` — if the GPU idles between steps or `nvidia-smi` shows utilisation sawtoothing, raise num_workers, use pinned memory + non_blocking copies, prefetch 2–4 batches, and move CPU decode to DALI or GPU.

</details>


### [06-15] Fault tolerance at scale: how often do you checkpoint, what is elastic training, and how do you handle NaNs and silent data corruption?  (difficulty 2; tags: fault-tolerance, checkpointing, elastic, reliability)

- Follow-up: how would you even detect a GPU that computes wrong answers?


<details><summary>Answer</summary>

At 512 GPUs a hardware failure every few hours is normal (Meta reported ~1 failure per 3 hours on 16k H100s), so the question is how much work you lose and how fast you restart.

**Checkpoint cadence**: the Young/Daly rule gives the optimal interval $\tau \approx \sqrt{2 \delta \cdot \text{MTBF}}$, where $\delta$ is the time to write a checkpoint. 160 GB of state for 10B params written asynchronously (copy to host, then to storage on a background thread, distributed/sharded save so every rank writes its shard) takes $\delta \sim 1$ min; MTBF 3 h → $\tau \approx 20$ min. Save sharded (FSDP `state_dict_type=SHARDED`, torch.distributed.checkpoint) and consolidate offline; keep the last few plus periodic permanent ones.

**Elastic training**: `torchrun --max-restarts` with a c10d rendezvous: when a rank dies, the agent kills all workers, re-rendezvouses (possibly with fewer nodes if `--nnodes=min:max`), re-assigns ranks, and every worker reloads the last checkpoint. Automated node health checks pull bad nodes before the job restarts on them. Restart time is dominated by loading the checkpoint and re-initialising NCCL, so keep checkpoints close (local NVMe or a burst buffer).

**NaN handling**: check the global gradient norm every step (you already compute it for clipping). On inf/NaN: skip the optimizer step (what `GradScaler` does for fp16) and count; a few per thousand steps is tolerable, a burst means a bad batch, an LR problem or a broken GPU. Never let a NaN reach the weights — one corrupted parameter propagates to all ranks through the all-reduce.

**Silent data corruption (SDC)**: a GPU that produces wrong numbers with no error. Detection is by redundancy: periodically run the same micro-batch on two ranks and compare, or compare per-rank gradient norms before the all-reduce — an outlier rank is either seeing a strange batch or is broken. Log Xid errors from the driver, run DCGM diagnostics on nodes with anomalies, and keep the training loss/grad-norm curves so a divergence can be traced back to the exact checkpoint to roll back to.

</details>


### [06-16] War story: your 512-GPU run hangs at step 500, no error, GPUs at 100% utilisation. Walk me through how you debug it, and how you would have profiled the run beforehand.  (difficulty 3; tags: debugging, nccl, profiling, hang)

- Follow-up: why do hung NCCL kernels show 100% GPU utilisation?
- Follow-up: how do you find a straggler rank?


<details><summary>Answer</summary>

Step 500 is not random — it is almost certainly the first time something periodic runs: evaluation, checkpoint save, logging with a gathered metric, or an LR-schedule boundary. A hang with 100% utilisation is the signature of a **collective mismatch**: some ranks are spinning in an NCCL kernel waiting for peers that never arrive (NCCL kernels busy-wait, hence the 100%).

Procedure:
1. **Get stack traces from every rank** without killing the job: `py-spy dump --pid` on each worker (a wrapper script over all nodes), or the built-in `TORCH_NCCL_TRACE_BUFFER_SIZE` flight recorder + `TORCH_NCCL_DUMP_ON_TIMEOUT`, which dumps the last $N$ collectives per rank with their sequence numbers and sizes. Set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and a sane timeout so hangs become exceptions instead of eternity.
2. **Diff the traces**: the rank whose stack differs (e.g. sitting in `save_checkpoint` or `evaluate` while the others are in `all_reduce`) is the culprit. Classic causes: rank 0 does `if rank == 0: save()` which internally calls a collective (FSDP state dict all-gathers!) that the others never join; uneven data — one rank's dataloader ran out of samples and it exited the loop; a data-dependent branch (`if loss.isnan(): continue`) taken on one rank; `find_unused_parameters` mismatch; a `torch.distributed.barrier()` in a rank-conditional block.
3. If all stacks are in the same collective, check the hardware: `NCCL_DEBUG=INFO` for link errors, `dmesg`/Xid for a fallen-off NVLink or NIC, `ibstat`, and a standalone nccl-tests all-reduce on the suspect node.

**Profiling beforehand**: `torch.profiler` with CUPTI for one rank for a few steps → look for gaps between kernels (dataloader, python overhead), NCCL kernels not overlapped with compute (comm exposed), small GEMMs (TP too high). `nsys profile` across ranks with `--capture-range` for the multi-GPU timeline. **Stragglers**: log per-rank step time and the wait time inside the first collective of each step (`torch.cuda.Event` timing before/after all-reduce); a rank whose compute is consistently slow (thermal throttling, a bad HBM stack, a noisy neighbour on the network) will show up as everyone else waiting on it. Meta's practice is to run this straggler detector continuously and evict the node.

</details>


### [06-17] The loss spikes and sometimes diverges once you scale to 10B. List the likely causes and the fixes you would try, in order.  (difficulty 3; tags: stability, loss-spikes, optimization, dit)

- Follow-up: what does QK-norm fix mechanically?
- Follow-up: what is μP and why does it help here?


<details><summary>Answer</summary>

Spikes at scale come from a few well-known mechanisms; the order below is the order I'd check them.

1. **Learning rate too high for the width**: the optimal LR shrinks roughly as $1/\text{width}$ for Adam under standard parameterisation. Fix: lower LR, longer warmup, or adopt **μP** (maximal update parameterisation): scale init and per-layer LRs so that activations and updates stay $O(1)$ as width grows, which makes the small-model LR sweep transfer to the big model. Also lower Adam $\beta_2$ from 0.999 to 0.95 — with $\beta_2 = 0.999$ a rare large gradient inflates $v$ slowly and the following updates over-shoot.
2. **Attention logit growth**: as training proceeds, $\|q\| \cdot \|k\|$ grows, the softmax saturates, gradients through it vanish for most tokens and explode for a few — the entropy collapse that precedes a spike. **QK-norm** (LayerNorm/RMSNorm on $q$ and $k$ before the dot product, standard in ViT-22B, SD3, WAN) bounds the logits to a learnable temperature. Related: bf16 rounding on large logits — compute attention scores and softmax in fp32 (flash attention already does).
3. **Output-logit / modulation drift**: in LLMs the $z$-loss ($10^{-4} \log^2 Z$) keeps the softmax normaliser near 1; in DiTs the analogue is the adaLN modulation vectors (scale/shift/gate) blowing up — zero-init the gate, RMSNorm the modulation input, and apply weight decay to them.
4. **Gradient clipping** at global norm 1.0 is a must; watch the pre-clip norm — a rising trend predicts the spike a few hundred steps early.
5. **Data**: a batch of corrupted latents or an aspect bucket with an extreme token count. Log per-batch loss with sample ids; when a spike is data-driven, PaLM's fix works — roll back to the previous checkpoint and **skip ~100–200 batches**.
6. **Numerics**: fp16 without loss scaling, fp8 outliers, or a LayerNorm $\epsilon$ that is too small relative to bf16 activations. Check for inf in the gradient norm before the clip.
7. Diffusion-specific: the loss at low-noise timesteps is tiny and at high-noise huge; use logit-normal or SD3-style timestep sampling and loss weighting so one $t$-range does not dominate the gradient, and keep an EMA of the weights so sampling quality is not hostage to the spike.

If it still diverges: reduce LR by 2×, increase warmup, or init the residual branches with $1/\sqrt{2L}$ scaling (or zero) so the residual stream does not grow with depth.

</details>


### [06-18] Get concrete at the API level: what does `torch.distributed.run` do, what are process groups, and sketch how you'd implement a ring-attention step with send/recv.  (difficulty 3; tags: torch-distributed, api, ring-attention, implementation)

- Follow-up: why `batch_isend_irecv` rather than blocking `send`/`recv`?
- Follow-up: how do you merge two partial attention outputs correctly?


<details><summary>Answer</summary>

**`torch.distributed.run` (torchrun)** is the elastic launcher: on each node it starts an agent that spawns `--nproc_per_node` worker processes, performs a rendezvous through a c10d TCP store at `MASTER_ADDR:MASTER_PORT`, and sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE` in each worker's environment. The worker then calls `dist.init_process_group("nccl")`, which reads those variables, exchanges NCCL unique ids through the store, and builds the default (world) communicator. `--max-restarts` and `--nnodes=min:max` give the elastic behaviour: on a worker failure the agent tears everything down and re-rendezvouses. Always set `torch.cuda.set_device(LOCAL_RANK)`.

**Process groups** are subsets of ranks with their own NCCL communicator. Every parallelism dimension is a process group: for CP=8 × DP=64 the DP group of a rank is the 64 ranks with the same position inside the node, the CP group its 8 node-mates. `dist.new_group(ranks)` must be called by *all* ranks in the same order (it is collective), and the modern way is `init_device_mesh("cuda", (64, 8), mesh_dim_names=("dp", "cp"))`, whose `mesh["cp"].get_group()` hands you the group; FSDP and the CP ops take it as an argument. Each rank belongs to one group per dimension.

**Ring attention with send/recv** (per layer; $q, k, v$ are $[b, s/P, h_{\text{local}}, d]$ on each rank of the cp group):

```python
def ring_attn(q, k, v, group):
    P, r = group.size(), group.rank()
    nxt, prv = (r + 1) % P, (r - 1) % P
    out, lse = None, None
    k_recv, v_recv = torch.empty_like(k), torch.empty_like(v)
    for step in range(P):
        if step < P - 1:  # post next KV exchange before computing
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, k, nxt, group), dist.P2POp(dist.irecv, k_recv, prv, group),
                dist.P2POp(dist.isend, v, nxt, group), dist.P2POp(dist.irecv, v_recv, prv, group)])
        o_blk, lse_blk = flash_attn_fwd(q, k, v, return_lse=True)  # local block
        out, lse = merge(out, lse, o_blk, lse_blk)              # online softmax
        if step < P - 1:
            for rq in reqs: rq.wait()
            k, v, k_recv, v_recv = k_recv, v_recv, k, v
    return out
```

The merge is the flash-attention rescale: with running $(o, l)$ and new block $(o', l')$, $l_{\text{new}} = \operatorname{logaddexp}(l, l')$, $o_{\text{new}} = o \exp(l - l_{\text{new}}) + o' \exp(l' - l_{\text{new}})$. Backward runs the ring in the same direction, passing $\mathrm{d}K/\mathrm{d}V$ accumulators along.

Why `batch_isend_irecv`: blocking `send` on every rank at once deadlocks (all wait to send), and posting the send and receive as one batch lets NCCL group them into a single kernel and lets the transfer run on the comm stream while the local flash-attention block computes — the overlap that makes ring attention viable. Use double buffers so the tensor being sent is not overwritten, and `torch.cuda.stream` events (or `.wait()` before the swap) to keep the compute stream from reading a buffer still in flight.

</details>


### [06-19] One minute: what is data parallelism?  (difficulty 1; tags: data-parallel, distributed, rapid-fire)


<details><summary>Answer</summary>

**Data parallelism** replicates the full model on every GPU, gives each GPU a different slice of the batch, runs forward/backward independently, then **all-reduces the gradients** so every replica applies the same averaged update and stays in sync. It exists because it is the simplest way to scale throughput: no model code changes, and the communication (one gradient-sized all-reduce per step, overlapped with backward in DDP) is cheap relative to compute for large per-GPU batches. Limits: every GPU must hold the full model, gradients, and optimizer state, so it caps out at a few billion parameters in mixed-precision Adam (~16 bytes/param) — which is where ZeRO/FSDP take over. Trap: global batch size grows with GPU count, so you must rescale LR and warm up.

</details>


### [06-20] What is an all-reduce?  (difficulty 1; tags: all-reduce, nccl, collectives, rapid-fire)


<details><summary>Answer</summary>

An **all-reduce** is a collective where every rank starts with a tensor and ends with the element-wise reduction (usually sum) of all ranks' tensors — every rank gets the full result. It is the primitive under data-parallel gradient averaging. The efficient implementation is **ring all-reduce** = reduce-scatter (each rank ends up owning the reduced $1/N$ chunk) followed by all-gather (everyone collects the chunks): each rank sends and receives $2\frac{N-1}{N} \times$ tensor bytes, roughly 2× the tensor size regardless of $N$, so it is bandwidth-bound, not latency-bound, for large tensors. Follow-up: on 8 GPUs with NVLink, $2 \times$ (gradient bytes) at ~450 GB/s per direction tells you the floor for one DDP step's communication; DDP hides it by bucketing and overlapping with backward.

</details>


### [06-21] One minute: what is tensor parallelism?  (difficulty 1; tags: tensor-parallel, megatron, rapid-fire)


<details><summary>Answer</summary>

**Tensor parallelism** splits individual weight matrices across GPUs so one layer's matmul runs on several devices at once. Megatron's recipe for a transformer block: split the first MLP matrix by columns and the second by rows, so each GPU computes a slice and a single **all-reduce** recombines the output; attention splits by heads the same way. That gives two all-reduces per block in forward and two in backward, on activation-sized tensors every layer, so it needs very high bandwidth — which is why TP stays **inside a node on NVLink** (TP $\le 8$). It exists because a single layer of a 70B–400B model does not fit, or is too slow, on one GPU. Trap: TP reduces per-GPU memory for weights and activations but does not touch optimizer-state replication; combine with ZeRO/FSDP for that.

</details>


### [06-22] One minute: what is pipeline parallelism?  (difficulty 1; tags: pipeline-parallel, distributed, rapid-fire)


<details><summary>Answer</summary>

**Pipeline parallelism** cuts the model into stages by depth — layers 0–15 on GPU 0, 16–31 on GPU 1, etc. — and streams micro-batches through, passing only the activations at stage boundaries (point-to-point send/recv, small compared with TP's all-reduces, so it works across nodes). It exists to fit very deep models across many GPUs with low communication. The cost is the **bubble**: stages idle while the pipeline fills and drains, with fraction $\approx \frac{p-1}{m+p-1}$ for $p$ stages and $m$ micro-batches, so you need many micro-batches per step. 1F1B interleaves forward and backward to bound activation memory; interleaved/virtual stages shrink the bubble further. Trap: the layers must be balanced in compute and the last stage often carries the extra loss-head work.

</details>


### [06-23] One minute: what is ZeRO / FSDP?  (difficulty 1; tags: zero, fsdp, memory, rapid-fire)


<details><summary>Answer</summary>

**ZeRO** (DeepSpeed) and **FSDP** (PyTorch) are data parallelism without the replication: instead of every GPU holding a full copy of parameters, gradients, and optimizer state, each GPU owns $1/N$ of them. Stage 1 shards optimizer state, stage 2 also gradients, stage 3 (= FSDP full shard) also parameters. During forward and backward each layer's weights are **all-gathered** just before use and freed after, and gradients are **reduce-scattered** instead of all-reduced. Memory per GPU drops from ~16 bytes/param to ~$16/N$, so a 10B model fits on 8 GPUs with no model-parallel code. The cost is ~1.5× the communication of plain DP (an extra all-gather in backward) and that it is latency-sensitive across nodes. Trap: activations are not sharded — you still need checkpointing or sequence parallelism for long-token video models.

</details>


### [06-24] What is gradient checkpointing, briefly?  (difficulty 1; tags: activation-checkpointing, memory, rapid-fire)


<details><summary>Answer</summary>

**Gradient (activation) checkpointing** trades compute for memory: instead of storing every intermediate activation from the forward pass for backward, you keep only the inputs to selected blocks and **recompute** the rest during backward. Applied per transformer block, memory drops from $O(\text{layers} \times \text{per-layer activations})$ to $O(\text{layers} \times \text{block inputs})$ plus one block's worth of live activations, at the cost of roughly one extra forward pass (~33% more compute). It exists because for long-token models (video DiTs at 32k+ tokens) activations, not weights, dominate memory. Practical points: wrap it at the transformer-block level (`torch.utils.checkpoint`); **selective** recomputation recomputes only the cheap, memory-heavy parts (attention softmax, norms, GELU) and keeps matmul outputs, cutting the overhead to a few percent. Trap: RNG state must be saved for dropout to recompute identically.

</details>


### [06-25] One minute: what is mixed-precision training?  (difficulty 1; tags: mixed-precision, bf16, rapid-fire)


<details><summary>Answer</summary>

**Mixed precision** runs the forward and backward matmuls in 16-bit (bf16 today, fp16 previously) for tensor-core speed and halved activation memory, while keeping an **fp32 master copy** of the weights and fp32 optimizer state so small updates ($\text{lr} \times \text{grad} \ll \text{weight}$) are not rounded away. bf16 has fp32's exponent range with 8 bits of mantissa, so it needs no loss scaling; fp16 has more mantissa but a tiny range, so gradients underflow and you need dynamic loss scaling. Per-parameter memory in mixed Adam is ~16 bytes: $2\,(\text{bf16 weight}) + 2\,(\text{bf16 grad}) + 4 + 4 + 4\,(\text{fp32 master}, m, v)$. Trap: reductions and softmax should still accumulate in fp32, and the loss/norms stay fp32 — `autocast` handles the op list. Follow-up: fp8 on Hopper adds per-tensor scaling and delayed scaling to keep the range usable.

</details>


### [06-26] What are a process group, rank, and world size?  (difficulty 1; tags: torch-distributed, api, rapid-fire)


<details><summary>Answer</summary>

In `torch.distributed`, **world size** is the total number of processes in the job (typically one per GPU), **rank** is a process's unique integer id in $0 \dots \text{world\_size} - 1$, and **local rank** is its index within its node (used to pick the CUDA device). A **process group** is a named subset of ranks that a collective runs over; the default group is all ranks, and you build sub-groups for each parallelism axis — e.g. with 512 GPUs, TP=8, PP=4, DP=16, every rank belongs to one TP group of 8, one PP group of 4, and one DP group of 16, and an all-reduce on the DP group only involves those 16 ranks. `torchrun` sets the env vars (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR) and `init_process_group("nccl")` does the rendezvous. Trap: every rank must call `new_group` for every group, even ones it is not in.

</details>


### [06-27] What is NCCL?  (difficulty 1; tags: nccl, collectives, rapid-fire)


<details><summary>Answer</summary>

**NCCL** (NVIDIA Collective Communications Library) is the GPU-native library that implements the collectives — all-reduce, all-gather, reduce-scatter, broadcast, send/recv — directly between GPU memories over NVLink, PCIe, and InfiniBand/RoCE, without staging through the host. It is the backend PyTorch uses for `init_process_group("nccl")`, and DDP, FSDP, and Megatron all sit on top of it. It exists because MPI's CPU-centric collectives could not saturate GPU interconnects; NCCL builds topology-aware ring and tree algorithms and runs them as CUDA kernels on a communication stream so they overlap with compute. Practical knowledge: NCCL_DEBUG=INFO shows the topology it picked, collectives are blocking on the GPU stream (hangs when one rank misses a call), and NCCL_ALGO/PROTO and NVLink SHARP affect large all-reduce bandwidth.

</details>


### [06-28] What is MFU?  (difficulty 1; tags: mfu, throughput, rapid-fire)


<details><summary>Answer</summary>

**MFU (Model FLOPs Utilization)** is the fraction of the hardware's peak throughput your training run actually converts into useful model FLOPs: $\text{MFU} = (\text{model FLOPs per second achieved}) / (\text{peak FLOP/s} \times \text{GPUs})$. Model FLOPs per token $\approx 6 \times \text{parameters}$ (2 forward, 4 backward) plus the attention term $12 \times \text{layers} \times \text{hidden} \times \text{sequence}$; multiply by tokens/s. It exists because "GPU utilisation 100%" says nothing about whether tensor cores are busy; MFU is the honest efficiency metric across systems. Good numbers: ~40–50% on H100 for dense LLMs; video DiTs with long sequences and heavy attention often land lower. Trap: **HFU** counts recomputation from activation checkpointing as real work, so it is higher than MFU — quote MFU when comparing, and remember bf16 peak (~990 TFLOP/s dense for H100) is the denominator, not the sparse marketing number.

</details>


## Long Context & Causal Video Generation


### [07-01] Why can't a standard bidirectional video diffusion model like WAN 2.1 stream frames, and what has to change architecturally to make it causal / autoregressive?  (difficulty 1; tags: causal-video, autoregressive, kv-cache)

- Follow-up: What is the difference between frame-by-frame and chunk-wise causal generation?


<details><summary>Answer</summary>

A bidirectional video DiT denoises **all frames jointly**: every token attends to every other token in space and time, and every denoising step refines the whole clip. So frame 1 is not finished until frame 80 is finished — there is no point at which you can emit frame 1 and keep going. Latency is the full clip's 50 steps × full-clip attention, and the length is fixed at training time.

To stream, you need three changes:
- **Causal attention over time**: frame (or chunk) $k$ only attends to frames $\le k$. Then earlier frames' KV are fixed once generated and can be **cached**, exactly like an LLM decoder.
- **Per-frame (or per-chunk) denoising**: fully denoise chunk $k$ conditioned on the clean cache of chunks $< k$, then append its KV and move on. This is the CausVid / Self-Forcing / MAGI-1 recipe.
- **Few-step denoising** (DMD or consistency distillation), because you now pay the sampler cost once per chunk, so 50 steps × chunks is too slow for real time.

Frame-by-frame gives lowest latency but the weakest intra-chunk coherence and the most AR steps (more error accumulation); **chunk-wise** (e.g. 3–5 latent frames = 12–20 pixel frames) lets the chunk be denoised bidirectionally inside itself while remaining causal across chunks, which is the sweet spot most systems use. The common follow-up is "what does the cache hold?" — the K/V projections of clean context tokens at every layer, which is what makes memory grow linearly with rollout length.

</details>


### [07-02] Explain Diffusion Forcing. How does it unify autoregressive next-token prediction and full-sequence diffusion, and what does it buy you at sampling time?  (difficulty 2; tags: diffusion-forcing, causal-video, training)

- Follow-up: Why does it need causal attention rather than full attention?


<details><summary>Answer</summary>

**Diffusion Forcing** (Chen et al. 2024) trains a causal sequence model where **each frame gets its own independent noise level** $t_k \sim U[0,1]$, instead of one shared $t$ for the whole clip. The model learns to denoise frame $k$ given noisy frame $k$ and the (arbitrarily noisy) history of frames $< k$.

Why that unifies the two paradigms:
- If $t_{<k} = 0$ (clean history) and $t_k = 1$, you recover **teacher-forced next-frame prediction** — the AR objective.
- If all $t_k$ are equal, you recover **full-sequence diffusion**.
- Every mixture in between is also trained, so at test time you can pick any **noise schedule over the time axis**: e.g. a "staircase" where near frames are almost clean and far frames are mostly noise, which is a principled way to do long-horizon rollout with uncertainty growing with horizon.

What it buys you at sampling time:
- **Stable AR rollout** — because the model was trained with noisy history, it is robust to its own imperfect past frames (partial fix for exposure bias).
- **Flexible horizon / planning**: you can denoise a future chunk under a goal constraint and use it as a plan, or do guidance on some frames but not others.
- **Variable-length generation** without retraining.

Causal attention is needed so that frame $k$'s denoising doesn't leak information from the (independently noised) future frames — otherwise the per-frame noise levels aren't a well-defined conditional problem and you can't cache. The trap to mention: the independent-noise trick makes the training distribution much larger (every $t$-combination), so it costs sample efficiency; practical systems restrict to a few patterns (clean-history, staircase, uniform).

</details>


### [07-03] What is exposure bias in autoregressive video generation, why does it hurt more for video than for text, and what are the main remedies?  (difficulty 2; tags: exposure-bias, causal-video, error-accumulation)

- Follow-up: Which of these remedies does Self-Forcing actually implement?


<details><summary>Answer</summary>

**Exposure bias** is the train/test mismatch: at training the model conditions on ground-truth past frames (teacher forcing); at inference it conditions on its **own** generated frames, which carry small errors. Those errors compound over steps, and for video the compounding is brutal because the conditioning is high-dimensional and continuous: a slight color shift or blur in frame $k$ is copied and amplified into frame $k+1$, so after a few hundred frames you get saturation, drift, frozen motion, or texture collapse — the classic "AR video degrades after 10 s".

Text tolerates it better because tokens are discrete (small logit errors don't accumulate as continuous drift) and language has strong local re-anchoring.

Remedies, roughly by strength:
- **Noise augmentation of context** (Diffusion Forcing, Cosmos-style): add noise to conditioning frames during training so the model sees "imperfect history"; simple and works, but only matches errors that look like Gaussian noise.
- **Rollout / self-forcing training**: actually **generate the history with the model itself** (few-step sampler, KV cache) during training and train on the resulting sequence — the training distribution equals the test distribution. Self-Forcing does this with a **DMD-style distribution-matching loss** against a bidirectional teacher so you don't need ground-truth continuations of your own rollouts.
- **Scheduled sampling / mixed teacher-student history** as a cheaper approximation.
- **Discrete-ish anchoring**: periodically re-condition on a clean keyframe or a compressed memory token so drift can't run away.

Self-Forcing implements rollout training + DMD distillation + KV-cached causal student, and reports that even a few-frame rollout horizon in training removes most drift. The follow-up trap: rollout training needs backprop through the sampler, so you truncate gradients to the last chunk or two and use a small number of denoising steps (4) for memory.

If pushed on the distillation itself: DMD minimizes $\mathrm{KL}(p_{\text{student}} \,\|\, p_{\text{teacher}})$ with gradient $\propto (s_{\text{fake}} - s_{\text{real}})$, where $s_{\text{real}}$ is the frozen bidirectional teacher's score and $s_{\text{fake}}$ is a copy fine-tuned online on the student's samples; applying it to the student's *rolled-out* clip is what makes the teacher judge whole-video coherence while the student stays causal. Gradients are truncated to the last chunk or two for memory.

</details>


### [07-04] Do the memory arithmetic: a causal video DiT with 30 layers, hidden size 3072, 3 latent frames per chunk, 1560 tokens per latent frame, bf16. How much KV cache per chunk, and how long a rollout fits in 80 GB alongside the model?  (difficulty 1; tags: kv-cache, memory, systems)

- Follow-up: How does GQA/MQA change the picture?


<details><summary>Answer</summary>

KV cache per token per layer $= 2\,(\text{K and V}) \times d_{\text{model}} \times 2\,\text{bytes (bf16)} = 2 \times 3072 \times 2 =$ **12,288 B ≈ 12 KB**.
Per token over 30 layers: $12\,\text{KB} \times 30 =$ **360 KB**.
Tokens per chunk: $3\ \text{frames} \times 1560 = 4680$ tokens → $4680 \times 360\,\text{KB} \approx$ **1.68 GB per chunk**.

If the model is ~14B params in bf16 that's ~28 GB, leaving ~50 GB. Without activations you'd fit ~30 chunks ≈ 90 latent frames ≈ 360 pixel frames (at 4× temporal VAE compression) ≈ **15 s at 24 fps**. Realistically activations and the fake-score model (if training) take half of that, so a **naive cache is good for 5–10 s of video** — which is exactly why long rollouts need a window or eviction.

Follow-up: with **GQA** (say 8 KV heads instead of 24), the cache drops 3× because only KV heads are stored; MQA drops it by $n_{\text{heads}}\times$. Most video DiTs (WAN, Cosmos) still use full MHA, so this is a free 3–4× on the table. Also mention that attention FLOPs scale with cache length: per new chunk you do $O(n_{\text{new}} \times n_{\text{cache}})$ attention, so a 30-chunk cache makes the last chunk 30× more expensive than the first — time per chunk grows linearly, total cost quadratically in rollout length.

</details>


### [07-05] Explain sliding-window attention with attention sinks (StreamingLLM) and heavy-hitter eviction (H2O). Why does naive sliding-window fail, and what carries over to video?  (difficulty 2; tags: kv-cache, streaming, attention)


<details><summary>Answer</summary>

**Naive sliding window** keeps only the last $W$ tokens' KV. It fails catastrophically the moment the very first tokens leave the window — perplexity explodes. StreamingLLM's finding: softmax needs somewhere to dump attention mass when nothing is relevant, and models learn to use the **first few tokens as "attention sinks"** (huge attention scores, near-zero information). Evict them and every attention distribution is mis-normalized. Fix: **keep the first ~4 tokens permanently + a sliding window**, and assign positions relative to the cache (not absolute) so RoPE doesn't extrapolate.

**H2O (Heavy-Hitter Oracle)** is content-aware: track the accumulated attention each cached token has received; a small set of "heavy hitters" gets most of the mass (power-law). Keep the top-$k$ heavy hitters + a recent window, evict the rest. It's a greedy, per-layer / per-head budget policy and gives ~5–10× cache reduction on LLMs with little loss.

**What carries over to video:**
- Sinks exist too — usually the first chunk's tokens and any conditioning (text) tokens. Keeping the **first chunk as a permanent anchor** also helps identity/scene persistence, so the sink and the semantic reason coincide.
- Heavy-hitter statistics are noisier per token because video has 1000s of tokens per frame and attention is spread spatially; you'd aggregate per patch across heads or over a chunk.
- Video has structure text doesn't: **static background tokens are redundant across frames** (predictable), so eviction should be by *information*, not just by attention — which motivates surprise-based criteria.
- A cheap baseline that's surprisingly hard to beat for video: keep chunk 1 + last $N$ chunks. Always report it.

</details>


### [07-06] You've built surprise-gated KV eviction for a world-action model. Describe the criterion, why you intersect it with action-expert attention, and then critique it — where does it fail?  (difficulty 3; tags: kv-cache, world-models, project, eviction)

- Follow-up: Why running quantiles instead of z-score thresholds?


<details><summary>Answer</summary>

**Criterion.** During an AR rollout the world model *imagines* the next latent $\hat{x}_{k+1}$ before it sees the ground-truth observation $x_{k+1}$. The per-token **surprise** is $\|x_{k+1} - \hat{x}_{k+1}\|$ in a **standardized VAE latent space** (per-channel mean/std so no channel dominates). Tokens the model predicted well carry no new information — the model can regenerate them from what it already has — so they're evicted; surprising tokens are kept.

**Why intersect with action-expert attention.** Surprise alone keeps tokens that are unpredictable but irrelevant (flickering background, shadows). The action expert's cross-attention over video tokens tells you which tokens **the policy actually reads**. Keeping only $\text{surprising} \cap \text{attended}$ tokens removes both redundant and irrelevant tokens: ~44–54% KV reduction with **task success unchanged** on LIBERO and only a 0.3–1.7 dB PSNR hit on imagined video.

**Critique (what an interviewer wants to hear me say):**
- **It needs ground truth.** Surprise requires the real next observation; in pure generation (no sensor) there's no surprise signal. It's a *world-action-model / robot* trick, not a video-generation trick; a proxy is denoiser residual or teacher-vs-student disagreement.
- **Attention is a lagging signal.** A token unattended now may be needed in 50 steps (object hidden behind the gripper). Eviction is irreversible — I'd want a small "cold" cache or compressed summary rather than deletion.
- **Confounded evaluation**: a robot that stalls produces a static scene that's trivially predictable → high PSNR and low surprise. I had to report on successful episodes and paired seeds.
- **Per-step overhead** of computing the criterion is non-trivial in a real-time loop; the intersection also needs the action expert's attention maps, which some fused-attention kernels don't expose.
- **Budget stability**: it's not a fixed-size cache; a novel scene floods it. A hard cap with heavy-hitter fallback is needed.

**Running quantiles vs $z$-score.** Surprise is heavy-tailed and non-stationary (scene changes shift the whole distribution). A $z$-score threshold $\mu + c\sigma$ assumes Gaussianity — the mean and std are dragged by outliers, so the threshold is either too loose in calm scenes or too tight in busy ones. A **running quantile** (e.g. keep top 40% by a streaming $P^2$ / reservoir estimate) is distribution-free and self-calibrates to "keep a fixed fraction," which is what you actually want for a memory budget. Median-based MAD is in between but still needs a scale assumption.

</details>


### [07-07] Compare ring attention, blockwise/sparse attention, sliding-window + global tokens, linear attention / SSMs (Mamba), and TTT layers for long video. What does each trade?  (difficulty 2; tags: long-context, attention, ssm, ttt)


<details><summary>Answer</summary>

Framing: full attention is $O(N^2)$ in compute and $O(N)$ in KV memory; every method attacks one of those, and each pays in **either exactness, expressivity, or parallelism**.

- **Ring attention**: exact full attention, distributed — shard the sequence across GPUs, rotate K/V blocks around a ring while overlapping communication with the local block compute. Trades **communication + hardware** for exactness; nothing is approximated. It's what you use to *train* on minute-long clips at all. Cost per device stays $O(N^2/P)$; comm volume per layer $\approx 2 \times N \times d \times \text{bytes}$.
- **Blockwise / sparse attention** (local blocks + strided/dilated, or learned sparse patterns like NSA/MoBA): reduces FLOPs to $O(NB)$. Trades **exactness for speed**; quality depends on whether the pattern matches the true dependency structure — good for video because dependencies are mostly local in space-time.
- **Sliding window + global tokens** (Longformer-style; for video: local temporal window + a few "memory"/register tokens): $O(NW)$. Trades **long-range recall for bounded cost**; the global tokens are a bottleneck everything long-range must pass through.
- **Linear attention / SSMs (Mamba, RWKV)**: $O(N)$ compute, **$O(1)$ state** — the state is a fixed-size matrix, so memory doesn't grow with rollout at all. Trades **exact retrieval for compression**: a fixed state cannot losslessly remember 10,000 frames; they're weak at "copy the object from 30 s ago" tasks. Great for a streaming backbone with hybrid attention layers for recall.
- **TTT layers**: the hidden state is a small neural net's weights, updated by gradient descent on a self-supervised loss over the incoming tokens; expressivity between SSM and attention. Trades **compute per token (an inner optimization) and implementation complexity** for a state that's more expressive than a linear RNN. The TTT-Video work showed minute-long Tom & Jerry episodes from a 5B backbone by inserting TTT layers into a local-attention DiT — the usual follow-up is "how is it different from fine-tuning at test time?": same principle, but scoped to one layer's state and with the update baked into training so the outer model learns to exploit it.

Rapid-fire answer: *train* with ring attention; *stream* with a hybrid of local attention + compressive state (SSM/TTT) + a few global anchor tokens.

</details>


### [07-08] Your model was trained on 81-frame clips with RoPE. A user asks for 300 frames at inference. Why does it break and what do RoPE base scaling, NTK-aware scaling, and YaRN each do about it?  (difficulty 2; tags: rope, positional-encoding, length-generalization)

- Follow-up: Is the same fix right for the spatial axes at super-native resolution?


<details><summary>Answer</summary>

**Why it breaks.** RoPE rotates each 2-D pair of $q/k$ dimensions by angle $p\theta_i$ with $\theta_i = \text{base}^{-2i/d}$. The low-frequency dimensions (large $i$) complete a small fraction of a period over 81 frames, so the model has only ever seen a narrow arc of those rotations. At position 300 those dimensions land on angles never seen in training → attention logits are out-of-distribution and you get repetition, motion freezing, or noise.

- **Position interpolation (linear)**: scale positions by $81/300$ so the maximum stays in range. Keeps low-freq dims in-distribution but **compresses the high-frequency dims** too, so neighboring frames become hard to distinguish — blurrier local motion.
- **NTK-aware base scaling**: instead of scaling positions, raise the base (e.g. $10{,}000 \to 10{,}000 \cdot s^{d/(d-2)}$). This scales rotation speed **non-uniformly**: high-freq dims barely change (local structure preserved), low-freq dims are compressed to stay in range. Training-free and usually the best zero-shot choice.
- **YaRN**: NTK-by-parts — leave dims with many periods in training untouched, interpolate dims with $< 1$ period, ramp in between; plus a **temperature on attention logits** ($\propto 1/\sqrt{\text{scale}}$ style) to counteract the entropy increase from a longer softmax. Best when you also do a short fine-tune at the target length.

The honest answer for video: any of these gives you maybe 2–3× length before quality falls off; beyond that you need training at length (ring attention) or a causal model where the *cache* positions are relative and never exceed the window.

**Spatial follow-up (from SPEED):** at 2× native resolution the same extrapolation problem appears on the spatial RoPE axes; NTK scaling helps but the sharper fix is to **spend most denoising steps at native resolution** and only expand late, which is why staged resolution beat full-res low-pass at 960p in my work.

</details>


### [07-09] How would you give a long video generator "memory" of scene state without keeping every frame? Compare latent memory tokens, recurrent state, and compressive transformers.  (difficulty 3; tags: memory, world-state, compression, long-context)


<details><summary>Answer</summary>

The problem: exact KV grows linearly and attention over it grows quadratically, but the *information* you need to keep — scene layout, object identities, camera pose, what's behind you — is roughly constant. So compress the history into a bounded **world-state**.

- **Latent memory / register tokens**: append $M$ learned tokens that attend to the current chunk and are carried to the next (Memorizing-Transformer / Perceiver style). Cheap, transformer-native, differentiable; the model learns what to write. Weakness: fixed $M$ means saturation, and without a write/erase rule old content decays uncontrollably; plus there's no explicit geometry, so "look back at the room" is learned, not guaranteed.
- **Recurrent state (SSM / GRU / TTT weights)**: state updated every chunk; $O(1)$ memory, streams forever. Weakness: lossy in a way you can't inspect; recall of specific past frames is poor. Good for *dynamics* (velocity, momentum), bad for *episodic* recall.
- **Compressive transformer**: keep a recent exact window, and when tokens age out, **compress them** (pool, conv, or a learned compressor) into a smaller set that stays attendable — a two-tier cache. Gives graceful degradation: recent = exact, old = coarse. Training needs an auxiliary reconstruction loss so the compressor keeps useful content. This is closest to what I'd build: exact last $N$ chunks + compressed older chunks + a permanent first-chunk anchor.

Two extra options worth naming: **explicit 3D memory** (a point map / Gaussian / voxel state updated from generated frames — WorldMem / Genie 3-style memory) which makes revisiting a location consistent by construction, and **surprise-based selective retention** (keep only tokens the model couldn't predict — my KV work), which is compressive in a content-adaptive way.

The design question an interviewer will push on: memory vs. **consistency vs. controllability**. Recurrent state gives smooth dynamics but forgets; explicit 3D memory gives revisitation consistency but assumes a static-ish world. A hybrid is the current frontier.

</details>


### [07-10] How do you evaluate a minute-long generated video? FVD on a 16-frame window doesn't tell you much.  (difficulty 2; tags: evaluation, fvd, long-video, drift)


<details><summary>Answer</summary>

Right — FVD on 16 frames measures per-clip realism at a single scale and is blind to the two failure modes of long video: **drift** (slow degradation) and **inconsistency** (subject/scene changes). I'd use a battery:

- **Windowed FVD over time**: FVD of window $k$ vs. real windows, plotted against $k$. A rising curve = drift; report the slope, not just the mean. Also FVD between the first and last window (self-consistency).
- **Subject persistence**: track the main subject (DINO / CLIP features or an identity embedding) and report cosine similarity to the first-window embedding over time; for humans, a face-ID model. Also object count / position consistency via a detector.
- **Scene consistency on revisits**: force a camera loop (turn 360°) and compare the returned view to the original — this is the honest test for memory.
- **Motion statistics drift**: optical-flow magnitude over time — most drifted models either freeze (flow $\to 0$) or explode. Also color histogram / saturation over time for the classic "AR videos turn orange" failure.
- **VBench-style decomposed scores** (subject consistency, background consistency, temporal flicker, imaging quality) reported per minute, not per clip.
- **Human preference** on pairs, at fixed timestamps (0 s, 30 s, 60 s) — still the deciding evidence.

Traps: (1) FVD is I3D-based and saturates/prefers static video; pair it with a motion score. (2) Always evaluate the *same* number of frames and windows for every model — window count silently changes FVD. (3) A model that freezes has excellent "consistency" — every consistency metric must be paired with a motion or task metric.

</details>


### [07-11] In your KV-compression evals you compared imagined-video PSNR between methods. Why must the comparison be paired-seed, and what confounds did you hit?  (difficulty 3; tags: evaluation, determinism, project, world-models)


<details><summary>Answer</summary>

Because a **stochastic rollout's PSNR variance across seeds is larger than the effect size**. With unpaired seeds, method A on seed 1 vs method B on seed 2 differ in initial noise, sampled actions, and physics outcomes; a 0.5 dB difference is pure noise. With **paired seeds** — same initial noise, same environment seed, same action sampling seed — the only difference is the KV policy, and you can do a paired t-test on per-episode deltas, which shrinks the CI by the between-seed variance.

Confounds I actually hit:
- **Flash-attention non-determinism**: even with identical seeds, the backward pass (and some forward reductions, atomics in split-K) is non-bitwise-deterministic, so trajectories diverge after a few steps and "paired" isn't exactly paired. Fixes: `torch.use_deterministic_algorithms`, deterministic attention kernels or SDPA math backend for eval, fixed batch composition, and — more honestly — report the *inherent* seed-to-seed spread of the same method as a noise floor, and only claim differences above it.
- **Failed episode = static scene = high PSNR**: when the policy stalls, the world barely changes, so imagined frames are trivially accurate. A KV policy that *hurts* the policy can score *better* PSNR. So PSNR must be reported **on the same episode outcome stratum** (success-only, or with success rate as the primary metric and PSNR secondary), and paired by episode.
- **PSNR is a bad video metric anyway** — it rewards blur. I added LPIPS / DINO distance and, most importantly, the downstream task success as the primary metric.
- **Ordering effects**: the running-quantile threshold is stateful, so the first episodes in a run behave differently; warm-up episodes are excluded.

The general principle: for stochastic generators, the comparison unit is the (seed, method) pair, and every "quality" metric needs a task metric next to it that exposes the "do nothing" degenerate solution.

</details>


### [07-12] Design question: build a minute-long, consistent, prompt-following video generator that runs at interactive speed on 8 GPUs. Walk me through the design and the tradeoffs.  (difficulty 3; tags: design, long-video, causal-video, systems)


<details><summary>Answer</summary>

I'd structure it as **a bidirectional teacher + a causal, few-step, memory-augmented student**, and be explicit about what I'm trading at each layer.

**1. Backbone and data.** Start from a strong bidirectional video DiT (WAN-class, ~14B) as the teacher — that's where quality and prompt-following come from. Curate long clips with scene-cut detection and per-shot captions; long-video data is the real bottleneck.

**2. Causal student.** Convert to chunk-wise causal attention (3–4 latent frames per chunk, bidirectional within chunk), KV-cached. Distill with **DMD against the teacher on self-generated rollouts** (Self-Forcing) so the student is trained on its own errors; 4 denoising steps per chunk. Tradeoff: some loss of diversity from DMD and a small quality gap vs. the teacher, in exchange for ~real-time.

**3. Memory.** Three tiers: permanent first-chunk anchor (identity + attention sink), exact KV for the last ~8 chunks, and **compressed older context** (learned pooling or selected surprising tokens) capped at a fixed budget so cost per chunk is constant. Positions are cache-relative so RoPE never extrapolates. Tradeoff: exact recall of the far past is lost; if the product needs "walk around and come back," add an explicit 3D memory.

**4. Speed on 8 GPUs.** Tensor/sequence parallel across the 8 GPUs per chunk, or — better for latency — pipeline chunks so the VAE decode of chunk $k$ overlaps denoising of chunk $k+1$. Add spatial efficiency: SPEED-style spectral progressive resolution (early denoising steps at low res) is training-free and gives ~2× on the few steps that dominate; foveated/mixed-resolution tokens if the app has a natural attention center. Target ≈ 1 chunk (0.5 s of video) in < 0.5 s.

**5. Prompt-following over time.** Support per-chunk prompt updates (the text KV is re-encoded per chunk), and use a lower CFG in the student than the teacher to avoid saturation drift.

**6. Evaluation** built in from day one: windowed FVD slope, subject-persistence curve, motion-magnitude drift, revisit consistency, paired-seed comparisons for every ablation.

**Where it fails:** compounding still happens over minutes (rollout training covers ~10 chunks, not 120), so I'd add periodic re-anchoring — re-denoise a "keyframe" bidirectionally every $N$ seconds with the teacher and re-seed the cache. That's the design lever I'd be least sure about and would ablate first.

</details>


### [07-13] What is autoregressive video generation?  (difficulty 1; tags: autoregressive, causal-video, rapid-fire)


<details><summary>Answer</summary>

**Generating a video in temporal order, each new frame or chunk conditioned only on what has already been generated**: $p(x) = \prod_k p(x_k \mid x_{<k})$. Contrast with bidirectional diffusion, which denoises the whole clip jointly and cannot emit frame 1 before frame 80 is done. It exists because it enables streaming, variable length, and interactive conditioning — you can inject an action or a prompt change at every step, which is what a world model needs. Each AR step can itself be a diffusion process (CausVid, Self-Forcing, MAGI-1), so "autoregressive" does not imply discrete tokens. The trap: errors compound over steps (exposure bias), so quality drifts after a few hundred frames unless you train on your own rollouts.

</details>


### [07-14] What is a KV cache in the video-diffusion setting?  (difficulty 1; tags: kv-cache, causal-video, rapid-fire)


<details><summary>Answer</summary>

The **stored key and value projections, at every layer, of the already-generated context tokens**, so the next chunk can attend to its history without recomputing it — the LLM idea applied to video latents. Two differences from text: each latent frame is ~1500 tokens, so the cache grows by gigabytes per second of video, and the cache is filled from **clean** frames, i.e. after a chunk's denoising finishes, not from noisy intermediates. Trap: the cache is built at a single noise level ($t = 0$), so Self-Forcing runs an extra forward pass on the finished clean chunk to populate it — and cache length, not model size, is what bounds rollout length in memory.

</details>


### [07-15] Diffusion forcing in one sentence.  (difficulty 1; tags: diffusion-forcing, training, rapid-fire)


<details><summary>Answer</summary>

**Train a causal sequence model where every frame gets its own independent noise level**, so the model learns to denoise frame $k$ given an arbitrarily noisy history — which makes teacher-forced next-frame prediction (clean past, fully noisy current) and full-sequence diffusion (one shared $t$) two corners of the same training distribution. It exists to make autoregressive rollout robust: because the model has seen noisy pasts in training, feeding it its own imperfect frames at test time is in-distribution. It also lets you choose any noise schedule along the time axis at sampling time (a staircase where far frames are noisier). The follow-up: it needs causal attention, otherwise a frame's denoising leaks information from independently noised future frames.

</details>


### [07-16] What is exposure bias?  (difficulty 1; tags: exposure-bias, error-accumulation, rapid-fire)


<details><summary>Answer</summary>

**The train/test mismatch in autoregressive models: training conditions on ground-truth history (teacher forcing), inference conditions on the model's own outputs.** Small errors in generated frame $k$ enter the conditioning for $k+1$, get copied and amplified, and after enough steps the video saturates, blurs, or freezes. It hurts video more than text because the conditioning is continuous and high-dimensional — a slight color shift compounds, whereas a discrete token error is either corrected or not. Remedies: noise-augment the history in training (diffusion forcing), or **train on your own rollouts** (Self-Forcing, with a DMD loss against a bidirectional teacher) so the training distribution equals the test distribution. Trap: scheduled sampling is the cheap approximation, but it biases toward the mode.

</details>


### [07-17] What is sliding-window attention?  (difficulty 1; tags: attention, streaming, kv-cache, rapid-fire)


<details><summary>Answer</summary>

**Each query attends only to the most recent $W$ keys, so compute per token is $O(W)$ instead of $O(N)$ and the KV cache is bounded at $W$ entries** — the simplest way to stream indefinitely with constant memory. Stacked over $L$ layers the receptive field is $LW$, so information can still propagate further than the window. Why it is not enough on its own: the moment the very first tokens leave the window, attention distributions collapse (the sink problem), so practical systems keep a few permanent tokens plus the window, and use cache-relative positions so RoPE never sees positions beyond $W$. Trap: exact long-range recall is gone — anything older than $W$ is forgotten unless you add memory tokens.

</details>


### [07-18] What is an attention sink?  (difficulty 1; tags: attention-sink, streaming, rapid-fire)


<details><summary>Answer</summary>

**A token — usually the first one or two in the sequence — that receives a large share of attention mass regardless of content, acting as a "no-op" target the softmax can dump probability on when nothing is relevant.** It exists because softmax must sum to one: with no explicit null option, the model learns to use an always-visible, low-information token as the null. Consequence (StreamingLLM): naive sliding-window eviction removes the sink, every head's attention becomes mis-normalized, and perplexity explodes. Fix: keep the first ~4 tokens permanently alongside the window. For video the sink is typically the first chunk and the text tokens, and keeping the first chunk doubles as an identity anchor.

</details>


### [07-19] Ring attention in one sentence.  (difficulty 2; tags: ring-attention, sequence-parallel, long-context, rapid-fire)


<details><summary>Answer</summary>

**Shard a long sequence across $P$ devices, and compute exact full attention by passing each device's K/V blocks around a ring while overlapping the communication with the local attention compute** — each device ends up having seen every K/V block. It exists because a minute-long video is millions of tokens, which does not fit one GPU's activation memory; ring attention makes context length scale linearly with device count with **no approximation**. Cost: per layer, each device sends and receives roughly $2Nd$ bytes of K/V, so it only pays off when compute per block exceeds transfer time. Trap: it is a training/prefill tool; it does not help streaming inference, where the bottleneck is cache length.

</details>


### [07-20] What is RoPE extrapolation, and why does length generalization fail?  (difficulty 2; tags: rope, length-generalization, positional-encoding, rapid-fire)


<details><summary>Answer</summary>

**RoPE encodes position by rotating each 2-D pair of $q/k$ dimensions by angle $p\theta_i$, $\theta_i = \text{base}^{-2i/d}$; extrapolation is asking the model to attend at positions $p$ larger than anything seen in training.** It fails because the low-frequency dimensions rotate only a fraction of a period over the training length, so the model has only seen a narrow arc of those angles; at 4× the training length, those dimensions land on unseen angles, the dot products become out-of-distribution, and you get repetition, frozen motion, or noise. Fixes: position interpolation, NTK-aware base scaling, YaRN. Trap: the same effect appears on the spatial axes when you generate at super-native resolution, which is why SPEED spends most steps at native resolution.

</details>


## World Models, Physical AI & Robotics


### [08-01] What is a world model? Contrast Ha & Schmidhuber's original, Dreamer's latent dynamics, and video-prediction world models like Cosmos, Genie, and GAIA.  (difficulty 1; tags: world-models, dreamer, video-prediction)

- Follow-up: Which of these can you plan in, and which can you only sample from?


<details><summary>Answer</summary>

A **world model** is a learned model of the environment's dynamics: $p(o_{t+1} \mid o_{\le t}, a_t)$ — given history and an action, predict what happens next — used either to **plan/learn a policy in imagination** or as a **simulator / data engine**.

- **Ha & Schmidhuber (2018)**: VAE compresses frames to $z$; an MDN-RNN predicts $z_{t+1}$ from $(z_t, a_t, h_t)$; a tiny linear controller is trained *inside the dream* with evolution strategies. Established the V-M-C decomposition and "train in imagination."
- **Dreamer (v1–v3)**: RSSM latent dynamics with a deterministic GRU state $h_t$ plus a stochastic $z_t$ (categorical in v2/v3), trained with reconstruction + KL; the policy and value are trained by **backprop through imagined latent rollouts** (actor-critic in latent space). Never decodes pixels at policy-training time; compact latents make long imagination cheap. Dreamer v3 solved Minecraft diamonds with fixed hyperparameters.
- **Video-prediction world models** (Cosmos, Genie, GAIA-1/2, Sora as "world simulator"): a large video diffusion / autoregressive transformer that predicts *pixels/latent-video* conditioned on actions (Genie: latent actions learned unsupervised from video; GAIA: ego-vehicle actions + text; Cosmos: diffusion & AR variants, camera/robot-action conditioned post-training). They're pretrained on internet-scale video so they carry visual and physical priors, and their output is human-viewable — useful for synthetic data, policy evaluation, and as a pretrained backbone. They're heavy (billions of params, seconds per rollout) so planning inside them is expensive.

Follow-up: Dreamer-style latent models are designed for planning (cheap rollouts, gradients through dynamics). Video-prediction models are mostly sampled from (data generation, evaluation) or used as a **frozen representation**; planning in them is emerging (MPC with a few candidate rollouts) but cost-limited. The trend is to close the gap: video-pretrained models with a compact action-aware latent for control.

</details>


### [08-02] Describe a world-action model that jointly predicts next video latents and actions. How does the action expert read the video context, and what's the difference between conditioning on imagined vs. ground-truth latents?  (difficulty 2; tags: world-action-models, vla, world-models, project)

- Follow-up: Why not just predict actions from the current frame?


<details><summary>Answer</summary>

A **world-action model** (lingbot-VA style, also Unified-Video-Action / WorldVLA) is one autoregressive transformer that at each step emits **both** the next observation latent $\hat{x}_{k+1}$ and the action chunk $a_k$, sharing a backbone. Two heads: a **video (dynamics) branch** — a diffusion or flow head that denoises the next video latent conditioned on the KV cache of past latents and actions — and an **action expert**, typically a smaller flow-matching head that **cross-attends (or attends via joint attention with its own token stream) to the video tokens** in the shared cache, including the just-imagined future latent.

Why joint prediction helps: the video objective forces the backbone to learn physics and object permanence from video alone (which is abundant), and the action head reads an explicit "what will happen" representation instead of inferring it from a single frame. It also gives you a **free evaluator**: imagined frames can be compared to what actually happened.

**Imagined vs. ground-truth latents.** In closed loop, the real camera gives you $x_{k+1}$ after acting, so you can insert the GT latent into the cache (**teacher-forced at test time**). Alternatively you condition the next step on the *imagined* latent. Tradeoffs:
- GT latents: accurate, no drift, but the action expert must handle the mismatch between the imagined latent it planned on and the GT it now sees; also requires a sensor at that frame rate.
- Imagined latents: needed when the world model runs ahead of the sensor (planning multiple steps, low-latency), but errors compound exactly like AR video.
- Hybrid (what I used): GT replaces imagined in the cache, and the **difference** $\|x - \hat{x}\|$ is a surprise signal — used both for diagnostics and for choosing which tokens to keep.

Follow-up: single-frame action prediction is a reactive policy; without a dynamics objective it never learns *why* an action works, is brittle to distribution shift, and gives no lookahead. The joint model's cost is inference latency (video head is expensive), which is why decoupled async or sparse video prediction is an active topic.

</details>


### [08-03] Explain the design of modern VLA models: OpenVLA vs. π0. What is action chunking and why does it help? Tokenized vs. continuous actions?  (difficulty 2; tags: vla, pi0, flow-matching, action-chunking)


<details><summary>Answer</summary>

**OpenVLA (2024)**: a 7B Llama-2 + fused DINOv2/SigLIP vision encoder, fine-tuned on Open X-Embodiment to emit **discretized actions as text tokens** (each of 7 DoF binned into 256 values, remapped to rarely-used vocab tokens). Simple, uses the LLM loss unchanged; but one action per forward pass, autoregressive over 7 tokens → slow (~5 Hz) and binning quantizes fine motion.

**π0 (Physical Intelligence, 2024)**: PaliGemma VLM backbone + a separate ~300M **action expert** connected via joint attention, trained with **flow matching** to output a **chunk of 50 continuous actions** in one denoising process (10 steps). Continuous, multimodal, 50 Hz-capable; the action expert is a separate set of weights so the VLM's language knowledge isn't overwritten by robot data.

**Action chunking** (ACT, Diffusion Policy, π0): predict the next $H$ actions (e.g. 16–50) and execute several before re-planning. Why it helps:
- **Temporal consistency**: humans demonstrate with pauses and idiosyncratic timing; per-step policies learn noisy, jittery actions and get stuck at "pause" states. Chunks commit to a coherent motion.
- Reduces the **effective horizon** of compounding errors by $H\times$ (fewer decisions).
- Handles **non-Markovian** demonstrations (the demonstrator's intent spans multiple steps).
- Amortizes inference cost of a big backbone. Downside: less reactive; fixed with temporal ensembling or receding-horizon re-planning.

**Tokenized vs. continuous**: tokens are plug-and-play with LLM infrastructure and cross-entropy handles multimodality natively, but quantize precision and need autoregressive decoding per dimension. Continuous with diffusion/flow gives high precision and multimodality via the generative head at the cost of iterative denoising. FAST (π0-FAST) tokenizes with a DCT + BPE over action chunks — a compression scheme very close in spirit to my spectral work — to get the best of both: ~5× shorter token sequences with continuous precision.

</details>


### [08-04] Imitation learning vs. RL for manipulation: when would you use behavior cloning, why does Diffusion Policy work better than a regression head, and how do you RL-fine-tune a diffusion policy?  (difficulty 2; tags: imitation-learning, rl, diffusion-policy, manipulation)


<details><summary>Answer</summary>

**BC** is the default when you have demonstrations and a stable environment: supervised, sample-efficient, safe. It fails on **covariate shift** (small errors lead to unseen states) and on **multimodal demonstrations**.

**Why Diffusion Policy beats regression.** Human demos are multimodal: go around the obstacle left or right; grasp from above or side. An MSE regression head predicts the **mean of the modes**, which is often an invalid action (straight into the obstacle). A diffusion / flow head models the **full conditional distribution** $p(a_{t:t+H} \mid o_t)$ and samples one coherent mode. Extra benefits: iterative denoising handles high-dimensional action chunks gracefully, training is stable (no GAN, no mixture-count choice as in MDNs), and classifier-free guidance / constraints can be applied at test time.

**RL after BC** addresses the covariate shift and lets the policy exceed the demonstrator. Options:
- **Advantage-weighted / filtered BC** (RWR, AWR, "ReinboT"): reweight demonstration or rollout samples by return — keeps the diffusion loss, cheapest and most stable.
- **Policy gradient through the denoising chain** (DPPO): treat the $K$ denoising steps as an MDP inside the environment MDP and run PPO on it; works but expensive.
- **Q-guided sampling / Q-score matching**: train a critic, then guide diffusion samples with $\nabla_a Q$ — decouples critic learning from the generative head.
- **Residual RL**: freeze the diffusion policy, learn a small residual action with RL.

What breaks: RL on real robots is data-starved and unsafe, so it's usually done in sim with a large domain-randomized wrapper, or as offline RL on logged data. The interviewer follow-up is "why not RL from scratch?" — sparse rewards and exploration in 7-DoF contact-rich tasks are hopeless without a demonstration prior; BC gives the prior, RL polishes.

</details>


### [08-05] Where is the sim-to-real gap for a vision-based manipulation policy — perception or dynamics? What do domain randomization, system ID, and real-to-sim digital twins each address?  (difficulty 2; tags: sim2real, domain-randomization, digital-twin)


<details><summary>Answer</summary>

Direct answer: for **vision-based manipulation** the gap is dominated by **perception** (rendering, lighting, textures, camera model), for **contact-rich or dynamic** tasks (insertion, in-hand, locomotion) it's dominated by **dynamics** (friction, compliance, actuator delay, contact solver artifacts). You need to know which one you're fighting.

- **Domain randomization** attacks both by making the policy invariant: randomize textures, lighting, camera pose, distractors (visual DR) and mass, friction, motor gains, latency (dynamics DR). It trades peak in-sim performance for robustness. Overdoing it produces a conservative policy; the modern practice is **automatic DR** (widen ranges as the policy succeeds) and **structured randomization** informed by real measurements.
- **System identification** narrows the dynamics gap: fit friction, damping, inertia, actuator model to real trajectories so the simulator is *centered* on reality, then randomize around it. Essential for locomotion and dexterous hands; less useful for vision.
- **Real-to-sim digital twins**: reconstruct the real scene (Gaussian splatting / mesh + physical params) so the simulator matches your specific lab, objects, and camera. Closes the visual gap by construction and lets you evaluate in a faithful replica; the risk is over-fitting to one twin. NVIDIA's "real-to-sim-to-real" pipelines (Isaac Lab + neural reconstruction) are exactly this.

Practical knobs I'd add: use **depth or segmentation as the policy input** to sidestep appearance; **pretrained visual encoders** (DINO/SigLIP) that already generalize; and **matching the sensor pipeline** — for my foveated-sensing project the sim rendered the same ROI-crop readout the 200 MP sensor performs, so the policy's *observation interface* was identical in sim and real, which mattered more than photorealism.

</details>


### [08-06] MuJoCo vs. Isaac Sim / Isaac Lab: how do the contact models and parallelism differ, and what did you actually do in MuJoCo for foveated perception with manipulation?  (difficulty 2; tags: simulators, mujoco, isaac, project)


<details><summary>Answer</summary>

**MuJoCo**: CPU, soft-contact model solved as a convex optimization (contacts are slightly compliant by design, tunable via solref/solimp), very stable and smooth gradients → the standard for RL research and system-ID; recent **MJX** runs it on GPU/TPU via JAX for thousands of parallel envs. Rendering is basic (OpenGL/OSMesa), no ray tracing.

**Isaac Sim / Lab**: PhysX GPU-rigid-body with rigid contacts (TGS solver), designed for **massively parallel** RL (tens of thousands of envs on one GPU), plus **photorealistic RTX rendering**, USD scene format, sensor simulation (cameras, lidar), and ready-made robot assets. Contact behavior can be more jittery and needs solver-parameter tuning; but for vision-based sim-to-real the rendering and asset ecosystem is the decisive advantage.

Rule of thumb: MuJoCo for algorithms and contact-accuracy research; Isaac Lab for scale and vision realism; both are fine for tabletop manipulation.

**What I did.** I built the foveated perception benchmark in MuJoCo: a manipulation task (pick-and-place style with a 7-DoF arm) observed by a simulated ultra-high-resolution camera. The full-res image is never given to the task policy; instead a **sensing policy** picks a small set of high-resolution ROIs (plus a low-res global view) each step, mimicking the ROI readout of the real 200 MP sensor. The manipulation policy then operates on that foveated observation. The sim let me (a) sweep the bandwidth budget (full res down to < 1/8 pixels) and measure task success, (b) render exactly the sensor's readout mode for sim-to-real transfer, and (c) get ground truth for tracking targets so I could also train and evaluate the sensing policy on object tracking and text recognition. The main sim gotcha was headless rendering (OSMesa/EGL) and keeping the high-res render cost from dominating RL throughput — I rendered at full res only for the selected ROIs.

</details>


### [08-07] Formulate foveated / active sensing as a policy. Why RL rather than supervised, how do you design the reward, and how do you evaluate on tracking, OCR, and manipulation?  (difficulty 3; tags: active-perception, foveation, rl, project)


<details><summary>Answer</summary>

**Formulation.** The sensor is a POMDP: state = full scene; observation = what you chose to read out ($K$ high-res ROIs + a low-res global view) within a pixel budget $B$; action = where to place the ROIs next (continuous centers/sizes or a discrete grid); reward = downstream task performance. The sensing policy $\pi_s(\text{ROIs}_{t+1} \mid \text{history of foveated observations})$ has to learn **saccades** (jump to a new salient region) and **pursuit** (track a moving target) as emergent behaviors.

**Why RL (or at least sequential decision learning) rather than supervised.** Supervised needs a label for "where should I look"; you can derive proxies (saliency, target position) but they don't capture the *sequential* value of looking — a glance now saves bandwidth later, and the task cost is non-differentiable through the crop operation. RL optimizes the actual objective under the budget. That said, I'd say honestly: with differentiable relaxations (soft attention over a grid, Gumbel-softmax over ROI choices) a **supervised / end-to-end** variant works for tasks with dense proxies, is far more sample-efficient, and I used it as a strong baseline; RL wins when the task reward is sparse (grasp success) or the horizon is long.

**Reward design.** Task reward (tracking IoU, OCR accuracy, manipulation success/shaped distance) minus a **bandwidth cost** $\lambda \cdot (\text{pixels read} / \text{budget})$, with the trade-off swept via $\lambda$ to draw the accuracy–bandwidth curve. Add a small penalty on ROI jitter to avoid oscillation. The trap: if the global low-res view alone solves the task, the policy learns to ignore ROIs — so the tasks were chosen to *need* high-res detail (small text, small objects, fine grasp alignment).

**Evaluation.** Report **task metric vs. pixel bandwidth** curves against baselines: full-res, uniform downsample, random ROIs, saliency-based ROIs, and a center-fixed fovea. Tracking: success/precision vs. budget on moving targets; OCR: character accuracy on text at varying scale; manipulation: success rate over ≥ 10 seeds × tasks with paired seeds across sensing policies. The headline: task performance held with < 1/8 of full-res bandwidth, and the real 200 MP prototype reproduced the sim trend.

</details>


### [08-08] Video generation models as data engines for robotics: how would you generate synthetic training data with them, and what are the risks?  (difficulty 2; tags: synthetic-data, video-generation, robotics)


<details><summary>Answer</summary>

**Pipelines.**
- **Visual augmentation of real demos** (Cosmos-Transfer / GR00T-Dreams style): keep real trajectories and actions, re-render the video with a controllable video model conditioned on depth/segmentation/edges to vary lighting, textures, objects, backgrounds. Actions stay ground truth, so it's the safest use.
- **Video → action via inverse dynamics**: generate new task videos from an image + text instruction with a fine-tuned video model, then label actions with a learned **inverse dynamics model** or a **latent action model** (Genie / LAPA), and train the policy on the pseudo-labeled data. Scales to tasks with no demos; quality depends entirely on the IDM.
- **World-model-based evaluation / RL**: use an action-conditioned video model as a simulator to score policies or roll out RL — data engine for *experience*, not just demos.
- **Sim + generative refinement**: render in Isaac Sim, then use a video model to photorealize (sim-to-real via generation) while keeping sim's perfect labels.

**Risks.**
- **Physics hallucinations**: objects merge, hands pass through things, contact events are wrong; a policy trained on these learns impossible strategies. Needs physics filtering (a VLM/physics critic, or a sim check).
- **Action–video mismatch**: pseudo-labeled actions may not achieve what the video shows; IDM errors are systematic, not random.
- **Distribution collapse**: generated data reflects the video model's priors (canonical viewpoints, common objects); it can reduce diversity while looking like it adds it — monitor coverage with embedding statistics.
- **Evaluation contamination**: if the same generative model is used to make training data and to evaluate, you measure agreement, not competence. Real-robot eval is non-negotiable for the final claim.
- Cost: minutes of GPU per clip; compare against cheap sim DR before scaling.

</details>


### [08-09] LIBERO-style benchmarks: why do people run 10 seeds × N tasks, how do you get a confidence interval on success rate, and when does sim eval mislead you about real performance?  (difficulty 1; tags: benchmarks, evaluation, libero)


<details><summary>Answer</summary>

**Success rate is a Bernoulli estimate**, so its standard error is $\sqrt{p(1-p)/n}$. With $10\ \text{tasks} \times 50\ \text{episodes} = 500$ rollouts at $p = 0.8$, SE $\approx 1.8\%$, so two methods 3 points apart are barely distinguishable; with 10 episodes per task the SE is ~13% per task. **Seeds** matter because both the initial layout (object poses) and the policy's stochasticity vary; multiple seeds average over the layout distribution and let you report a CI across seeds (t-interval over per-seed means) rather than pooling episodes as if independent. Using **the same seeds for every method** (paired) removes layout variance from the comparison. Also report per-task results — averages hide that a method fixed one task and broke another.

Sim eval misleads when:
- The sim's **success predicate** is loose (object "in bowl" tolerance) or exploitable.
- The policy overfits sim rendering — LIBERO's fixed textures/lighting make vision trivially easy.
- The benchmark's initial-state distribution is narrow; a policy that memorizes 50 layouts looks great.
- **Failure modes differ**: in sim the robot stalls harmlessly; on real hardware the same policy collides.
- Timing: sim runs open-loop at fixed control rate; real inference latency changes the closed loop.

So: sim numbers for ablations and ranking, with paired seeds and CIs; real-robot trials (even 20–50) for the headline claim.

</details>


### [08-10] How would you measure and improve the physical consistency of a video world model?  (difficulty 3; tags: physics, world-models, evaluation, latent-actions)


<details><summary>Answer</summary>

**Measure.**
- **Physics-probe benchmarks**: Physics-IQ / VideoPhy-style tests — generate the continuation of a real clip with a known physical outcome (collision, drop, pour, occlusion) and compare to ground truth with **event-level metrics** (did the ball fall, when did contact happen, spatial IoU over time) rather than pixel MSE. Report by category: gravity, momentum, solidity, permanence.
- **Object permanence / occlusion tests**: hide an object and check it reappears.
- **Counterfactual action sensitivity**: for action-conditioned models, vary the action and measure whether the outcome changes in the right direction ($\partial\,\text{outcome}/\partial\,\text{action}$ sign); many models ignore the action and just continue the video.
- **Simulator-grounded scoring**: recreate the scene in a physics engine, track objects in the generated video, and measure deviation from the sim trajectory.

**Improve.**
- **Physics sim as teacher**: co-train or fine-tune on large volumes of sim-rendered videos with perfect dynamics and randomized appearance (Isaac Sim, MuJoCo + rendering) — the model learns dynamics from sim and appearance from real video. Risk: sim's look leaks into generations; fix with controllable-generation conditioning (depth/segmentation from sim, appearance from the model).
- **Latent action models**: learn a discrete latent action between consecutive frames (Genie / LAPA VQ bottleneck) so the model is forced to represent *what changed* explicitly, then condition on it; this sharpens causal structure and makes action conditioning stronger.
- **Physics-aware losses / rewards**: RL fine-tune the video model with a physics critic reward (VLM judge or detector-based consistency), same machinery as RLHF for diffusion (DPO / GRPO on video).
- **Architectural inductive bias**: object-centric latents, explicit 3D memory, or a differentiable physics prior for rigid bodies; costly and domain-limiting, so I'd try data and rewards first.

The honest caveat: current video models learn *typical* physics, not laws; extrapolation to unusual masses or frictions is poor. For robotics I'd treat the video model as a strong perceptual prior and keep a physics simulator in the loop for contact.

</details>


### [08-11] Planning with a world model: how does MPC in imagination work, and how do you choose the rollout horizon when the model's errors compound?  (difficulty 2; tags: planning, mpc, world-models)


<details><summary>Answer</summary>

**MPC in imagination**: at each control step, sample $N$ candidate action sequences of length $H$, roll them through the world model from the current latent, score them with a learned reward/value (or a goal-image distance), pick the best (CEM / MPPI refinement over a few iterations), execute the first action(s), re-plan. PlaNet and TD-MPC2 are the canonical latent-space versions; Dreamer instead amortizes planning into a policy trained by imagined rollouts, which is cheaper at test time.

**Horizon choice.** The imagined return has two error sources: **model error** growing with horizon (compounding) and **value-bootstrap error** at the truncation point. The trade-off is like an n-step return: short $H$ leans on the value function; long $H$ leans on the model. Practical rule: $H \approx$ the horizon over which the model's rollout error stays below the reward's sensitivity — measure it directly by rolling the model on held-out trajectories and plotting prediction error vs. step; pick $H$ where the error curve knees. Typical $H$ is 5–15 latent steps in Dreamer/TD-MPC. Mitigations: **ensembles** to penalize disagreement (MBPO-style pessimism), **short rollouts branched from real states** rather than long ones from the start, and **receding horizon** so only the first action is ever executed.

With a **video world model** the same logic holds but rollouts cost seconds, so you use very few candidates ($N \approx 4\text{–}16$), coarse actions (subgoals), and a VLM/critic scorer; the latency makes it a high-level planner, with a fast reactive policy underneath. The follow-up: "why not backprop through the dynamics instead of sampling?" — you can (Dreamer's actor loss), but gradients through long imagined horizons are chaotic; sampling-based MPC is robust to a non-smooth model.

</details>


### [08-12] Using VLMs as the perception backbone of a robot policy — freeze or fine-tune? What does each choice break?  (difficulty 2; tags: vlm, foundation-models, robot-learning)


<details><summary>Answer</summary>

**Freeze** (train only the action head / adapter): preserves the VLM's open-vocabulary grounding and language understanding, avoids catastrophic forgetting from small, narrow robot datasets, cheap to train, and robust to distribution shift in *semantics*. Breaks: the frozen features are tuned for captioning/VQA, not for **precise spatial and metric information** (where exactly is the handle, how far); resolution is often low (224–448) and fine manipulation suffers; no adaptation to egocentric/wrist cameras.

**Full fine-tune**: best in-distribution task performance and spatial precision. Breaks: forgets language and generalization (OpenVLA's authors found fine-tuning the vision encoder was necessary for performance yet it hurt zero-shot semantics); needs large robot datasets (OXE-scale) or it overfits; expensive.

**The middle ground most systems use**: LoRA or partial fine-tuning of the VLM, a **separate action expert** (π0) so robot data flows through its own weights while joint attention reads the VLM features, **co-training** with VLM data (captioning/VQA) mixed into the robot batches to anchor the language ability, and adding **higher-resolution or multi-scale visual tokens** for precision (which is where foveated / mixed-resolution tokenization directly applies — spend tokens where the task needs detail).

Rapid follow-up: "what does the VLM give you that a DINO encoder doesn't?" — language grounding for instruction following and object generalization; for a single fixed task, a DINO + small transformer policy is often equal and 10× cheaper.

</details>


### [08-13] Quick take on humanoid loco-manipulation: how are today's systems built, and where does the data come from?  (difficulty 1; tags: humanoid, locomotion, whole-body-control)


<details><summary>Answer</summary>

Today's stack is usually **two-tier**:
- **Low-level whole-body controller** trained with **RL in simulation** (Isaac Lab, thousands of parallel envs, heavy domain randomization + system ID for actuators), taking proprioception and a command (target root velocity, upper-body joint targets or end-effector poses) and outputting joint targets at ~50 Hz through a PD loop. It handles balance, stepping, and disturbance rejection. Motion priors from human mocap (AMP / adversarial motion priors, or tracking retargeted motions) make the gait natural.
- **High-level policy** — a VLA or a diffusion policy on vision — outputs the commands to the low-level controller. Manipulation-heavy work often trains the high-level policy by **BC on teleoperation data** collected with VR / mocap suits / exoskeletons that retarget human motion to the humanoid (GR00T, the Unitree/Tesla-style setups).

**Where the data comes from**: (1) sim for locomotion (essentially unlimited), (2) **teleop** for manipulation (expensive, ~hours to hundreds of hours), (3) **human video** — egocentric datasets and retargeted human motion as a pretraining prior, with the embodiment gap bridged by latent actions or retargeting, (4) increasingly synthetic video from world models (GR00T-Dreams) for augmentation.

Key difficulties to mention: the whole-body controller must remain stable while the upper body does arbitrary things (coupled dynamics), latency between tiers, and the sim-to-real gap in contact with the environment during manipulation (not just feet). NVIDIA's positioning — Isaac Lab + GR00T + Cosmos — is exactly this stack.

</details>


### [08-14] Concrete pipeline: you have a video generation model and a mediocre manipulation policy with 200 demos. How would you use the video model to improve the policy, and how would you prove it worked?  (difficulty 3; tags: design, video-generation, robot-learning, synthetic-data)


<details><summary>Answer</summary>

I'd go in three stages of increasing risk, each with a measurable gate.

**Stage 1 — Visual augmentation (low risk, actions untouched).** Fine-tune a controllable video model (depth/segmentation-conditioned, Cosmos-Transfer style) on the 200 demos, then re-render each demo 10–50× with varied lighting, textures, backgrounds, distractors, and camera jitter while keeping the exact action trajectory. Train the policy on real + augmented. Gate: success rate on a **perturbed-appearance** eval set (new lighting/objects) with paired seeds; expect the biggest win here for a vision-brittle policy.

**Stage 2 — New trajectories with pseudo-labeled actions (medium risk).** Fine-tune the video model as an image+instruction → video predictor on the demos (plus any related web/robot video). Generate rollouts from *new* initial states and instructions. Label actions with an **inverse dynamics model** trained on the 200 demos + sim play data (the IDM is the crux; validate its accuracy on held-out real demos first). Filter generations with a physics/plausibility critic (VLM + object tracker: object moves only when in contact, gripper closes on the object). Train on filtered data. Gate: success on new initial-state distributions, plus ablate "no filter" to show the filter matters.

**Stage 3 — World model in the loop (highest risk, highest ceiling).** Action-condition the video model (or use the world-action-model form) and use it as an evaluator: run the current policy in imagination over many initial states, score with a VLM/success detector, and do **filtered BC / advantage-weighted fine-tuning** on the imagined successes, or short-horizon RL. Gate: correlation between imagined success rate and real success rate across policy checkpoints must be high before I trust it for training — if the world model can't rank policies it can't improve them.

**Proof.** Real robot: ≥ 3 conditions (in-distribution, new appearance, new layouts) × ≥ 30 trials, same seeds/layouts across policies, blind evaluator; report the whole curve of success vs. amount of synthetic data. Ablations: real-only, sim-DR-only (the cheap competitor everyone will ask about), each stage cumulatively. And report failure analysis — if the policy learned a physically impossible strategy from hallucinated video, that's the risk I'd flag up front and the reason the filter and the Stage-3 correlation gate exist.

</details>


### [08-15] What is a world model, in one sentence?  (difficulty 1; tags: world-models, rapid-fire)


<details><summary>Answer</summary>

**A learned model of environment dynamics, $p(o_{t+1} \mid o_{\le t}, a_t)$, that predicts what happens next given history and an action** — so an agent can plan, learn, or be evaluated in imagination instead of in the real world. It exists because real interaction is slow, expensive, and unsafe, whereas a model can roll out thousands of futures in parallel. Two flavors: compact latent dynamics (Dreamer, TD-MPC — cheap to plan in) and pixel/latent-video predictors (Cosmos, Genie — expensive but human-viewable, pretrained on internet video). The trap: a video model without action conditioning is a video generator, not a world model — the test is whether varying the action changes the predicted outcome in the right direction.

</details>


### [08-16] What is a VLA?  (difficulty 1; tags: vla, robot-learning, rapid-fire)


<details><summary>Answer</summary>

**A vision-language-action model: a vision-language model fine-tuned to output robot actions, so a single network maps camera images plus a language instruction directly to motor commands.** It exists to reuse the VLM's open-vocabulary grounding and web-scale visual knowledge for robots, where task-specific data is tiny. Two design axes: how actions are emitted — as discretized text tokens (OpenVLA, RT-2) or as continuous chunks from a separate diffusion/flow "action expert" (π0) — and how much of the VLM is fine-tuned versus frozen. Trap: a VLA is still imitation learning — it inherits covariate shift from behavior cloning and only sees what the teleop data covered; the language ability is what makes it generalize across objects, not across skills.

</details>


### [08-17] What is action chunking?  (difficulty 1; tags: action-chunking, imitation-learning, rapid-fire)


<details><summary>Answer</summary>

**Predicting a sequence of the next $H$ actions (e.g. 16–50 steps) in one forward pass and executing several before re-planning, instead of one action per step.** It exists because human demonstrations are non-Markovian and noisy in timing: a per-step policy learns jittery actions and gets stuck at pause states, whereas a chunk commits to a coherent motion. It also cuts the number of decisions by $H$, which shrinks compounding error, and amortizes the cost of a big backbone. The trade-off is reactivity — the robot is open-loop for $H$ steps — mitigated by receding-horizon execution (run only the first few actions) or temporal ensembling of overlapping chunks.

</details>


### [08-18] What is behavior cloning?  (difficulty 1; tags: behavior-cloning, imitation-learning, rapid-fire)


<details><summary>Answer</summary>

**Supervised learning of a policy from demonstrations: fit $\pi(a \mid o)$ to expert $(o, a)$ pairs with a regression or likelihood loss.** It is the default for manipulation because it is sample-efficient, stable, and safe to train. Its two failure modes are the whole story: **covariate shift** — small errors put the robot in states the demonstrator never visited, where the policy has no idea what to do, so errors compound (DAgger fixes this by querying the expert in those states); and **multimodality** — an MSE head averages the demonstrator's modes into an invalid action, which is why diffusion policies replaced regression heads. Trap: BC cannot exceed the demonstrator; RL fine-tuning or filtered BC is needed for that.

</details>


### [08-19] What is the sim-to-real gap?  (difficulty 1; tags: sim2real, rapid-fire)


<details><summary>Answer</summary>

**The drop in performance when a policy trained in simulation is deployed on real hardware, caused by every way the simulator differs from reality.** It splits into a **perception gap** (rendering, lighting, textures, camera and sensor noise) and a **dynamics gap** (friction, compliance, actuator delay, contact modeling). Which one dominates depends on the task: vision-based tabletop manipulation is mostly perception; locomotion and insertion are mostly dynamics. Tools: domain randomization (make the policy invariant), system identification (center the sim on measured reality), and real-to-sim digital twins (reconstruct the actual scene). Trap: matching the observation interface matters more than photorealism — in my foveated-sensing work the sim reproduced the sensor's ROI readout exactly, and that transferred.

</details>


### [08-20] What is domain randomization?  (difficulty 1; tags: domain-randomization, sim2real, rapid-fire)


<details><summary>Answer</summary>

**Training in simulation while randomizing the parameters you cannot match to reality — textures, lighting, camera pose, distractors, mass, friction, motor gains, latency — so the policy learns to be invariant to them and the real world looks like one more sample.** It exists because closing the sim-to-real gap by making the simulator perfect is impossible; widening the training distribution is cheap. This can also reduce overfitting. Visual DR attacks the perception gap, dynamics DR the dynamics gap. Trap: too much randomization produces a conservative, low-performance policy that hedges against everything; the fix is automatic DR (widen ranges as the policy succeeds) and system ID so you randomize around the true values rather than across a huge box.

</details>


### [08-21] What is a diffusion policy?  (difficulty 1; tags: diffusion-policy, imitation-learning, rapid-fire)


<details><summary>Answer</summary>

**A behavior-cloning policy whose action head is a conditional diffusion (or flow-matching) model: it samples an action chunk $a_{t:t+H}$ by iteratively denoising from noise, conditioned on the observation.** It exists because demonstrations are multimodal — go left or right around an obstacle — and a regression head predicts the invalid mean; a generative head models the full distribution and samples one coherent mode. Compared to mixture density networks or GANs it is stable to train and needs no choice of mixture count. Cost: 10–100 denoising steps per decision, so it pairs with action chunking. Trap: it is still supervised on demos, so covariate shift is unchanged; RL fine-tuning of diffusion policies (DPPO, Q-guided sampling) is the follow-up.

</details>


### [08-22] What is a POMDP?  (difficulty 2; tags: pomdp, rl, active-perception, rapid-fire)


<details><summary>Answer</summary>

**A partially observable Markov decision process: an MDP (states, actions, transitions, rewards) where the agent does not see the state, only an observation drawn from $p(o \mid s)$.** The optimal policy must therefore act on a belief — a distribution over states given the history — rather than on the current observation, which is why memory is needed. Active perception makes the observation itself part of the action — my foveated-sensing policy chooses where to look, so the reward includes the information value of the next observation. Trap: an observation-only reactive policy can still work if the task is effectively Markov in the observation, which is why frame stacking is the first thing to try.

</details>


## Computer Vision & 3D


### [09-01] Write down the pinhole camera model. What are intrinsics and extrinsics, and how do you go from a 3D point in the world to a pixel?  (difficulty 1; tags: camera, geometry, 3d)

- Follow-up: what does the principal point do, and why is the skew term almost always zero?
- Follow-up: how do you go the other way — pixel plus depth to a 3D point?


<details><summary>Answer</summary>

**A pixel is a 3D point projected through the optical centre onto the image plane: $u = K[R \mid t]\,X$.**

- **Extrinsics** $[R \mid t]$ (a $3 \times 4$ rigid transform, 6 DoF) map a world point $X_w$ to camera coordinates $X_c = R X_w + t$, with the camera at the origin looking down $+z$. Careful with the convention: $R, t$ here are world-to-camera; the "camera pose" people store (camera-to-world, as in NeRF datasets) is the inverse: $C = -R^\top t$.
- **Perspective division**: $(x, y, z) \to (x/z, y/z)$. This is the non-linear step and the reason projection is written in homogeneous coordinates.
- **Intrinsics** $K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ map normalised image coordinates to pixels: $f_x, f_y$ are focal lengths in pixels (physical focal length divided by pixel pitch; different if pixels are non-square), $(c_x, c_y)$ the principal point where the optical axis hits the sensor ($\approx$ image centre), $s$ the skew (non-orthogonal pixel axes, zero for any modern sensor). Real lenses add radial/tangential distortion applied in normalised coordinates before $K$.

Full chain: $u = K(R X_w + t)$, then divide by the third coordinate. Inverse: given pixel $(u, v)$ and depth $z$, $X_c = z K^{-1} [u, v, 1]^\top$, $X_w = R^\top (X_c - t)$ — this is how you unproject a depth map into a point cloud, and the ray direction $d = R^\top K^{-1} [u, v, 1]^\top$ is what NeRF and Gaussian splatting march along. Focal length in pixels and the field of view are the same information: $f = (W/2)/\tan(\mathrm{fov}_x/2)$.

</details>


### [09-02] NeRF versus 3D Gaussian Splatting: write the volume-rendering equation, explain why NeRF needs positional encoding and why it is slow, and explain why 3DGS is fast and how gradients reach the Gaussians.  (difficulty 3; tags: nerf, gaussian-splatting, neural-rendering)

- Follow-up: what does 3DGS lose compared to NeRF?
- Follow-up: what is the role of densification and pruning?


<details><summary>Answer</summary>

**Volume rendering** along a ray $r(t) = o + t\,d$ with density $\sigma$ and colour $c$:

$$
\begin{aligned}
C(r) &= \int T(t)\,\sigma(t)\,c(t)\,\mathrm{d}t, \qquad T(t) = \exp\!\left(-\int_0^t \sigma(s)\,\mathrm{d}s\right) \\
\text{discretised:}\quad C &= \sum_i T_i\,\bigl(1 - e^{-\sigma_i \delta_i}\bigr)\,c_i, \qquad T_i = \prod_{j<i} e^{-\sigma_j \delta_j}
\end{aligned}
$$

$\alpha_i = 1 - e^{-\sigma_i \delta_i}$ is the opacity of sample $i$; $T_i$ is transmittance (probability the ray got this far). It is differentiable in $\sigma$ and $c$, so you can fit them from photos with an L2 loss on pixels.

**NeRF** represents $(\sigma, c)$ with an MLP $F(x, d)$. A plain MLP on raw coordinates is spectrally biased toward smooth functions, so it cannot fit fine detail; **positional encoding** $\gamma(x) = [\sin(2^k \pi x), \cos(2^k \pi x)]_k$ lifts inputs to a high-frequency basis (the NTK view: it makes the kernel stationary with tunable bandwidth). Why slow: every pixel needs ~64 + 128 MLP evaluations along its ray, so a 1080p image is ~400M network evaluations; training takes hours-to-a-day and rendering seconds per frame. Instant-NGP fixed most of that with hash-grid features + a tiny MLP, but rendering still requires marching samples.

**3D Gaussian Splatting** replaces the field with an explicit set of anisotropic 3D Gaussians (mean $\mu$, covariance $\Sigma = R S S^\top R^\top$, opacity, spherical-harmonic colour). Rendering is **rasterisation, not ray-marching**: project each Gaussian to a 2D Gaussian ($\Sigma' = J W \Sigma W^\top J^\top$ with $J$ the Jacobian of the projection), sort by depth per 16×16 tile, and alpha-composite front-to-back with the *same* discrete equation, $C = \sum_i \alpha_i c_i \prod_{j<i}(1-\alpha_j)$. It is fast because the work is proportional to the number of Gaussians touching each tile rather than samples per ray, there is no network, and the tile-based CUDA rasteriser is bandwidth-friendly: real-time (100+ fps) at 1080p.

**Gradients**: the compositing is a differentiable function of $\alpha_i$ and $c_i$; $\alpha_i$ depends on the projected 2D Gaussian evaluated at the pixel, which depends on $\mu$, $\Sigma$, and opacity through the projection Jacobian. The custom backward pass walks tiles back-to-front, accumulating $\partial L/\partial c$, $\partial L/\partial \alpha \to \partial L/\partial \mu$, $\partial L/\partial \Sigma$ (via $\partial \Sigma'/\partial \Sigma$ and the rotation/scale parameterisation), $\partial L/\partial(\text{opacity})$. Because the scene is explicit, gradient magnitudes on positions tell you where the model is under-fitting: **densification** clones/splits Gaussians with large view-space position gradient, and **pruning** removes near-transparent ones. Losses vs NeRF: no continuous field (worse for extrapolated views and thin structures), memory grows with scene detail (millions of Gaussians, hundreds of MB), and floaters near the cameras.

</details>


### [09-03] How do diffusion models do novel view synthesis? Explain the Zero-1-to-3 idea, camera-conditioned video models, and why world models are "NVS without geometry".  (difficulty 2; tags: nvs, diffusion, world-models, camera)

- Follow-up: how do you inject the camera pose into a DiT?
- Follow-up: what do you gain and lose versus reconstructing with 3DGS first?


<details><summary>Answer</summary>

**The move is to replace explicit geometry with a generative prior: condition an image/video diffusion model on the source view and a relative camera, and let it hallucinate what the target view should look like.**

**Zero-1-to-3**: fine-tune Stable Diffusion on rendered Objaverse pairs with the conditioning = CLIP embedding of the input image + relative $(\Delta\theta, \Delta\phi, \Delta r)$ spherical camera offset, concatenated in the cross-attention, plus the input image channel-concatenated to the latent. It learned a viewpoint prior from 2D pretraining with only ~800k synthetic objects; the output is one view per sample and consecutive samples are inconsistent, which is what SyncDreamer, MVDream and Zero123++ fix by generating all views jointly with cross-view attention (treat views as a sequence, like frames).

**Camera-conditioned video models** (CameraCtrl, MotionCtrl, CamCo, ReCamMaster, GEN3C): a video DiT is conditioned on the per-frame camera trajectory. The pose is injected as **Plücker ray embeddings** — for each pixel, the 6-D $(d, o \times d)$ ray computed from $K, R, t$ — patchified and added to (or channel-concatenated with) the latent tokens, so each token knows which ray it is rendering. That is the same signal NeRF gets, but the model can fill in unseen content. GEN3C goes further and renders a 3D point-cloud cache from estimated depth to feed a warped guide video to the DiT, so consistency is enforced geometrically and the generator only in-paints.

**World models as NVS**: a world model (Genie, Cosmos, Oasis, or the action-conditioned models in robotics) predicts the next frames given actions; if the action is "camera moved by $\Delta T$", that is NVS in autoregressive form. The scene is stored implicitly in the KV cache / latent state instead of in a 3DGS point set. Gains: works on unobserved regions, dynamic scenes, any sensor; losses: no guaranteed multi-view consistency (drift on loops), no explicit geometry to query, cost per frame is a full DiT pass, and evaluation must test "revisit consistency" — return to a previous pose and compare. The hybrid that seems to be winning is diffusion for content + an explicit 3D cache (or a learned memory) for consistency.

</details>


### [09-04] Monocular depth estimation: why is scale ambiguous, what do DPT / Depth Anything actually predict, and what changes for metric depth?  (difficulty 2; tags: depth, monocular, dpt)

- Follow-up: what is the scale-and-shift-invariant loss?
- Follow-up: how would you get metric depth into a video generation pipeline?


<details><summary>Answer</summary>

**A single image cannot fix absolute scale: a scene twice as large twice as far away produces the same pixels.** Focal length compounds it — a longer lens on a farther scene looks like a shorter lens on a nearer one — so a network trained across cameras can only learn *relative* depth up to a global scale (and, for disparity-like targets, a shift).

**DPT** (Dense Prediction Transformer, MiDaS v3) is a ViT encoder with a convolutional "reassemble + fusion" decoder that produces a dense map at multiple resolutions; it predicts **inverse depth (disparity) up to affine transform** and is trained with the **scale-and-shift-invariant loss**: for each image, solve the least-squares $(s, t)$ aligning prediction to ground truth, then take the L1/L2 error, plus a multi-scale gradient-matching term for sharp edges. That is what lets MiDaS mix datasets with incompatible depth units (stereo, LiDAR, SfM).

**Depth Anything** keeps the DPT head but swaps the encoder for DINOv2 and scales data: 1.5M labelled + 62M pseudo-labelled images, a teacher-student loop with strong augmentation on the student, and a feature-alignment loss to DINOv2 so semantics survive. V2 replaces real labels with synthetic renders for the teacher (sharper, no sensor noise) and distils to real images through a big teacher.

**Metric depth**: you need a scale cue — the camera intrinsics (Metric3D, UniDepth, Depth Pro predict focal length or take it as input and canonicalise the image to a fixed focal length), an absolute reference (LiDAR, stereo baseline, a known object), or dataset-specific fine-tuning (indoor NYU vs outdoor KITTI heads in ZoeDepth). Metric heads generalise worse across cameras precisely because they had to commit to a scale. For video, consistency across frames matters more than metric accuracy: Video Depth Anything / DepthCrafter add temporal attention; to use depth as a conditioning signal for a DiT you can pass relative disparity, normalised per clip, and let the generator ignore scale — but to build a point-cloud cache for NVS (GEN3C-style) you need metric depth aligned with the camera poses, typically by least-squares-fitting the monocular depth to sparse SfM points.

</details>


### [09-05] Optical flow: what is the brightness-constancy assumption, how does RAFT work, and how would you use flow to measure temporal consistency of generated video?  (difficulty 2; tags: optical-flow, raft, video-metrics)

- Follow-up: why do the classical methods fail at large displacements?
- Follow-up: what is warping error and what is wrong with it as a metric?


<details><summary>Answer</summary>

**Brightness constancy**: a moving point keeps its intensity, $I(x+u, y+v, t+1) = I(x, y, t)$. Linearising gives the optical-flow constraint $I_x u + I_y v + I_t = 0$ — one equation for two unknowns per pixel (the **aperture problem**: only the flow component along the gradient is observable), so classical methods add a prior: Lucas-Kanade assumes constant flow in a window, Horn-Schunck a global smoothness term. The linearisation only holds for sub-pixel motion, hence coarse-to-fine pyramids and their failure on large displacements of small objects.

**RAFT** (Teed & Deng 2020) is the design every modern flow model inherits: (1) a CNN extracts per-pixel features for both frames at 1/8 resolution; (2) an **all-pairs 4D correlation volume** $C[i, j] = \langle f_1(i), f_2(j) \rangle$ is built once, with a pyramid of pooled versions so large motions are visible; (3) a **GRU-based iterative update** starts from zero flow and, at each iteration, looks up correlation values around the current estimate, and predicts a flow *update* $\Delta f$. Because the lookup is around the current estimate, it handles large displacements without a pyramid over flow itself; iterations share weights (like an unrolled optimiser) and the loss is applied at every iteration with exponentially increasing weight. Learned convex upsampling gets back to full resolution. Successors: GMA (attention for occlusions), FlowFormer, SEA-RAFT (faster, mixture-of-Laplace loss), and for video-generation use cases usually RAFT-large or UniMatch off the shelf.

**Using flow for video consistency**: compute forward and backward flow between generated frames, warp frame $t+1$ back to $t$ with the flow, and measure the **warping error** on pixels that pass the forward-backward consistency check (occlusion mask). Low error = temporally stable. Failure modes as a metric: a static video scores perfectly (so pair it with a motion-magnitude score or the "dynamic degree" of VBench); the flow network itself is fooled by flicker and blur, and a model that blurs everything also reduces warping error. Better proxies for coherence: flow-field smoothness over time, and comparing the flow statistics of generated video to real video (a Fréchet distance on flow features). In our SPEED evaluation the concern is exactly this — that per-step frequency masking traded sharpness for a deceptively low warping error.

</details>


### [09-06] SLAM in one breath — and how do learned methods (DROID-SLAM, DUSt3R/MASt3R) change the picture?  (difficulty 1; tags: slam, 3d, geometry)

- Follow-up: what is bundle adjustment optimising?


<details><summary>Answer</summary>

**SLAM = estimate the camera trajectory and a map of the scene simultaneously from a stream of sensor data, in real time.** Classical visual SLAM (ORB-SLAM) is a pipeline: detect and match features → estimate relative pose (essential matrix / PnP) → triangulate landmarks → **bundle adjustment** (nonlinear least squares over poses and 3D points minimising reprojection error, sparse via Schur complement) on a sliding window of keyframes → loop closure via place recognition and pose-graph optimisation to kill drift. The core insight is that poses and structure are coupled and you alternate or jointly refine them.

Learned methods keep the optimisation but replace the brittle parts:
- **DROID-SLAM** replaces feature matching with RAFT-style dense flow between keyframe pairs and puts a differentiable dense bundle adjustment layer inside the network; a GRU iteratively refines flow, and the BA layer converts flow updates into pose + per-pixel depth updates. Much more robust than ORB features, still needs iterations.
- **DUSt3R / MASt3R**: a ViT that takes two images and directly regresses a pointmap for each (3D coordinates of every pixel in the first camera's frame), with no intrinsics or poses given. Poses and depth fall out of aligning pointmaps; MASt3R adds a matching head for metric-scale, and MASt3R-SLAM / CUT3R / VGGT run this feed-forward over sequences at real-time rates. The picture shifts from "geometry solved by optimisation, learning for features" to "geometry predicted by a foundation model, optimisation as a light global alignment".

Why it matters for generative work: these are the tools that give you camera poses and depth for any video, which is exactly the conditioning data a camera-controlled video DiT or a GEN3C-style 3D cache needs, and they turn any video collection into 4D training data.

</details>


### [09-07] Multi-view consistency in 3D generation: explain Score Distillation Sampling, why it produces over-saturated and Janus-faced objects, and how multi-view diffusion fixed it.  (difficulty 3; tags: sds, dreamfusion, 3d-generation, multi-view-diffusion)

- Follow-up: write the SDS gradient and say which term is dropped and why.
- Follow-up: where does the "mode-seeking" behaviour come from?


<details><summary>Answer</summary>

**SDS (DreamFusion)** optimises the parameters $\theta$ of a 3D representation (NeRF, later 3DGS) so that renders from random cameras look like samples of a pretrained 2D text-to-image diffusion model. Render $x = g(\theta)$, add noise $x_t = \alpha_t x + \sigma_t \epsilon$, and take

$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon}\left[ w(t)\,\bigl(\hat{\epsilon}_\phi(x_t; y, t) - \epsilon\bigr)\,\frac{\partial x}{\partial \theta} \right]
$$

i.e. push the render toward what the denoiser thinks a cleaner image looks like. The Jacobian of the U-Net $\partial \hat{\epsilon}/\partial x_t$ is **dropped** — it is expensive and empirically hurts — so this is not the gradient of the diffusion loss but a score-matching-like update. Because the noise prediction $\hat{\epsilon} - \epsilon \propto -\sigma_t \nabla \log p_t(x_t)$, SDS is doing gradient ascent on $\log p_t$ averaged over $t$: it is **mode-seeking** ($\mathrm{KL}(q_\theta \,\|\, p)$ direction), which combined with the huge CFG scale (~100) needed to make it work yields over-saturated, over-smoothed, low-diversity results.

**Failure modes**:
- **Janus problem**: the 2D model has a strong prior that "a dog" is seen from the front. Every camera view is pulled toward the canonical view, so the object grows multiple faces. Root cause: per-view optimisation with no view awareness in the 2D prior (view-dependent prompts "back view of…" help only marginally).
- **Over-saturation / floaters**: high CFG + mode-seeking; VSD (ProlificDreamer) fixes it by training a LoRA to model the distribution of *renders* and using $\hat{\epsilon}_{\text{pretrained}} - \hat{\epsilon}_{\text{LoRA}}$ as the gradient (a particle-based variational objective), giving sharp, diverse results.
- Slow (hours per asset), and the geometry is only as good as the NeRF density field.

**Multi-view diffusion** attacks the root cause: fine-tune the 2D model to generate several views **jointly** with camera conditioning — MVDream (4 orthogonal views with 3D self-attention across views), Zero123++, SyncDreamer, Wonder3D (normals + colour), and video models repurposed as "orbit" generators (SV3D). The views are consistent by construction because they attend to each other, so a fast reconstruction (sparse-view 3DGS/NeRF, or a feed-forward LRM-style transformer that regresses triplanes or Gaussians directly) replaces SDS. Today's pipeline is: multi-view diffusion → feed-forward reconstructor (LGM, InstantMesh, TRELLIS) → optional short SDS/texture refinement, in under a minute. Remaining issues: back views are still hallucinated, and consistency is only among the generated views, not with arbitrary novel ones — which is why native 3D latent diffusion (TRELLIS, Hunyuan3D) is now displacing the multi-view route.

</details>


### [09-08] PSNR, SSIM, LPIPS: define them, and give me a case where each one misleads.  (difficulty 2; tags: metrics, image-quality, evaluation)

- Follow-up: which would you report for a video-generation speedup paper, and why not just FID?


<details><summary>Answer</summary>

- **PSNR** $= 10 \log_{10}(\mathrm{MAX}^2 / \mathrm{MSE})$, in dB, $\mathrm{MAX} = 1$ for $[0, 1]$ images. A pure per-pixel L2 measure. Misleads because it prefers the **blurry mean**: a slightly shifted (by one pixel) sharp image can score worse than a Gaussian-blurred one, and a generative model that regresses to the conditional mean wins PSNR while looking terrible. In our KV-cache work a stalled robot gave high imagined-video PSNR for the wrong reason — the future was trivially predictable.
- **SSIM** compares local means, variances and covariance in an 11×11 Gaussian window: $\mathrm{SSIM} = \frac{(2\mu_x \mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$, range $[-1, 1]$, averaged over the image (MS-SSIM adds scales). Better at structure and less sensitive to global brightness. Misleads on texture: it is still a local-alignment metric, so a realistic re-synthesised texture that is not pixel-aligned scores low, and it saturates at high quality; also it is computed on luminance by default and ignores colour errors.
- **LPIPS** = distance between deep features (VGG/AlexNet) at several layers, with learned per-channel weights calibrated on human 2AFC judgements. Correlates far better with perceived similarity and tolerates small misalignments. Misleads when the *reference* is wrong (it is still a full-reference metric), when the content is out of the training distribution of the backbone (it is fooled by adversarial-style textures), and it says nothing about realism per se — a plausible but different image scores badly.

None of them measures *realism without a reference*; that's what **FID / FVD** (Fréchet distance between Inception / I3D feature Gaussians) do, at the cost of needing thousands of samples, being biased by sample count, and being insensitive to fine detail and temporal artefacts (FVD's I3D features barely see motion quality). For a speedup paper like SPEED the honest set is: paired metrics vs the full-cost baseline with the **same seed** (PSNR/SSIM/LPIPS quantify "did we change the sample"), plus distributional metrics vs real data (FID/FVD, or CMMD which is sample-efficient), plus a human or VLM preference study — and for video, a temporal metric (warping error or VBench's consistency scores) because spatial metrics cannot see flicker.

</details>


### [09-09] What are camera intrinsics?  (difficulty 1; tags: camera, geometry, rapid-fire)


<details><summary>Answer</summary>

**The parameters that map a point in normalized camera coordinates to a pixel: the $3 \times 3$ matrix $K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$.** $f_x, f_y$ are the focal length in pixels (physical focal length over pixel pitch), $(c_x, c_y)$ is the principal point where the optical axis hits the sensor, and $s$ is the skew, zero on any modern sensor. They exist separately from extrinsics because they are a property of the camera, not its pose, so you calibrate once and reuse. Trap: focal length and field of view are the same information, $f = (W/2)/\tan(\mathrm{fov}/2)$, and a camera-conditioned video model gets intrinsics through the Plücker ray embeddings, which are computed from $K^{-1}$.

</details>


### [09-10] What is a NeRF, in one sentence?  (difficulty 1; tags: nerf, neural-rendering, rapid-fire)


<details><summary>Answer</summary>

**A neural radiance field: an MLP that maps a 3D position and viewing direction to density and color, trained from posed photos by differentiable volume rendering so that ray-marched renders match the pixels.** It exists because it turned novel view synthesis into a continuous optimization with no explicit mesh — the scene is stored in the weights and rendered by integrating $C = \sum_i T_i \alpha_i c_i$ along each ray. The two enabling tricks: positional encoding (a plain MLP is biased toward smooth functions) and hierarchical sampling. Trap: it is slow — hundreds of MLP evaluations per pixel — which is what Instant-NGP's hash grids and then Gaussian splatting addressed, and it needs accurate camera poses from SfM to begin with.

</details>


### [09-11] What is 3D Gaussian splatting, in one sentence?  (difficulty 1; tags: gaussian-splatting, neural-rendering, rapid-fire)


<details><summary>Answer</summary>

**An explicit scene representation — millions of anisotropic 3D Gaussians with position, covariance, opacity, and spherical-harmonic color — rendered by projecting each Gaussian to the image, sorting by depth per tile, and alpha-compositing front to back, then optimized from photos by gradients through that rasterizer.** It exists because ray-marching a NeRF is too slow for real time; splatting does work proportional to visible Gaussians per tile, with no network, and hits 100+ fps at 1080p. Densification clones and splits Gaussians where position gradients are large; pruning removes transparent ones. Trap: memory grows with detail (hundreds of MB), floaters appear near cameras, and there is no continuous field for extrapolated views.

</details>


### [09-12] What is optical flow?  (difficulty 1; tags: optical-flow, video, rapid-fire)


<details><summary>Answer</summary>

**A dense field of 2-D displacement vectors $(u, v)$ per pixel describing how image content moves between two consecutive frames.** It exists because motion is the primary signal for tracking, video compression, and temporal consistency. Classically derived from brightness constancy, $I(x+u, y+v, t+1) = I(x, y, t)$, which gives one equation for two unknowns per pixel — the aperture problem — so a smoothness prior is required. Modern methods (RAFT) learn it: an all-pairs correlation volume between per-pixel features plus a GRU that iteratively refines the flow. Trap: flow is 2-D apparent motion, not 3-D scene motion, and as a metric for generated video it is fooled by static or blurry outputs, so pair warping error with a motion-magnitude score.

</details>


### [09-13] What is epipolar geometry?  (difficulty 2; tags: epipolar, geometry, multi-view, rapid-fire)


<details><summary>Answer</summary>

**The geometric constraint between two views of the same scene: a point in image 1 must lie on a specific line — the epipolar line — in image 2, because the 3D point sits somewhere along image 1's ray, and that ray projects to a line.** Algebraically, $x_2^\top F x_1 = 0$ with $F$ the fundamental matrix (uncalibrated) or $x_2^\top E x_1 = 0$ with $E = [t]_\times R$ the essential matrix (calibrated). It exists because it turns 2-D correspondence search into a 1-D search and gives you relative pose: estimate $E$ from 5+ correspondences, decompose into $R, t$, then triangulate. Trap: $t$ is only recovered up to scale, which is the monocular scale ambiguity again.

</details>


### [09-14] SSIM and LPIPS in one breath.  (difficulty 1; tags: metrics, image-quality, rapid-fire)


<details><summary>Answer</summary>

**SSIM compares local means, variances, and covariance of two images in a sliding Gaussian window — a structural, luminance-normalized similarity in $[-1, 1]$; LPIPS is the distance between deep features (VGG or AlexNet) at several layers, with per-channel weights calibrated on human two-alternative judgments.** Both exist because PSNR rewards the blurry mean: SSIM at least captures local structure, and LPIPS matches human perception far better and tolerates small misalignments. Both are full-reference metrics, so they say nothing about realism without a ground truth. Traps: SSIM still penalizes any re-synthesized texture that is not pixel-aligned and ignores color by default; LPIPS depends on the backbone's training distribution and can be gamed by adversarial textures.

</details>


## Project Grill: SPEED (Spectral Progressive Diffusion)


### [10-01] Give me the two-minute pitch for SPEED. What is the one-sentence insight, and what is the number I should remember?  (difficulty 1; tags: speed, pitch, diffusion)

- Follow-up: Who cares? Why is this not just "run the model at low res first"?


<details><summary>Answer</summary>

**Insight: a diffusion sampler is already a coarse-to-fine process in frequency space, so it should not pay full-resolution compute while it is only resolving low frequencies.**

The natural-image (and latent) power spectrum is a power law, $P(f) = A f^{-\beta}$ ($\beta \approx 1.9$ for FLUX latents, $\approx 2.4$ for WAN / PixelGen). Under flow matching, $x_t = (1-t)\,x_0 + t\,\epsilon$, noise has flat unit power per orthonormal DCT bin, so at large $t$ every high-frequency bin is pure noise — the network's output there carries no information, but a DiT still spends $O(N^2)$ attention on those tokens. SPEED measures the spectrum once, derives the flow time at which each frequency band's SNR crosses 1, and runs the sampler at a resolution whose Nyquist frequency matches what is currently resolvable: low res early, native res late. Stage transitions are done in the DCT domain (embed low-res coefficients into the low-frequency corner of the larger grid, fill the new band with $t \cdot \text{noise}$, fix the SNR with a scale), so no architecture change, no retraining; optionally a small LoRA fixes the model's low-res regime.

Numbers: **up to 7.09× wall-clock on FLUX.1-dev, 2.54× on WAN 2.1**, same step count, quality preserved on standard metrics; also works pixel-space (PixelGen) and on Z-Image / Qwen-Image.

Why not "just run low-res first" (cascades)? Cascades train separate models and a super-resolution stage; SPEED reuses one model and one trajectory, and the schedule is derived from measured statistics rather than tuned by hand. The control experiment that convinced me: a full-resolution staged box low-pass gives the same quality and **zero speedup** — the gain comes precisely from moving tokens out of the early steps.

</details>


### [10-02] Derive it for me. Under $x_t = (1-t)\,x_0 + t\,\epsilon$ and $P(f) = A f^{-\beta}$, at what flow time does frequency $f$ \"turn on\"?  (difficulty 2; tags: speed, derivation, spectrum, snr)

- Follow-up: What happens to frequencies with $P(f) < 1$? Do they ever get $\text{SNR} > 1$ before $t = 0$?


<details><summary>Answer</summary>

Work per orthonormal DCT bin. Signal power at frequency $f$ is $(1-t)^2 P(f)$; noise power is $t^2 \cdot 1$ because $\epsilon \sim \mathcal{N}(0, I)$ and an orthonormal transform keeps unit variance per bin. **A band "activates" when signal power equals noise power**:

$$
\begin{aligned}
(1-t)^2\,P(f) &= t^2 \\
(1-t)\,\sqrt{P(f)} &= t \\
t^*(f) &= \frac{\sqrt{P(f)}}{1 + \sqrt{P(f)}} = \frac{\sqrt{A}\,f^{-\beta/2}}{1 + \sqrt{A}\,f^{-\beta/2}}
\end{aligned}
$$

Properties I would point out:
- $t^*(f)$ is **monotonically decreasing in $f$**: low frequencies emerge first ($t$ near 1), high frequencies last ($t$ near 0). That is the coarse-to-fine ordering — it is a property of the data spectrum plus the interpolant, not of the network.
- $P(f) = 1$ is the **$\text{SNR}=1$ line at $t = 0.5$**; a bin with $P(f) < 1$ ($f$ above $A^{1/\beta}$) has $t^* < 0.5$. Every bin eventually activates because $t \to 0$ makes noise vanish, but the steepest part of the sigmoid-like curve happens in a narrow window of $t$, which is why a few stages capture most of the benefit.
- Steeper spectra (larger $\beta$) push high-frequency activation later, so video latents ($\beta \approx 2.4$) allow relatively more time at low resolution than FLUX ($\beta \approx 1.9$).

Trap the interviewer may set: if you used a variance-preserving DDPM interpolant, $x_t = \sqrt{\bar{\alpha}}\,x_0 + \sqrt{1-\bar{\alpha}}\,\epsilon$, the same argument holds with $\bar{\alpha} P(f) = 1 - \bar{\alpha}$; the only thing that changes is the time reparameterization. Also, \"$\text{SNR} = 1$\" is a convention — what matters is that the network cannot predict the bin better than its prior; $\delta$ (next question) absorbs that slack.

</details>


### [10-03] Why DCT rather than FFT? Isn't this a detail?  (difficulty 2; tags: speed, dct, signal-processing)

- Follow-up: What does DCT index $k$ correspond to in cycles per sample?


<details><summary>Answer</summary>

It is not a detail — it is what makes the resolution embedding clean.

- **Real-valued.** DCT-II of a real image is real; FFT gives complex coefficients with Hermitian symmetry, so "embed into the low-frequency corner" needs careful handling of conjugate pairs and the Nyquist bin. With DCT the low-frequency block is literally the top-left corner of the coefficient array.
- **Boundary handling.** DFT assumes periodic extension, so an image with different left/right edges has a jump that leaks energy into all high frequencies (spectral leakage). DCT-II assumes even-symmetric extension: no jump at the boundary, so energy compacts better into low frequencies — the fitted power law is cleaner and the embedding does not create seam artifacts.
- **Orthonormal ⇒ energy preserved (Parseval).** With the ortho normalization, $\|x\|^2 = \|\mathrm{DCT}(x)\|^2$ and i.i.d. unit Gaussian noise in pixel space is i.i.d. unit Gaussian in coefficient space. That is exactly why $P(f) = 1$ is the $\text{SNR}=1$ line and why \"fill new bins with $t \cdot \text{noise}$\" is statistically correct.
- **Frequency mapping.** DCT-II basis $k$ is $\cos\!\left(\frac{\pi k (2n+1)}{2N}\right)$, i.e. **index $k \leftrightarrow$ frequency $k/2$ cycles per $N$ samples** — half the FFT's $k/N$. So the top DCT index $N-1$ corresponds to Nyquist, and the low-res $N_s \times N_s$ block maps exactly onto the frequencies below the low-res grid's Nyquist. For the radial spectrum I use $f = \sqrt{k_x^2 + k_y^2}$ normalized to the full-res Nyquist so schedules are comparable across models.

Practical note: I compute the 2D (or 3D for video) DCT with separable orthonormal transforms along each axis; it costs $O(N \log N)$ and is negligible compared to one DiT forward.

</details>


### [10-04] Walk me through how the resolution schedule is derived from the fitted spectrum. What exactly does $\delta$ do, and how sensitive are results to it?  (difficulty 2; tags: speed, schedule, hyperparameters)

- Follow-up: How many stages, and who chooses them?


<details><summary>Answer</summary>

The pipeline is: **measure → fit → invert → add margin.**

1. Encode a few hundred images/videos with the model's VAE, take the orthonormal DCT of the latents, radially average $|\text{coef}|^2$ into a 1D $P(f)$ with $f$ normalized to the full-res Nyquist. Fit $\log P = \log A - \beta \log f$ by least squares over the mid band (exclude DC and the last few bins).
2. Choose candidate stage resolutions the model can actually run at (multiples of the patch size and VAE stride, e.g. 1/4, 1/2, 1 of native for FLUX). Each stage $s$ has Nyquist $f_s = r_s f_{\text{Nyq}}$ where $r_s$ is the resolution ratio.
3. Invert the activation formula: $t_s = t^*(f_s) = \frac{\sqrt{P(f_s)}}{1+\sqrt{P(f_s)}}$. The sampler runs at stage $s$ while $t > t_s$ — i.e. until the current grid's own Nyquist is about to carry signal — then transitions.
4. **$\delta$** is a small margin added to $t_s$ so the switch happens slightly *earlier* (larger $t$) than the $\text{SNR}=1$ crossing. Its purpose is to make sure the first high-frequency coefficients above the low-res Nyquist are still essentially pure noise when we synthesize them as $t\,\epsilon$; if you switch late, you are throwing away structure the model would already have started to commit to, and you also give the new band too few steps.

Sensitivity: results are flat over a reasonable $\delta$ range; too small $\Rightarrow$ slight softness (new band under-stepped); too large $\Rightarrow$ you leave speedup on the table since you spend more steps at full res. I report the schedule with the step allocation explicitly (e.g. how many of the 50 steps at each res) so people can see where the speedup comes from. The number of stages is a user choice constrained by the model: below some resolution the pretrained model is out of regime (the LoRA question), so in practice 2–4 stages.

</details>


### [10-05] Precisely what happens at a stage transition, and why would a naive bilinear resize of $x_t$ be wrong?  (difficulty 3; tags: speed, transition, dct, snr)

- Follow-up: Where does the $1/r_{\text{eff}}$ factor come from? Derive it.


<details><summary>Answer</summary>

At transition from an $N_{\text{src}}$-pixel grid to an $N_{\text{tgt}}$-pixel grid (per spatial dimension ratio $r = \sqrt{N_{\text{tgt}}/N_{\text{src}}}$ — for 2D, $r_{\text{eff}} = \sqrt{N_{\text{tgt}}/N_{\text{src}}}$ with $N$ counting all pixels):

1. **DCT** the current low-res state $x_t$ (orthonormal).
2. **Embed** those coefficients into the low-frequency corner of a zero-initialized $N_{\text{tgt}}$ DCT grid.
3. **Fill** all new high-frequency coefficients with $t\,\epsilon$, $\epsilon \sim \mathcal{N}(0,1)$ i.i.d. — because per the activation argument those bins are below $\text{SNR}=1$ and, marginally, $x_t$ there is just $t \cdot \text{noise}$ (the signal contribution $(1-t)\,x_0$ is negligible).
4. **Rescale** the embedded block by $\kappa$ (the $1/r_{\text{eff}}$ correction) and align the solver time to $\tilde{t}$ so the SNR of the embedded block matches what the target-resolution trajectory would have at that time.
5. **Inverse DCT** → the new full-res $x_t$, continue sampling.

Why $\kappa$? Orthonormal DCT of a constant patch of amplitude $c$ over $N$ pixels has DC coefficient $c\sqrt{N}$. Put that coefficient into a grid of $N_{\text{tgt}}$ pixels and inverse-transform: amplitude $c\sqrt{N_{\text{src}}/N_{\text{tgt}}} = c/r_{\text{eff}}$. So embedding **attenuates the low band by $1/r_{\text{eff}}$** relative to what a genuine target-res image would have; you must multiply by $\kappa = r_{\text{eff}}$ (per-band, the same factor for all embedded bins). But $\kappa$ scales the noise inside those bins too (it was $t\,\epsilon$ at low res, becomes $\kappa t \epsilon$), while the freshly filled bins have $t\,\epsilon$. The state is therefore not exactly on the $(1-t)\,x_0 + t\,\epsilon$ manifold; **$\tilde{t}$ is the effective time whose SNR matches the embedded block**, and I hand $\tilde{t}$ (not $t$) to the model / solver at the first full-res step [fill in: whether you renormalize the block to $(1-t)/t$ ratio or shift the solver time — state your exact choice and the formula].

Why not bilinear-resize $x_t$? (a) It is a low-pass in the wrong domain: the interpolation kernel attenuates the top of the low band unevenly (not a brick wall) and interpolates the *noise* too, making the noise spatially correlated — the model was trained on white noise, so it sees an off-distribution noise covariance and produces smooth, blurry output. (b) It gives no principled way to inject exactly the missing band at exactly variance $t^2$. The DCT route keeps the low band bit-exact and the new band exactly white.

</details>


### [10-06] You say continuous per-step frequency masking blurs but staged reveal doesn't. Why? And what did the box low-pass control experiment prove?  (difficulty 3; tags: speed, ablation, blur, frequency)

- Follow-up: Why not just use a smooth (Gaussian) mask instead of a brick wall?


<details><summary>Answer</summary>

I tried the "obvious" version first: at every step, radially reveal frequencies up to the current activation cutoff (a brick-wall mask on x_t or on the prediction). It **consistently blurs**. Two mechanisms:

- **Late frequencies are starved of steps.** With per-step reveal, the highest band is unmasked only in the last few steps of the schedule; the model has to synthesize fine texture in, say, 3 Euler steps at low noise where it is supposed to be doing small corrections. Result: under-resolved high frequencies = softness.
- **Brick-wall ringing.** A hard radial cutoff applied every step is a sinc kernel in the spatial domain; each step re-introduces Gibbs ringing at edges, the model partially "corrects" it, and the next mask re-introduces it — an oscillating fixed point that averages to blur and halo artifacts.

**Staged reveal** solves both: a whole band is unmasked once at a transition and then receives *all* remaining steps, and there is exactly one discontinuity, which is immediately buried under $t \cdot \text{noise}$ ($\text{SNR}<1$) so the model never sees a sharp edge in a band that matters yet. A smooth (Gaussian) mask helps ringing but not step starvation, and it also attenuates real signal in the pass band.

The control experiment: keep everything at full resolution and apply the *same* staged brick-wall low-pass at the same transition times. Quality **matches** the resolution-staged run — so the staged masking itself is quality-neutral — but wall-clock is **unchanged**, since token count never dropped. That decomposition is important: it shows (1) the coarse-to-fine schedule is harmless, and (2) 100% of the speedup comes from actually removing tokens. The one place staged resolution *beats* the full-res control is super-native generation (WAN 1.3B at 960p): 3.8× faster *and* better, because the model spends most steps at its native resolution instead of in RoPE extrapolation.

</details>


### [10-07] Training-free vs. the LoRA variant — what does the LoRA fix and how did you train it? Isn't "we needed fine-tuning" an admission the training-free story doesn't hold?  (difficulty 2; tags: speed, lora, training)

- Follow-up: Why LoRA and not full fine-tune? Why not fine-tune only at low res?


<details><summary>Answer</summary>

The training-free version already gives most of the speedup. The LoRA addresses one specific failure: **the low-resolution regime mismatch.** A model like FLUX.1-dev was trained mostly around 1 MP; at 1/4 resolution with the corresponding RoPE coordinates it is out of distribution — global composition is fine but it tends to produce fewer, larger objects and slightly wrong scale statistics at the first stage, and those errors are frozen in by the time we transition. For video, WAN at 240p is well out of its training distribution.

LoRA training: [fill in: exact rank, targets, steps, data] — the recipe was: sample data at the stage resolutions (downsample in the DCT domain the same way as inference, so the training distribution matches what the sampler sees), train a LoRA on attention projections with the standard flow-matching loss restricted to the timestep range where each resolution is used. The LoRA sees exactly the (resolution, t) pairs it will be asked about at inference, and nothing else; it does not need to learn the transition itself because the transition is analytic. Training is cheap (hours on a few GPUs) and the base weights are untouched, so it composes with other LoRAs.

Why LoRA: it cannot destroy the base model's full-res behaviour (which we rely on for the last stage), it is small enough to ship, and the target is a distribution-shift correction, which is low-rank in practice. Why not full fine-tune at low res: you would then have two models — that's a cascade, and you lose the single-model guarantee that the trajectory is consistent across stages.

So no, not an admission: the training-free result stands on its own for image models; the LoRA is a knob that pushes the earliest stage lower (more speedup) without losing quality.

</details>


### [10-08] How did you measure speedup fairly? And why 7.09× on FLUX but only 2.54× on WAN 2.1 — where does video's time actually go?  (difficulty 1; tags: speed, evaluation, benchmarks, video)

- Follow-up: What metrics did you use, and which ones would you distrust?


<details><summary>Answer</summary>

Fair measurement rules I used:
- **Wall-clock**, end-to-end on the same GPU, same batch, same number of sampling steps (50), same solver, same CFG, warm cache, median over runs. Not FLOPs, because attention efficiency and kernel launch overhead do not track FLOPs.
- Baseline is the model's own standard pipeline at native resolution; SPEED uses identical step count, so the only difference is where tokens are spent.
- Quality: standard benchmark metrics for the model class [fill in: e.g. GenEval / DPG / ImageReward / FID-style metrics for images, VBench for video], plus paired visual comparisons on the same seeds. I distrust FID at small n and any metric that rewards smoothness — the whole failure mode here is blur, so I always report a sharpness-sensitive metric or high-frequency energy ratio alongside.

Why FLUX ≈ 7× and WAN ≈ 2.5×:
- **Tokens removed is the lever.** FLUX at 1 MP is 4096 tokens with attention dominating; 1/4-res stages cut tokens by 16× and attention by ~256× — the early steps become almost free, and FLUX tolerates going down to very low res.
- **Video is spatiotemporal.** A *spatial* schedule reduces $H \times W$ but leaves the temporal token axis untouched; with 480p×81f WAN has ~30k tokens, and attention share is huge, but the spatial floor is higher (WAN below 240p is out of regime), so fewer/less-aggressive stages.
- **Fixed costs don't shrink**: 3D VAE decode, text encoder, CFG doubling the DiT calls (CFG 6 for WAN, two passes), and per-step overhead all stay constant, so Amdahl's law caps the ratio.
- The joint spatiotemporal extension recovers some of that (1.85× for a spatial-first 3-stage schedule on top of the spatial-only baseline), but the temporal axis is much more energetic so it activates early and cannot be reduced for long.

</details>


### [10-09] Convince me this isn't just cascaded diffusion, Matryoshka, progressive growing, frequency-domain diffusion, DeepCache, or Pyramid Flow with better PR.  (difficulty 3; tags: speed, related-work, novelty)

- Follow-up: What would you cite as the closest prior work, and what is the one experiment that separates you from it?


<details><summary>Answer</summary>

I'll take them in turn — the distinguishing claim is: **one pretrained model, one trajectory, a schedule derived from measured spectra, and a transition operator that is SNR-exact.**

- **Cascaded diffusion (Imagen, SR3, eDiff-I):** separate models per resolution, each trained; the low-res sample is a *conditioning input* to an SR model that starts from fresh noise. SPEED has no SR model and no restart: the low-res $x_t$ *is* the state that continues. Also cascades typically spend full steps at every stage (they are slower, not faster).
- **Matryoshka Diffusion:** trains a nested multi-res architecture jointly. Needs training from scratch or heavy fine-tuning; SPEED changes nothing about the model and is training-free.
- **Progressive growing (ProGAN-style / progressive distillation at res):** a *training* curriculum. SPEED is an *inference* schedule.
- **Frequency-domain / wavelet diffusion (WaveDiff, spectral diffusion, blurring diffusion):** they change the forward process or the representation the model is trained on. SPEED keeps the standard Gaussian forward process; the DCT is only used to *analyze* the process and to build the transition. My activation-time derivation is the same math that motivates blurring diffusion — but I use it to save compute, not to redefine the process.
- **DeepCache / TeaCache / feature caching:** reuse features across steps; orthogonal and composable (cache within a stage). Caching does not reduce tokens, so it can't touch attention cost the way resolution does; and it introduces approximation error every step.
- **Pyramid Flow (pyramidal flow matching):** the closest in spirit: multi-res stages in a single flow. But it *trains* the model with a pyramidal interpolant and renoising between stages; the transition is a learned/heuristic renoise. SPEED is post-hoc on an unmodified model, and the transition's scale/time alignment is derived from Parseval, not tuned. I would cite Pyramid Flow and blurring/frequency diffusion as closest prior work.

The separating experiment: the **full-res box low-pass control** (same quality, no speedup) plus the **training-free FLUX result** — no baseline above can be applied to a frozen FLUX with zero training and give a 7× wall-clock reduction at matched steps.

</details>


### [10-10] Where does SPEED fail? Give me the honest limitation list, and tell me which ones are fundamental versus fixable.  (difficulty 2; tags: speed, limitations, failure-modes)

- Follow-up: The VAE latent spectrum isn't a natural-image spectrum. Why does your argument still hold?


<details><summary>Answer</summary>

Failure modes I would volunteer before being asked:

1. **Text rendering and small high-contrast detail.** Glyphs are broadband; their layout is decided at low res where they don't exist yet, so letter placement and spelling degrade with aggressive schedules. Fixable by a later first transition (less speedup) or by the LoRA; fundamentally, anything whose *semantics* live in high frequencies gets less say in composition.
2. **Fine texture / hair / foliage** with very aggressive schedules: the last band gets fewer steps. Fixable via $\delta$ / step reallocation, and visible in sharpness metrics.
3. **Super-native resolution and RoPE.** Above the training resolution, the full-res stage extrapolates RoPE; SPEED actually *helps* here (WAN 1.3B at 960p: 3.8×, better quality) because most steps happen in-regime, but the final stage is still out of regime — that is a model limitation, not ours.
4. **Low-res floor.** Below ~1/4 native (image) or 240p (video) the pretrained model is out of regime; training-free stops there, LoRA extends it. Not fundamental but costs training.
5. **Latent spectra are not natural-image spectra.** True — the VAE's latent is a learned code with its own power law ($\beta \approx 1.9$ for FLUX vs $\approx 2$ for pixels; 2.4 for WAN) and channel-dependent structure. But the argument needs only two things: that the *measured* latent spectrum is monotone decreasing and roughly power-law, and that the *noise* is white in the same basis. Both hold, which is why I fit the spectrum per model rather than assuming the natural-image exponent. What does not transfer is "resolution = pixels": in latent space, a resolution change is a change of latent grid, and the VAE decoder's own receptive field sets the smallest useful grid.
6. **Few-step / distilled samplers.** With 4–8 steps the argument that early steps are "wasted" at full res weakens because each step already covers a wide $t$ range; you may have only one step per stage. This is the biggest open question for production (last question).
7. **Batching heterogeneity**: within a batch all samples must share the schedule; fine for T2I services, awkward for mixed workloads.

</details>


### [10-11] Video: how does the argument change with a temporal axis, and why does spatial-first win over temporal-first?  (difficulty 3; tags: speed, video, spatiotemporal, 3d-dct)

- Follow-up: What does "the temporal axis is 50× more energetic" mean concretely, and what does it imply for a temporal-only schedule?
- Follow-up: Why Euler with a step-index reset? What went wrong with UniPC?


<details><summary>Answer</summary>

Video latents have a **joint spatiotemporal spectrum**, and it is anisotropic. Fitting separately along each axis with frequency normalized to that axis's Nyquist gives **$\beta_s \approx 2.65$ (spatial) and $\beta_t \approx 1.57$ (temporal)**, and at an equal normalized frequency the temporal axis carries **~50× more power**. Concretely: the highest temporal frequency of a 16 fps latent still has substantial signal (motion is sharp in time; frames change abruptly), whereas the highest spatial frequencies are nearly noise. So temporal frequencies **activate early** (large $t$), which means you cannot keep the frame count low for long without corrupting motion.

Implementation: 3D orthonormal DCT (separable over $t, y, x$), the same embedding-into-a-corner trick in 3D, new coefficients filled with $t\,\epsilon$, $\kappa = \sqrt{N_{\text{tgt}}/N_{\text{src}}}$ counting all voxels, and **Euler with a step-index reset at each transition**. The solver point is a correctness bug I hit, not a preference: multistep solvers (FlowUniPCMultistep, DPM-Solver++) extrapolate from a history of previous model outputs; after a resolution change that history has the wrong shape and, worse, the wrong statistics — mixing a low-res prediction with a full-res one injects an error the size of the newly revealed band, giving artifacts or divergence at the first full-res step. Fix: reset the solver history/step index at every transition so the first step after a switch is first-order (UniPC degrades gracefully with no history), or use stateless Euler; and always hand the scheduler the aligned time $\tilde{t}$, not the old step index (same bug class in dynamic-shift FlowMatchEuler for Qwen-Image).

Stage orders (three stages each, 50 steps, WAN 2.1):
- **Spatial-first:** 240p·40f → 480p·40f → 480p·80f: **1.85×** over the spatial-only baseline.
- **Temporal-first:** 240p·40f → 240p·80f → 480p·80f: 1.56×.

Why spatial-first wins on compute: tokens are $T \cdot H \cdot W$; an early stage at low spatial res *and* low temporal res is the cheap regime, and the question is which axis you are allowed to keep small the longest. Because the temporal spectrum is so energetic, temporal Nyquist activates early, so the temporal-first order is forced to spend its middle stage at 240p·80f (twice the tokens of 240p·40f) while spatial-first sits at 480p·40f — same token count, but the spatial band is what actually activates around then. Equivalently: put the expensive dimension (time) back last, because the schedule can tolerate the temporal reveal being late only if there is real spatial detail to resolve, which there is. A purely temporal schedule barely helps — the 50× energy gap means the temporal Nyquist activates at $t$ close to 1.

</details>


### [10-12] You're at NVIDIA and asked to ship this in a production video model. What do you integrate with, what interacts badly, and where might the whole argument break?  (difficulty 3; tags: speed, production, scaling, distillation, cfg, fp8)

- Follow-up: With a 4-step distilled model, is there any speedup left?


<details><summary>Answer</summary>

What I would do in order:

1. **Composability first.** SPEED is orthogonal to feature caching (DeepCache/TeaCache), to sparse/linear attention, and to fp8 — each reduces per-step cost while SPEED reduces tokens in early steps. I'd stack them and measure, expecting roughly multiplicative gains since they touch different axes.
2. **CFG.** With classical CFG (two passes), the low-res stages are the cheap place to run guidance and the expensive full-res stage is where guidance-interval tricks (drop CFG at low noise) already apply; combining both gives the biggest saving. Note guidance changes the *effective* spectrum of the prediction (it sharpens), so re-check the transition time with CFG on.
3. **Batching and shapes.** Stage resolutions must be multiples of patch × VAE stride, and every stage's shape must be a compiled/cuda-graph shape; I would fix 2–3 stage shapes per product resolution and precompile. Serving systems that pad to fixed shapes lose part of the benefit — with continuous batching, group requests by stage.
4. **fp8 / quantization.** The low-res stages run at high noise where activations are noise-dominated; fp8 there is benign. Quantization error at the first full-res step could bias the newly injected band — verify with the sharpness metrics.
5. **Training-side.** In a production model I'd fold the LoRA into pretraining as a multi-resolution, timestep-conditioned curriculum (each resolution only sees its t range), so the model is natively in-regime at low res and the schedule could go lower.

Where it can break: **few-step distilled models.** With 4 steps the sampler doesn't "spend" early steps — each step spans a huge $t$ range, and the distilled model was trained to jump from noise to data at full res; its predictions at the first step already contain committed high frequencies (distillation collapses the coarse-to-fine ordering). Options: distill *with* the SPEED trajectory (teacher runs staged, student learns staged, so step 1 is a low-res step), or accept that speedup for a 4-step model is $\le 4/3\times$ and focus SPEED on the 20–50 step quality tier and on video, where step counts remain high. I would say that plainly rather than oversell.

</details>


### [10-13] In one sentence, what does SPEED do?  (difficulty 1; tags: speed, rapid-fire, pitch)


<details><summary>Answer</summary>

**SPEED denoises at low resolution early and raises resolution along the trajectory**, using the latent power spectrum to decide exactly when each frequency band is worth computing, so a pretrained diffusion/flow model spends most of its steps on far fewer tokens with no architecture change. The why: high frequencies are pure noise at high $t$, so full-resolution steps there are wasted attention. Result: up to 7.09× wall-clock speedup on FLUX.1-dev and 2.54× on WAN 2.1 with quality preserved, training-free or with a light LoRA. Follow-up to expect: "how do you pick the switch times?" — from the fitted spectrum $P(f) = A f^{-\beta}$, not by hand.

</details>


### [10-14] What is a power spectrum, and how did you measure it for a latent diffusion model?  (difficulty 1; tags: speed, rapid-fire, spectrum, signal-processing)


<details><summary>Answer</summary>

The **power spectrum is the energy of a signal per frequency**: take a 2D transform (DCT or FFT), square the coefficients, and average over rings of equal radial frequency $f$ to get a 1D curve $P(f)$. I measured it on clean latents $x_0$ from the VAE encoder over a few thousand images, with an **ortho-normalized** transform so that unit Gaussian noise has power exactly 1 in every bin. Natural images and their latents follow a power law, $P(f) = A f^{-\beta}$: $\beta \approx 1.92$ for FLUX, 2.42 for WAN 2.1, 2.45 for PixelGen, 2.24 for Qwen-Image. Trap: normalization matters — without ortho scaling, \"$P(f) = 1$ is the noise floor\" is false and the schedule is off.

</details>


### [10-15] What is the DCT, in one minute?  (difficulty 1; tags: speed, rapid-fire, dct, signal-processing)


<details><summary>Answer</summary>

The **discrete cosine transform expresses a signal as a sum of cosines** of increasing frequency; it is a real, orthogonal linear map (DCT-II is the JPEG one), so it is its own inverse up to transpose and preserves energy (Parseval). Why I use it instead of the FFT: it is real-valued, its implicit even-symmetric extension avoids the wrap-around discontinuity of the FFT so there is no ringing at image borders, and its coefficients are laid out on a plain 2D grid where low frequencies sit in one corner — which makes "embed a low-res grid into the low-frequency corner of a high-res grid" a literal array copy. Follow-up: energy scaling when you change grid size (the $1/r_{\text{eff}}$ factor).

</details>


### [10-16] What does it mean that "a frequency activates"?  (difficulty 1; tags: speed, rapid-fire, snr, flow-matching)


<details><summary>Answer</summary>

Under the interpolant $x_t = (1-t)\,x_0 + t\,\epsilon$, the signal at frequency $f$ has power $(1-t)^2 P(f)$ and the noise has power $t^2$ (unit per bin). A frequency **activates at the flow time $t^*$ where signal first exceeds noise**, i.e. $(1-t)^2 P(f) = t^2$, so $t^*(f) = \frac{\sqrt{P(f)}}{1 + \sqrt{P(f)}}$. Because $P(f)$ decays as $f^{-\beta}$, low frequencies activate early (at high $t$) and high frequencies activate late. Before its activation time a band is essentially noise and the model cannot extract information from it, so computing it is wasted. This is the whole justification for the schedule. Follow-up: it is $\text{SNR} = 1$, a threshold, not a hard cutoff; $\delta$ adds margin.

</details>


### [10-17] What is a stage transition?  (difficulty 1; tags: speed, rapid-fire, transition, dct)


<details><summary>Answer</summary>

A **stage transition is the moment we jump from resolution $r_k$ to $r_{k+1}$ mid-trajectory**. Mechanically: DCT the current low-res noisy latent, copy it into the low-frequency corner of the larger DCT grid, fill the new high-frequency coefficients with fresh noise scaled by $t$, inverse-DCT, and continue denoising at the higher resolution. A $\kappa$ / $\tilde{t}$ correction fixes the SNR, because embedding into a bigger grid attenuates the signal by $1/r_{\text{eff}}$ with $r_{\text{eff}} = \sqrt{N_{\text{target}}/N_{\text{source}}}$. The transition is triggered when the current stage's Nyquist frequency activates. Trap: bilinear-resizing $x_t$ instead would also blur the noise, breaking the marginal the model was trained on.

</details>


### [10-18] What is the difference between the training-free mode and the LoRA mode?  (difficulty 1; tags: speed, rapid-fire, lora, training)


<details><summary>Answer</summary>

**Training-free** means we run the pretrained model unchanged and only change the sampler: the resolution schedule, the DCT transition, and the SNR correction. It already gives most of the speedup because the transition is designed to land on the model's own training marginal. **LoRA mode** additionally fine-tunes low-rank adapters on the staged trajectory so the model sees low-resolution noisy inputs at high $t$ and mixed states right after a transition, which cleans up residual artifacts (slight texture mismatch right after a jump) and recovers the last bit of quality. Same architecture, same inference code; the LoRA is optional. Follow-up: "so training-free does not fully hold?" — it holds; the LoRA closes a small gap, it does not enable the method.

</details>


### [10-19] What does $\delta$ control?  (difficulty 2; tags: speed, rapid-fire, hyperparameters, schedule)


<details><summary>Answer</summary>

**$\delta$ is a margin on the activation criterion that controls how early each transition happens.** The rule is \"switch to the next resolution when the current stage's Nyquist frequency activates\"; $\delta$ shifts that threshold so we switch slightly before $\text{SNR} = 1$ rather than exactly at it, giving the new band a few steps of headroom before it carries signal. Larger $\delta$ means earlier transitions: more steps at high resolution, safer quality, less speedup. Smaller $\delta$ means later transitions and more speedup, until the highest band is starved of steps and detail goes soft. It is the one knob on the speed–quality curve; everything else follows from the fitted $\beta$. Trap: one scalar shared across stages, not per-stage tuning.

</details>


### [10-20] What was the speedup, and on which models?  (difficulty 1; tags: speed, rapid-fire, benchmarks, results)


<details><summary>Answer</summary>

**Up to 7.09× wall-clock on FLUX.1-dev (latent image) and 2.54× on WAN 2.1 (latent video)**, with quality preserved on standard metrics; also evaluated on Z-Image (latent image) and PixelGen (pixel-space image), and I later ported it to Qwen-Image. Video speedup is lower because tokens scale with frames as well as pixels and the 3D VAE / non-attention costs are a larger fraction of time; a 3-stage spatiotemporal schedule (240p40f→480p40f→480p80f) gave 1.85× on WAN. At super-native resolution (WAN 1.3B at 960p) it was 3.8× and better quality than full-res, since most steps run at native resolution. All numbers are measured wall-clock at 50 steps, same solver and CFG as baseline.

</details>


## Project Grill: Foveated Imaging & Foveated Diffusion


### [11-01] Pitch the SIGGRAPH foveated imaging paper in two minutes. What is the problem, what is the policy, what is the headline number?  (difficulty 1; tags: foveated, pitch, active-perception)

- Follow-up: Why is this a graphics/imaging paper and not a robotics paper?


<details><summary>Answer</summary>

**Problem: a 200-megapixel sensor produces far more pixels than any interface, ISP, or downstream network can consume at frame rate, yet most of those pixels are irrelevant to the task at hand.** Reading the whole sensor at full resolution is bandwidth-bound; uniform downsampling throws away the small, distant, or text-bearing detail that the task needs.

**Solution: policy-based foveation.** The sensor is read in two streams — a coarse global view and one or more high-resolution regions of interest (ROIs) — and a learned policy decides, every frame, where the ROIs go, conditioned on the coarse view, the previous ROIs and the task. The policy learns saccade-like behaviour (jump to a new target) and pursuit-like behaviour (track a moving one) without being told to. It is trained in simulation against task reward and transferred to a real 200 MP sensor prototype with a readout that supports programmable ROIs.

Headline: **task performance matched to full-resolution readout while consuming < 1/8 of the pixel bandwidth**, across three tasks: object tracking, text recognition (OCR) and MuJoCo robotic manipulation, and demonstrated on hardware.

Why imaging/graphics rather than robotics: the contribution is a **sensing** primitive — closing the loop between a task and where the sensor spends its bandwidth — analogous to foveated rendering in VR (spend shading where the eye looks), just on the capture side. The robotics tasks are downstream consumers that make the benefit measurable.

</details>


### [11-02] Formulate it as a sequential decision problem. State, action, reward, horizon — and why is it sequential at all rather than a per-frame saliency map?  (difficulty 2; tags: foveated, rl, formulation, pomdp)

- Follow-up: Is this a POMDP? What is the hidden state?


<details><summary>Answer</summary>

It is a **POMDP**: the true scene at full resolution is the hidden state; the agent only observes what it chose to read.

- **Observation:** the coarse global frame (e.g. full sensor downsampled by a large factor), the previously read high-res ROI crops, and a recurrent/temporal memory of previous observations [fill in: exact memory — recurrent state, stack of past crops, or feature buffer]. Optionally the task context (target description, OCR query, robot proprioception).
- **Action:** the ROI parameters for the next frame — centre $(x, y)$ and, if supported, scale / number of ROIs — under a hard bandwidth budget (ROI pixels + global pixels $\le B$ per frame). Continuous in the simulator; quantized to the sensor's readout grid on hardware.
- **Reward:** downstream task metric — tracking IoU / centre error, OCR character accuracy, manipulation success or dense distance-to-goal — possibly minus a small penalty for large ROI jumps (readout latency) [fill in: exact shaping terms].
- **Horizon:** episodic (a tracking clip, a manipulation episode); the policy acts every frame.

Why sequential and not a saliency map: (1) **the information needed to place the ROI is itself only available at high resolution** — a small distant object or a line of text is invisible in the coarse view, so the agent must *search* (saccade) then *hold* (pursue); that is a memory-dependent strategy, not a function of the current frame. (2) The reward is delayed: reading the wrong region now costs you the next several frames of tracking. (3) Bandwidth budget across time is a constraint a per-frame heuristic cannot reason about. Saliency is one of our baselines and it loses precisely on small targets and text.

</details>


### [11-03] How is the policy trained, and why that choice? I'll push: why RL instead of supervised learning from an oracle that knows the full-res frame?  (difficulty 3; tags: foveated, rl, imitation, training)

- Follow-up: How did you deal with credit assignment for a reward that only arrives at the end of a manipulation episode?


<details><summary>Answer</summary>

[fill in: the exact algorithm and architecture from the paper — e.g. PPO / SAC / DAgger-style imitation, the policy backbone, and training budget. Adapt the reasoning below to what you actually did.]

The argument I'd make for **RL with task reward, warm-started or regularized by an oracle where one exists**:

- **An oracle exists for some tasks, not others.** For tracking, the simulator knows the target's location, so "put the ROI on the target" is a perfect supervised label — behaviour cloning from that oracle is fast and stable. For OCR the oracle is "wherever the text is", also known in sim. For manipulation there is **no oracle for where to look** — the policy must discover that looking at the gripper–object contact, then at the goal, is what helps; only task reward defines that.
- **Imitating an oracle teaches you to look where the answer is, not how to *find* it.** The oracle never needs to search; a cloned policy therefore never learns the saccade behaviour needed when the target is lost or initially unknown, and it fails on exactly the frames where foveation matters. RL (or DAgger-style on-policy correction) exposes the policy to its own mistakes.
- **Reward shaping:** tracking uses dense per-frame IoU / negative distance; OCR uses per-step character accuracy of a frozen recognizer on the ROI; manipulation uses the downstream controller's success plus dense progress terms; all plus a bandwidth penalty if the budget is soft. Because the *perception* policy and the *control* policy can be decoupled (the controller consumes the foveated observation), I could keep the controller fixed and train only the sensing policy, which massively simplifies credit assignment.
- **Credit assignment for sparse rewards:** short effective horizon via discounting and dense proxy rewards (e.g. does the downstream perception module's estimate improve after this ROI), plus the pre-training from the oracle for tasks that have one.

Honest weaknesses to volunteer: RL is sample-hungry (fine in MuJoCo, not on hardware), and rewards from a frozen downstream model make the policy overfit that model's failure modes. That is the "why not X" I would expect, and the answer is: the oracle/imitation baseline *is* in the paper's comparison [fill in: numbers], and RL wins where search is required.

</details>


### [11-04] Describe the MuJoCo manipulation setup and, honestly, how much of the sim result actually transferred to the real 200 MP prototype.  (difficulty 2; tags: foveated, mujoco, sim-to-real, hardware)

- Follow-up: What is the largest sim-to-real gap for a *sensing* policy, as opposed to a control policy?


<details><summary>Answer</summary>

Simulation: a MuJoCo scene rendered at very high resolution (emulating the 200 MP sensor's field of view), with a robot arm doing [fill in: the task — e.g. pick-and-place / peg insertion] where the relevant object is small in the global view. The foveated observation (coarse global + ROI crops) is what the manipulation controller sees; the sensing policy chooses the ROI per step. The controller is [fill in: a fixed pre-trained visuomotor policy or a scripted/IK controller with a learned perception head]. Metric: task success at a given pixel-bandwidth budget, compared to full-res and to uniform downsampling at the same budget.

Sim-to-real: the hardware prototype is a 200-megapixel sensor with programmable ROI readout [fill in: sensor / interface details], and we ran tracking and text recognition on it live [fill in: which tasks ran on hardware; whether manipulation was hardware or sim-only]. What transferred well: the *behaviour* — search-then-lock, pursuit on moving targets, revisiting text — because the policy operates on coarse images whose statistics we can match with randomization (blur, noise, exposure, colour) and the ROI is a geometric action that is exact on hardware.

The largest gap for a sensing policy is **timing, not appearance**: on hardware the ROI you request is applied to the *next* frame after readout and transport latency, so the target you pointed at has moved. In sim the action is instantaneous unless you model that. We had to [fill in: model a one-frame action delay in sim / predict the target's next position]. The second gap is the readout constraint set (ROI grid alignment, max number of windows, fixed row bandwidth) which the simulator must enforce exactly or the policy learns unrealizable actions.

</details>


### [11-05] What are the baselines, and how exactly is "< 1/8 of the bandwidth" measured? Convince me the comparison isn't rigged.  (difficulty 2; tags: foveated, baselines, evaluation, bandwidth)

- Follow-up: What happens to the curve as you push the budget to 1/32 or 1/64?


<details><summary>Answer</summary>

**Bandwidth is counted in pixels read from the sensor per frame** (global stream + all ROI streams), which is the quantity that limits the sensor interface, the ISP and any downstream network. Full-res is 200 MP per frame; our operating point reads ≤ 1/8 of that. It is a hard per-frame budget applied identically to every method — no method gets to read more pixels than another.

Baselines at the *same* budget:
- **Uniform downsampling:** the whole sensor at 1/8 the pixels — the industry default. Loses small targets and text.
- **Random ROI:** same global stream plus randomly placed high-res windows — controls for "any high-res crop helps".
- **Saliency / detector-driven ROI:** place the ROI on the most salient region of the coarse view (a strong non-learned heuristic).
- **Center-fixed ROI** (a fixed fovea, the naive foveation).
- **Oracle:** ROI placed on the ground-truth target — an upper bound telling us how much of the gap the learned policy closes.
- **Full-resolution readout:** the quality ceiling, at 8× the bandwidth.

The rigging checks I would offer: same downstream perception/control model for all methods; task-specific metrics that punish detail loss (character accuracy, not word-level fuzzy match; centre error in full-res pixels); reporting a *curve* of accuracy vs. budget rather than one point; and reporting the learned policy against the oracle so the reader can see the ceiling.

Behaviour of the curve: performance is flat down to roughly 1/8, then the global stream becomes so coarse that the policy can no longer *find* targets and search time dominates — the knee is set by the detectability of targets in the global view, not by ROI size. That is exactly the argument for a multi-scale (more than two levels) readout as future work.

</details>


### [11-06] On real hardware, the policy's action affects the *next* frame. What closed-loop and latency problems did you hit and how did you handle them?  (difficulty 3; tags: foveated, latency, hardware, control)

- Follow-up: What is the maximum target speed you can track, and what sets it?


<details><summary>Answer</summary>

The loop per frame is: read global + ROI → policy inference → send ROI coordinates → sensor applies them to the next exposure → readout. Three latency sources: exposure/readout of a 200 MP sensor (even at reduced bandwidth, the row scan is slow), transport, and policy inference. Together they mean the policy is always acting on an observation that is at least one frame old.

What this did to a policy trained with an instantaneous simulator: on fast targets it **lags and oscillates** — it places the ROI where the target *was*, loses it, saccades, re-acquires, and the effective tracking accuracy collapses. Fixes, in order of how much they helped [fill in: which you actually used]:
1. **Model the delay in simulation:** the action is applied $k$ frames later, with $k$ matched to the measured hardware pipeline. The policy then learns to lead moving targets (predictive pursuit), which is what biological smooth pursuit does.
2. **Lightweight policy** so inference is a small fraction of the frame time; the global view is small, so this is easy.
3. **Slightly larger ROI than the target** as a margin — a bandwidth/robustness trade-off that is exactly the fovea/parafovea structure.
4. Asynchronous pipeline: policy runs on frame $n$ while frame $n+1$ is exposing.

Maximum trackable speed is set by $(\text{ROI margin}) / (\text{loop latency})$: a target must not leave the ROI within one loop delay. Increasing the ROI helps linearly but eats budget; reducing latency helps everywhere. On the prototype the binding constraint was sensor readout time, not the network.

</details>


### [11-07] Connect this to biological foveation and to the active-perception literature. What did you borrow, and what is different from the classic "active vision" work?  (difficulty 2; tags: foveated, biology, active-perception, related-work)


<details><summary>Answer</summary>

**Borrowed from biology:** the retina samples with a high-acuity fovea and a low-acuity periphery, and the brain compensates with eye movements — **saccades** (fast ballistic jumps to a new target selected from the periphery) and **smooth pursuit** (keep a moving target on the fovea). Our two-stream readout is the same architecture: coarse global view = periphery, ROI = fovea; and the learned policy reproduced saccade/pursuit behaviour without those being engineered, which is a nice sanity check that the objective (task reward under a bandwidth budget) is the right one — the visual system is solving the same optimization with a fixed optic-nerve bandwidth.

**Active perception / active vision** (Bajcsy, Aloimonos, Ballard's "animate vision", later Mnih's recurrent attention model, hard-attention / glimpse networks): the idea that the perceiver chooses its observations to reduce task uncertainty. Recurrent attention models are the closest ML ancestor: a glimpse policy trained with REINFORCE on classification. Also foveated rendering in graphics, which is the *display-side* dual.

What is different:
- The constraint is a **real hardware readout budget** on a real ultra-high-res sensor, not a synthetic glimpse on a 28×28 image; actions must be realizable on the sensor's ROI grid, and latency matters.
- **Multiple downstream tasks with a fixed controller**, including closed-loop manipulation, rather than static classification.
- The bandwidth-accuracy trade-off is quantified as a curve, with an oracle upper bound.
- Sim-to-real for a *sensing* policy, which brings the delay-modelling issues that classic active-vision papers did not face.

The honest open question: biological vision also has a smooth eccentricity-dependent resolution falloff and multiple scales; ours is two-level. That is where the foveated diffusion work's multi-scale tokenization ideas come back in.

</details>


### [11-08] Foveated Diffusion: how do you actually put mixed-resolution tokens into a pretrained DiT? Patchify, position embeddings, attention cost — and why does a LoRA suffice?  (difficulty 3; tags: foveated-diffusion, dit, tokens, rope, lora)

- Follow-up: Show me the cost model that gives 2× on images and 4× on video.


<details><summary>Answer</summary>

Setup: FLUX.2 and WAN 2.1, unchanged architecture, LoRA post-training only.

**Patchify at two scales.** The latent is split into a foveal region and a periphery (a mask). Foveal latents are patchified at the native patch size $p$; peripheral latents at $2p$ (image) or a larger spatiotemporal patch (video), so a peripheral token covers 4× (image) or up to 8×+ (video) the area of a foveal token. The patch-embedding linear layer for the coarse scale is [fill in: a new small projection trained with the LoRA / the native projection applied after average-pooling the latent]. Tokens from both scales are concatenated into one sequence; the transformer is agnostic to their origin.

**Position embeddings.** FLUX and WAN use 2D/3D RoPE indexed by latent coordinates. Each token gets the RoPE of its **patch centre in the full-resolution coordinate frame** — a coarse token at rows 8–11 gets position 9.5. That keeps relative distances between foveal and peripheral tokens geometrically correct, which is what RoPE encodes; the model only needs to learn that some tokens are "blurrier", which is a small distribution shift — hence LoRA. [fill in: whether you also add a scale indicator.]

**Cost model.** Let $N$ be the full-res token count and $a$ the foveal area fraction, periphery at $1/s$ tokens per area. Tokens $N' = N\,(a + (1-a)/s)$. Attention $\propto N'^2$, MLP $\propto N'$. Image, $a = 0.25$, $s = 4$: $N' = 0.44N \to$ attention $\approx 0.19\times$, MLP $\approx 0.44\times$; with attention $\approx$ half the FLOPs at 4k tokens, total $\approx 0.3\times$, minus overhead $\approx$ **2× wall-clock**. Video, $s = 8$ spatiotemporal and attention dominating at ~30k tokens: $N' \approx 0.34N$, attention $\approx 0.12\times \to$ **~4×**. Attention share is why video gains more.

**Why LoRA is enough.** The pretrained model already generates the periphery content; what changes is the *input/output statistics* of coarse tokens (they look like a low-passed latent) and their unusual density in the sequence. That is a low-rank adaptation of the projections, learned in hours from the model's own data with a random fovea. The final image is assembled by upsampling the periphery's coarse latent (it is genuinely lower resolution there) and blending at the boundary before VAE decoding.

</details>


### [11-09] Where does the fovea come from at generation time, what does the periphery look like, and where does foveated diffusion fail? Then: how would you combine it with SPEED?  (difficulty 3; tags: foveated-diffusion, speed, failure-modes, combination)


<details><summary>Answer</summary>

**Fovea source.** Three practical choices: (1) **user gaze** — the VR/AR case, the periphery is literally not looked at, so its lower quality is invisible; (2) **saliency / subject mask** from the prompt or a first low-res pass — put the fovea on faces, text, the main object; (3) **text-driven** — the prompt (or a VLM) names what deserves detail. In the paper we [fill in: which of these you evaluated and how the fovea was sampled during training — random rectangles / random positions and sizes].

**Periphery quality.** It is a genuinely lower-resolution generation: coarse tokens mean the model can only represent frequencies up to the periphery's Nyquist. Upsampled to full res it looks like a mild blur, which is fine for backgrounds, sky, out-of-focus regions, and for gaze-contingent display; it is not fine for a second face in the periphery. Boundary handling matters: without care you get a visible seam of sharpness change; blending tokens across scales at the border and having the LoRA see mixed boundaries during training removes it.

**Failure cases.** Small important content outside the fovea (a second subject, text) is rendered coarsely; semantic consistency across the boundary can slip (an object crossing the border is sharp on one side); a wrong fovea choice is unrecoverable within the sample; strong periphery downsampling on video can produce temporally flickering coarse regions.

**Combining with SPEED.** They are complementary axes: SPEED is *temporal* allocation (resolution as a function of denoising time), foveated diffusion is *spatial* allocation (resolution as a function of location). The natural combination: run SPEED's early stages uniformly at low resolution — at high noise nobody has high frequencies anyway, so foveation buys nothing there — and only in the last stage(s), when high frequencies activate, switch to the mixed-resolution token layout with the fovea at native res. In DCT terms, the transition embeds the low-res state and fills the new band with $t \cdot \text{noise}$ **only inside the fovea**; the periphery simply never gets a high band. Speedups should roughly multiply (SPEED cuts early steps, foveation cuts late steps), which is the experiment I would run first. The wrinkle is that the LoRA has then to handle both the low-res regime and the mixed-res layout — a joint LoRA with both augmentations.

</details>


### [11-10] You are on Cosmos / Isaac at NVIDIA. What is the follow-up to these two papers, and what should the sensor and the world model share?  (difficulty 2; tags: foveated, nvidia, cosmos, isaac, future)

- Follow-up: Which of the two papers is more useful to a robot, and why?


<details><summary>Answer</summary>

The thread connecting both papers is **allocate resolution where and when it carries information** — sensing side and generation side. The follow-up I would propose:

1. **Foveated world models.** A video world model (Cosmos-style) for a robot does not need uniform resolution: the gripper–object contact region needs detail, the rest of the scene does not. Combine foveated diffusion tokens with an *action-aware fovea* (predicted from the policy's attention or from proprioception), and SPEED-style schedules for the denoising. That gives faster imagination rollouts, which matters for model-based planning where you roll out many futures.
2. **Learned sensing + world model in the loop.** The foveated-imaging policy decides where to read; the world model predicts what will be there. A world model with an uncertainty estimate is an excellent reward signal for the sensing policy: look where prediction is worst — this is exactly the surprise idea I have used for KV-cache compression, and it does not need an oracle.
3. **Isaac Sim / Lab** as the simulator: physically-based rendering at high res, programmable virtual sensors with realistic readout and latency models, and thousands of parallel environments for RL — solving the sample-efficiency problem I had in MuJoCo. The sim-to-real recipe (delay modelling, readout constraints) carries over; the appearance gap is smaller with a proper renderer.
4. **Synthetic data**: a foveated generator can produce high-detail training data for exactly the regions a perception model is weak on, at a fraction of the cost.

Which is more useful to a robot: the sensing policy, immediately — bandwidth and latency are the binding constraints on any embodied system with high-res cameras, and it plugs in front of an existing VLA. Foveated/spectral generation matters once the robot *uses* a world model in the loop, where imagination cost dominates. I would argue NVIDIA is in the rare position to own both the sensor readout path and the model, and that is exactly where the two papers meet.

</details>


### [11-11] What is foveated imaging, in one sentence?  (difficulty 1; tags: foveated, rapid-fire, active-perception)


<details><summary>Answer</summary>

**Foveated imaging reads a small region of the sensor at full resolution and the rest at low resolution, and moves that region over time, like the human eye's fovea and saccades.** The why: a 200-megapixel sensor cannot be read out, transmitted, or processed at full rate, but most of those pixels are irrelevant for any given task at any given moment; a task-driven fovea gets the useful pixels for a fraction of the bandwidth. Follow-up you will get: "how is that different from cropping a region of interest?" — the difference is that we also keep a coarse global view, and the ROI is chosen by a learned policy that acts sequentially rather than by a fixed heuristic.

</details>


### [11-12] What is the "policy" in your paper?  (difficulty 1; tags: foveated, rapid-fire, rl, policy)


<details><summary>Answer</summary>

The **policy is a learned network that, at each timestep, looks at the current coarse global view plus the previous foveal crop and outputs where to place the next high-resolution region**. It is the decision-maker in a sequential problem: state is what has been observed so far, action is the fovea location (and size), reward is downstream task performance under a bandwidth budget. It is trained with RL / policy learning rather than a per-frame saliency map because good behavior is temporal — pursuit to track a moving object, saccades to re-check something uncertain — and depends on what the task needs next. Follow-up: it is separate from the task network, so it plugs in front of an existing model.

</details>


### [11-13] What does "bandwidth" mean here, and what was the number?  (difficulty 1; tags: foveated, rapid-fire, bandwidth, evaluation)


<details><summary>Answer</summary>

**Bandwidth is the number of pixels read off the sensor per frame** (equivalently bits per second from readout to compute), the resource that limits ultra-high-resolution sensors: readout, transmission, and inference all scale with it. Our foveated policy keeps task performance on manipulation, tracking, and text recognition using **less than 1/8 of the full-resolution pixel bandwidth**; the budget counts the coarse global view plus the high-res fovea, so the comparison is against reading the whole 200-megapixel frame. Trap: the honest baseline is not just "full-res vs ours" but also uniform downsampling and a fixed-center or saliency fovea at the same budget, which is what we show.

</details>


### [11-14] What simulator did you use, and for which task?  (difficulty 1; tags: foveated, rapid-fire, mujoco, simulation)


<details><summary>Answer</summary>

**MuJoCo**, for robotic manipulation: a simulated arm with a camera where the policy chooses which region to sense at high resolution while a downstream controller does the task, so fine detail (a small object, a grasp point) is only paid for where it matters. We also validated on object tracking and text recognition, and on a real 200-megapixel sensor prototype. Why MuJoCo: fast, deterministic physics with a controllable renderer, so we can generate large amounts of policy rollouts cheaply and control the bandwidth accounting exactly. Follow-up: sim-to-real — the sensing policy transferred better than a control policy would, because it only has to decide where to look, and the rendering gap is smaller than the dynamics gap.

</details>


### [11-15] What is foveated diffusion, in one sentence?  (difficulty 1; tags: foveated-diffusion, rapid-fire, dit, efficiency)


<details><summary>Answer</summary>

**Foveated diffusion generates an image or video with a pretrained diffusion transformer using full-resolution tokens only in a foveal region and coarse tokens in the periphery, cutting the quadratic attention cost.** Since attention cost scales with the square of token count, replacing most tokens with a few coarse ones gives a large saving; the model is adapted with a LoRA post-training on FLUX.2 and WAN 2.1, giving about 2× image and 4× video wall-clock speedup. Follow-up: "where does the fovea come from at generation time?" — from a user or task-specified region of interest, or from attention/saliency, and the periphery is upsampled and blended.

</details>


### [11-16] What is a mixed-resolution token?  (difficulty 2; tags: foveated-diffusion, rapid-fire, tokens, patchify)


<details><summary>Answer</summary>

A diffusion transformer patchifies the latent into fixed-size tokens; a **mixed-resolution token sequence uses small patches (many tokens) in the fovea and large patches (few tokens covering more area) in the periphery**, so one sequence contains tokens at different spatial scales. Two things must be handled: position embeddings, since a coarse token covers a region rather than a point, so its RoPE position is the region's center at a scale consistent with the fine tokens; and patchify/unpatchify at multiple scales, so the periphery is decoded at low resolution and upsampled. Weights are shared; a LoRA teaches the model to accept the mixed sequence. Trap: this is a token-count reduction, so the speedup is on attention, not on the VAE.

</details>


## Project Grill: Single-Photon Imaging, LLM Work & Math


### [12-01] Give me the two-minute pitch for the opportunistic single-photon time-of-flight paper, and tell me exactly what you contributed.  (difficulty 2; tags: spad, tof, computational-imaging, project)

- Follow-up: why timestamps instead of histograms?
- Follow-up: what does "opportunistic" buy you over a pulsed laser?


<details><summary>Answer</summary>

**Pitch.** A SPAD is a pixel that detects individual photons and timestamps each arrival with ~tens-of-picosecond resolution. Classic single-photon ToF fires a laser pulse and histograms photon arrival times against the pulse; the peak position gives round-trip time and hence depth. **Opportunistic ToF** drops the controlled laser: we recover depth from **ambient or uncontrolled light sources** — flickering/modulated lights, a screen, an unsynchronized source — whose temporal structure we do not control and often do not know in advance. The key idea is that if a source has *any* temporal structure, the scene returns are a delayed copy of it, so depth is encoded in the **cross-correlation** between a reference measurement of the source waveform and each pixel's photon stream: $\text{depth} = c \cdot \tau^*/2$ where $\tau^* = \arg\max$ of the correlation. Ultra-wideband sources (very short temporal features) sharpen that correlation peak, which is where the USRA-funded ultra-wideband work comes in.

**Histogram vs timestamp processing.** Histogramming bins arrivals into a fixed grid and then peak-finds — simple, but you throw away sub-bin timing and you must know the repetition period. **Working directly on raw timestamps** (per-photon likelihood or correlation against a continuous reference) keeps full timing precision, handles non-periodic sources, and lets you do MLE under a Poisson model rather than a heuristic peak-fit.

**My contribution** (say it in the first person, concretely): [fill in: e.g., built the timestamp-domain correlation / MLE pipeline, the pile-up-aware forward model, hardware capture experiments with the SPAD array, or the simulation and evaluation harness]. Be ready to name one specific experiment you ran and one number you own.

Common trap: don't say "we used a SPAD camera" and stop — interviewers want the *signal model*: photon arrivals are an inhomogeneous Poisson process with rate $\lambda(t) = \alpha \cdot s(t - \tau) + b$, and everything downstream is inference on $\tau$.

</details>


### [12-02] Explain pile-up in single-photon detection. How does it bias depth, and what are the fixes?  (difficulty 2; tags: spad, poisson, statistics)

- Follow-up: write the corrected estimator.


<details><summary>Answer</summary>

**What it is.** Per laser cycle a SPAD detects at most **one** photon, then is dead for the rest of the cycle. If the expected photon count per cycle is not $\ll 1$, early photons win and later ones are never recorded — the measured histogram is skewed toward **earlier** times. Since depth $\propto$ arrival time, pile-up **biases depth toward the camera** for bright pixels, and the bias grows with flux, so it is scene-dependent (a systematic, not just noise).

**Statistics.** With true rate $\lambda_i$ in bin $i$ (Poisson), the probability the first detection lands in bin $i$ is

$$
\begin{aligned}
p_i &= (1 - \exp(-\lambda_i)) \cdot \prod_{j<i} \exp(-\lambda_j) \\
    &= (1 - e^{-\lambda_i}) \cdot e^{-\sum_{j<i} \lambda_j}
\end{aligned}
$$

The $e^{-\sum_{j<i} \lambda_j}$ factor is the "survival" term that crushes late bins.

**Fixes.**
- **Attenuate** so the per-cycle detection probability is ~1–5% (the classic "low-flux regime"); costs acquisition time.
- **Coates correction / MLE inversion**: invert the formula above — estimate $\lambda_i$ from counts $h_i$ and the number of cycles $N$ via $\hat\lambda_i = -\ln\!\left(1 - \frac{h_i}{N - \sum_{j<i} h_j}\right)$. This is the maximum-likelihood inverse of the pile-up model and removes the bias when $N$ is large.
- **Asynchronous acquisition** (shift the gate start randomly each cycle) so no bin is systematically shadowed; or **gated SPADs**.
- **Model it in the likelihood** — for opportunistic sources the best answer is to put the dead-time survival term into the per-photon likelihood and do MLE for $\tau$ directly.

Trap: pile-up is *not* fixed by averaging more cycles — it is a bias, so more data only makes you more confidently wrong.

</details>


### [12-03] You come from computational imaging. Why is that background relevant to a generative-modeling internship?  (difficulty 2; tags: inverse-problems, diffusion, posterior-sampling)

- Follow-up: sketch how you would use a pretrained diffusion model as a prior for a reconstruction problem.


<details><summary>Answer</summary>

**Direct answer:** computational imaging *is* inverse problems with an explicit forward model plus a prior — and diffusion models are the best learned priors we have. The habits transfer: write down the forward model $y = A(x) + \text{noise}$, be honest about the noise statistics, and separate what the sensor knows from what the prior assumes.

Three concrete links:
- **Sensor-aware generation.** In foveated imaging we asked "which pixels are worth reading"; in foveated diffusion we ask "which tokens are worth denoising at full resolution." Same bandwidth-vs-fidelity question, same answer shape (spend compute where the task needs it). SPEED is the same idea in the frequency domain: don't compute frequencies the model can't resolve yet.
- **Diffusion as a prior (posterior sampling / DPS idea).** Sampling from $p(x \mid y) \propto p(y \mid x) \cdot p(x)$: run the reverse diffusion with the score of the prior, $s_\theta(x_t, t)$, and add a likelihood gradient. DPS approximates $\nabla_{x_t} \log p(y \mid x_t) \approx \nabla_{x_t} \log p(y \mid \hat{x}_0(x_t))$ using Tweedie's estimate $\hat{x}_0 = (x_t + \sigma_t^2 \cdot s_\theta)/\alpha_t$, giving the update $x_{t-1} \leftarrow \text{DDPM step} - \zeta \cdot \nabla_{x_t} \|y - A(\hat{x}_0)\|^2$. Cheap, training-free, works for deblurring, inpainting, super-resolution, and — relevant to my SPAD work — photon-limited (Poisson) likelihoods.
- **Knowing when the prior lies.** Diffusion posterior samplers hallucinate confidently when the likelihood is weak; imaging people are trained to check consistency with the measurement (data-fidelity residuals), which is exactly the discipline generative models for science / world models need.

Follow-up trap: DPS uses a point estimate $\hat{x}_0$ inside the likelihood, so it's a biased approximation of the true posterior score; better methods (e.g., ΠGDM, particle/SMC guidance) account for the variance of $x_0 \mid x_t$.

</details>


### [12-04] At Bell you built a document-retrieval system and fine-tuned open LLMs. Walk me through the RAG design and the fine-tuning, and tell me what you'd do differently now.  (difficulty 2; tags: rag, llm, finetuning, project)

- Follow-up: how did you measure hallucination?
- Follow-up: SFT vs LoRA — how did you choose?


<details><summary>Answer</summary>

**RAG design (state it as a pipeline):**
- **Ingest & chunk**: internal docs / code → chunks of ~[fill in: 300–800 tokens] with overlap, split on structural boundaries (headings, functions) rather than fixed windows; attach metadata (source, section, date) so retrieval can filter.
- **Embed & index**: [fill in: embedding model] → vector index; hybrid retrieval (dense + BM25) because internal jargon and identifiers are exact-match problems that dense embeddings miss.
- **Retrieve & rerank**: top-k by cosine, optional cross-encoder rerank, then pack into the prompt with citations.
- **Retrieval eval**: a hand-labeled set of (query → gold chunk) pairs; report recall@k and MRR. Retrieval is where most failures came from, so I evaluated it separately from generation.
- **Hallucination control**: constrain the model to answer only from provided context with explicit "not found" behavior, require citations to chunk ids, and check faithfulness with [fill in: manual review / LLM-as-judge]. Reduce temperature; ask for abstention.

**Fine-tuning open LLMs:** [fill in: model, e.g., Llama-2 7B]. Data curation was the real work: deduplicate, filter low-quality pairs, format consistently as instruction/response. **LoRA over full SFT** because of GPU budget and because we wanted to keep the base model's general capability — low-rank adapters (rank 8–64 on attention projections) trained on a few thousand examples. Eval: held-out task accuracy plus a small human-rated set, and a regression suite to catch capability loss.

**What I'd do differently now:**
- Build the eval set *first*, before touching retrieval or training.
- Use longer-context models and smarter chunking (semantic / late chunking) instead of tuning chunk size by hand.
- Add an explicit **abstention** objective and log retrieval confidence, so hallucination is measured, not eyeballed.
- Prefer DPO / preference tuning on real user feedback over plain SFT for style and refusal behavior.
- Treat it as a systems problem: latency, caching of embeddings, and monitoring drift.

</details>


### [12-05] Quick linear algebra: state the SVD, Eckart–Young, and condition number, and explain why each matters for LoRA and PCA.  (difficulty 2; tags: linear-algebra, math, lora)

- Follow-up: why is LoRA rank-r a reasonable inductive bias for fine-tuning?


<details><summary>Answer</summary>

**SVD.** Any $A \in \mathbb{R}^{m \times n}$ factors as $A = U \Sigma V^\top$ with $U, V$ orthonormal columns and $\Sigma = \operatorname{diag}(\sigma_1 \ge \sigma_2 \ge \dots \ge 0)$. Columns of $U$/$V$ are left/right singular vectors; $\sigma_i^2$ are eigenvalues of $A^\top A$.

**Eckart–Young (–Mirsky).** The best rank-$r$ approximation of $A$ in Frobenius or spectral norm is the truncated SVD $A_r = \sum_{i \le r} \sigma_i u_i v_i^\top$, with error $\|A - A_r\|_F^2 = \sum_{i>r} \sigma_i^2$ and $\|A - A_r\|_2 = \sigma_{r+1}$. **This is PCA**: center the data, take the top-$r$ right singular vectors $\to$ directions of maximal variance, and the captured variance fraction is $\sum_{i \le r} \sigma_i^2 / \sum_i \sigma_i^2$.

**Condition number.** $\kappa(A) = \sigma_{\max}/\sigma_{\min}$. It bounds how much relative error in $b$ amplifies into $x$ when solving $Ax = b$, and governs gradient-descent convergence on quadratics (rate $\sim (\kappa-1)/(\kappa+1)$). Ill-conditioned $\Rightarrow$ slow, noise-sensitive optimization; this is why we whiten inputs, use LayerNorm, and prefer Adam-style preconditioning.

**Why it matters for LoRA.** LoRA writes the weight update as $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$ — an explicit low-rank parameterization. The bet, justified empirically by low intrinsic dimension of fine-tuning updates, is that the *optimal* $\Delta W$ has a fast-decaying singular spectrum, so by Eckart–Young a rank-$r$ factor captures most of it with $r \cdot (d+k)$ parameters instead of $d \cdot k$. Practical corollaries: initialize $B = 0$ so $\Delta W$ starts at zero; scaling $\alpha/r$ keeps the update magnitude stable across ranks; and if you inspect a merged $\Delta W$'s singular values and they are flat, your rank is too low. Trap: LoRA constrains the *update* to be low rank, not the weights — the full $W$ stays full rank.

</details>


### [12-06] Probability rigor: derive Gaussian conditioning, the KL between two Gaussians, and explain where that KL appears in the DDPM loss. Then explain the reparameterization trick.  (difficulty 2; tags: probability, gaussian, ddpm, vae)


<details><summary>Answer</summary>

**Gaussian conditioning.** For jointly Gaussian $(x, y)$ with means $\mu_x, \mu_y$ and covariance blocks $\Sigma_{xx}, \Sigma_{xy}, \Sigma_{yy}$:

$$
x \mid y \;\sim\; \mathcal{N}\!\left( \mu_x + \Sigma_{xy} \Sigma_{yy}^{-1} (y - \mu_y),\; \Sigma_{xx} - \Sigma_{xy} \Sigma_{yy}^{-1} \Sigma_{yx} \right)
$$

The conditional mean is linear in $y$; the conditional covariance is the Schur complement and does not depend on $y$. This is exactly what gives the closed-form DDPM posterior $q(x_{t-1} \mid x_t, x_0)$: $x_t$ and $x_{t-1}$ are jointly Gaussian given $x_0$.

**KL between Gaussians.** For $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$ in $d$ dims:

$$
\mathrm{KL} = \tfrac12 \left[ \operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^\top \Sigma_2^{-1} (\mu_2-\mu_1) - d + \ln\frac{\det \Sigma_2}{\det \Sigma_1} \right]
$$

For equal isotropic covariance $\sigma^2 I$ it collapses to $\|\mu_1 - \mu_2\|^2 / (2\sigma^2)$.

**Where it lives in DDPM.** The ELBO decomposes into $L_T + \sum_t L_{t-1} + L_0$ where each $L_{t-1} = \mathrm{KL}\big( q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t) \big)$. Both are Gaussians with the same (fixed) variance $\sigma_t^2$, so the KL is a weighted squared distance between means; substituting the $\epsilon$-parameterization of the mean gives the familiar weighted $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$. Ho et al. drop the weight ("simple" loss) — that reweighting is a choice, not the ELBO.

**Reparameterization trick.** To get gradients through a sample $z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$, write $z = \mu_\phi + \sigma_\phi \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$. The randomness is now an input, so $\nabla_\phi \mathbb{E}[f(z)] = \mathbb{E}[\nabla_\phi f(\mu_\phi + \sigma_\phi \epsilon)]$ — low-variance pathwise gradients, versus the high-variance score-function (REINFORCE) estimator. Diffusion uses the same idea when it writes $x_t = \alpha_t x_0 + \sigma_t \epsilon$.

</details>


### [12-07] Rapid fire: Jacobian of softmax; gradient of $\|Ax - b\|^2$; $\mathbb{E}\|\epsilon\|^2$ for $\epsilon \sim \mathcal{N}(0, I_d)$.  (difficulty 1; tags: math, calculus, warmup)

- Follow-up: why does $\mathbb{E}\|\epsilon\|^2 = d$ matter for diffusion / for attention scaling?


<details><summary>Answer</summary>

**Softmax Jacobian.** With $s = \operatorname{softmax}(z)$, $s_i = e^{z_i}/\sum_j e^{z_j}$:

$$
\frac{\partial s_i}{\partial z_j} = s_i (\delta_{ij} - s_j) \qquad \text{i.e.}\quad J = \operatorname{diag}(s) - s s^\top
$$

$J$ is symmetric, PSD, rank $d-1$ (rows sum to zero — softmax is shift-invariant). Vector-Jacobian product for backprop: $J^\top g = s \odot (g - \langle g, s \rangle)$. Combined with cross-entropy the whole thing simplifies to $s - y$, which is why we fuse them.

**Gradient of $\|Ax - b\|^2$.** $f(x) = (Ax-b)^\top(Ax-b)$, so $\nabla f = 2A^\top(Ax - b)$, Hessian $2A^\top A$. Setting $\nabla f = 0$ gives the normal equations $A^\top A x = A^\top b \to x^* = A^+ b$. Note the condition number of the problem is $\kappa(A)^2$, which is why you solve via QR/SVD rather than forming $A^\top A$.

**$\mathbb{E}\|\epsilon\|^2$ for $\epsilon \sim \mathcal{N}(0, I_d)$.** Each $\epsilon_i^2$ has mean 1, so $\mathbb{E}\|\epsilon\|^2 = d$; $\|\epsilon\|^2 \sim \chi^2_d$, variance $2d$, so $\|\epsilon\| \approx \sqrt{d} \pm O(1)$ — Gaussian samples concentrate on a thin shell of radius $\sqrt{d}$.

**Why it matters.** (1) Attention: $q \cdot k$ for random unit-variance $q, k$ has variance $d_k$, so scores are divided by $\sqrt{d_k}$ to keep softmax from saturating. (2) Diffusion: at $t = 1$, $x_t \approx \epsilon$ lives on a shell of radius $\sqrt{d}$; the ortho-normalized DCT preserves norms, so noise has unit power per frequency bin — that's the $P(f) = 1$ SNR line in SPEED. (3) The "simple" DDPM loss $\|\epsilon - \epsilon_\theta\|^2$ has a natural scale of $d$ at init.

</details>


### [12-08] Explain your Virasoro / diffeomorphism math paper in one paragraph for a non-mathematician, then tell me what it trained you in.  (difficulty 1; tags: math, background, communication)


<details><summary>Answer</summary>

**One paragraph, plain language.** Take all the smooth ways to stretch and reparameterize a circle — these form an infinite-dimensional group of *diffeomorphisms*, and its algebra of infinitesimal stretches has a famous "central extension" called the **Virasoro algebra**, which is the symmetry algebra of 2D conformal field theory and string theory. The paper asks what happens when the reparameterizations are allowed to have **breaks** — points where the map is continuous but not smooth, or has jumps in derivative. We work out how the Virasoro-type extension (the extra term that measures how much two stretches fail to commute in a way the classical picture misses — the Gelfand–Fuks cocycle) must be modified to make sense on this larger, less regular space. [fill in: the concrete result — e.g., classification of the admissible cocycles / how the central charge term picks up boundary contributions at the breaks.]

**What it trained me in** (this is what the interviewer wants):
- **Working with infinite-dimensional objects carefully** — the same instinct that says "what regularity does this actually need" is what I use when reasoning about score functions, continuous-time flows, and probability-flow ODEs.
- **Cohomology / extension thinking**: what is the *minimal* extra structure that makes a construction consistent. That shows up in ML as "what is the minimal change to the sampler that keeps it correct" — e.g., resetting the multistep solver history at a resolution change in SPEED.
- **Proof hygiene**: stating assumptions explicitly, finding the counterexample first. In experiments that becomes: design the control that would falsify the claim before running the headline number.
- **Communicating abstract ideas** — I TA'd advanced linear algebra, which is where my habit of always giving the 2×2 example comes from.

</details>


### [12-09] What is a SPAD?  (difficulty 1; tags: spad, rapid-fire, sensors)


<details><summary>Answer</summary>

A **single-photon avalanche diode is a photodetector biased above breakdown so that a single photon triggers a self-sustaining avalanche, giving a digital click with picosecond-scale timing**. Instead of integrating charge like a CMOS pixel, it outputs a stream of photon-arrival timestamps, which is what makes it useful for time-of-flight depth, low-light imaging, and fluorescence lifetime. Costs: after each detection it is dead for tens of nanoseconds, so detections are lost at high flux (pile-up), there is dark-count noise, and fill factor is low. Follow-up: the data is a Poisson process, so the whole processing pipeline is statistical — histogramming timestamps, then estimating the depth peak.

</details>


### [12-10] What is time-of-flight depth?  (difficulty 1; tags: tof, rapid-fire, depth, computational-imaging)


<details><summary>Answer</summary>

**Time-of-flight depth measures distance by timing how long light takes to travel to a surface and back: $d = c \cdot \tau/2$.** Direct ToF sends a short pulse and timestamps the return (a SPAD histograms many returns and finds the peak); indirect ToF modulates a continuous source and measures phase shift. Our paper's twist was **opportunistic** ToF: rather than a controlled laser, use ambient or uncontrolled light sources whose temporal structure we do not own, and recover depth by cross-correlating the received photon stream with a reference measurement of the source. Trap for a one-minute answer: mention the timing resolution to depth conversion, 1 ns is about 15 cm round-trip, so picosecond timing is what makes millimeter depth possible.

</details>


### [12-11] What is pile-up, in one sentence?  (difficulty 1; tags: spad, rapid-fire, pile-up, poisson)


<details><summary>Answer</summary>

**Pile-up is the bias that occurs when a SPAD detects the first photon of each cycle and then goes dead, so at high flux early photons are over-represented and the measured histogram is skewed toward earlier times, biasing depth closer.** The math: the measured histogram is the true arrival distribution times the probability that no photon arrived earlier, a survival term like $\exp(-\Lambda(t))$, which is Poisson statistics. Fixes: attenuate the light so photons per cycle are well below one, invert the distortion analytically using the known Poisson model, or use asynchronous / free-running acquisition so the dead time is decorrelated from the cycle. Follow-up: pile-up is not noise, it is a deterministic bias, so averaging more does not remove it.

</details>


### [12-12] What is RAG, in one sentence?  (difficulty 1; tags: rag, rapid-fire, llm)


<details><summary>Answer</summary>

**Retrieval-augmented generation retrieves relevant documents for a query, usually by embedding similarity, and puts them in the LLM's context so the model answers from those sources instead of only its weights.** Why: it gives up-to-date, private, or domain-specific knowledge without retraining, and makes answers attributable. At Bell I built this over internal documents and code with LangChain: chunk documents, embed, store in a vector index, retrieve top-k, then prompt. The practical traps are retrieval quality (chunking, hybrid keyword plus dense search, reranking) and evaluation, since a fluent answer over the wrong chunks is the common failure. Follow-up: when to fine-tune instead — for style and format, not for facts that change.

</details>


### [12-13] What is LoRA fine-tuning of an LLM, in one sentence?  (difficulty 1; tags: lora, rapid-fire, finetuning, llm)


<details><summary>Answer</summary>

**LoRA freezes the pretrained weights and trains a low-rank update $\Delta W = B \cdot A$ ($A$ is $r \times d$, $B$ is $d \times r$, $r$ small) added to selected linear layers, usually attention projections, so fine-tuning touches a fraction of a percent of the parameters.** Why it works: the weight change needed to adapt a large model has low intrinsic rank, so rank 8 to 64 captures most of it. Benefits: tiny checkpoints, no inference change once merged ($W + BA$), and far less memory since optimizer states exist only for $A$ and $B$. I used it at Bell on open LLMs and in SPEED and foveated diffusion on DiTs. Trap: LoRA is weak for adding new knowledge; it is best for style, format, and behavior.

</details>


### [12-14] What is the SVD, in one sentence, and what is it used for?  (difficulty 1; tags: linear-algebra, rapid-fire, svd, math)


<details><summary>Answer</summary>

**The singular value decomposition writes any matrix as $A = U \cdot \Sigma \cdot V^\top$, with $U$ and $V$ orthogonal and $\Sigma$ diagonal with non-negative singular values, so $A$ is a rotation, an axis-aligned scaling, and another rotation.** It exists for every matrix, square or not. Uses: the best rank-$k$ approximation is the truncated SVD (Eckart–Young), which underlies PCA, low-rank compression, and the intuition behind LoRA; the ratio of largest to smallest singular value is the condition number, which tells you how numerically stable a solve or a gradient will be; and the pseudo-inverse and least-squares solutions come from it directly. One-liner follow-up: singular values of $A$ are square roots of eigenvalues of $A^\top A$.

</details>


## Research Story & Behavioral


### [13-01] Walk me through your research in about three minutes.  (difficulty 1; tags: research-story, behavioral, pitch)

- Follow-up: what is the single thread connecting these projects?


<details><summary>Answer</summary>

Structure it as **one thread, three chapters, one destination**. The thread: *spend sensing and compute only where the signal is.*

**Chapter 1 — Foveated sensing (SIGGRAPH 2026).** Cameras now have 200-megapixel sensors, but no downstream model can consume that bandwidth. We trained a **policy** that decides which regions of an ultra-high-res sensor to read at full resolution, like a saccading eye, for tasks such as MuJoCo manipulation, tracking, and text recognition. It keeps task performance with **under 1/8 of the pixel bandwidth**, and we validated it on a real 200 MP prototype. Lesson: adaptive allocation of a scarce resource, driven by the task.

**Chapter 2 — Efficient generation (SPEED + Foveated Diffusion, 2026).** Same idea applied to diffusion models. In **SPEED**, I measured that latents have a power-law spectrum $P(f) \propto f^{-\beta}$, so at high noise levels high frequencies are below the noise floor — the model literally cannot see them. So we run early denoising steps at low resolution and grow resolution in stages when each band's SNR crosses 1. Training-free or with a small LoRA; **up to 7× wall-clock speedup on FLUX and 2.5× on WAN 2.1**, quality preserved. **Foveated Diffusion** does the spatial analogue: full-res tokens in the fovea, coarse in the periphery, 2×/4× image/video speedups.

**Chapter 3 — World models for physical AI (ongoing).** Robots need world models that run long horizons cheaply. My current work compresses the KV cache of a world-action model by keeping only tokens that are *surprising* and *attended by the action head* — ~44–54% less KV with task success unchanged. Plus the CVPR 2025 oral on single-photon ToF, which is where my sensor-physics instincts come from.

**Destination:** efficient, sensor-aware generative world models that a robot can run in real time — which is why NVIDIA's Cosmos / Isaac stack is where I want to be. [fill in: one sentence on your intended thesis framing.]

Land within three minutes; the follow-up "what's the thread?" should already be answered.

</details>


### [13-02] Tell me about a result that didn't work, and what you learned.  (difficulty 2; tags: behavioral, speed, debugging, research-story)

- Follow-up: how did you know it was the method and not a bug?


<details><summary>Answer</summary>

Use the real SPEED story; it shows diagnosis, not just persistence.

**Situation.** The first version of SPEED was the "obvious" one: a continuous per-step frequency mask — at every denoising step, reveal exactly the frequencies whose SNR has crossed 1 (a brick-wall radial cut that grows each step). Theory said this should be optimal.

**Result.** Images came out **blurry**, consistently, across seeds and prompts. Speed looked fine; quality did not.

**Diagnosis — controls, not vibes.** I separated the two things that changed at once: the *masking schedule* and the *solver's view of the trajectory*.
- Control A: full resolution, no masking, same solver → sharp. So the base pipeline was fine.
- Control B: full resolution, staged box low-pass (whole band revealed at a transition, then left alone) → sharp, no speedup. So *staged* reveal is fine; the problem is *continuous* reveal.
- Control C: continuous masking with Euler vs. with UniPC → both blurred, so it was not a solver-history artifact (though that check later found a separate real bug: multistep solvers keep stale history across a resolution change, which we fixed with a step-index reset).
Conclusion: with per-step reveal, the **last frequencies to unlock get only a handful of denoising steps**, and the moving brick-wall cut introduces ringing the model has to spend steps undoing. Both starve the high-frequency detail.

**Fix.** Staged transitions: grow resolution in a few discrete stages timed to when each stage's Nyquist frequency activates, so every band gets all remaining steps. That is what gave sharp results *and* the 7× speedup, because only actual resolution reduction (not masking) saves compute.

**What I learned.** (1) When a theoretically-motivated method fails, the theory is usually right about the *what* and wrong about the *discretization*. (2) Change one thing at a time and keep a "no-speedup but same quality" control — it told us which half of the idea carried the quality. (3) Write the solver check early; the bug it caught would have poisoned every video result.

</details>


### [13-03] What would you want to work on at NVIDIA, and why here rather than another lab?  (difficulty 1; tags: behavioral, nvidia, motivation)

- Follow-up: which team, concretely?


<details><summary>Answer</summary>

**Direct answer.** I want to work on **efficient world models for physical AI** — video/world foundation models that are fast enough and long-context enough to sit inside a robot's control loop — and NVIDIA is the one place where the model (Cosmos), the simulator (Isaac Sim / Isaac Lab), and the hardware co-design all live under one roof.

Three concrete threads I'd bring, matched to teams:
- **GenAI / Cosmos world foundation models.** My SPEED and foveated-diffusion work is about *where* a diffusion model spends compute — spectrally and spatially. World models are the regime where that matters most: long rollouts, high resolution, tight latency. I'd like to push resolution-progressive and token-adaptive generation into autoregressive/causal video, and into the tokenizer + diffusion-decoder stack.
- **Robotics / Isaac, world-action models.** My KV-cache compression for a world-action model (surprise-gated, ~50% less cache with unchanged task success) is a small step toward long-horizon rollouts on-device. Isaac's sim-to-real and GR00T-style VLA work is exactly where I'd test whether "predict only what surprises you" scales.
- **Diffusion fundamentals.** I care about the spectral / SNR view of flow matching — it explains resolution schedules, why super-native generation works when you stay in the native-resolution regime, and it hints at better noise schedules and solvers. That's basic research NVIDIA's GenAI group actually publishes.

**Why here.** (1) Scale: the questions I care about only get answered at 10B-parameter video models on hundreds of GPUs, and NVIDIA runs those. (2) Research-to-product path: my speedups are only interesting if they ship in an inference stack; NVIDIA turns papers into Cosmos releases and TensorRT paths. (3) The people: [fill in: 1–2 specific NVIDIA papers/people you've read — e.g., Cosmos WFM tech report, EDM/EDM2 lineage, the Isaac Lab / GR00T team].

Keep it honest: mention the second-choice lab only if asked, and then say why NVIDIA's combo is unique.

</details>


### [13-04] An experiment gives you an ambiguous result. How do you decide what to try next?  (difficulty 2; tags: behavioral, methodology, experiments)

- Follow-up: give a concrete example from your work.


<details><summary>Answer</summary>

**Principle: turn ambiguity into a factorial design before running more of the same.** Ambiguous usually means two variables changed at once or the noise floor is unknown.

My checklist, in order:
1. **Estimate the noise floor first.** Re-run the baseline with N seeds and report the spread. Half of "ambiguous" results are within seed variance. In the KV-compression work I used **paired seeds** — same seed for compressed and uncompressed rollouts — because LIBERO task success at 20–50 episodes has a wide confidence interval and pairing removes the shared variance.
2. **Isolate the axis.** Write down the ≤3 things that differ between the good and bad condition, then build the 2×2 (or 2×2×2) of controls. In SPEED the axes were {continuous vs staged masking} × {resolution reduction vs full-res low-pass} × {Euler vs UniPC}; running the grid told us quality came from staging and speed came from resolution — two separate findings.
3. **Look for a confound that flips the metric's meaning.** Example: in world-action rollouts, failure episodes had *higher* imagined-video PSNR because a stalled robot is trivial to predict. Split metrics by success/failure before believing any average.
4. **Check determinism.** Flash-attention kernels are non-deterministic across runs; I verify bitwise reproducibility of the baseline before attributing a 0.3 dB change to the method.
5. **Pick the experiment that would change my decision.** Not the one that's easiest. If neither outcome changes what I'd do next, skip it.
6. **Time-box.** If the ablation grid doesn't disambiguate in a day, step back and re-derive the expected effect size from the model — if theory says the effect should be tiny, the experiment isn't ambiguous, it's null.

Follow-up example: the per-step-masking blur (see the "result that didn't work" answer) was resolved entirely by step 2.

</details>


### [13-05] Describe a time you collaborated closely, or disagreed with a co-author, and how it resolved.  (difficulty 1; tags: behavioral, collaboration)

- Follow-up: what would you do differently?


<details><summary>Answer</summary>

Pick one real story and tell it in four beats: context → the disagreement → how you resolved it with evidence → what changed in you.

**Suggested story (SPEED / Foveated Diffusion are co-authored with the same group — Chao, Yariv, Wetzstein):** [fill in: the actual disagreement. Plausible shapes: (a) whether to ship SPEED as training-free only vs. add the LoRA variant — one side wanted a clean training-free story, the other wanted the best numbers; (b) staged vs continuous masking, where the "elegant" continuous version was someone's preference and the data said staged; (c) how to split credit/scope between SPEED and Foveated Diffusion, which share machinery.]

Model structure:
- **Context.** Two of us had different intuitions about [fill in]. Both were reasonable; neither had data.
- **Disagreement.** I made my position falsifiable: "if X, then we should see Y in control Z." I proposed the smallest experiment that would settle it — not a debate.
- **Resolution.** [fill in: who was right]. If I was wrong, say so plainly and name what convinced you — interviewers rate that higher than being right. If I was right, credit the co-author for the pressure that made the control exist.
- **What changed.** We adopted a habit: every design disagreement gets a control in the ablation table. It made the paper stronger because the reviewers' "why not X" was already answered.

**Collaboration texture to include:** shared codebase across FLUX / WAN / Qwen-Image ports meant agreeing on interfaces early (schedulers, DCT utilities); I owned [fill in] and reviewed [fill in]. Say what you did to make co-authors' lives easier (reproducible scripts, seed-pinned configs).

Follow-up: "differently" — raise the disagreement earlier and in writing, with the proposed experiment attached, rather than letting it sit in a meeting.

</details>


### [13-06] How do you read and keep up with papers, and what recent paper impressed you and why?  (difficulty 2; tags: behavioral, literature, video-generation, vla, world-models)

- Follow-up: what's the weakness of the paper you just praised?


<details><summary>Answer</summary>

**Process.** Daily arXiv skim by keyword (diffusion, flow matching, video, world model, VLA), a weekly deeper read of 2–3 papers where I reproduce the key equation on paper and note "what would break this." I keep a running notes file per topic and re-read it before starting a project. I also read *outside* the lane deliberately — neuroscience of foveation, signal processing — because that is where SPEED's spectral view came from.

**Three directions that impressed me (pick two, know them cold):**

- **Self-forcing / causal autoregressive video.** Training a causal video model by rolling out its *own* generations during training (rather than teacher-forcing on ground truth) closes the train-test gap that causes drift in long rollouts, and pairs with KV-cache-based streaming generation. Why it matters to me: it's the bridge from bidirectional video diffusion to a *world model* you can run interactively; and my KV-compression work is about what to keep in exactly that cache. Weakness: exposure to its own errors makes training expensive and sensitive to the rollout horizon; quality still lags bidirectional models.

- **VLAs with flow-matching action heads.** Predicting continuous action chunks with a flow/diffusion head on top of a VLM backbone (π0-style) instead of discretizing actions into tokens. Why it impressed me: it's the cleanest evidence that diffusion fundamentals transfer to control — multimodal action distributions, few-step sampling, and the same guidance tricks. Weakness: the vision backbone is still frozen-ish and low-res; sensing is not adaptive — which is where foveated perception should plug in.

- **World-action models** (jointly predicting future video latents and actions, one transformer). Why: it unifies the "imagine, then act" loop, and gives you a prediction error signal for free — the surprise signal I use for KV compression. Weakness: imagined video is expensive, and in practice failure cases have *easier* video (stalled robot), so the video loss can mislead the policy.

**The framing I'd give:** the field is converging on *causal, long-context, action-conditioned video models*, and the open problem is making them cheap enough to run in the loop — which is my research thread.

[fill in: exact titles/authors of the two you choose; be able to state each one's key equation or training objective in one line.]

</details>


### [13-07] What questions do you have for us?  (difficulty 1; tags: behavioral, questions-for-interviewer)


<details><summary>Answer</summary>

Have five ready; ask two or three, tailored to the interviewer. Good ones for an NVIDIA research team:

**About the work**
- "What does a successful intern project look like on this team — a paper, a component that ships in Cosmos / Isaac, or both? How do you decide which?"
- "How much of the team's research is done at foundation-model scale versus at scales an intern can iterate on daily? What compute does an intern actually get?"
- "Which open problem in world models / video generation is the team most divided about right now?"
- "How does research here move into product — what's the path from a paper result to something in Cosmos, Isaac, or TensorRT, and how involved are researchers?"

**About mentorship & fit**
- "Who would I be working with day to day, and how do mentorship and paper authorship typically work for interns?"
- "What did the last intern on this team work on, and where did it go?"
- "Is there a chance to work with the hardware/systems side — e.g., getting an inference speedup into an actual deployment path?"

**About the interviewer**
- "What's the thing you're personally most excited to see solved in the next two years?"

Avoid: compensation, vacation, anything answerable from the job posting. If the interviewer works on a paper you know, ask a specific technical question about it — that lands better than any generic question.

</details>


### [13-08] Tell me about your internship at Bell Canada, and what you learned about shipping software.  (difficulty 1; tags: behavioral, bell, industry, shipping)

- Follow-up: how does that change how you do research code?


<details><summary>Answer</summary>

**Setup.** Sixteen months as a software intern; I built internal **document-retrieval and code-generation tools** (Python, Ruby, SQL, LangChain) and **fine-tuned open-source LLMs** on internal data. Real users, real deadlines, a team that had to maintain it after I left.

**What I actually did** (be concrete): [fill in: the tool, who used it, one usage number or outcome]. The technical core was a RAG pipeline plus LoRA fine-tunes with an eval set I built.

**Three lessons about shipping:**
1. **The eval set is the product.** The first version of the retrieval tool "felt" good in demos and failed on real queries. Building a labeled query set and tracking recall@k turned arguments into numbers. I now build the metric before the method — the same discipline that made SPEED's ablation table exist.
2. **Boring reliability beats clever.** Hybrid BM25 + dense retrieval, explicit "not found" answers, and logging beat a fancier model. Users trust a tool that fails loudly over one that hallucinates quietly.
3. **Hand-off is part of the work.** Documentation, reproducible configs, tests, and a clean interface mattered more to the team than my last 5% of accuracy. I left with the tool still running.

**How it changed my research code.** Seed-pinned configs, one-command reproductions, and scripts a co-author can run on a different model family (the SPEED code was ported across FLUX, WAN, and Qwen-Image because of that). I also learned to time-box: a 16-month deadline horizon teaches you to pick the experiment that changes the decision.

Follow-up trap: don't oversell it as research. It was engineering, and the point is that you know the difference and can do both.

</details>


### [13-09] Describe your research in one sentence.  (difficulty 1; tags: behavioral, rapid-fire, pitch)


<details><summary>Answer</summary>

**I make visual generation and visual sensing spend compute only where the signal is: spectral resolution schedules for diffusion models, learned foveation for sensors and diffusion transformers, and photon-level imaging that recovers depth from whatever light is available.** If they ask for the throughline: the same idea, that information in images is unevenly distributed over frequency, space, and time, applied to both the sensor and the generator, which is why my interests point at world models and physical AI, where the model and the camera sit in one loop. Keep it to one breath, then stop and let them pick which thread to pull.

</details>


### [13-10] What is the single most important result you have?  (difficulty 1; tags: behavioral, rapid-fire, results)


<details><summary>Answer</summary>

**SPEED: a training-free 7.09× wall-clock speedup on FLUX.1-dev, and 2.54× on WAN 2.1 video, with quality preserved, from a resolution schedule derived from the latent power spectrum $P(f) = A \cdot f^{-\beta}$ rather than tuned by hand.** Why I rank it first: it is principled (the schedule falls out of an $\mathrm{SNR} = 1$ criterion), it transfers across four model families without architecture changes, and the ablation that continuous frequency masking blurs while staged reveal is sharp taught us something about how diffusion models use steps. The follow-up I expect is "does this survive distillation?" and the honest answer is partially: for 4-step models the ceiling is about 4/3×, so the value is in the 20-50 step tier and in video.

</details>


### [13-11] What is your biggest weakness as a researcher?  (difficulty 1; tags: behavioral, rapid-fire, weakness)


<details><summary>Answer</summary>

**I tend to over-derive before I run the cheap experiment.** My math background pulls me toward getting the SNR-correction or the schedule exactly right on paper first; on SPEED I spent days on the transition scaling before a one-hour sweep would have shown which regime mattered. What I do about it now: I write the falsifying experiment first — the control that would kill the idea, such as the full-resolution box low-pass baseline that showed only real resolution reduction gives speedup — and run it before the derivation is polished. Related: I underestimate engineering time for video pipelines, so I now time-box and build the smallest end-to-end version first. Do not say "perfectionism"; give the incident and the fix.

</details>


### [13-12] Where do you see video generation and world models in three years?  (difficulty 2; tags: behavioral, rapid-fire, world-models, video-generation, future)


<details><summary>Answer</summary>

**Video generation becomes a component inside interactive world models rather than a standalone product: causal, streaming, action-conditioned, and evaluated on whether a robot or agent using it gets better, not on FVD.** Three concrete bets. First, the cost problem gets solved by spending compute unevenly, resolution schedules, foveation, and KV-cache compression that keeps only surprising tokens, since a world model in a control loop cannot afford 50 full-resolution steps per frame. Second, distillation plus long-context memory makes minute-scale consistent rollouts routine. Third, the data bottleneck shifts to physics correctness, so simulation and real-to-sim become the training signal. The open risk: we still lack an evaluation that measures physical plausibility, and that will decide who wins.

</details>


## Practical Training & PyTorch Rapid-Fire


### [14-01] What is gradient (activation) checkpointing, and what do you have to be careful about?  (difficulty 1; tags: memory, pytorch, training, rapid-fire)

- Follow-up: what does "selective" checkpointing mean?


<details><summary>Answer</summary>

**Activation checkpointing** drops intermediate activations in the forward pass and recomputes them during backward, trading ~30% extra compute for a large memory cut (activations dominate memory for long sequences). In PyTorch: `torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)` wrapping a block, typically one transformer layer. Memory goes from $O(L)$ layer activations to $O(\sqrt{L})$ or $O(1)$ per checkpointed segment. **Selective** checkpointing recomputes only the cheap-but-large ops (attention scores, GELU, layernorm outputs) and keeps the expensive matmul outputs, which recovers most of the memory at a fraction of the recompute cost. Traps: the recomputed forward must be bit-identical, so **dropout/RNG state** must be replayed (PyTorch preserves RNG by default, `preserve_rng_state=True`); do not put in-place mutations or side effects inside the checkpointed function; and with `use_reentrant=True` inputs that do not require grad break gradient flow silently.

</details>


### [14-02] What is gradient accumulation, and what does it NOT reproduce exactly compared to a real large batch?  (difficulty 1; tags: training, pytorch, rapid-fire)


<details><summary>Answer</summary>

**Gradient accumulation** runs $K$ micro-batches, calls `backward()` on each (gradients sum into `.grad`), and steps the optimizer once, so the update equals a batch of $K\times$ the micro-batch size at the memory of one. Divide the loss by $K$ (or average after) so the effective LR is unchanged. What it does not reproduce: **BatchNorm** statistics are computed per micro-batch, so BN sees a batch of $B$ not $K \cdot B$ (transformers with LayerNorm/RMSNorm are unaffected); anything with cross-sample interaction (contrastive losses, in-batch negatives) sees a smaller batch; and it does not save the activation memory of a single micro-batch. Traps: per-token losses need token-count-weighted averaging, not per-micro-batch mean, when sequence lengths differ; and under DDP use `no_sync()` on all but the last micro-batch or you pay $K$ all-reduces.

</details>


### [14-03] What does torch.autocast do, why does bf16 need no loss scaling, and what must stay in fp32?  (difficulty 1; tags: precision, pytorch, training, rapid-fire)


<details><summary>Answer</summary>

`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` is a context manager that casts op inputs per an **allow-list**: matmuls, convolutions and linear layers run in the low-precision dtype (tensor cores), while reductions and numerically sensitive ops (softmax, layernorm, loss functions, sums, exp/log) are promoted to fp32. Weights stay fp32 (**master weights**); autocast makes low-precision copies on the fly. **bf16** has the same 8-bit exponent as fp32, so gradients of 1e-30 do not underflow and no loss scaling is needed; fp16 has a 5-bit exponent (min normal ~6e-5), so small gradients flush to zero and you need a GradScaler. Keep fp32 for the loss, norms, softmax/attention logits (or use flash-attn's fp32 accumulation), optimizer state, and master weights. Trap: bf16 has only 8 mantissa bits, so accumulating a running sum or an EMA directly in bf16 loses updates smaller than 1/256 of the value.

</details>


### [14-04] What is a GradScaler and what happens when it sees an inf gradient?  (difficulty 1; tags: precision, pytorch, rapid-fire)


<details><summary>Answer</summary>

`torch.cuda.amp.GradScaler` implements **dynamic loss scaling** for fp16 training: it multiplies the loss by a scale factor $S$ (initially 65536) before backward, so small gradients stay representable in fp16, then `scaler.step(optimizer)` unscales `.grad` by $1/S$ before the update. On each step it checks the gradients for **inf/NaN**; if found, it **skips the optimizer step entirely**, halves $S$, and continues (no crash). If no overflow occurs for a growth interval (2000 steps by default) it doubles $S$. So an occasional skipped step is normal early in training; a scale that keeps collapsing means real instability. Order of calls matters: `scaler.scale(loss).backward()`, then `scaler.unscale_(optimizer)` if you clip gradients, then `scaler.step(optimizer)`, then `scaler.update()`. Trap: the LR scheduler will warn about stepping before the optimizer when a step is skipped; that is benign.

</details>


### [14-05] torch.no_grad vs torch.inference_mode vs requires_grad_(False): what is the difference?  (difficulty 1; tags: pytorch, autograd, rapid-fire)


<details><summary>Answer</summary>

All three stop gradient computation but at different scopes. **`torch.no_grad()`** is a context: ops inside do not record an autograd graph, saving memory and time, but the outputs are normal tensors that can be used in a graph later. **`torch.inference_mode()`** is stricter and faster: it also disables version counters and view tracking, and the resulting tensors are "inference tensors" that **cannot be used in autograd later** (you get an error if you try). Use it for pure eval/serving. **`param.requires_grad_(False)`** is per-tensor and permanent: that parameter never gets a `.grad`, the optimizer will not update it, and autograd does not store activations for its own weight-gradient, though activations are still stored if gradients must flow through to earlier trainable layers. Trap: neither context manager changes dropout or BatchNorm behavior; that is `model.eval()`.

</details>


### [14-06] What does model.eval() actually change, and what does it not?  (difficulty 1; tags: pytorch, training, rapid-fire)


<details><summary>Answer</summary>

`model.eval()` sets `self.training = False` recursively, which only matters for modules that branch on it: **Dropout** becomes identity (no scaling needed because train-time uses inverted dropout), **BatchNorm** uses running mean/var instead of batch statistics and stops updating them, and a few others (stochastic depth, some data-augmentation layers, `nn.MultiheadAttention` fast path). It does **not** disable gradient tracking, so a forward in eval mode still builds the graph and stores activations; you need `torch.no_grad()` or `inference_mode()` for that. Traps: forgetting `model.train()` after validation silently freezes BN statistics and turns dropout off for the rest of training; and evaluating a BN model in train mode with batch size 1 crashes or gives garbage. In frozen-backbone fine-tuning, call `backbone.eval()` every epoch because `model.train()` re-enables BN updates.

</details>


### [14-07] What is a torch.autograd.Function and when do you write one?  (difficulty 2; tags: pytorch, autograd, rapid-fire)


<details><summary>Answer</summary>

A `torch.autograd.Function` subclass defines a custom op with an explicit `forward(ctx, ...)` and `backward(ctx, grad_out)`; `ctx.save_for_backward` stores what backward needs. You write one when autograd cannot or should not do it for you: **(1) a custom backward** for a fused CUDA/Triton kernel (flash-attention is exactly this), **(2) memory savings** by recomputing or storing a compact intermediate instead of everything autograd would keep, **(3) non-differentiable ops** where you want a surrogate gradient, e.g. the **straight-through estimator** for quantization or argmax (forward rounds, backward passes the gradient through unchanged), and **(4) gradient reversal** for domain adaptation.
```python
class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.round()
    @staticmethod
    def backward(ctx, g): return g
```
Trap: check it with `torch.autograd.gradcheck` in float64; a silently wrong backward trains but converges to nonsense.

</details>


### [14-08] What are forward and backward hooks, and give three practical uses.  (difficulty 1; tags: pytorch, debugging, rapid-fire)


<details><summary>Answer</summary>

Hooks are callbacks attached to modules or tensors: `module.register_forward_hook(fn)` runs after the module's forward with (module, input, output); `register_forward_pre_hook` runs before; `register_full_backward_hook` runs when gradients w.r.t. module inputs are computed; `tensor.register_hook(fn)` intercepts a tensor's gradient and can replace it. Three uses: **(1) feature extraction** from intermediate layers without editing the model (e.g. DINOv2 patch tokens for a probe); **(2) per-layer gradient clipping or scaling**, e.g. a tensor hook that clamps gradients into an embedding table; **(3) NaN hunting**: a forward hook that asserts `torch.isfinite(output).all()` on every module tells you exactly which layer first produced a NaN. Also used for activation statistics and for DDP's gradient bucketing internally. Trap: hooks return handles; forgetting `handle.remove()` leaks memory and doubles work, and modifying `output` in place inside a hook confuses autograd.

</details>


### [14-09] Why do in-place ops cause autograd errors like "a leaf Variable that requires grad is being used in an in-place operation" or version-counter mismatches?  (difficulty 2; tags: pytorch, autograd, rapid-fire)


<details><summary>Answer</summary>

Autograd saves tensors needed for backward (the input of a `sigmoid`, the weight of a matmul) and tags each with a **version counter**. If an in-place op (`x.add_()`, `x[mask] = 0`, `relu_`) later modifies a saved tensor, the counter increments and backward raises "modified by an inplace operation", because the saved value is now wrong. The leaf error is different: a **leaf parameter with requires_grad** cannot be modified in place while tracking, since that would corrupt the root of the graph; do such updates under `torch.no_grad()`, which is what optimizers do. Fixes: use out-of-place versions, clone before mutating, or wrap intentional mutation (EMA update, buffer masking) in `no_grad`. Trap: `x += y` on a view of a saved tensor is still in-place, and the error surfaces at backward, far from the offending line; `torch.autograd.set_detect_anomaly(True)` points to it.

</details>


### [14-10] Why do we zero gradients every step, and what does set_to_none=True do?  (difficulty 1; tags: pytorch, training, rapid-fire)


<details><summary>Answer</summary>

`backward()` **accumulates** into `.grad` (it adds, it does not assign), which is what makes gradient accumulation and multi-loss training possible, so you must clear gradients before the next backward or updates compound. `optimizer.zero_grad(set_to_none=True)`, now the default, sets each `.grad` to `None` instead of filling it with zeros. Benefits: it skips a memset kernel per parameter, the next backward allocates and writes directly rather than reading and adding (a `=` instead of `+=`), and the gradient memory is actually released between steps, which matters when activations peak at a different time than gradients. Traps: code that inspects `p.grad` between zero and backward now sees `None`; parameters that received no gradient this step stay `None` and optimizers skip them, which changes behavior for Adam's step count and weight decay on rarely-used embeddings.

</details>


### [14-11] Gradient clipping by norm vs by value, and where in the step does it go?  (difficulty 1; tags: training, pytorch, rapid-fire)


<details><summary>Answer</summary>

**Clip by global norm** (`torch.nn.utils.clip_grad_norm_(params, max_norm)`) rescales all gradients by $\min(1, \text{max\_norm} / \|g\|_2)$ so the update direction is unchanged but its length is bounded; standard for transformers with max_norm around 1.0. **Clip by value** clamps each element into $[-c, c]$, which changes the direction and is mostly used in RNNs. The order matters: clipping must see the **true gradients**, so it goes **after** all accumulation micro-batches have been summed, **after** `scaler.unscale_(optimizer)` if using fp16 loss scaling (otherwise you clip scaled values), and **before** `optimizer.step()`. Under DDP or FSDP call it after the reduction so the norm is over the global gradient (FSDP needs `model.clip_grad_norm_`). Trap: `clip_grad_norm_` returns the pre-clip norm; log it, since a rising gradient norm is the earliest sign of divergence and a norm that is always clipped means the LR is too high.

</details>


### [14-12] Weight decay: which parameters should you exclude and how?  (difficulty 1; tags: training, optimizers, rapid-fire)


<details><summary>Answer</summary>

Standard practice is to apply weight decay only to **matrix weights** (linear and conv kernels) and exclude **biases, LayerNorm/RMSNorm gains, and often embeddings**. Reason: decay pulls parameters toward zero, and for 1-D parameters this is not a meaningful regularizer; shrinking a norm gain just rescales the following layer, shrinking a bias shifts activations, and there is no overfitting to fight in those few parameters. Implementation is via **parameter groups**:
```python
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
```
Note that AdamW applies **decoupled** decay ($p \mathrel{-}= \text{lr} \cdot \text{wd} \cdot p$), so the effective decay scales with LR and follows the schedule. Trap: with tied embeddings the embedding matrix is also the output projection, so decide once; and LoRA adapters usually get no decay.

</details>


### [14-13] What are parameter groups, and how do you do layer-wise LR decay or discriminative fine-tuning?  (difficulty 2; tags: training, optimizers, finetuning, rapid-fire)


<details><summary>Answer</summary>

A PyTorch optimizer takes a list of dicts, each with its own `params` and overrides (lr, weight_decay, betas); unspecified values fall back to the optimizer default. This is the mechanism for **discriminative learning rates**: a small LR for a pretrained backbone and a 10× larger one for a freshly initialized head, so the head catches up without wrecking pretrained features. **Layer-wise LR decay** (BEiT/MAE fine-tuning) generalizes it: layer $i$ of $L$ gets $\text{lr} \cdot \text{decay}^{(L-i)}$ with decay 0.65 to 0.85, so early layers holding generic features move least. Schedulers multiply each group's base LR, so ratios survive warmup and cosine. Traps: every parameter must be in exactly one group (duplicates error, omissions silently freeze); and `optimizer.load_state_dict` restores group hyperparameters on resume, overwriting a changed LR in your config unless you reset it afterward.

</details>


### [14-14] LR schedules in practice: warmup plus cosine vs constant plus cooldown, how do you pick warmup length, and what is an LR finder?  (difficulty 2; tags: training, optimizers, rapid-fire)


<details><summary>Answer</summary>

**Warmup + cosine** is the default: linear ramp over the first 1 to 3% of steps, then cosine decay to ~10% of peak. Warmup exists because Adam's second-moment estimate is unreliable at step 0 and early gradients are large, so a full LR immediately damages the initialization. **Warmup-Stable-Decay (WSD)** holds the LR constant and decays (linear or 1-sqrt) over the last 10 to 20%; you need not fix the step count in advance and can branch a cooldown from any checkpoint. Warmup length: longer for larger batch or LR; if the loss spikes as warmup ends, peak LR is too high. An **LR finder** sweeps LR exponentially for a few hundred steps and picks about 10× below where the loss starts rising. Trap: when fine-tuning from a checkpoint with fresh optimizer state, warm up again or expect a loss bump.

</details>


### [14-15] What is weight tying between input and output embeddings, why do it, and when does it hurt?  (difficulty 1; tags: transformers, training, rapid-fire)


<details><summary>Answer</summary>

**Weight tying** uses one matrix for the token embedding ($\text{vocab} \times d$ lookup) and the output projection ($d \to \text{vocab}$ logits): `lm_head.weight = tok_emb.weight`. Motivation: both map between token identity and the same semantic space, so sharing halves that parameter count (in GPT-2 small the embedding is ~30% of all parameters) and regularizes small models, improving perplexity. When it hurts: at scale the two roles diverge (input embeddings want to be well spread, output logits want calibration), so most models above a few billion parameters (LLaMA and successors) **untie**; with tying you must also manage scale, since the matrix is used at two very different magnitudes (hence the $\sqrt{d}$ input scaling in the original Transformer). Trap: tying breaks if you resize the vocab or apply different decay/init to the two uses, and FSDP or `torch.compile` wrapping must keep the parameter shared rather than cloning it.

</details>


### [14-16] What is label smoothing and what does it do to calibration and logits?  (difficulty 1; tags: training, loss, rapid-fire)


<details><summary>Answer</summary>

**Label smoothing** replaces the one-hot target with $(1-\epsilon) \cdot \text{one\_hot} + \epsilon/K$, so the cross-entropy asks for probability $1-\epsilon+\epsilon/K$ on the correct class and $\epsilon/K$ on every other class; typical $\epsilon = 0.1$. Why: with hard targets the loss is minimized only as the correct logit goes to $+\infty$, so the model keeps inflating logit magnitudes and becomes overconfident; smoothing gives a finite optimum and bounds the logit gap at $\log((1-\epsilon)(K-1)/\epsilon)$. Effects: better **calibration** (ECE drops, predictions are less overconfident), a small accuracy gain in classification and machine translation, and features that cluster tightly per class. It hurts knowledge distillation (the teacher's dark knowledge is flattened) and it is not used for language-model pretraining, where the next-token distribution is genuinely soft already. Trap: `nn.CrossEntropyLoss(label_smoothing=0.1)` smooths uniformly across all classes including padding tokens unless you also set `ignore_index`.

</details>


### [14-17] What are EMA weights, how do you choose the decay, and why keep them in fp32?  (difficulty 1; tags: training, diffusion, rapid-fire)


<details><summary>Answer</summary>

An **exponential moving average** of weights, $\theta_{\text{ema}} \leftarrow \beta \cdot \theta_{\text{ema}} + (1-\beta) \cdot \theta$ after every step, is cheap iterate averaging that smooths optimizer noise and gives better samples; nearly all diffusion models (DDPM, Stable Diffusion, EDM, WAN) evaluate and ship EMA weights. Decay: $\beta = 0.999$ to $0.9999$; the horizon is roughly $1/(1-\beta)$ steps, so 0.9999 averages the last ~10k steps. EDM2 instead parameterizes the horizon as a fraction of training length so it scales with the run. Keep the EMA in **fp32**: with $\beta = 0.9999$ the update is 1e-4 of the weight, below bf16's 1/256 resolution, so a bf16 EMA never moves. Traps: warm up $\beta$ ($\beta_t = \min(\beta, (1+t)/(10+t))$) so the EMA is not anchored at initialization; save the EMA in checkpoints; evaluate with EMA but monitor loss with raw weights, since the EMA lags by its horizon.

</details>


### [14-18] What does torch.compile do, what breaks it, and when do you use mode="max-autotune"?  (difficulty 2; tags: pytorch, performance, rapid-fire)


<details><summary>Answer</summary>

`torch.compile` traces the function with **Dynamo** (bytecode-level graph capture), lowers to **Inductor**, which fuses element-wise ops and reductions into generated Triton kernels and plans memory, and can wrap the graph in **CUDA graphs** to remove launch overhead. Typical wins are 1.3 to 2× on transformers, mostly from fusing bandwidth-bound glue (norms, activations, residual adds, RoPE). What breaks it: **graph breaks** on unsupported Python (data-dependent control flow, `.item()`, prints, C extensions) fall back to eager for that segment; **dynamic shapes** trigger recompiles per new shape unless `dynamic=True`; input mutation can also recompile. `mode="reduce-overhead"` enables CUDA graphs for small-batch inference; **`mode="max-autotune"`** additionally benchmarks many matmul/conv kernel configs, costing minutes of compile time for a few percent, worth it for fixed-shape production jobs. Trap: measure after warmup and run `torch._dynamo.explain` to find graph breaks before blaming the compiler.

</details>


### [14-19] What is channels_last memory format and why does it matter for convolutions on tensor cores?  (difficulty 2; tags: performance, pytorch, rapid-fire)


<details><summary>Answer</summary>

`channels_last` stores a 4-D tensor logically as NCHW but physically as **NHWC** (channel is the fastest-varying dimension), set by `x.to(memory_format=torch.channels_last)` and the same for the model. cuDNN's tensor-core convolution kernels for fp16/bf16 are written for NHWC, so with NCHW input PyTorch inserts a transpose before and after every conv, which is pure memory traffic; NHWC avoids it and typically yields 1.2 to 2× on conv nets in mixed precision. It also makes the per-pixel channel vector contiguous, which is the natural layout for the implicit-GEMM formulation of convolution. Transformers do not care (they are already token-major). Traps: mixing formats causes silent conversions on every op; `.view()` and some custom ops assume NCHW strides and will error or copy; check `x.is_contiguous(memory_format=torch.channels_last)`; and this is a strides trick, so `x.shape` still reads (N, C, H, W).

</details>


### [14-20] How do you tune a DataLoader, and how do you tell a run is data-bound?  (difficulty 1; tags: performance, pytorch, data, rapid-fire)


<details><summary>Answer</summary>

Knobs: **`num_workers`** (CPU processes for decode/augment; start at 4 to 8 per GPU, bounded by cores and RAM), **`pin_memory=True`** (page-locked host memory enables async copies with `.to(device, non_blocking=True)`), **`persistent_workers=True`** (do not respawn workers every epoch), and **`prefetch_factor`** (batches queued per worker, default 2). The bottleneck is usually JPEG/video decode and augmentation on CPU, or random reads from a network filesystem, not the GPU. Detecting a data-bound run: GPU utilization oscillates or sits low; the profiler shows gaps on the GPU timeline with `DataLoader.__next__` on the critical path; or time the loop with one cached batch repeated versus real loading, and the difference is your stall. Fixes: pre-decode to tensors or tar shards (WebDataset), GPU decode (DALI) for video, more workers and prefetch. Trap: too many workers thrash RAM (each holds `prefetch_factor` batches) and can be slower.

</details>


### [14-21] How do you make a PyTorch run reproducible, and why are some kernels nondeterministic?  (difficulty 2; tags: reproducibility, pytorch, debugging, rapid-fire)


<details><summary>Answer</summary>

Seed everything: `torch.manual_seed` (covers CUDA), `numpy.random.seed`, `random.seed`, and the DataLoader via `generator` and `worker_init_fn`; under DDP use the same seed for weights and a rank-offset seed for data. Then `torch.backends.cudnn.deterministic=True`, `benchmark=False` (benchmark autotunes kernels per shape, and different kernels round differently), or globally `torch.use_deterministic_algorithms(True)`, which errors on ops with no deterministic path and needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Why nondeterminism exists: float addition is not associative, and any kernel that reduces with **atomics** (scatter_add, index_add, embedding backward, some conv backward) sums in thread-completion order. **Flash-attention's backward** accumulates dQ with atomic adds across key blocks, so it is nondeterministic by design (a deterministic flag exists at a speed cost). In practice, chase bit-exactness only for debugging; for paired comparisons use identical seeds and report variance over several seeds.

</details>


### [14-22] What must a training checkpoint contain to resume correctly, and what are the state_dict prefix pitfalls?  (difficulty 2; tags: training, pytorch, checkpointing, rapid-fire)


<details><summary>Answer</summary>

A resumable checkpoint holds: **model** state_dict, **optimizer** state (Adam moments; resuming without them causes a loss spike), **LR scheduler**, **GradScaler** scale, **EMA** weights, global **step/epoch**, **RNG states** (torch CPU/CUDA, numpy, python), and the **dataloader position** (epoch plus sample offset, or a resumable sampler) so you do not re-see data. Prefix pitfalls: DDP wraps the model so keys become `module.layer...`, and `torch.compile` wraps it as `_orig_mod.layer...`; save `model.module.state_dict()` or the uncompiled model, or strip prefixes on load, otherwise `load_state_dict` fails with missing/unexpected keys, and with `strict=False` it silently loads nothing. Load with `map_location="cpu"` (or the local rank's device) so all ranks do not deserialize onto GPU 0 and OOM. Trap: write to a temp file and rename so a preemption mid-write does not leave a corrupt checkpoint; keep the last two.

</details>


### [14-23] Walk me through your OOM debugging checklist.  (difficulty 2; tags: memory, debugging, pytorch, rapid-fire)


<details><summary>Answer</summary>

First identify **which memory**: parameters, gradients and optimizer state are fixed (16 bytes/param for Adam in mixed precision); activations scale with batch × sequence × layers and are usually the culprit. `torch.cuda.memory_summary()` and `max_memory_allocated()` show peak vs reserved; a large gap means **fragmentation**, fixed by `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Then in order: smaller micro-batch with gradient accumulation; activation checkpointing; flash/SDPA attention so the $L \times L$ score matrix is never materialized; bf16 autocast; `zero_grad(set_to_none=True)`; fused optimizer to avoid temporaries; 8-bit Adam or Adafactor to shrink optimizer state 4×; FSDP/ZeRO sharding; CPU offload last. Check for leaks: accumulating loss tensors for logging without `.item()` keeps whole graphs alive. Trap: `torch.cuda.empty_cache()` frees nothing in use; it only returns cached blocks to the driver, so it never fixes a real OOM and just slows the next allocation.

</details>


### [14-24] Estimate the GPU memory to train a 1B-parameter model with Adam in mixed precision.  (difficulty 1; tags: memory, training, estimation, rapid-fire)


<details><summary>Answer</summary>

Per parameter with AdamW in mixed precision: **fp32 master weight (4) + fp32 gradient (4) + Adam m (4) + Adam v (4) = 16 bytes**, plus a bf16 working copy of weights (2) and often bf16 gradients (2), so 16 to 20 bytes/param. For 1B parameters that is **16 to 20 GB** before any activation. Activations are roughly $34 \cdot s \cdot b \cdot h$ bytes per transformer layer without flash attention (plus the $a \cdot s^2 \cdot b$ score matrix), or 10 to 15 bytes per token per hidden unit per layer with it; for a 1B model ($h = 2048$, 24 layers) at 8 × 2k tokens that is 20 to 30 GB, comparable to the static state. So 1B fits on one 80 GB GPU; 7B needs 112+ GB of static state alone and must be sharded or use 8-bit optimizer state. Trap: inference needs only 2 bytes/param plus KV cache.

</details>


### [14-25] What are fused kernels and fused/foreach optimizers, and why do they help?  (difficulty 1; tags: performance, optimizers, pytorch, rapid-fire)


<details><summary>Answer</summary>

Most training ops outside the matmuls are **memory-bandwidth bound**: each element-wise kernel reads and writes the whole tensor from HBM, so a chain of five small ops costs five round trips. A **fused kernel** does them in one pass while data sits in registers (fused LayerNorm, bias+GELU, RoPE, cross-entropy, and flash attention as the extreme case). Optimizers are the worst offender: naive Adam launches ~10 tiny kernels per parameter tensor, and a model has hundreds of tensors. `foreach=True` (now default) uses multi-tensor kernels that process a list of parameters per launch; **`fused=True`** does the entire Adam update in one CUDA kernel per group, 2 to 3× faster for the step with less temporary memory. NVIDIA Apex `FusedAdam` predates it and is equivalent. Trap: `fused=True` requires all parameters on CUDA in float dtypes; compiling the optimizer step with `torch.compile` gets similar fusion.

</details>


### [14-26] Is dropout still used in transformers, what is stochastic depth, and when do you need regularization vs more data?  (difficulty 2; tags: regularization, training, rapid-fire)


<details><summary>Answer</summary>

For large-scale pretraining dropout is **mostly off** (GPT-3 onward, LLaMA, DiT, WAN use dropout 0): in the single-epoch, data-abundant regime the model never overfits, and dropout just adds noise, slows convergence, and muddies scaling laws. **Stochastic depth** (drop-path) randomly skips whole residual blocks per sample with a rate increasing linearly with depth (0.1 to 0.3 for ViTs); it is the regularizer of choice for ViTs and ConvNeXts trained many epochs on ImageNet, where overfitting is real, and doubles as an ensemble of shallower nets. Rule: regularization (dropout, drop-path, weight decay, augmentation, label smoothing) matters when the model sees the same data many times; when you can scale data, scale data instead. Trap: for fine-tuning a big model on small data, LoRA plus early stopping often regularizes better than dropout, and enabling dropout in a model pretrained without it shifts activation statistics.

</details>


### [14-27] How do you freeze part of a model for fine-tuning, and what still consumes memory?  (difficulty 2; tags: finetuning, memory, pytorch, rapid-fire)


<details><summary>Answer</summary>

Set `p.requires_grad_(False)` on the frozen parameters and pass only trainable ones to the optimizer (LoRA, a head, or the last $N$ blocks). Frozen parameters then have no gradient and no optimizer state, so static memory drops to 2 bytes/param for a bf16 trunk. But **activations are still stored for any frozen layer that gradients must flow through**: if the trainable part is at the output (a head, or LoRA in every block), autograd needs each frozen layer's inputs to compute gradients with respect to its inputs, so activation memory barely changes. To avoid it, make sure nothing trainable sits before the trunk: run the trunk under `torch.no_grad()` and **`.detach()`** its output, or when training only the last $N$ blocks, wrap the first $L-N$ in no_grad. Trap: `model.train()` still updates BatchNorm running stats in the frozen trunk, so call `.eval()` on it.

</details>


### [14-28] DDP vs DataParallel, find_unused_parameters, SyncBatchNorm, gradient bucketing and no_sync(): explain briefly.  (difficulty 2; tags: distributed, pytorch, rapid-fire)


<details><summary>Answer</summary>

**DataParallel** is single-process multi-thread: it scatters the batch, replicates the model each step, and gathers outputs on GPU 0, so it is GIL-bound; never use it. **DistributedDataParallel** is one process per GPU with its own model copy; gradients are averaged with an **all-reduce** overlapped with backward, packed into **buckets** (default 25 MB) and reduced while earlier layers are still computing. `find_unused_parameters=True` makes DDP find parameters that got no gradient this step (conditional branches, a frozen encoder) so the reduction does not hang; it costs overhead and hides bugs. **SyncBatchNorm** computes BN statistics across ranks because per-GPU batches may be too small (LayerNorm models do not need it). **`model.no_sync()`** skips the all-reduce on accumulation micro-batches so you reduce once on the last. Trap: DDP broadcasts rank 0's weights at construction, and every rank must run the same number of forward calls or it deadlocks.

</details>
