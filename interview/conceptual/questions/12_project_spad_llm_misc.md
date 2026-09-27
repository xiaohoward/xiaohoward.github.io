# Project Grill: Single-Photon Imaging, LLM Work & Math
<!-- weight: 2 -->

## Q: Give me the two-minute pitch for the opportunistic single-photon time-of-flight paper, and tell me exactly what you contributed. {diff=2 tags=spad,tof,computational-imaging,project}
- Follow-up: why timestamps instead of histograms?
- Follow-up: what does "opportunistic" buy you over a pulsed laser?
### A
**Pitch.** A SPAD is a pixel that detects individual photons and timestamps each arrival with ~tens-of-picosecond resolution. Classic single-photon ToF fires a laser pulse and histograms photon arrival times against the pulse; the peak position gives round-trip time and hence depth. **Opportunistic ToF** drops the controlled laser: we recover depth from **ambient or uncontrolled light sources** — flickering/modulated lights, a screen, an unsynchronized source — whose temporal structure we do not control and often do not know in advance. The key idea is that if a source has *any* temporal structure, the scene returns are a delayed copy of it, so depth is encoded in the **cross-correlation** between a reference measurement of the source waveform and each pixel's photon stream: $\text{depth} = c \cdot \tau^*/2$ where $\tau^* = \arg\max$ of the correlation. Ultra-wideband sources (very short temporal features) sharpen that correlation peak, which is where the USRA-funded ultra-wideband work comes in.

**Histogram vs timestamp processing.** Histogramming bins arrivals into a fixed grid and then peak-finds — simple, but you throw away sub-bin timing and you must know the repetition period. **Working directly on raw timestamps** (per-photon likelihood or correlation against a continuous reference) keeps full timing precision, handles non-periodic sources, and lets you do MLE under a Poisson model rather than a heuristic peak-fit.

**My contribution** (say it in the first person, concretely): [fill in: e.g., built the timestamp-domain correlation / MLE pipeline, the pile-up-aware forward model, hardware capture experiments with the SPAD array, or the simulation and evaluation harness]. Be ready to name one specific experiment you ran and one number you own.

Common trap: don't say "we used a SPAD camera" and stop — interviewers want the *signal model*: photon arrivals are an inhomogeneous Poisson process with rate $\lambda(t) = \alpha \cdot s(t - \tau) + b$, and everything downstream is inference on $\tau$.

## Q: Explain pile-up in single-photon detection. How does it bias depth, and what are the fixes? {diff=2 tags=spad,poisson,statistics}
- Follow-up: write the corrected estimator.
### A
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

## Q: You come from computational imaging. Why is that background relevant to a generative-modeling internship? {diff=2 tags=inverse-problems,diffusion,posterior-sampling}
- Follow-up: sketch how you would use a pretrained diffusion model as a prior for a reconstruction problem.
### A
**Direct answer:** computational imaging *is* inverse problems with an explicit forward model plus a prior — and diffusion models are the best learned priors we have. The habits transfer: write down the forward model $y = A(x) + \text{noise}$, be honest about the noise statistics, and separate what the sensor knows from what the prior assumes.

Three concrete links:
- **Sensor-aware generation.** In foveated imaging we asked "which pixels are worth reading"; in foveated diffusion we ask "which tokens are worth denoising at full resolution." Same bandwidth-vs-fidelity question, same answer shape (spend compute where the task needs it). SPEED is the same idea in the frequency domain: don't compute frequencies the model can't resolve yet.
- **Diffusion as a prior (posterior sampling / DPS idea).** Sampling from $p(x \mid y) \propto p(y \mid x) \cdot p(x)$: run the reverse diffusion with the score of the prior, $s_\theta(x_t, t)$, and add a likelihood gradient. DPS approximates $\nabla_{x_t} \log p(y \mid x_t) \approx \nabla_{x_t} \log p(y \mid \hat{x}_0(x_t))$ using Tweedie's estimate $\hat{x}_0 = (x_t + \sigma_t^2 \cdot s_\theta)/\alpha_t$, giving the update $x_{t-1} \leftarrow \text{DDPM step} - \zeta \cdot \nabla_{x_t} \|y - A(\hat{x}_0)\|^2$. Cheap, training-free, works for deblurring, inpainting, super-resolution, and — relevant to my SPAD work — photon-limited (Poisson) likelihoods.
- **Knowing when the prior lies.** Diffusion posterior samplers hallucinate confidently when the likelihood is weak; imaging people are trained to check consistency with the measurement (data-fidelity residuals), which is exactly the discipline generative models for science / world models need.

Follow-up trap: DPS uses a point estimate $\hat{x}_0$ inside the likelihood, so it's a biased approximation of the true posterior score; better methods (e.g., ΠGDM, particle/SMC guidance) account for the variance of $x_0 \mid x_t$.

## Q: At Bell you built a document-retrieval system and fine-tuned open LLMs. Walk me through the RAG design and the fine-tuning, and tell me what you'd do differently now. {diff=2 tags=rag,llm,finetuning,project}
- Follow-up: how did you measure hallucination?
- Follow-up: SFT vs LoRA — how did you choose?
### A
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

## Q: Quick linear algebra: state the SVD, Eckart–Young, and condition number, and explain why each matters for LoRA and PCA. {diff=2 tags=linear-algebra,math,lora}
- Follow-up: why is LoRA rank-r a reasonable inductive bias for fine-tuning?
### A
**SVD.** Any $A \in \mathbb{R}^{m \times n}$ factors as $A = U \Sigma V^\top$ with $U, V$ orthonormal columns and $\Sigma = \operatorname{diag}(\sigma_1 \ge \sigma_2 \ge \dots \ge 0)$. Columns of $U$/$V$ are left/right singular vectors; $\sigma_i^2$ are eigenvalues of $A^\top A$.

**Eckart–Young (–Mirsky).** The best rank-$r$ approximation of $A$ in Frobenius or spectral norm is the truncated SVD $A_r = \sum_{i \le r} \sigma_i u_i v_i^\top$, with error $\|A - A_r\|_F^2 = \sum_{i>r} \sigma_i^2$ and $\|A - A_r\|_2 = \sigma_{r+1}$. **This is PCA**: center the data, take the top-$r$ right singular vectors $\to$ directions of maximal variance, and the captured variance fraction is $\sum_{i \le r} \sigma_i^2 / \sum_i \sigma_i^2$.

**Condition number.** $\kappa(A) = \sigma_{\max}/\sigma_{\min}$. It bounds how much relative error in $b$ amplifies into $x$ when solving $Ax = b$, and governs gradient-descent convergence on quadratics (rate $\sim (\kappa-1)/(\kappa+1)$). Ill-conditioned $\Rightarrow$ slow, noise-sensitive optimization; this is why we whiten inputs, use LayerNorm, and prefer Adam-style preconditioning.

**Why it matters for LoRA.** LoRA writes the weight update as $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$ — an explicit low-rank parameterization. The bet, justified empirically by low intrinsic dimension of fine-tuning updates, is that the *optimal* $\Delta W$ has a fast-decaying singular spectrum, so by Eckart–Young a rank-$r$ factor captures most of it with $r \cdot (d+k)$ parameters instead of $d \cdot k$. Practical corollaries: initialize $B = 0$ so $\Delta W$ starts at zero; scaling $\alpha/r$ keeps the update magnitude stable across ranks; and if you inspect a merged $\Delta W$'s singular values and they are flat, your rank is too low. Trap: LoRA constrains the *update* to be low rank, not the weights — the full $W$ stays full rank.

## Q: Probability rigor: derive Gaussian conditioning, the KL between two Gaussians, and explain where that KL appears in the DDPM loss. Then explain the reparameterization trick. {diff=2 tags=probability,gaussian,ddpm,vae}
### A
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

## Q: Rapid fire: Jacobian of softmax; gradient of $\|Ax - b\|^2$; $\mathbb{E}\|\epsilon\|^2$ for $\epsilon \sim \mathcal{N}(0, I_d)$. {diff=1 tags=math,calculus,warmup}
- Follow-up: why does $\mathbb{E}\|\epsilon\|^2 = d$ matter for diffusion / for attention scaling?
### A
**Softmax Jacobian.** With $s = \operatorname{softmax}(z)$, $s_i = e^{z_i}/\sum_j e^{z_j}$:

$$
\frac{\partial s_i}{\partial z_j} = s_i (\delta_{ij} - s_j) \qquad \text{i.e.}\quad J = \operatorname{diag}(s) - s s^\top
$$

$J$ is symmetric, PSD, rank $d-1$ (rows sum to zero — softmax is shift-invariant). Vector-Jacobian product for backprop: $J^\top g = s \odot (g - \langle g, s \rangle)$. Combined with cross-entropy the whole thing simplifies to $s - y$, which is why we fuse them.

**Gradient of $\|Ax - b\|^2$.** $f(x) = (Ax-b)^\top(Ax-b)$, so $\nabla f = 2A^\top(Ax - b)$, Hessian $2A^\top A$. Setting $\nabla f = 0$ gives the normal equations $A^\top A x = A^\top b \to x^* = A^+ b$. Note the condition number of the problem is $\kappa(A)^2$, which is why you solve via QR/SVD rather than forming $A^\top A$.

**$\mathbb{E}\|\epsilon\|^2$ for $\epsilon \sim \mathcal{N}(0, I_d)$.** Each $\epsilon_i^2$ has mean 1, so $\mathbb{E}\|\epsilon\|^2 = d$; $\|\epsilon\|^2 \sim \chi^2_d$, variance $2d$, so $\|\epsilon\| \approx \sqrt{d} \pm O(1)$ — Gaussian samples concentrate on a thin shell of radius $\sqrt{d}$.

**Why it matters.** (1) Attention: $q \cdot k$ for random unit-variance $q, k$ has variance $d_k$, so scores are divided by $\sqrt{d_k}$ to keep softmax from saturating. (2) Diffusion: at $t = 1$, $x_t \approx \epsilon$ lives on a shell of radius $\sqrt{d}$; the ortho-normalized DCT preserves norms, so noise has unit power per frequency bin — that's the $P(f) = 1$ SNR line in SPEED. (3) The "simple" DDPM loss $\|\epsilon - \epsilon_\theta\|^2$ has a natural scale of $d$ at init.

## Q: Explain your Virasoro / diffeomorphism math paper in one paragraph for a non-mathematician, then tell me what it trained you in. {diff=1 tags=math,background,communication}
### A
**One paragraph, plain language.** Take all the smooth ways to stretch and reparameterize a circle — these form an infinite-dimensional group of *diffeomorphisms*, and its algebra of infinitesimal stretches has a famous "central extension" called the **Virasoro algebra**, which is the symmetry algebra of 2D conformal field theory and string theory. The paper asks what happens when the reparameterizations are allowed to have **breaks** — points where the map is continuous but not smooth, or has jumps in derivative. We work out how the Virasoro-type extension (the extra term that measures how much two stretches fail to commute in a way the classical picture misses — the Gelfand–Fuks cocycle) must be modified to make sense on this larger, less regular space. [fill in: the concrete result — e.g., classification of the admissible cocycles / how the central charge term picks up boundary contributions at the breaks.]

**What it trained me in** (this is what the interviewer wants):
- **Working with infinite-dimensional objects carefully** — the same instinct that says "what regularity does this actually need" is what I use when reasoning about score functions, continuous-time flows, and probability-flow ODEs.
- **Cohomology / extension thinking**: what is the *minimal* extra structure that makes a construction consistent. That shows up in ML as "what is the minimal change to the sampler that keeps it correct" — e.g., resetting the multistep solver history at a resolution change in SPEED.
- **Proof hygiene**: stating assumptions explicitly, finding the counterexample first. In experiments that becomes: design the control that would falsify the claim before running the headline number.
- **Communicating abstract ideas** — I TA'd advanced linear algebra, which is where my habit of always giving the 2×2 example comes from.

<!-- rapid-fire -->
## Q: What is a SPAD? {diff=1 tags=spad,rapid-fire,sensors}
### A
A **single-photon avalanche diode is a photodetector biased above breakdown so that a single photon triggers a self-sustaining avalanche, giving a digital click with picosecond-scale timing**. Instead of integrating charge like a CMOS pixel, it outputs a stream of photon-arrival timestamps, which is what makes it useful for time-of-flight depth, low-light imaging, and fluorescence lifetime. Costs: after each detection it is dead for tens of nanoseconds, so detections are lost at high flux (pile-up), there is dark-count noise, and fill factor is low. Follow-up: the data is a Poisson process, so the whole processing pipeline is statistical — histogramming timestamps, then estimating the depth peak.

## Q: What is time-of-flight depth? {diff=1 tags=tof,rapid-fire,depth,computational-imaging}
### A
**Time-of-flight depth measures distance by timing how long light takes to travel to a surface and back: $d = c \cdot \tau/2$.** Direct ToF sends a short pulse and timestamps the return (a SPAD histograms many returns and finds the peak); indirect ToF modulates a continuous source and measures phase shift. Our paper's twist was **opportunistic** ToF: rather than a controlled laser, use ambient or uncontrolled light sources whose temporal structure we do not own, and recover depth by cross-correlating the received photon stream with a reference measurement of the source. Trap for a one-minute answer: mention the timing resolution to depth conversion, 1 ns is about 15 cm round-trip, so picosecond timing is what makes millimeter depth possible.

## Q: What is pile-up, in one sentence? {diff=1 tags=spad,rapid-fire,pile-up,poisson}
### A
**Pile-up is the bias that occurs when a SPAD detects the first photon of each cycle and then goes dead, so at high flux early photons are over-represented and the measured histogram is skewed toward earlier times, biasing depth closer.** The math: the measured histogram is the true arrival distribution times the probability that no photon arrived earlier, a survival term like $\exp(-\Lambda(t))$, which is Poisson statistics. Fixes: attenuate the light so photons per cycle are well below one, invert the distortion analytically using the known Poisson model, or use asynchronous / free-running acquisition so the dead time is decorrelated from the cycle. Follow-up: pile-up is not noise, it is a deterministic bias, so averaging more does not remove it.

## Q: What is RAG, in one sentence? {diff=1 tags=rag,rapid-fire,llm}
### A
**Retrieval-augmented generation retrieves relevant documents for a query, usually by embedding similarity, and puts them in the LLM's context so the model answers from those sources instead of only its weights.** Why: it gives up-to-date, private, or domain-specific knowledge without retraining, and makes answers attributable. At Bell I built this over internal documents and code with LangChain: chunk documents, embed, store in a vector index, retrieve top-k, then prompt. The practical traps are retrieval quality (chunking, hybrid keyword plus dense search, reranking) and evaluation, since a fluent answer over the wrong chunks is the common failure. Follow-up: when to fine-tune instead — for style and format, not for facts that change.

## Q: What is LoRA fine-tuning of an LLM, in one sentence? {diff=1 tags=lora,rapid-fire,finetuning,llm}
### A
**LoRA freezes the pretrained weights and trains a low-rank update $\Delta W = B \cdot A$ ($A$ is $r \times d$, $B$ is $d \times r$, $r$ small) added to selected linear layers, usually attention projections, so fine-tuning touches a fraction of a percent of the parameters.** Why it works: the weight change needed to adapt a large model has low intrinsic rank, so rank 8 to 64 captures most of it. Benefits: tiny checkpoints, no inference change once merged ($W + BA$), and far less memory since optimizer states exist only for $A$ and $B$. I used it at Bell on open LLMs and in SPEED and foveated diffusion on DiTs. Trap: LoRA is weak for adding new knowledge; it is best for style, format, and behavior.

## Q: What is the SVD, in one sentence, and what is it used for? {diff=1 tags=linear-algebra,rapid-fire,svd,math}
### A
**The singular value decomposition writes any matrix as $A = U \cdot \Sigma \cdot V^\top$, with $U$ and $V$ orthogonal and $\Sigma$ diagonal with non-negative singular values, so $A$ is a rotation, an axis-aligned scaling, and another rotation.** It exists for every matrix, square or not. Uses: the best rank-$k$ approximation is the truncated SVD (Eckart–Young), which underlies PCA, low-rank compression, and the intuition behind LoRA; the ratio of largest to smallest singular value is the condition number, which tells you how numerically stable a solve or a gradient will be; and the pseudo-inverse and least-squares solutions come from it directly. One-liner follow-up: singular values of $A$ are square roots of eigenvalues of $A^\top A$.
