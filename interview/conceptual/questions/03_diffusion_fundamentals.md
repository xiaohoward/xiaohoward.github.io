# Diffusion & Flow Fundamentals
<!-- weight: 5 -->

## Q: Write down the DDPM forward process and derive the closed form of $q(x_t \mid x_0)$. Why does that closed form matter for training? {diff=1 tags=diffusion,forward-process,derivation}
- Follow-up: what is $\bar\alpha_T$ for the original linear schedule, and is it exactly zero?
### A
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

## Q: Derive the simplified $\epsilon$-prediction loss from the ELBO. Why is dropping the per-timestep weight okay, and how does $\epsilon$-prediction relate to score matching? {diff=2 tags=diffusion,elbo,score-matching,derivation}
- Follow-up: write the score of $q(x_t \mid x_0)$ in terms of $\epsilon$.
### A
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

## Q: Give the SDE view of diffusion and derive the probability-flow ODE. Why does the ODE give deterministic sampling, and when would you still prefer the SDE? {diff=3 tags=diffusion,sde,ode,derivation}
- Follow-up: what does "same marginals" mean here, and how do you show it?
### A
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

## Q: Show that DDIM is a discretization of the probability-flow ODE. Why is it better than plain Euler on the raw ODE? {diff=2 tags=diffusion,ddim,ode,samplers}
- Follow-up: what does $\eta$ do, and what happens to inversion when $\eta > 0$?
### A
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

## Q: Derive the flow-matching / rectified-flow objective for $x_t = (1-t)\,x_0 + t\,\epsilon$. How does it relate to v-prediction and to diffusion, and why do "straight paths" help few-step sampling? {diff=2 tags=flow-matching,rectified-flow,derivation}
- Follow-up: is the learned marginal velocity field straight?
### A
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

## Q: Compare the $\epsilon$, $x_0$, $v$ and flow-velocity parameterizations. Give the conversions, and tell me which is numerically stable at which SNR and why. {diff=2 tags=diffusion,parameterization,numerics}
- Follow-up: what does EDM's preconditioning do differently?
### A
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

## Q: Compare the linear and cosine noise schedules. What is the log-SNR view, and what is the "zero terminal SNR" problem? {diff=1 tags=diffusion,noise-schedule,snr}
- Follow-up: does the training schedule have to match the sampling schedule?
### A
**A schedule is just a monotone map from time to log-SNR $\lambda = \log(\alpha^2/\sigma^2)$; linear wastes steps near pure noise and doesn't reach SNR=0, cosine spends time more evenly, and both facts matter for sampling.**

- Linear $\beta$ (DDPM): $\beta$ from $10^{-4}$ to $0.02$. $\bar\alpha_t$ decays fast; the last ~20% of timesteps are visually pure noise — wasted capacity — yet $\bar\alpha_T \approx 4 \times 10^{-5} \neq 0$.
- Cosine (Nichol & Dhariwal): $\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$, $s = 0.008$. $\bar\alpha$ decreases nearly linearly in the middle, more steps at mid-SNR where structure is decided; better at low resolution.
- **Log-SNR view** (Kingma; EDM): what matters is the distribution of $\lambda$ over training timesteps and the weighting $w(\lambda)$. Two schedules with the same $p(\lambda)\, w(\lambda)$ train the same model; the schedule in $t$ is an implementation detail. FM-linear: $\lambda = 2 \log\big((1-t)/t\big)$, logistic; VP-cosine: $\lambda = -2 \log \tan(\pi t/2)$.

Zero terminal SNR (Lin et al. 2024): at inference you start from $\mathcal{N}(0, I)$ (SNR = 0) but the model was trained at $\mathrm{SNR}_T > 0$, where $x_T$ still carries mean/low-frequency information about $x_0$ (in latent space the DC component is huge). Result: the model assumes a nonzero mean is present, so it can't produce very dark or very bright images and everything drifts to medium brightness. Fix: rescale $\sqrt{\bar\alpha}$ so $\bar\alpha_T = 0$ exactly, switch to **$v$-prediction** ($\epsilon$ is degenerate at SNR 0), sample starting from the last timestep, and rescale CFG to avoid over-exposure.

Follow-up: no — sampling steps are chosen separately (Karras $\rho$-schedule, "shift", trailing spacing). Training $p(\lambda)$ affects what the model learned; sampling $\sigma_i$ only affects discretization error.

## Q: Why does a higher-resolution (or video) model need "more noise", and how do you derive the timestep shift used by SD3/FLUX/WAN? Tie it to the power spectrum. {diff=3 tags=diffusion,noise-schedule,resolution,spectral,shift}
- Follow-up: FLUX uses a "dynamic shift" that depends on token count — what is it doing?
### A
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

## Q: Uniform timestep sampling versus logit-normal, min-SNR-$\gamma$, P2 — what problem is each fixing, and how do they fit into one framework? {diff=2 tags=diffusion,loss-weighting,training}
- Follow-up: why is uniform-$t$ $\epsilon$-loss "secretly" weighted toward high SNR?
### A
**All of them are choosing the effective weight $w(\lambda)\, p(\lambda)$ over log-SNR. Uniform $t$ with $\epsilon$-loss puts most of the gradient budget on high-SNR, perceptually cheap timesteps; the alternatives move it to the mid-SNR region where structure is learned and gradients across timesteps conflict least.**

- **Uniform $t$, $\epsilon$-loss**: the implicit weight relative to an $x_0$ loss is $\mathrm{SNR}(t)$. With linear/cosine schedules, most of the timesteps sit at high SNR, so training is dominated by "remove a little noise" — fine texture — while the hard mid-SNR steps get little signal.
- **Logit-normal (SD3, FLUX)**: $t = \operatorname{sigmoid}(u)$, $u \sim \mathcal{N}(m, s)$. For velocity prediction the target $\epsilon - x_0$ is easy at both ends (at $t \approx 1$ you only need $\mathbb{E}[x_0] \approx 0$; at $t \approx 0$ $\epsilon$ is unpredictable noise, irreducible loss) and hard in the middle, so sampling concentrates there. SD3 found $m=0$, $s=1$ best; shifting $m$ with resolution is another way to express the timestep shift.
- **Min-SNR-$\gamma$** (Hang et al.): weight the $\epsilon$-loss by $\min(\mathrm{SNR}, \gamma)/\mathrm{SNR}$, $\gamma = 5$. Views timesteps as a multi-task problem with conflicting gradients; clipping stops the high-SNR tasks from dominating; 3× faster convergence.
- **P2** (Choi et al.): weight $1/(k + \mathrm{SNR})^\gamma$, explicitly down-weighting high SNR ("imperceptible details") in favor of the "content" and "coarse" stages.
- **EDM**: sample $\ln \sigma \sim \mathcal{N}(-1.2, 1.2^2)$ and weight by $(\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\, \sigma_{\text{data}})^2$ so the loss is roughly uniform in the unit-variance target — log-normal in $\sigma$ is the same idea as logit-normal in $t$.

Unified view (Kingma & Gao): the objective is $\int w(\lambda) \cdot \mathbb{E}\|x_0 - \hat x_0\|^2\, p(\lambda)\, \mathrm{d}\lambda$; schedule, parameterization, and sampling density are three ways to set the same curve. Practical follow-up: when you fine-tune at a new resolution, re-check the $\lambda$ distribution — the schedule that trained the base may put almost no mass where the new resolution's structure is decided.

## Q: Derive classifier-free guidance. What distribution does it sample from, and why does high guidance oversaturate and lose diversity? {diff=2 tags=diffusion,cfg,guidance,derivation}
- Follow-up: what does the model actually compute per sampling step with CFG?
### A
**CFG replaces a classifier gradient with the difference between conditional and unconditional scores, extrapolating past the conditional prediction; the result sharpens $p(c \mid x)$ like a temperature, which is exactly what removes diversity and pushes values out of range.**

Derivation: Bayes gives $\nabla \log p(x_t \mid c) = \nabla \log p(x_t) + \nabla \log p(c \mid x_t)$. Classifier guidance scales the second term: $\nabla \log p(x_t) + w \cdot \nabla \log p(c \mid x_t)$. Without a classifier, use $\nabla \log p(c \mid x_t) = \nabla \log p(x_t \mid c) - \nabla \log p(x_t)$:
$$
\tilde s = s_u + w\,(s_c - s_u) \quad \Leftrightarrow \quad \tilde\epsilon = \epsilon_u + w\,(\epsilon_c - \epsilon_u) = (1-w)\,\epsilon_u + w\,\epsilon_c
$$
The unconditional model is the same network with the condition dropped ($\varnothing$ token) 10–20% of the time in training. $w=1$ is pure conditional, $w>1$ extrapolates.

What it samples: the guided score corresponds to $\tilde p(x \mid c) \propto p(x \mid c)^w \cdot p(x)^{1-w} = p(x) \cdot p(c \mid x)^w$ — a **tempered posterior over $c$**. Caveat: the diffused version of this sharpened density is not what you get by sharpening the diffused scores at each $t$, so the guided trajectory is not an exact sampler of anything; it is a heuristic that happens to work.

Why oversaturation: $(\epsilon_c - \epsilon_u)$ at high noise is mostly a low-frequency "move toward the class mode" vector; multiplying it by $w$ = 7–10 gives predicted $\hat x_0$ values that leave the data range $[-1, 1]$ (or unit-variance latent range). Each step compounds this — colors clip, contrast blows up, textures become "crispy". Diversity: $p(c \mid x)^w$ with large $w$ is a near-delta on the most prototypical $x$ for $c$; samples collapse toward the mode (the mode-seeking direction of the extrapolation), which is why FID rises while CLIP score rises.

Per step: two forward passes (conditional + unconditional, usually batched as $2 \times B$), so **CFG doubles NFE**; with a negative prompt or an extra condition (image + text, InstructPix2Pix) it is three.

## Q: How do you get the benefits of strong CFG without the saturation? Compare guidance interval, dynamic thresholding, APG, autoguidance, and guidance distillation — and what does distillation cost you? {diff=3 tags=diffusion,cfg,guidance,distillation}
- Follow-up: why can't you stack CFG on top of a guidance-distilled model like FLUX.1-dev?
### A
**Each fix attacks a different symptom: where in the trajectory guidance is applied, whether the guided $\hat x_0$ stays in range, which component of the guidance vector you keep, or what "bad" model you subtract. Distillation removes the two-pass cost but freezes one guidance behavior into the weights.**

- **Guidance interval** (Kynkäänniemi et al. 2024): apply CFG only for mid noise levels. At high $\sigma$ the guided direction chooses a mode and kills diversity; at low $\sigma$ it is unnecessary and only adds saturation. Turning it off at both ends improves FID and lets you raise $w$.
- **Dynamic thresholding** (Imagen): at each step clip $\hat x_0$ to the $s$-th percentile of $|\hat x_0|$ and divide by $s$ ($s>1$). Keeps the prediction in range; pixel-space only, meaningless in latents.
- **CFG rescale** (Lin et al.): rescale the guided output's per-channel std back to the conditional's std, blended by $\phi \approx 0.7$. Cheap, standard with zero-terminal-SNR.
- **APG (adaptive projected guidance)**: decompose $\Delta = \epsilon_c - \epsilon_u$ into components parallel and orthogonal to $\epsilon_c$. The parallel part mostly scales the prediction (saturation); the orthogonal part carries the "which mode" quality signal. Keep orthogonal, down-weight parallel ($\eta \approx 0$), add momentum and a norm cap. Lets you use $w \approx 15$ without burn.
- **Autoguidance** (Karras 2024): guide with a smaller/less-trained version of the same model: $\tilde\epsilon = \epsilon_{\text{bad}} + w(\epsilon_{\text{good}} - \epsilon_{\text{bad}})$. Decouples "sharpen $p(c \mid x)$" from "remove the errors a weak model makes"; improves quality without the diversity loss because you are not tempering the class posterior.
- **Guidance distillation** (Meng et al.; FLUX.1-dev, SD3-turbo lineage): train a student $\epsilon_\theta(x, t, c, w)$ to regress the two-pass guided teacher output with $w$ embedded. Halves inference cost.

Costs of distillation: the model's distribution is the guided one — slightly mode-collapsed and saturated; you lose the unconditional branch, so **negative prompts and true CFG are gone**. Stacking CFG on FLUX.1-dev "double-guides" a model that already has guidance baked in; you get burnt images. That is why the candidate's Qwen-Image experiments use true two-pass CFG while FLUX-dev runs single-pass. Follow-up on cost: true CFG doubles NFE and, batched, doubles activation memory; sequential halves memory but doubles latency.

## Q: Compare Euler, Heun, DPM-Solver++ and UniPC. What history does each keep, and what breaks if you change the state mid-trajectory — say you change resolution — without resetting the solver? {diff=3 tags=diffusion,samplers,ode-solvers,multistep}
- Follow-up: in diffusers, what exactly do you reset?
### A
**Euler and Heun are memoryless single-step methods; DPM-Solver++(2M) and UniPC are multistep — they build higher-order accuracy from a buffer of previous model outputs. That buffer assumes one continuous trajectory in one state space, so any discontinuity in $x$ makes the stored derivative estimates wrong.**

Work in EDM form: $\mathrm{d}x/\mathrm{d}\sigma = (x - \hat x_0(x, \sigma))/\sigma$.
- **Euler**: $x_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) \cdot d_i$. First order, 1 NFE/step, no history.
- **Heun**: Euler predictor, then average slope at both ends; second order, 2 NFE/step. EDM found it better than Euler at equal NFE; still memoryless.
- **DPM-Solver++**: exploits the semilinear structure — integrates the linear part exactly (exponential integrator in $\lambda$ = log-SNR) and Taylor-expands $\hat x_0(\lambda)$. The 2M (multistep) variant estimates the first derivative of $\hat x_0$ from the **previous step's model output**; 1 NFE/step, second order. Keeps a buffer of 1 (order 2) or 2 (order 3) past $\hat x_0$'s plus their $\lambda$'s.
- **UniPC**: unified predictor–corrector; the corrector reuses the *new* evaluation to refine the previous step, so it is one NFE for higher effective order. Keeps $k$ past outputs (`model_outputs`), their timesteps, and an order counter that ramps 1→2→3 at the start (`lower_order_nums`).

What breaks: the finite-difference derivative $(\hat x_0^{(i)} - \hat x_0^{(i-1)})/(\lambda_i - \lambda_{i-1})$ assumes both outputs are samples of the same smooth function along one trajectory. If you upsample the latent, re-noise, swap conditioning or change the prompt mid-way, the buffered outputs are (a) the wrong shape — crash, or (b) the right shape after resizing but from a different trajectory and a different SNR — the extrapolation is garbage, you get ringing, ghosted structure, or blown contrast that shows up one or two steps after the transition and looks like a model bug. Euler has no such problem, which is why staged-resolution pipelines either use Euler across a transition or **reset the solver**: clear the output buffer, set the order counter to 0 so the next steps are first-order again, and set the step index so the $\sigma$ schedule is picked up at the right place. In diffusers that is `scheduler.model_outputs = [None]*k`, `lower_order_nums = 0`, `_step_index` / `set_begin_index`.

Independent of solver: the $\sigma$ grid (Karras $\rho=7$, shift, trailing spacing) is a separate choice; a good grid matters as much as solver order at ≤20 steps.

## Q: Going from 50 steps to 10, what degrades and why? Separate discretization error from model error, and explain exposure bias and how autoguidance or noise augmentation address it. {diff=2 tags=diffusion,samplers,exposure-bias,error-analysis}
- Follow-up: why does quality plateau at ~30–50 steps and sometimes get *worse* with more deterministic steps?
### A
**Three error sources: discretization error (shrinks with more steps and solver order), model error (independent of steps — the network is not the true score), and exposure bias (the model is evaluated on its own imperfect $x_t$, which is off the training distribution). Cutting steps mostly raises the first; the plateau is set by the other two.**

- **Discretization**: the ODE is integrated with steps of size $h$; local error $O(h^{p+1})$, global $O(h^p)$ for a $p$-th order solver. The trajectory is curved because the marginal velocity changes along the path — at large steps the model gives an *instantaneous* velocity where you need the *average* one. Symptoms at 10 Euler steps: blur, missing high-frequency detail, oversmoothed textures; the coarse layout is fine because it is decided by the first few steps anyway.
- **Model error**: $\epsilon_\theta \neq$ true $\epsilon$. Does not shrink with steps; at 50+ steps the deterministic sampler is essentially converged and the remaining gap is this. Stochastic samplers can *reduce* it: the Langevin term re-projects onto $p_t$, which is why EDM churn / SDE samplers beat ODE samplers at high NFE.
- **Exposure bias**: training uses $q(x_t \mid x_0)$ — exact noising of real data — but at inference $x_t$ is the sampler's own output, carrying the accumulated errors. The network extrapolates; errors compound over steps. This is also why more deterministic steps can get slightly worse: you take more opportunities to drift.

Fixes:
- **Noise augmentation / input perturbation** in training: add extra Gaussian noise to $x_t$ (or to the conditioning signal in cascades — Imagen, SVD, SDXL refiner) so the model sees inputs that look like imperfect predictions, and condition on the augmentation level.
- **Autoguidance**: $\tilde\epsilon = \epsilon_{\text{bad}} + w(\epsilon_{\text{good}} - \epsilon_{\text{bad}})$. The weak model's errors are amplified versions of the strong model's, so the difference cancels systematic bias — including the drift from being off-distribution.
- **$\epsilon$-scaling** at inference or SDE "churn" to re-noise and let the model correct.
- **Distillation** is the real few-step answer: teach the student the average velocity over a big step so discretization error disappears by construction.

## Q: Compare progressive distillation, consistency models, DMD, and adversarial distillation. What do you give up when you go to one step? {diff=3 tags=diffusion,distillation,few-step}
- Follow-up: why is DMD mode-seeking, and is that good or bad?
### A
**All of them teach a student to jump along the teacher's trajectory with one or a few evaluations; they differ in whether they supervise the trajectory (progressive, consistency) or the distribution (DMD, GAN), and that determines diversity, training cost, and what downstream features survive.**

- **Progressive distillation** (Salimans & Ho): student learns one step = two teacher DDIM steps; halve the step count each round. Needs $v$-parameterization (bounded targets at low step counts). Error compounds over rounds; $\log_2(T)$ training phases; keeps the deterministic map so it inherits teacher diversity.
- **Consistency models** (Song et al.): learn $f(x_t, t) = x_0$ for all $t$ on the same ODE trajectory with boundary condition $f(x, \epsilon) = x$ enforced by a skip parameterization. Train by consistency distillation (adjacent points from one teacher ODE step) or from scratch. One-step by construction; multistep by re-noising. LCM applies this to latents with CFG scale as input. Weakness: matching at adjacent points accumulates error along the trajectory; blurry at one step.
- **DMD** (Yin et al.): distribution matching. A one-step generator $G$ is trained so the score of its output distribution matches the teacher's: $\nabla_\theta \approx \mathbb{E}\big[(s_{\text{real}}(x_t) - s_{\text{fake}}(x_t)) \cdot \partial G/\partial \theta\big]$, where $s_{\text{fake}}$ is an auxiliary diffusion model kept trained on $G$'s outputs (same idea as VSD). DMD2 drops the regression loss, uses two-timescale updates and adds a GAN loss. It minimizes reverse KL → **mode-seeking**: excellent fidelity, reduced diversity. Three networks in memory.
- **Adversarial** (ADD/SDXL-Turbo, LADD): discriminator on the student's outputs (in pixel space with a DINO backbone, or in latent space using teacher features) plus a score-distillation term. Fast, sharp, but GAN instability and mode dropping.

What one step costs: diversity and mode coverage (reverse-KL or GAN objectives); no ability to trade steps for quality; CFG is baked in (you cannot change $w$, use negative prompts, or apply guidance interval); inversion and editing tricks break because there is no trajectory; ControlNet/adapter compatibility often degrades; artifacts in fine text and hands. The practical sweet spot is 4–8 steps, where a distilled model keeps most of the teacher's controllability. Follow-up: reflow (rectified flow) is the FM-native alternative — straighten first, then a one-step student is almost exact.

## Q: Why does EMA of the weights matter so much for diffusion models, and what can go wrong with it? {diff=1 tags=diffusion,training,ema}
- Follow-up: how should the EMA half-life scale with training length?
### A
**The diffusion loss is intrinsically noisy — the regression target is a random $\epsilon$ that cannot be predicted, so the minimum loss is nonzero and every minibatch gradient carries large target variance. Raw weights jitter around the basin; EMA averages that jitter out and gives a lower-variance estimate of the denoiser, which can be worth several FID points, sometimes 2×.**

$\theta_{\text{ema}} \leftarrow \beta\, \theta_{\text{ema}} + (1-\beta)\, \theta$, $\beta \approx 0.9999$, i.e. an average over roughly $1/(1-\beta) \approx 10\text{k}$ steps. This is Polyak averaging; for a noisy quadratic it converges to the basin center while the iterate oscillates. Diffusion is worse than classification here because the target noise never decreases with training, and because sample quality is very sensitive to small weight perturbations at high-SNR timesteps.

Things that go wrong:
- **No warm-up**: with $\beta = 0.9999$ from step 0 the EMA is dominated by random init for thousands of steps; use a $\beta$ ramp or start EMA later.
- **Wrong length for the run**: too short and you keep the noise; too long and you average over pre-convergence weights (a common reason a short fine-tune "does nothing" — the EMA barely moved). Karras et al. 2024 show the optimal EMA length scales with training length and batch size, and propose power-function EMA plus **post-hoc EMA** (store a few snapshots, reconstruct any EMA length after the fact).
- Precision: keep the EMA copy in fp32; bf16 EMA loses the small updates.
- Sampling with the raw weights by mistake during eval — one of the first things to check when "the loss is fine but samples are bad".
- With LoRA or DMD-style distillation the EMA copy should be of the trained parameters only.

## Q: Latent versus pixel diffusion — what is the VAE's role, why is the KL weight so small, and what is the scaling factor for? {diff=1 tags=diffusion,latent,vae}
- Follow-up: why do 16- or 32-channel VAEs make the diffusion model's job harder?
### A
**The VAE is a perceptual compressor that gives the diffusion model an 8× (spatially) smaller, roughly unit-variance, smooth space; the KL term is a light regularizer, not a generative prior; the scaling factor makes the latents match the noise schedule's assumption that data has unit variance.**

Cost motivation: a DiT's attention is quadratic in tokens. 1024² pixels → 128² latents (8×) → 64² tokens with 2×2 patchify = 4096 tokens. Video VAEs (WAN: 4× temporal, 8× spatial, 16 channels) matter even more.

VAE training: encoder gives $\mu, \sigma$; reconstruction with L1 + LPIPS + a patch GAN loss for sharpness; the KL to $\mathcal{N}(0, I)$ has weight ~1e-6. Reason: a real VAE with a strong prior gets posterior collapse / blurry reconstructions; here we only want the latent space **smooth and centered without holes**, so the diffusion model doesn't have to learn a jagged manifold. It is essentially a regularized autoencoder.

Scaling factor: latents come out with std ≈ 5.5 for SD1.x, so `0.18215 = 1/std` scales them to unit variance. Diffusion's SNR definition assumes unit-variance data; without the scaling the whole schedule shifts (too much signal at every $t$). FLUX/SD3 use a per-model shift and scale (e.g. shift 0.1159, scale 0.3611). Forgetting to apply — or double-applying — this factor is a classic "trains but samples garbage" bug.

Trade-offs: latent diffusion has a quality ceiling set by the decoder (text, faces, thin structures) and the 8× compression; pixel diffusion has none but must use cascades or heavy patching (PixelGen in the candidate's evaluation). More channels (16, 32 in DC-AE) improve reconstruction but the latent distribution gets harder to model — more information per token, less spatially redundant — so you need more diffusion compute or representation alignment (VA-VAE, REPA-style). Latents still show a power-law spectrum, just with a different $\beta$ than pixels.

## Q: Derive why diffusion generates coarse-to-fine. Under $x_t = (1-t)\,x_0 + t\,\epsilon$ with data power spectrum $P(f)$, at what time does frequency $f$ become resolvable, and what does that imply for compute? {diff=3 tags=diffusion,spectral,coarse-to-fine,derivation,speed}
- Follow-up: what does the optimal denoiser do to frequencies below the noise floor?
### A
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

## Q: Your diffusion model trains with a healthy loss curve but samples garbage. Walk me through your debugging checklist. {diff=2 tags=diffusion,debugging,training,practical}
- Follow-up: what single test localizes a train/inference mismatch fastest?
### A
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

<!-- rapid-fire -->

## Q: What is a diffusion model, in one sentence? {diff=1 tags=diffusion,definition,rapid-fire}
### A
**A diffusion model learns to reverse a fixed process of gradually adding Gaussian noise to data, by training a network to denoise at every noise level, then generates by starting from pure noise and denoising step by step.**

Why it works: instead of learning the whole distribution in one shot (GAN, VAE), it breaks generation into many small, easy conditional steps — each a regression problem with a plain MSE loss, stable to train and scalable. The network implicitly learns the score $\nabla_x \log p_t(x)$ at every noise level, and sampling is numerical integration of an ODE or SDE driven by it.

Trap: the cost moves to inference — tens of network evaluations per sample instead of one — which is why samplers, distillation, and coarse-to-fine tricks like SPEED are a research area.

## Q: What is the noise schedule? {diff=1 tags=diffusion,noise-schedule,rapid-fire}
### A
**The noise schedule says how much signal and noise are mixed at each timestep: $x_t = \alpha_t\, x_0 + \sigma_t\, \epsilon$, from clean data at $t = 0$ to pure noise at $t = T$.**

Common choices: DDPM's linear $\beta$ schedule (variance-preserving, $\alpha_t^2 + \sigma_t^2 = 1$), cosine, EDM's log-normal over $\sigma$, and the flow-matching interpolant $\alpha_t = 1-t$, $\sigma_t = t$. It defines *which* denoising problems the network trains on and how densely — a distribution over difficulty. The invariant is log-SNR $\lambda_t = \log(\alpha_t^2/\sigma_t^2)$; schedules with the same $\lambda$ range differ only in step spacing.

Trap: it must reach SNR ≈ 0 at $T$ or the model never sees pure noise and produces low-contrast images (zero-terminal-SNR), and it must be shifted at higher resolution because per-pixel noise averages out over bigger images.

## Q: What is the SNR at a timestep? {diff=1 tags=diffusion,snr,noise-schedule,rapid-fire}
### A
**$\mathrm{SNR}(t) = \alpha_t^2 / \sigma_t^2$, the ratio of signal to noise power in $x_t = \alpha_t\, x_0 + \sigma_t\, \epsilon$; log-SNR $\lambda_t$ is the natural coordinate for everything in diffusion.**

It matters because the schedule's index ($\beta$, $t \in [0,1]$, $\sigma$) is arbitrary but SNR is not: the denoising task at a given SNR is the same task however you index it. Loss weightings (min-SNR, P2), the resolution timestep shift, parameterization choice ($\epsilon$ at high SNR, $x_0$ at low), and step spacing are all cleanly stated in $\lambda$. Under flow matching, $\mathrm{SNR} = \big((1-t)/t\big)^2$.

Trap: SNR is per-pixel, but perception is per-frequency: low frequencies carry far more power, so at a given $t$ coarse structure is already resolved while fine detail is still buried — the coarse-to-fine behavior, and the reason high-res models need a shifted schedule.

## Q: What is the score function? {diff=1 tags=diffusion,score,rapid-fire}
### A
**The score is the gradient of the log density with respect to the data, $\nabla_x \log p(x)$: a vector field pointing toward higher-probability regions.**

It exists because it sidesteps the normalizing constant — $\nabla \log p$ doesn't depend on $Z$ — so you can learn it without ever computing a likelihood. For the noised marginal $p_t$ the score is exactly what a denoiser gives you: $\nabla_x \log p_t(x_t) = -\hat\epsilon / \sigma_t$ (Tweedie), so $\epsilon$-prediction *is* score estimation up to scale. Sampling then follows the score: Langevin dynamics, the reverse SDE, or the probability-flow ODE.

Trap: the raw data score is ill-defined off the data manifold (zero density), which is precisely why we noise the data — $p_t$ is smooth with full support for $\sigma > 0$, so the score exists everywhere and points back toward the manifold.

## Q: What is Tweedie's formula? {diff=2 tags=diffusion,tweedie,denoising,rapid-fire}
### A
**Tweedie: the posterior mean of the clean signal given a Gaussian-corrupted observation is the observation plus noise variance times the score, $\mathbb{E}[x_0 \mid x_t] = \big(x_t + \sigma_t^2\, \nabla \log p_t(x_t)\big) / \alpha_t$.**

It links the two views of diffusion in one line: the optimal MSE denoiser and the score are the same object. From it, $\hat\epsilon = -\sigma_t \nabla \log p_t$, $\hat x_0 = (x_t - \sigma_t \hat\epsilon)/\alpha_t$, and every parameterization conversion follows. It's also the tool behind training-free guidance and inverse problems (DPS): you get an $x_0$ estimate at any step to compute a measurement gradient against.

Trap: $\hat x_0$ is a *posterior mean*, so at high noise it's a blurry average of all plausible images — a one-step $\hat x_0$ at $t = T$ is grey mush. Sampling re-noises or integrates the ODE precisely so you land on a mode, not the mean.

## Q: What is v-prediction? {diff=1 tags=diffusion,parameterization,v-prediction,rapid-fire}
### A
**$v$-prediction has the network output $v = \alpha_t\, \epsilon - \sigma_t\, x_0$, the velocity of $x_t$ along the trajectory; for a variance-preserving schedule, $\hat\epsilon = \alpha_t\, \hat v + \sigma_t\, x_t$ and $\hat x_0 = \alpha_t\, x_t - \sigma_t\, \hat v$.**

It exists because $\epsilon$-prediction blows up at low SNR ($x_0 = (x_t - \sigma_t \hat\epsilon)/\alpha_t$ divides by a tiny $\alpha_t$ and amplifies error) and $x_0$-prediction is useless at high SNR (the target is trivially $x_t$). $v$ mixes both, so the target has unit variance and the implied loss weight is well-behaved at *every* SNR — what made progressive distillation and zero-terminal-SNR sampling work.

Trap: for the flow-matching interpolant $x_t = (1-t)\,x_0 + t\,\epsilon$ the velocity is $\epsilon - x_0$ — same idea, different sign — so pairing a $v$-pred checkpoint with an FM scheduler silently flips the sign: the classic "healthy loss, garbage samples" bug.

## Q: Explain classifier-free guidance in one minute. {diff=1 tags=diffusion,cfg,guidance,rapid-fire}
### A
**CFG trains one network for conditional and unconditional denoising by randomly dropping the condition (~10%), then at sampling extrapolates: $\tilde\epsilon = \epsilon(\varnothing) + w \cdot (\epsilon(c) - \epsilon(\varnothing))$ with $w > 1$.**

Why: unguided conditional samples follow the prompt weakly. The difference $\epsilon(c) - \epsilon(\varnothing)$ is proportional to $\nabla \log p(c \mid x)$, an implicit classifier gradient, so amplifying it sharpens toward the condition — sampling from $p(x \mid c) \cdot p(c \mid x)^{w-1}$, a lower-temperature distribution in prompt-relevant directions. It replaced classifier guidance because it needs no separate noise-aware classifier.

Traps: two forward passes per step (batch them); too-high $w$ oversaturates and collapses diversity because the extrapolated $\epsilon$ leaves the manifold — hence guidance intervals, rescaling, and distillation into one pass. The null condition must be the same tensor at train and inference.

## Q: DDPM versus DDIM sampling — in one breath. {diff=1 tags=diffusion,ddim,ddpm,samplers,rapid-fire}
### A
**DDPM reverses the Markov chain one step at a time with fresh noise injected each step — stochastic, ~1000 steps; DDIM uses the same trained network and marginals but a deterministic, non-Markovian update that lets you skip steps — same model, ~20–50 steps.**

The DDIM step: predict $\hat x_0$ from $\hat\epsilon$, then jump to the next timestep, $x_{t'} = \alpha_{t'}\, \hat x_0 + \sigma_{t'}\, \hat\epsilon$ — Euler on the probability-flow ODE in a good coordinate. Determinism means a fixed seed gives a fixed image, enabling inversion (encode a real image to its noise) and smooth latent interpolation.

Trap: DDIM's $\eta$ knob interpolates back toward DDPM stochasticity; a little noise corrects accumulated error at many steps, but at few steps deterministic wins. Neither retrains anything — sampling is a property of the solver, not the model.

## Q: What is a consistency model? {diff=2 tags=diffusion,consistency,distillation,few-step,rapid-fire}
### A
**A consistency model learns $f(x_t, t)$ that maps any point on a probability-flow ODE trajectory directly to that trajectory's endpoint $x_0$, with the constraint that all points on one trajectory give the same output — self-consistency.**

Why: a diffusion model reaches $x_0$ only after integrating the ODE over many steps; if $f$ is consistent, one evaluation from pure noise is a sample, and you trade quality for steps by re-noising and calling $f$ again. Training enforces $f(x_t, t) \approx f(x_{t'}, t')$ for adjacent points on a trajectory, via a teacher solver (consistency distillation) or a one-sample trajectory estimate (consistency training). The boundary condition $f(x_0, 0) = x_0$ is baked into the parameterization.

Trap: multi-step consistency sampling doesn't converge to the teacher — errors don't cancel — and training is unstable (step-count curriculum, EMA target). LCM is the latent version behind 4-step SDXL.

## Q: Flow matching versus diffusion — what's the difference, in one minute? {diff=1 tags=flow-matching,diffusion,comparison,rapid-fire}
### A
**Both learn a vector field transporting noise to data along a prescribed path; diffusion uses a Gaussian noising process (curved variance-preserving path, $\epsilon$ / score target), flow matching uses the straight interpolant $x_t = (1-t)\,x_0 + t\,\epsilon$ and regresses the constant velocity $\epsilon - x_0$.**

What actually differs: (1) the path — straight lines make the ODE nearly linear, so few Euler steps suffice; (2) the target — unit variance at all $t$, no SNR-dependent blow-up; (3) the framing — no SDE or ELBO, just "regress a velocity along a coupling", which extends trivially to non-Gaussian sources and paired couplings (reflow, OT couplings).

Trap: it's a reparameterization, not a new model class — FM with the linear schedule is diffusion with $\alpha_t = 1-t$, $\sigma_t = t$ and a $v$-like target; its score is recoverable via Tweedie. The wins in SD3/FLUX/WAN come from the straight path and timestep shift, not new theory.
