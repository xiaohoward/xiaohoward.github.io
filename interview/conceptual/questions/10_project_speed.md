# Project Grill: SPEED (Spectral Progressive Diffusion)
<!-- weight: 5 -->

## Q: Give me the two-minute pitch for SPEED. What is the one-sentence insight, and what is the number I should remember? {diff=1 tags=speed,pitch,diffusion}
- Follow-up: Who cares? Why is this not just "run the model at low res first"?
### A
**Insight: a diffusion sampler is already a coarse-to-fine process in frequency space, so it should not pay full-resolution compute while it is only resolving low frequencies.**

The natural-image (and latent) power spectrum is a power law, $P(f) = A f^{-\beta}$ ($\beta \approx 1.9$ for FLUX latents, $\approx 2.4$ for WAN / PixelGen). Under flow matching, $x_t = (1-t)\,x_0 + t\,\epsilon$, noise has flat unit power per orthonormal DCT bin, so at large $t$ every high-frequency bin is pure noise — the network's output there carries no information, but a DiT still spends $O(N^2)$ attention on those tokens. SPEED measures the spectrum once, derives the flow time at which each frequency band's SNR crosses 1, and runs the sampler at a resolution whose Nyquist frequency matches what is currently resolvable: low res early, native res late. Stage transitions are done in the DCT domain (embed low-res coefficients into the low-frequency corner of the larger grid, fill the new band with $t \cdot \text{noise}$, fix the SNR with a scale), so no architecture change, no retraining; optionally a small LoRA fixes the model's low-res regime.

Numbers: **up to 7.09× wall-clock on FLUX.1-dev, 2.54× on WAN 2.1**, same step count, quality preserved on standard metrics; also works pixel-space (PixelGen) and on Z-Image / Qwen-Image.

Why not "just run low-res first" (cascades)? Cascades train separate models and a super-resolution stage; SPEED reuses one model and one trajectory, and the schedule is derived from measured statistics rather than tuned by hand. The control experiment that convinced me: a full-resolution staged box low-pass gives the same quality and **zero speedup** — the gain comes precisely from moving tokens out of the early steps.

## Q: Derive it for me. Under $x_t = (1-t)\,x_0 + t\,\epsilon$ and $P(f) = A f^{-\beta}$, at what flow time does frequency $f$ \"turn on\"? {diff=2 tags=speed,derivation,spectrum,snr}
- Follow-up: What happens to frequencies with $P(f) < 1$? Do they ever get $\text{SNR} > 1$ before $t = 0$?
### A
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

## Q: Why DCT rather than FFT? Isn't this a detail? {diff=2 tags=speed,dct,signal-processing}
- Follow-up: What does DCT index $k$ correspond to in cycles per sample?
### A
It is not a detail — it is what makes the resolution embedding clean.

- **Real-valued.** DCT-II of a real image is real; FFT gives complex coefficients with Hermitian symmetry, so "embed into the low-frequency corner" needs careful handling of conjugate pairs and the Nyquist bin. With DCT the low-frequency block is literally the top-left corner of the coefficient array.
- **Boundary handling.** DFT assumes periodic extension, so an image with different left/right edges has a jump that leaks energy into all high frequencies (spectral leakage). DCT-II assumes even-symmetric extension: no jump at the boundary, so energy compacts better into low frequencies — the fitted power law is cleaner and the embedding does not create seam artifacts.
- **Orthonormal ⇒ energy preserved (Parseval).** With the ortho normalization, $\|x\|^2 = \|\mathrm{DCT}(x)\|^2$ and i.i.d. unit Gaussian noise in pixel space is i.i.d. unit Gaussian in coefficient space. That is exactly why $P(f) = 1$ is the $\text{SNR}=1$ line and why \"fill new bins with $t \cdot \text{noise}$\" is statistically correct.
- **Frequency mapping.** DCT-II basis $k$ is $\cos\!\left(\frac{\pi k (2n+1)}{2N}\right)$, i.e. **index $k \leftrightarrow$ frequency $k/2$ cycles per $N$ samples** — half the FFT's $k/N$. So the top DCT index $N-1$ corresponds to Nyquist, and the low-res $N_s \times N_s$ block maps exactly onto the frequencies below the low-res grid's Nyquist. For the radial spectrum I use $f = \sqrt{k_x^2 + k_y^2}$ normalized to the full-res Nyquist so schedules are comparable across models.

Practical note: I compute the 2D (or 3D for video) DCT with separable orthonormal transforms along each axis; it costs $O(N \log N)$ and is negligible compared to one DiT forward.

## Q: Walk me through how the resolution schedule is derived from the fitted spectrum. What exactly does $\delta$ do, and how sensitive are results to it? {diff=2 tags=speed,schedule,hyperparameters}
- Follow-up: How many stages, and who chooses them?
### A
The pipeline is: **measure → fit → invert → add margin.**

1. Encode a few hundred images/videos with the model's VAE, take the orthonormal DCT of the latents, radially average $|\text{coef}|^2$ into a 1D $P(f)$ with $f$ normalized to the full-res Nyquist. Fit $\log P = \log A - \beta \log f$ by least squares over the mid band (exclude DC and the last few bins).
2. Choose candidate stage resolutions the model can actually run at (multiples of the patch size and VAE stride, e.g. 1/4, 1/2, 1 of native for FLUX). Each stage $s$ has Nyquist $f_s = r_s f_{\text{Nyq}}$ where $r_s$ is the resolution ratio.
3. Invert the activation formula: $t_s = t^*(f_s) = \frac{\sqrt{P(f_s)}}{1+\sqrt{P(f_s)}}$. The sampler runs at stage $s$ while $t > t_s$ — i.e. until the current grid's own Nyquist is about to carry signal — then transitions.
4. **$\delta$** is a small margin added to $t_s$ so the switch happens slightly *earlier* (larger $t$) than the $\text{SNR}=1$ crossing. Its purpose is to make sure the first high-frequency coefficients above the low-res Nyquist are still essentially pure noise when we synthesize them as $t\,\epsilon$; if you switch late, you are throwing away structure the model would already have started to commit to, and you also give the new band too few steps.

Sensitivity: results are flat over a reasonable $\delta$ range; too small $\Rightarrow$ slight softness (new band under-stepped); too large $\Rightarrow$ you leave speedup on the table since you spend more steps at full res. I report the schedule with the step allocation explicitly (e.g. how many of the 50 steps at each res) so people can see where the speedup comes from. The number of stages is a user choice constrained by the model: below some resolution the pretrained model is out of regime (the LoRA question), so in practice 2–4 stages.

## Q: Precisely what happens at a stage transition, and why would a naive bilinear resize of $x_t$ be wrong? {diff=3 tags=speed,transition,dct,snr}
- Follow-up: Where does the $1/r_{\text{eff}}$ factor come from? Derive it.
### A
At transition from an $N_{\text{src}}$-pixel grid to an $N_{\text{tgt}}$-pixel grid (per spatial dimension ratio $r = \sqrt{N_{\text{tgt}}/N_{\text{src}}}$ — for 2D, $r_{\text{eff}} = \sqrt{N_{\text{tgt}}/N_{\text{src}}}$ with $N$ counting all pixels):

1. **DCT** the current low-res state $x_t$ (orthonormal).
2. **Embed** those coefficients into the low-frequency corner of a zero-initialized $N_{\text{tgt}}$ DCT grid.
3. **Fill** all new high-frequency coefficients with $t\,\epsilon$, $\epsilon \sim \mathcal{N}(0,1)$ i.i.d. — because per the activation argument those bins are below $\text{SNR}=1$ and, marginally, $x_t$ there is just $t \cdot \text{noise}$ (the signal contribution $(1-t)\,x_0$ is negligible).
4. **Rescale** the embedded block by $\kappa$ (the $1/r_{\text{eff}}$ correction) and align the solver time to $\tilde{t}$ so the SNR of the embedded block matches what the target-resolution trajectory would have at that time.
5. **Inverse DCT** → the new full-res $x_t$, continue sampling.

Why $\kappa$? Orthonormal DCT of a constant patch of amplitude $c$ over $N$ pixels has DC coefficient $c\sqrt{N}$. Put that coefficient into a grid of $N_{\text{tgt}}$ pixels and inverse-transform: amplitude $c\sqrt{N_{\text{src}}/N_{\text{tgt}}} = c/r_{\text{eff}}$. So embedding **attenuates the low band by $1/r_{\text{eff}}$** relative to what a genuine target-res image would have; you must multiply by $\kappa = r_{\text{eff}}$ (per-band, the same factor for all embedded bins). But $\kappa$ scales the noise inside those bins too (it was $t\,\epsilon$ at low res, becomes $\kappa t \epsilon$), while the freshly filled bins have $t\,\epsilon$. The state is therefore not exactly on the $(1-t)\,x_0 + t\,\epsilon$ manifold; **$\tilde{t}$ is the effective time whose SNR matches the embedded block**, and I hand $\tilde{t}$ (not $t$) to the model / solver at the first full-res step [fill in: whether you renormalize the block to $(1-t)/t$ ratio or shift the solver time — state your exact choice and the formula].

Why not bilinear-resize $x_t$? (a) It is a low-pass in the wrong domain: the interpolation kernel attenuates the top of the low band unevenly (not a brick wall) and interpolates the *noise* too, making the noise spatially correlated — the model was trained on white noise, so it sees an off-distribution noise covariance and produces smooth, blurry output. (b) It gives no principled way to inject exactly the missing band at exactly variance $t^2$. The DCT route keeps the low band bit-exact and the new band exactly white.

## Q: You say continuous per-step frequency masking blurs but staged reveal doesn't. Why? And what did the box low-pass control experiment prove? {diff=3 tags=speed,ablation,blur,frequency}
- Follow-up: Why not just use a smooth (Gaussian) mask instead of a brick wall?
### A
I tried the "obvious" version first: at every step, radially reveal frequencies up to the current activation cutoff (a brick-wall mask on x_t or on the prediction). It **consistently blurs**. Two mechanisms:

- **Late frequencies are starved of steps.** With per-step reveal, the highest band is unmasked only in the last few steps of the schedule; the model has to synthesize fine texture in, say, 3 Euler steps at low noise where it is supposed to be doing small corrections. Result: under-resolved high frequencies = softness.
- **Brick-wall ringing.** A hard radial cutoff applied every step is a sinc kernel in the spatial domain; each step re-introduces Gibbs ringing at edges, the model partially "corrects" it, and the next mask re-introduces it — an oscillating fixed point that averages to blur and halo artifacts.

**Staged reveal** solves both: a whole band is unmasked once at a transition and then receives *all* remaining steps, and there is exactly one discontinuity, which is immediately buried under $t \cdot \text{noise}$ ($\text{SNR}<1$) so the model never sees a sharp edge in a band that matters yet. A smooth (Gaussian) mask helps ringing but not step starvation, and it also attenuates real signal in the pass band.

The control experiment: keep everything at full resolution and apply the *same* staged brick-wall low-pass at the same transition times. Quality **matches** the resolution-staged run — so the staged masking itself is quality-neutral — but wall-clock is **unchanged**, since token count never dropped. That decomposition is important: it shows (1) the coarse-to-fine schedule is harmless, and (2) 100% of the speedup comes from actually removing tokens. The one place staged resolution *beats* the full-res control is super-native generation (WAN 1.3B at 960p): 3.8× faster *and* better, because the model spends most steps at its native resolution instead of in RoPE extrapolation.

## Q: Training-free vs. the LoRA variant — what does the LoRA fix and how did you train it? Isn't "we needed fine-tuning" an admission the training-free story doesn't hold? {diff=2 tags=speed,lora,training}
- Follow-up: Why LoRA and not full fine-tune? Why not fine-tune only at low res?
### A
The training-free version already gives most of the speedup. The LoRA addresses one specific failure: **the low-resolution regime mismatch.** A model like FLUX.1-dev was trained mostly around 1 MP; at 1/4 resolution with the corresponding RoPE coordinates it is out of distribution — global composition is fine but it tends to produce fewer, larger objects and slightly wrong scale statistics at the first stage, and those errors are frozen in by the time we transition. For video, WAN at 240p is well out of its training distribution.

LoRA training: [fill in: exact rank, targets, steps, data] — the recipe was: sample data at the stage resolutions (downsample in the DCT domain the same way as inference, so the training distribution matches what the sampler sees), train a LoRA on attention projections with the standard flow-matching loss restricted to the timestep range where each resolution is used. The LoRA sees exactly the (resolution, t) pairs it will be asked about at inference, and nothing else; it does not need to learn the transition itself because the transition is analytic. Training is cheap (hours on a few GPUs) and the base weights are untouched, so it composes with other LoRAs.

Why LoRA: it cannot destroy the base model's full-res behaviour (which we rely on for the last stage), it is small enough to ship, and the target is a distribution-shift correction, which is low-rank in practice. Why not full fine-tune at low res: you would then have two models — that's a cascade, and you lose the single-model guarantee that the trajectory is consistent across stages.

So no, not an admission: the training-free result stands on its own for image models; the LoRA is a knob that pushes the earliest stage lower (more speedup) without losing quality.

## Q: How did you measure speedup fairly? And why 7.09× on FLUX but only 2.54× on WAN 2.1 — where does video's time actually go? {diff=1 tags=speed,evaluation,benchmarks,video}
- Follow-up: What metrics did you use, and which ones would you distrust?
### A
Fair measurement rules I used:
- **Wall-clock**, end-to-end on the same GPU, same batch, same number of sampling steps (50), same solver, same CFG, warm cache, median over runs. Not FLOPs, because attention efficiency and kernel launch overhead do not track FLOPs.
- Baseline is the model's own standard pipeline at native resolution; SPEED uses identical step count, so the only difference is where tokens are spent.
- Quality: standard benchmark metrics for the model class [fill in: e.g. GenEval / DPG / ImageReward / FID-style metrics for images, VBench for video], plus paired visual comparisons on the same seeds. I distrust FID at small n and any metric that rewards smoothness — the whole failure mode here is blur, so I always report a sharpness-sensitive metric or high-frequency energy ratio alongside.

Why FLUX ≈ 7× and WAN ≈ 2.5×:
- **Tokens removed is the lever.** FLUX at 1 MP is 4096 tokens with attention dominating; 1/4-res stages cut tokens by 16× and attention by ~256× — the early steps become almost free, and FLUX tolerates going down to very low res.
- **Video is spatiotemporal.** A *spatial* schedule reduces $H \times W$ but leaves the temporal token axis untouched; with 480p×81f WAN has ~30k tokens, and attention share is huge, but the spatial floor is higher (WAN below 240p is out of regime), so fewer/less-aggressive stages.
- **Fixed costs don't shrink**: 3D VAE decode, text encoder, CFG doubling the DiT calls (CFG 6 for WAN, two passes), and per-step overhead all stay constant, so Amdahl's law caps the ratio.
- The joint spatiotemporal extension recovers some of that (1.85× for a spatial-first 3-stage schedule on top of the spatial-only baseline), but the temporal axis is much more energetic so it activates early and cannot be reduced for long.

## Q: Convince me this isn't just cascaded diffusion, Matryoshka, progressive growing, frequency-domain diffusion, DeepCache, or Pyramid Flow with better PR. {diff=3 tags=speed,related-work,novelty}
- Follow-up: What would you cite as the closest prior work, and what is the one experiment that separates you from it?
### A
I'll take them in turn — the distinguishing claim is: **one pretrained model, one trajectory, a schedule derived from measured spectra, and a transition operator that is SNR-exact.**

- **Cascaded diffusion (Imagen, SR3, eDiff-I):** separate models per resolution, each trained; the low-res sample is a *conditioning input* to an SR model that starts from fresh noise. SPEED has no SR model and no restart: the low-res $x_t$ *is* the state that continues. Also cascades typically spend full steps at every stage (they are slower, not faster).
- **Matryoshka Diffusion:** trains a nested multi-res architecture jointly. Needs training from scratch or heavy fine-tuning; SPEED changes nothing about the model and is training-free.
- **Progressive growing (ProGAN-style / progressive distillation at res):** a *training* curriculum. SPEED is an *inference* schedule.
- **Frequency-domain / wavelet diffusion (WaveDiff, spectral diffusion, blurring diffusion):** they change the forward process or the representation the model is trained on. SPEED keeps the standard Gaussian forward process; the DCT is only used to *analyze* the process and to build the transition. My activation-time derivation is the same math that motivates blurring diffusion — but I use it to save compute, not to redefine the process.
- **DeepCache / TeaCache / feature caching:** reuse features across steps; orthogonal and composable (cache within a stage). Caching does not reduce tokens, so it can't touch attention cost the way resolution does; and it introduces approximation error every step.
- **Pyramid Flow (pyramidal flow matching):** the closest in spirit: multi-res stages in a single flow. But it *trains* the model with a pyramidal interpolant and renoising between stages; the transition is a learned/heuristic renoise. SPEED is post-hoc on an unmodified model, and the transition's scale/time alignment is derived from Parseval, not tuned. I would cite Pyramid Flow and blurring/frequency diffusion as closest prior work.

The separating experiment: the **full-res box low-pass control** (same quality, no speedup) plus the **training-free FLUX result** — no baseline above can be applied to a frozen FLUX with zero training and give a 7× wall-clock reduction at matched steps.

## Q: Where does SPEED fail? Give me the honest limitation list, and tell me which ones are fundamental versus fixable. {diff=2 tags=speed,limitations,failure-modes}
- Follow-up: The VAE latent spectrum isn't a natural-image spectrum. Why does your argument still hold?
### A
Failure modes I would volunteer before being asked:

1. **Text rendering and small high-contrast detail.** Glyphs are broadband; their layout is decided at low res where they don't exist yet, so letter placement and spelling degrade with aggressive schedules. Fixable by a later first transition (less speedup) or by the LoRA; fundamentally, anything whose *semantics* live in high frequencies gets less say in composition.
2. **Fine texture / hair / foliage** with very aggressive schedules: the last band gets fewer steps. Fixable via $\delta$ / step reallocation, and visible in sharpness metrics.
3. **Super-native resolution and RoPE.** Above the training resolution, the full-res stage extrapolates RoPE; SPEED actually *helps* here (WAN 1.3B at 960p: 3.8×, better quality) because most steps happen in-regime, but the final stage is still out of regime — that is a model limitation, not ours.
4. **Low-res floor.** Below ~1/4 native (image) or 240p (video) the pretrained model is out of regime; training-free stops there, LoRA extends it. Not fundamental but costs training.
5. **Latent spectra are not natural-image spectra.** True — the VAE's latent is a learned code with its own power law ($\beta \approx 1.9$ for FLUX vs $\approx 2$ for pixels; 2.4 for WAN) and channel-dependent structure. But the argument needs only two things: that the *measured* latent spectrum is monotone decreasing and roughly power-law, and that the *noise* is white in the same basis. Both hold, which is why I fit the spectrum per model rather than assuming the natural-image exponent. What does not transfer is "resolution = pixels": in latent space, a resolution change is a change of latent grid, and the VAE decoder's own receptive field sets the smallest useful grid.
6. **Few-step / distilled samplers.** With 4–8 steps the argument that early steps are "wasted" at full res weakens because each step already covers a wide $t$ range; you may have only one step per stage. This is the biggest open question for production (last question).
7. **Batching heterogeneity**: within a batch all samples must share the schedule; fine for T2I services, awkward for mixed workloads.

## Q: Video: how does the argument change with a temporal axis, and why does spatial-first win over temporal-first? {diff=3 tags=speed,video,spatiotemporal,3d-dct}
- Follow-up: What does "the temporal axis is 50× more energetic" mean concretely, and what does it imply for a temporal-only schedule?
- Follow-up: Why Euler with a step-index reset? What went wrong with UniPC?
### A
Video latents have a **joint spatiotemporal spectrum**, and it is anisotropic. Fitting separately along each axis with frequency normalized to that axis's Nyquist gives **$\beta_s \approx 2.65$ (spatial) and $\beta_t \approx 1.57$ (temporal)**, and at an equal normalized frequency the temporal axis carries **~50× more power**. Concretely: the highest temporal frequency of a 16 fps latent still has substantial signal (motion is sharp in time; frames change abruptly), whereas the highest spatial frequencies are nearly noise. So temporal frequencies **activate early** (large $t$), which means you cannot keep the frame count low for long without corrupting motion.

Implementation: 3D orthonormal DCT (separable over $t, y, x$), the same embedding-into-a-corner trick in 3D, new coefficients filled with $t\,\epsilon$, $\kappa = \sqrt{N_{\text{tgt}}/N_{\text{src}}}$ counting all voxels, and **Euler with a step-index reset at each transition**. The solver point is a correctness bug I hit, not a preference: multistep solvers (FlowUniPCMultistep, DPM-Solver++) extrapolate from a history of previous model outputs; after a resolution change that history has the wrong shape and, worse, the wrong statistics — mixing a low-res prediction with a full-res one injects an error the size of the newly revealed band, giving artifacts or divergence at the first full-res step. Fix: reset the solver history/step index at every transition so the first step after a switch is first-order (UniPC degrades gracefully with no history), or use stateless Euler; and always hand the scheduler the aligned time $\tilde{t}$, not the old step index (same bug class in dynamic-shift FlowMatchEuler for Qwen-Image).

Stage orders (three stages each, 50 steps, WAN 2.1):
- **Spatial-first:** 240p·40f → 480p·40f → 480p·80f: **1.85×** over the spatial-only baseline.
- **Temporal-first:** 240p·40f → 240p·80f → 480p·80f: 1.56×.

Why spatial-first wins on compute: tokens are $T \cdot H \cdot W$; an early stage at low spatial res *and* low temporal res is the cheap regime, and the question is which axis you are allowed to keep small the longest. Because the temporal spectrum is so energetic, temporal Nyquist activates early, so the temporal-first order is forced to spend its middle stage at 240p·80f (twice the tokens of 240p·40f) while spatial-first sits at 480p·40f — same token count, but the spatial band is what actually activates around then. Equivalently: put the expensive dimension (time) back last, because the schedule can tolerate the temporal reveal being late only if there is real spatial detail to resolve, which there is. A purely temporal schedule barely helps — the 50× energy gap means the temporal Nyquist activates at $t$ close to 1.

## Q: You're at NVIDIA and asked to ship this in a production video model. What do you integrate with, what interacts badly, and where might the whole argument break? {diff=3 tags=speed,production,scaling,distillation,cfg,fp8}
- Follow-up: With a 4-step distilled model, is there any speedup left?
### A
What I would do in order:

1. **Composability first.** SPEED is orthogonal to feature caching (DeepCache/TeaCache), to sparse/linear attention, and to fp8 — each reduces per-step cost while SPEED reduces tokens in early steps. I'd stack them and measure, expecting roughly multiplicative gains since they touch different axes.
2. **CFG.** With classical CFG (two passes), the low-res stages are the cheap place to run guidance and the expensive full-res stage is where guidance-interval tricks (drop CFG at low noise) already apply; combining both gives the biggest saving. Note guidance changes the *effective* spectrum of the prediction (it sharpens), so re-check the transition time with CFG on.
3. **Batching and shapes.** Stage resolutions must be multiples of patch × VAE stride, and every stage's shape must be a compiled/cuda-graph shape; I would fix 2–3 stage shapes per product resolution and precompile. Serving systems that pad to fixed shapes lose part of the benefit — with continuous batching, group requests by stage.
4. **fp8 / quantization.** The low-res stages run at high noise where activations are noise-dominated; fp8 there is benign. Quantization error at the first full-res step could bias the newly injected band — verify with the sharpness metrics.
5. **Training-side.** In a production model I'd fold the LoRA into pretraining as a multi-resolution, timestep-conditioned curriculum (each resolution only sees its t range), so the model is natively in-regime at low res and the schedule could go lower.

Where it can break: **few-step distilled models.** With 4 steps the sampler doesn't "spend" early steps — each step spans a huge $t$ range, and the distilled model was trained to jump from noise to data at full res; its predictions at the first step already contain committed high frequencies (distillation collapses the coarse-to-fine ordering). Options: distill *with* the SPEED trajectory (teacher runs staged, student learns staged, so step 1 is a low-res step), or accept that speedup for a 4-step model is $\le 4/3\times$ and focus SPEED on the 20–50 step quality tier and on video, where step counts remain high. I would say that plainly rather than oversell.

<!-- rapid-fire -->
## Q: In one sentence, what does SPEED do? {diff=1 tags=speed,rapid-fire,pitch}
### A
**SPEED denoises at low resolution early and raises resolution along the trajectory**, using the latent power spectrum to decide exactly when each frequency band is worth computing, so a pretrained diffusion/flow model spends most of its steps on far fewer tokens with no architecture change. The why: high frequencies are pure noise at high $t$, so full-resolution steps there are wasted attention. Result: up to 7.09× wall-clock speedup on FLUX.1-dev and 2.54× on WAN 2.1 with quality preserved, training-free or with a light LoRA. Follow-up to expect: "how do you pick the switch times?" — from the fitted spectrum $P(f) = A f^{-\beta}$, not by hand.

## Q: What is a power spectrum, and how did you measure it for a latent diffusion model? {diff=1 tags=speed,rapid-fire,spectrum,signal-processing}
### A
The **power spectrum is the energy of a signal per frequency**: take a 2D transform (DCT or FFT), square the coefficients, and average over rings of equal radial frequency $f$ to get a 1D curve $P(f)$. I measured it on clean latents $x_0$ from the VAE encoder over a few thousand images, with an **ortho-normalized** transform so that unit Gaussian noise has power exactly 1 in every bin. Natural images and their latents follow a power law, $P(f) = A f^{-\beta}$: $\beta \approx 1.92$ for FLUX, 2.42 for WAN 2.1, 2.45 for PixelGen, 2.24 for Qwen-Image. Trap: normalization matters — without ortho scaling, \"$P(f) = 1$ is the noise floor\" is false and the schedule is off.

## Q: What is the DCT, in one minute? {diff=1 tags=speed,rapid-fire,dct,signal-processing}
### A
The **discrete cosine transform expresses a signal as a sum of cosines** of increasing frequency; it is a real, orthogonal linear map (DCT-II is the JPEG one), so it is its own inverse up to transpose and preserves energy (Parseval). Why I use it instead of the FFT: it is real-valued, its implicit even-symmetric extension avoids the wrap-around discontinuity of the FFT so there is no ringing at image borders, and its coefficients are laid out on a plain 2D grid where low frequencies sit in one corner — which makes "embed a low-res grid into the low-frequency corner of a high-res grid" a literal array copy. Follow-up: energy scaling when you change grid size (the $1/r_{\text{eff}}$ factor).

## Q: What does it mean that "a frequency activates"? {diff=1 tags=speed,rapid-fire,snr,flow-matching}
### A
Under the interpolant $x_t = (1-t)\,x_0 + t\,\epsilon$, the signal at frequency $f$ has power $(1-t)^2 P(f)$ and the noise has power $t^2$ (unit per bin). A frequency **activates at the flow time $t^*$ where signal first exceeds noise**, i.e. $(1-t)^2 P(f) = t^2$, so $t^*(f) = \frac{\sqrt{P(f)}}{1 + \sqrt{P(f)}}$. Because $P(f)$ decays as $f^{-\beta}$, low frequencies activate early (at high $t$) and high frequencies activate late. Before its activation time a band is essentially noise and the model cannot extract information from it, so computing it is wasted. This is the whole justification for the schedule. Follow-up: it is $\text{SNR} = 1$, a threshold, not a hard cutoff; $\delta$ adds margin.

## Q: What is a stage transition? {diff=1 tags=speed,rapid-fire,transition,dct}
### A
A **stage transition is the moment we jump from resolution $r_k$ to $r_{k+1}$ mid-trajectory**. Mechanically: DCT the current low-res noisy latent, copy it into the low-frequency corner of the larger DCT grid, fill the new high-frequency coefficients with fresh noise scaled by $t$, inverse-DCT, and continue denoising at the higher resolution. A $\kappa$ / $\tilde{t}$ correction fixes the SNR, because embedding into a bigger grid attenuates the signal by $1/r_{\text{eff}}$ with $r_{\text{eff}} = \sqrt{N_{\text{target}}/N_{\text{source}}}$. The transition is triggered when the current stage's Nyquist frequency activates. Trap: bilinear-resizing $x_t$ instead would also blur the noise, breaking the marginal the model was trained on.

## Q: What is the difference between the training-free mode and the LoRA mode? {diff=1 tags=speed,rapid-fire,lora,training}
### A
**Training-free** means we run the pretrained model unchanged and only change the sampler: the resolution schedule, the DCT transition, and the SNR correction. It already gives most of the speedup because the transition is designed to land on the model's own training marginal. **LoRA mode** additionally fine-tunes low-rank adapters on the staged trajectory so the model sees low-resolution noisy inputs at high $t$ and mixed states right after a transition, which cleans up residual artifacts (slight texture mismatch right after a jump) and recovers the last bit of quality. Same architecture, same inference code; the LoRA is optional. Follow-up: "so training-free does not fully hold?" — it holds; the LoRA closes a small gap, it does not enable the method.

## Q: What does $\delta$ control? {diff=2 tags=speed,rapid-fire,hyperparameters,schedule}
### A
**$\delta$ is a margin on the activation criterion that controls how early each transition happens.** The rule is \"switch to the next resolution when the current stage's Nyquist frequency activates\"; $\delta$ shifts that threshold so we switch slightly before $\text{SNR} = 1$ rather than exactly at it, giving the new band a few steps of headroom before it carries signal. Larger $\delta$ means earlier transitions: more steps at high resolution, safer quality, less speedup. Smaller $\delta$ means later transitions and more speedup, until the highest band is starved of steps and detail goes soft. It is the one knob on the speed–quality curve; everything else follows from the fitted $\beta$. Trap: one scalar shared across stages, not per-stage tuning.

## Q: What was the speedup, and on which models? {diff=1 tags=speed,rapid-fire,benchmarks,results}
### A
**Up to 7.09× wall-clock on FLUX.1-dev (latent image) and 2.54× on WAN 2.1 (latent video)**, with quality preserved on standard metrics; also evaluated on Z-Image (latent image) and PixelGen (pixel-space image), and I later ported it to Qwen-Image. Video speedup is lower because tokens scale with frames as well as pixels and the 3D VAE / non-attention costs are a larger fraction of time; a 3-stage spatiotemporal schedule (240p40f→480p40f→480p80f) gave 1.85× on WAN. At super-native resolution (WAN 1.3B at 960p) it was 3.8× and better quality than full-res, since most steps run at native resolution. All numbers are measured wall-clock at 50 steps, same solver and CFG as baseline.
