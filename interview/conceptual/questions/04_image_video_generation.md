# Image & Video Generation
<!-- weight: 5 -->

## Q: Walk me through the VAE in a latent diffusion model. Why did the field move from 4-channel to 16-channel latents, and what does the "scaling factor" do? {diff=1 tags=vae,latent-diffusion,architecture}
- Follow-up: KL-regularized vs VQ — which would you pick for a diffusion backbone and why?
- Follow-up: What happens if you skip the scaling factor?
### A
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

## Q: Compare cross-attention text conditioning (SD1/SDXL) with MMDiT-style joint attention (SD3/FLUX). Why did the field move to joint attention, and what's the cost? {diff=2 tags=conditioning,architecture,dit,text-to-image}
- Follow-up: Why keep both CLIP and T5 encoders?
- Follow-up: What are "in-context" conditioning tokens and when are they preferable?
### A
**Cross-attention injects a fixed text-encoder output as keys/values in every block; MMDiT concatenates text tokens and image tokens into one sequence with separate weights per modality and lets both streams update each other.**
 
 
- **Cross-attn:** image queries, text K/V. Strength: Text representation is frozen after the encoder; cheap (text has ~77-256 tokens, so cost is $O(N_{\text{img}} \times N_{\text{txt}})$). Weakness: text never sees the image, so compositional/spatial binding ("red cube left of blue sphere") is weak. 
- **MMDiT joint attention:** sequence = [text tokens; image tokens], one attention over both, with modality-specific QKV/MLP weights (two "streams") and shared attention. Text tokens are refined conditioned on the current image estimate — that's what gives SD3/FLUX their prompt-following jump. FLUX uses ~19 double-stream blocks then ~38 single-stream blocks (shared weights) to save parameters. 
- **Cost:** attention is now $O((N_{\text{img}} + N_{\text{txt}})^2)$ instead of $O(N_{\text{img}}^2) + O(N_{\text{img}} \cdot N_{\text{txt}})$. At 4096 image tokens + 512 text tokens that's ~27% more attention FLOPs. Also text tokens now sit in the RoPE grid (FLUX gives them position 0 on the image axes ($(0, 0, 0)$). 
- **Why keep CLIP + T5:** CLIP-L/G give a pooled global vector (aligned with images, good for style/global semantics) that feeds the timestep-modulation (adaLN) path; T5-XXL gives ~4.7B-param sequence tokens with real language understanding for long, compositional prompts. In T5, every token gets its own contextual vector embedding. Clip instead gives one global vector. SD3 ablation: dropping T5 hurts text rendering and complex prompts, dropping CLIP hurts aesthetics slightly. Newer models (Qwen-Image, Z-Image) use a VLM/LLM encoder as a single strong text tower.
- Aside: what is the adaLN path: for FLUX tells every transformer block what is the current timestep $t$, and $e_t = \mathrm{TimeEmbed}(t)$. then the embedding adds the clip vector and adds the guidance embedding. So adaLN first applies regular LN, $\hat{h} = \mathrm{LN}(h)$, then do $h_{\text{mod}} = (1 + \text{scale})\,\hat{h} + \text{shift}$. Then the gate controls how strongly the block's output is added back through the residual connection, $h_{\text{out}} = h + \text{gate} \cdot F(h_{\text{mod}})$. so adaLN is a small neural layer that convert the conditioning vector into per-feature modulation params.  
- **In-context tokens:** conditioning images (reference, edit source, ControlNet-like maps) are patchified and concatenated as extra tokens with their own position ids, rather than added as channels. Preferable when the condition isn't pixel-aligned with the output (reference subject, multi-image editing) — the model learns correspondence via attention. FLUX Kontext / OmniGen do this. Cost is token count, so it's exactly where token-reduction ideas matter.

## Q: What actually scales in a DiT? Depth, width, or token count — and how do Gflops relate to FID? {diff=1 tags=dit,scaling,architecture}
- Follow-up: Is patch size 2 vs 4 a "free" 4× compute saving?
### A
**Peebles & Xie's finding: FID is a near-monotone function of forward-pass Gflops, almost regardless of whether you get those Gflops from depth, width, or more tokens (smaller patch).** DiT-XL/2 at ~119 Gflops/forward hits FID 2.27 on ImageNet-256 with CFG; the S/B/L/XL × patch-8/4/2 grid lines up on one Gflops-vs-FID curve.
 
 
- **Depth/width:** parameter count scales as $\approx 12 L d^2$ for the transformer; per-token FLOPs $\approx 2 \times$ params. Both work; width is more hardware-efficient (bigger matmuls), depth gives more "sequential computation". 
- **Tokens (patch size):** halving patch size from 4 to 2 gives 4× tokens, ~4× MLP FLOPs and 16× attention FLOPs, with *zero* new parameters — and it improves FID more than adding parameters at equal Gflops. So no, patch 4 is not free: you lose fine detail because each token must explain 16× more pixels. This is the key insight behind token-count-driven methods (SPEED, Foveated Diffusion): tokens are where compute goes, and where quality comes from — so spend them where and when they matter. 
- **Trends at scale:** loss follows a power law in compute; SD3 showed validation loss correlates with human preference across 0.8B-8B. FID saturates and stops discriminating at large scale (see the metrics question). 
- Interviewer's real question: "given 4× more compute, what do you do?" Answer: mostly more tokens / higher res and more data, some width; depth beyond ~40-60 layers has diminishing returns at these scales, and you need the VAE to keep up. 
Once you already have a reasonably large DiT, extra compute is often better used to let the model process a richer image representation—more spatial tokens/higher resolution—while also increasing width (channel size) and training data, rather than simply stacking many more identical Transformer layers. And if you increase what the DiT can model, the autoencoder must preserve enough detail for those gains to survive into pixels.

## Q: Describe the architecture of a modern video diffusion model like WAN 2.1 or CogVideoX: the 3D VAE, the attention pattern, and the position encoding. {diff=2 tags=video,architecture,vae,attention,rope}
- Follow-up: Why a *causal* 3D VAE?
- Follow-up: Full spatiotemporal attention vs factorized — which wins, and why did the field move to full?
### A
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

## Q: Why does a diffusion model degrade when you sample above its training resolution, and what are the fixes? {diff=3 tags=resolution,rope,extrapolation,video}
- Follow-up: Explain NTK-aware / YaRN-style RoPE scaling in one breath.
- Follow-up: Why does "generate at native res for most of the trajectory, then upscale late" work so well?
### A
**Two failure modes: (1) positional extrapolation — RoPE has never seen positions beyond the training grid, so attention patterns become unreliable and you get repeated objects / duplicated limbs; (2) noise-schedule mismatch — at higher res the per-pixel SNR at a given $t$ is effectively higher (more redundant pixels average out noise), so the model's timestep semantics shift.**

- **RoPE extrapolation:** RoPE rotates q/k by angle $\text{pos} \times \theta_i$ with $\theta_i = \text{base}^{-2i/d}$. Low-frequency dims have periods longer than the training length and have only seen a fraction of a cycle; at 2× length they hit unseen phases. Fixes: **position interpolation** (scale positions by $1/s$ — keeps in-distribution but compresses fine detail), **NTK-aware** (raise the base so high-freq dims are untouched and low-freq dims are interpolated — "change base so the longest wavelength stretches by $s$"), **YaRN** (per-dimension: interpolate dims whose wavelength exceeds context, leave high-freq dims alone, blend in between, plus a temperature on attention logits). In video/image: FLUX/WAN users apply NTK scaling to the h/w axes; models like FiT/Lumina train with variable grids so RoPE sees many scales.
- **Schedule mismatch:** SD3 shifts the timestep schedule with resolution — shift factor $\propto \sqrt{N_{\text{tokens}} / N_{\text{base}}}$ — so at 4× tokens you spend more steps at high noise. Same logic as SPEED's SNR correction: embedding a signal into a bigger grid spreads energy, so the noise level that "means" a given structure level moves.
- **Why native-then-upscale works:** the coarse structure (layout, composition, low frequencies) is decided in the early, high-noise steps; that's exactly where the model must be in-distribution, and at native res it is. Late steps only add high frequencies, which are local — a few steps of RoPE-extrapolated high-res denoising with well-established low frequencies is easy (it's essentially guided super-resolution). SPEED's 960p WAN result: 3.8× faster *and* better than full-res, because ~80% of the trajectory never enters the extrapolation regime. The general principle: **the trajectory's low-frequency content is the extrapolation-sensitive part; keep it in-distribution.**

## Q: Compare progressive / multi-resolution generation strategies: cascaded diffusion, SDXL refiner, Matryoshka Diffusion, pixel-space models like PixelGen, your SPEED, and Foveated Diffusion. What's the axis each one cuts along? {diff=3 tags=progressive,resolution,speed,foveated,cascades}
- Follow-up: Why does continuous per-step frequency masking blur, but staged reveal doesn't?
- Follow-up: Could SPEED and Foveated Diffusion be combined?
### A
**They all exploit the same fact — coarse structure is cheap and determined early; detail is expensive and local — but cut along different axes: separate models per resolution (cascades), separate phase of one model (refiner), shared multi-scale weights (Matryoshka), *time* (SPEED), or *space* (Foveated).**

- **Cascaded diffusion (Imagen, DALL·E 2, Stable Cascade):** base 64² model + 1-2 super-res diffusion models conditioned on the low-res image, with noise-augmentation on the conditioning to bridge train/test gap. Pros: each stage is cheap and specialized. Cons: multiple models to train/serve, errors compound, SR stages hallucinate detail inconsistent with the base.
- **SDXL refiner:** same latent space, a second model specialized on the last ~20% of noise levels (img2img). It's a "denoising-expert" split in *time* — but no resolution change, so no speedup, only quality.
- **Matryoshka Diffusion:** one model jointly denoises a pyramid of resolutions in pixel space with nested U-Net features; progressive training schedule. Elegant but the multi-res target is still computed at full cost.
- **PixelGen / pixel-space DiTs:** skip the VAE; work directly on pixels with large patches or a pixel-pyramid. Removes VAE artifacts and decode cost but token counts explode — which is exactly why resolution scheduling is even more valuable there (SPEED evaluates on it).
- **SPEED (mine):** one pretrained model, no architecture change; the *resolution grows along the denoising trajectory*, with stage transitions placed where the spectrum says the next frequency band becomes signal-dominated ($(1-t)^2 P(f) = t^2$ with $P(f) = A f^{-\beta}$). Transitions are DCT embedding + noise fill for the new band + an SNR correction ($1/r_{\text{eff}}$ attenuation). Training-free or light LoRA. Speedup comes purely from fewer tokens per step early on: 7.09× on FLUX, 2.54× on WAN.
- **Foveated Diffusion (mine, co-author):** *spatial* axis — one forward pass mixes high-res tokens in a fovea and coarse tokens in the periphery; LoRA teaches the model mixed-res token statistics and position embeddings at multiple patch scales; 2× image / 4× video.

**Why continuous masking blurs but staged reveal doesn't:** with a brick-wall radial mask growing every step, the highest frequencies get revealed in the last 1-2 steps — they're step-starved — and per-step hard cutoffs create ringing that the model treats as signal. A staged reveal exposes an entire band at once at the point the spectrum says it's $\mathrm{SNR} \ge 1$, then gives it all remaining steps. Also: a low-pass at full res gives *no* speedup; only real token reduction does. **Combining:** yes — SPEED decides *when* tokens exist, Foveated decides *where*; the product of the two speedups is plausible since attention cost is quadratic in total tokens, and both are LoRA-compatible on the same backbone.

## Q: How do you build the training data for a text-to-image or text-to-video model? Talk about captions, aspect ratios, and resolution curriculum. {diff=2 tags=data,captioning,training,text-to-image,video}
- Follow-up: What goes wrong if you train only on synthetic captions?
### A
**Three levers: (1) dense synthetic captions from a VLM, (2) aspect-ratio bucketing so you never crop/distort, (3) a resolution and duration curriculum from low to high.**

- **Captions:** raw alt-text is short, noisy, and often unrelated. DALL·E 3's recipe: train a captioner to produce long, descriptive captions, then train the generator on ~95% synthetic / 5% original. Synthetic captions give far better prompt following and text rendering. Trap: train on 100% synthetic and you overfit to the captioner's phrasing — users write short prompts, so at inference you either get a prompt-style mismatch or you need an LLM prompt "upsampler". Mix caption lengths and styles; for video, caption motion and camera explicitly (VLMs describe frames, not dynamics — WAN/HunyuanVideo train dedicated video captioners).
- **Aspect-ratio bucketing (NovelAI / SDXL):** group images into buckets of equal token count but varying (h, w) — e.g., 1024², 832×1216, 1216×832 — so every batch has a uniform shape and no center-cropping ("headless people" problem). SDXL additionally conditions on original size and crop coordinates as micro-conditioning so the model learns not to reproduce low-res/cropped statistics.
- **Resolution / duration curriculum:** pretrain at 256², then 512², then 1024², with a small final high-quality/aesthetic stage; for video: images → low-res short clips → high-res long clips (WAN: 256p → 480p → 720p; HunyuanVideo similar), with joint image-video batches throughout so the model doesn't forget spatial quality. Low-res stages are where the compute goes — each 2× res step is 4× tokens and up to 16× attention FLOPs — which is the same intuition SPEED applies at inference time.
- **Filtering:** aesthetic score, watermark/OCR filters, CLIP similarity, dedup, motion-magnitude filters for video (drop static clips and hard cuts using scene detection).
- Data flywheel: the model itself + human preference data (see LoRA/RL post-training) is the last mile.

## Q: What's wrong with FID, and what would you use instead to evaluate a high-resolution T2I or a T2V model? {diff=2 tags=evaluation,metrics,fid,fvd,vbench}
- Follow-up: Why is FID especially misleading at high resolution or for a speedup paper?
### A
**FID measures Fréchet distance between Gaussians fit to Inception pool-3 features of 299×299 images — so it's (a) blind to anything Inception downsampling destroys, (b) biased toward ImageNet-like content, (c) assumes Gaussian features, (d) needs ~10-50k samples and is biased with fewer.**

- **Why bad at high res:** every image is resized to 299², so a 1024² image loses ~90% of its pixels; sharpness, text, and fine texture — the things high-res models are for — are invisible. Two models differing only in high-frequency detail can have identical FID. For a *speedup* paper this is dangerous: a method that blurs slightly (e.g., continuous frequency masking) can have *better* FID than the baseline because blur reduces feature variance. That's why SPEED reports multiple metrics and human/side-by-side comparisons and inspects the spectrum.
- **Alternatives for images:** CLIP score (prompt alignment, but saturates and prefers "literal" images), ImageReward / PickScore / HPSv2 (learned from human preferences, correlate best with humans), GenEval / T2I-CompBench / DPG-Bench (object count, color, spatial relations via detectors — tests prompt following structurally), DINOv2- or CLIP-based FD (FD_DINO is much more sensitive than FID), and OCR accuracy for text rendering. Always report at native resolution where possible.
- **Video:** FVD (I3D features, 16 frames at 224²) — same problems plus insensitivity to temporal coherence beyond 16 frames and a preference for static videos. VBench decomposes into ~16 dimensions (subject consistency, motion smoothness, dynamic degree, aesthetic, imaging quality, text alignment) — better, but gameable: "motion smoothness" rewards near-static video, so pair it with "dynamic degree". Human preference (side-by-side Elo, as in the Artificial Analysis / VideoArena style) is the gold standard.
- **For a speed method specifically:** paired comparison with the same seed/prompt against the full-cost baseline: PSNR/SSIM/LPIPS vs baseline output *plus* a reference-free quality score, plus wall-clock on identical hardware. PSNR-to-baseline alone rewards being a blurry copy.

## Q: You LoRA-fine-tune a pretrained DiT to change its input statistics — new resolution, mixed-resolution tokens. Which layers, what rank, why does LoRA suffice, and what LR/steps would you use? {diff=2 tags=lora,fine-tuning,dit,foveated,speed}
- Follow-up: Why not just full fine-tune?
- Follow-up: When does LoRA *not* suffice?
### A
**LoRA on the attention QKV/output projections (and usually the MLP) at rank 16-64 is enough because changing input statistics is a low-rank adjustment — the model already knows how to denoise; it just needs to recalibrate how it reads positions and token scales.**

- **Mechanism:** $W' = W + (\alpha/r)\, B A$ with $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $B$ initialized to zero. Trainable params for FLUX (12B) at rank 32 on all linear layers ≈ 100-300M (~1-2%).
- **Which layers:** attention Q/K/V/O first — resolution and mixed-res tokens change *where* attention should look (RoPE frequencies interact with q/k), so q/k adaptation matters most. Add MLP up/down for changes in token content statistics (new patch scale = new pixel-to-token mapping). The patch embedder and final unpatchify layer: train fully (they're tiny) if the patchify scale changes, as in Foveated Diffusion's multi-scale patchify. Modulation (adaLN) layers: often left alone.
- **Rank:** 16-64; ablations typically show rank 8 vs 128 differ little for distribution-shift tasks. Alpha = rank or 2×rank; rsLoRA scaling ($\alpha/\sqrt{r}$) helps at high rank.
- **LR / steps:** 1e-4 (LoRA tolerates ~10× higher LR than full FT since it's a scaled low-rank delta), AdamW, bf16, batch of ~32-64 at the target resolution, 2k-10k steps — a few hours on 8 GPUs. Warmup 100-500 steps; cosine or constant. Monitor with the validation loss at a fixed set of timesteps and visual samples every 500 steps.
- **Why not full FT:** memory (12B params × 16 bytes for Adam states ≈ 200 GB per replica → needs FSDP), catastrophic forgetting of prompt following, and no need. LoRA also ships as a 200 MB adapter that composes with other LoRAs.
- **When LoRA isn't enough:** a genuinely new modality/channel count (e.g., 16-ch → 32-ch VAE), new attention pattern requiring different inductive bias, or large distribution shifts where you want to change *what* the model generates, not *how it reads input*. Then: full FT of the affected blocks, or train new embedder + LoRA on the rest.

## Q: Break down where wall-clock time goes when sampling a 480p, 81-frame video with a 14B DiT like WAN 2.1. Attention vs MLP FLOPs, VAE decode, CFG. {diff=3 tags=inference,cost,video,attention,flops}
- Follow-up: At what token count does attention overtake the MLP?
- Follow-up: What would you optimize first?
### A
**Roughly: 50 steps × 2 CFG passes × 40 blocks of a 32k-token transformer ≈ 95% of the time; VAE decode ≈ 3-5%; text encoding negligible. Within a block at 32k tokens, attention FLOPs ≈ 2× the MLP FLOPs, so attention is ~60-65% of the transformer time.**

Numbers (WAN 2.1 14B: 40 blocks, $d = 5120$, heads 40, MLP 13824, patch (1,2,2)):
- Tokens: latent 21 × 60 × 104 → 21 × 30 × 52 = **32,760 tokens**.
- Per block, per token: QKVO projections $4 \cdot 2 \cdot d^2 \approx 210$ MFLOP; MLP $2 \cdot 2 \cdot d \cdot d_{\text{ff}} \approx 283$ MFLOP; cross-attn K/V is tiny (512 text tokens). So linear layers ≈ 0.5 GFLOP/token → ×32.8k tokens ≈ **16 TFLOP per block**.
- Self-attention: $QK^\top$ and $AV$ $= 2 \times 2 \times N^2 \times d = 4 \cdot (32.8\text{k})^2 \cdot 5120 \approx$ **22 TFLOP per block**. So attention:linear ≈ 1.4:1 at 480p; at 720p (~75k tokens) it's ~3:1.
- Crossover: attention = linear when $4 N d \approx 4 \cdot 2 \cdot d^2 + 4 \cdot d \cdot d_{\text{ff}}$ → $N \approx 2d + d_{\text{ff}} \approx$ **24k tokens**. Below that (images: FLUX at 1024² is 4096 tokens) the MLP dominates and cost is ~linear in tokens; above it, quadratic.
- Whole forward: ~40 × 38 ≈ 1.5 PFLOP. × 100 passes (50 steps, CFG) ≈ 150 PFLOP. At ~600 TFLOP/s effective bf16 on a B200 with FlashAttention → ~4 min; in practice ~5-8 min single-GPU on H100 because attention runs at lower MFU than matmul.
- VAE decode: a 3D conv decoder at 832×480×81 — ~5-10 s, plus memory pressure (chunked decode).

**What to optimize first:** (1) tokens — quadratic term — hence SPEED (fewer tokens for most of the trajectory) and Foveated (fewer tokens everywhere); (2) number of forward passes — distillation, CFG-free / guidance distillation, step caching (TeaCache); (3) attention kernel — FA3 / sparse tile attention; (4) parallelism — sequence parallel (Ulysses/ring) across 8 GPUs for latency. SPEED's 2.54× on WAN is consistent with spending most steps at 1/4 tokens: linear layers drop 4×, attention 16×.

## Q: What causes temporal flicker and inconsistency in generated video, and how do you fix it? {diff=2 tags=video,temporal-consistency,flicker}
- Follow-up: How would you diagnose whether flicker comes from the VAE or the diffusion model?
### A
**Flicker = high-frequency content that isn't temporally correlated: either the model denoises frames with insufficient cross-frame coupling, or the VAE decodes each frame with independent high-frequency errors, or the sampler injects uncorrelated noise per frame.**

Causes and fixes:
- **Architecture:** per-frame 2D models with weak temporal layers (AnimateDiff-style) — temporal attention only along each pixel's column can't track moving content. Fix: full 3D attention, 3D VAE with temporal compression (the latent itself is temporally smooth), 3D RoPE.
- **Noise correlation:** i.i.d. noise per frame at initialization biases toward uncorrelated detail. Fixes: shared/mixed noise (PYoCo's progressive noise: $\epsilon_t = \text{shared} \oplus \text{per-frame}$), FreeNoise for long video, or simply the 3D VAE, which makes "one latent frame = 4 pixel frames" so noise is shared.
- **Training data:** clips with cuts, camera shake, or compression artifacts teach flicker; filter by scene detection and motion statistics.
- **VAE:** 2D VAE decoding a video frame-by-frame produces flicker on textures; 3D causal VAE decoders with temporal receptive field fix it. Diagnosis: encode-decode a *real* video and measure per-frame PSNR/LPIPS and temporal LPIPS — if the reconstruction flickers, it's the VAE.
- **Sampler-level:** progressive methods must keep new noise consistent with the existing trajectory — in SPEED's video version, the 3D DCT expansion fills new spatiotemporal frequencies with $t\,\epsilon$ in the joint spectrum so temporal detail is revealed as a band, not per frame; and resetting the multistep solver's history at transitions matters because stale UniPC history across a resolution change produces exactly this artifact.
- **Long video / autoregressive:** error accumulation and drift. Fixes: overlapping-chunk or diffusion-forcing training (per-frame noise levels), KV-cached causal models (Self-Forcing, CausVid) trained with rollout so they see their own errors, anchor-frame conditioning.
- **Metric:** VBench temporal flickering / subject consistency, warped-frame error with optical flow, or the temporal power spectrum — flicker shows as excess high temporal frequency energy.

## Q: How do you do image editing or inpainting with a diffusion model? Cover RePaint, conditioning-channel inpainting, and ControlNet, and say when you'd use each. {diff=1 tags=editing,inpainting,controlnet,conditioning}
- Follow-up: Why does naive RePaint produce seams, and what's the resampling trick?
### A
**Three families: training-free masking of the sampling trajectory (RePaint), training with the mask and masked image as extra input channels (SD-inpainting), and a trainable side-network injecting spatial conditions (ControlNet).**

- **RePaint (training-free):** at every step, replace the known region with the forward-noised ground truth $x_t^{\text{known}} = (1-t)\,x_0 + t\,\epsilon$ and keep the model's prediction in the unknown region. Problem: the unknown region is denoised without ever "seeing" a harmonized known region — it only sees the noised version — so you get seams and semantically inconsistent fills. Fix: **resampling** — go forward a few steps (re-noise) and denoise again, ~10 times per step, so information propagates across the boundary. Cost: 10× slower. Good for quick experiments, no training.
- **Conditioning channels (SD-inpainting, FLUX-Fill):** concatenate [noisy latent (16ch), masked-image latent (16ch), mask (1ch)] → 33 input channels; train (or fine-tune) with random masks. Model learns to inpaint coherently in one pass. Best quality and speed; needs fine-tuning and a modified first layer (zero-init the new channels' weights so training starts from the pretrained behaviour).
- **ControlNet:** clone the encoder blocks, feed a spatial condition (edges, depth, pose) through them, add outputs to the frozen model via zero-initialized convolutions. Frozen backbone preserves generality; zero-init means the model starts identical to the base. In DiTs, the analog is a few copied blocks whose outputs are added to the residual stream. Use it for pixel-aligned structural control; for reference/style (non-aligned), use IP-Adapter-style decoupled cross-attention or in-context tokens.
- **Instruction editing (InstructPix2Pix, FLUX Kontext):** condition on the source image (channels or in-context tokens) + edit text, trained on synthetic edit pairs.
- **Trap:** in inpainting, blend with the original in pixel space at the end (Poisson or feathered) — the VAE round-trip changes unmasked pixels.

## Q: Autoregressive vs diffusion for images and video — VAR, MAR, and hybrids. Where does each win, and where is the field converging? {diff=2 tags=autoregressive,diffusion,var,mar,hybrid,video}
- Follow-up: Why does raster-order next-token AR do poorly on images, and how does VAR fix it?
- Follow-up: How does this relate to causal video generation and world models?
### A
**Diffusion wins on per-sample quality and parallelism at fixed length; AR wins on variable length, streaming/causality, and unification with LLMs. The convergence point is "AR over time, diffusion within a chunk" — next-frame (or next-scale) prediction where each step is a small diffusion.**

- **Raster AR (DALL·E 1, Parti, LlamaGen):** VQ tokens, next-token in scan order. Problems: a 1D order on 2D data is arbitrary, quantization loses information, and $N$ tokens = $N$ sequential steps (1024² → 4096 steps). Quality lags diffusion.
- **VAR (next-scale prediction):** predict the whole next resolution level conditioned on all coarser levels — 10 scales instead of 4096 tokens, each step parallel. Matches the coarse-to-fine inductive bias (the same one SPEED exploits in the diffusion setting), with better scaling laws than raster AR. Still VQ.
- **MAR / diffusion-loss AR:** keep AR ordering (random-order masked) but replace the categorical head with a small per-token diffusion MLP so tokens are continuous — no VQ. Bridges to diffusion.
- **Hybrids:** Transfusion / Show-o (one transformer, next-token for text, diffusion for images); for video, **diffusion forcing**, CausVid, Self-Forcing, MAGI-1: a causal transformer over frames/chunks with KV cache, each chunk denoised by a few diffusion steps, distilled from a bidirectional teacher. That's what streaming and interactive world models (Genie 3-style, lingbot-VA) need: the model must be *causal* in time so actions can be injected, and it must run at frame-rate — bidirectional 50-step video diffusion can't.
- **Practical numbers:** a distilled causal 1.3B model produces ~10-16 fps at 480p on one H100; bidirectional WAN 14B takes minutes per 5-s clip.
- **Where I'd bet:** continuous latents + flow matching stay for the "within chunk" denoiser; autoregression over time with KV cache for long horizon; the open problems are drift over long rollouts (exposure bias), KV memory growth (my surprise-gated KV-compression work: keep tokens that are surprising and attended by the action expert, ~44-54% reduction), and unifying training so the model sees its own errors.

## Q: Design a 10B-parameter text-to-video model from scratch. Give me the top-level decisions and the numbers behind them. {diff=3 tags=design,video,architecture,training,scaling}
- Follow-up: What would you change if the target is real-time interactive generation instead of offline quality?
### A
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


<!-- rapid-fire -->

## Q: One minute: what is a latent diffusion model? {diff=1 tags=latent-diffusion,vae,rapid-fire}
### A
A **latent diffusion model** runs the diffusion/flow process in the compressed latent space of a pretrained autoencoder instead of in pixel space. A VAE encoder maps a 1024×1024×3 image to something like 128×128×16 (8× spatial downsampling); the denoiser is trained on those latents; at sampling time you denoise in latent space and decode once at the end. It exists because pixel-space diffusion spends most of its compute on perceptually irrelevant high-frequency detail; the VAE handles that cheaply, and the denoiser's cost drops by ~64× per step. Trap: the VAE bounds quality — text, faces, and fine texture are limited by reconstruction, which is why the field moved to 16-channel latents. Follow-up: "why not train the VAE and denoiser jointly?" Latent drift makes it unstable; freeze the VAE.

## Q: One minute: what is a VAE, and what is the ELBO in words? {diff=1 tags=vae,elbo,rapid-fire}
### A
A **VAE** is an encoder-decoder trained as a latent-variable generative model: the encoder outputs a distribution $q(z \mid x)$ (mean and variance), you sample $z$ with the reparameterization trick, and the decoder reconstructs $x$ from $z$. The **ELBO** is a lower bound on $\log p(x)$ that you maximise instead of the intractable marginal: in words, "reconstruct the input well" (expected log-likelihood under the decoder) **minus** "keep the encoder's posterior close to the prior" ($\mathrm{KL}(q(z \mid x) \,\|\, \mathcal{N}(0, I))$). The KL term is what makes the latent space smooth and sampleable. Trap: in latent-diffusion VAEs the KL weight is tiny (~1e-6) and there is a GAN + LPIPS loss, so it is really a regularised autoencoder, not a generative model in its own right — the diffusion model is the prior.

## Q: One minute: what is a DiT? {diff=1 tags=dit,architecture,rapid-fire}
### A
A **DiT (Diffusion Transformer)** replaces the U-Net denoiser with a plain ViT: patchify the (latent) image into tokens (patch size 2 on a 128×128 latent gives 4096 tokens), add position embeddings, run $N$ transformer blocks with full self-attention, unpatchify to predict noise or velocity. Timestep and class/text conditioning enter through **adaLN-Zero**: a small MLP maps $(t, c)$ to per-block scale/shift/gate, with the gate initialised to zero so each block starts as identity. It exists because transformers scale predictably with compute (FID tracks Gflops) and have no convolutional inductive bias to fight at high resolution. Trap: cost is quadratic in tokens, so resolution and video length are the expensive axes — which is what token-reduction methods like SPEED and Foveated Diffusion attack.

## Q: What is a 3D causal VAE, and why "causal"? {diff=2 tags=video,vae,causal,rapid-fire}
### A
A **3D causal VAE** is the video autoencoder used by WAN, CogVideoX, and Hunyuan: 3D convolutions compress a video 8× spatially and 4× temporally (81 frames → 21 latent frames), and every temporal convolution is **causal** — padded only on the past, so latent frame $k$ depends on input frames $\le k$. Two reasons: (1) the first frame is encoded independently, so a single image is a valid 1-frame video and image and video training share one latent space; (2) you can encode and decode in chunks, streaming through arbitrary-length videos with constant memory by carrying a small cache of past features. Trap: temporal compression means one latent frame mixes 4 pixel frames, so fast motion blurs at reconstruction — the VAE, not the denoiser, caps temporal fidelity.

## Q: Temporal attention vs spatial attention in a video model — what's the difference, and which do modern models use? {diff=1 tags=video,attention,rapid-fire}
### A
**Spatial attention** lets tokens attend within a single frame; **temporal attention** lets each spatial position attend across frames at that same position. Early video models (AnimateDiff, VideoLDM, SVD) factorised attention this way — spatial layers inherited from an image model plus inserted temporal layers — because full attention over $T \times H \times W$ tokens was unaffordable and it let them reuse image weights. The cost is $T (HW)^2 + HW\, T^2$ instead of $(THW)^2$. Modern DiTs (WAN, Sora-style, CogVideoX) use **full 3D attention** over all spatiotemporal tokens with 3D RoPE, because factorised attention cannot model a moving object attending to where it was, which shows up as flicker and drift. Follow-up: "how do you make full 3D attention affordable?" FlashAttention, sequence parallelism, and token reduction.

## Q: One minute: what is LoRA? {diff=1 tags=lora,fine-tuning,rapid-fire}
### A
**LoRA (Low-Rank Adaptation)** freezes the pretrained weights $W$ and learns an additive update $\Delta W = B A$ with $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $r \ll d$ (typically 8–128), applied to the attention and sometimes MLP projections; the forward pass becomes $W x + (\alpha/r)\, B A x$. It exists because full fine-tuning of a 12B model needs optimizer state for every parameter and produces a full-size checkpoint per task; LoRA trains <1% of the parameters, fits on one GPU, and the adapter is a few hundred MB that can be merged into $W$ at zero inference cost. Trap: initialise $B$ to zero so training starts at the pretrained function. Follow-up: "why does low rank suffice?" Task adaptation is a small change relative to the pretraining manifold — which is exactly why LoRA sufficed for SPEED's and Foveated Diffusion's input-statistics shifts.

## Q: What is FID, in one minute? {diff=1 tags=fid,evaluation,rapid-fire}
### A
**FID (Fréchet Inception Distance)** embeds real and generated images with an Inception-v3 pool3 layer (2048-d), fits a Gaussian to each set, and reports the Fréchet (Wasserstein-2) distance between them: $\|\mu_r - \mu_g\|^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big)$. Lower is better; it captures both fidelity (mean) and diversity (covariance) in one number, which is why it became the default. Traps: it is biased by sample count (always compare at the same $N$, usually 5k–50k), the Gaussian assumption is crude, Inception is trained on ImageNet at 299px so it is blind to high-resolution detail and text rendering, and it is not comparable across resolutions or datasets. Follow-up: "what instead?" CLIP-FID or DINOv2-FD for features, plus human preference and prompt-alignment metrics.

## Q: What is CFG distillation, and what does it mean that FLUX-dev is "guidance-distilled"? {diff=2 tags=cfg,distillation,flux,rapid-fire}
### A
**Classifier-free guidance** needs two forward passes per step — conditional and unconditional — then extrapolates: $v = v_{\text{uncond}} + w\,(v_{\text{cond}} - v_{\text{uncond}})$. **CFG distillation** trains a student to output that guided prediction in one pass, taking the guidance scale $w$ as an extra conditioning input (embedded like the timestep), so inference is 2× cheaper for free. FLUX.1-dev is exactly this: the "guidance" argument in diffusers is an input to the network, not a second pass, and there is no negative prompt. Trap: the student only covers the range of $w$ it saw, and you lose the ability to guide on new conditions or negative prompts; that is why Qwen-Image or WAN still run true CFG and why SPEED's reported speedups must say whether the baseline counts one or two passes per step.
