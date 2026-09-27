# Deep Learning Architectures
<!-- weight: 3 -->

## Q: Derive a transformer block for me from scratch: the attention formula, why the $1/\sqrt{d}$ scaling, and what multi-head buys you. {diff=1 tags=transformer,attention,architecture}
- Follow-up: What are the shapes at every step for $B=1$, $N=1024$ tokens, $D=1024$, 16 heads?
### A
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

## Q: Self-attention vs cross-attention — what's the difference, and where does cross-attention appear in text-to-image models? Where has it disappeared? {diff=1 tags=attention,text-to-image,diffusion}
- Follow-up: Why did FLUX/SD3 drop cross-attention for joint attention?
### A
**Self-attention lets tokens in one sequence attend to each other ($Q, K, V$ all from $X$); cross-attention lets one sequence query another ($Q$ from $X$, $K$ and $V$ from a context $C$).**

Cross-attn: $A = \operatorname{softmax}\!\left(X W_Q (C W_K)^\top / \sqrt{d}\right) (C W_V)$. Its cost is $O(N \cdot M)$ for $N$ image tokens and $M$ context tokens — with $M = 77$ CLIP tokens that's negligible next to $O(N^2)$ self-attention.

Where it appears in T2I:
- **U-Net models (SD 1.x / 2.x / SDXL)**: every transformer block in the U-Net has self-attn over image tokens *then* cross-attn to the CLIP text embeddings (77×768 or 77×2048). The text is a fixed, non-updated context; only image features change through depth. Prompt-to-prompt / attention-map editing works by manipulating these cross-attn maps.
- **PixArt-α, early DiT-based T2I**: DiT blocks with self-attn + an added cross-attn to T5 text embeddings, timestep via adaLN.
- Class-conditional DiT has no cross-attn at all — class and timestep go in through adaLN.

Where it disappeared: **MMDiT (SD3, FLUX)** concatenates text and image tokens into one sequence and runs *joint self-attention* — each modality has its own Q/K/V/MLP weights (dual-stream) but they share one attention matrix. Text tokens are now updated layer by layer and can attend back to the image, which improved text rendering and compositional binding. FLUX uses 19 dual-stream blocks followed by 38 single-stream blocks where text and image share all weights. WAN 2.1 goes the other way: T5 text enters via cross-attention in every block (cheaper for 32k video tokens, and text doesn't need image feedback).

Practical consequence for efficiency work: with cross-attn, text cost is fixed; with joint attention, text tokens (512 T5 tokens in FLUX) add to the quadratic term — but at 4k image tokens it's a minor (~25%) overhead.

## Q: Compare positional encodings: sinusoidal, learned, and RoPE. How exactly does RoPE encode relative position, and how do you extend it to 2D images and 3D video? {diff=2 tags=positional-encoding,rope,transformer,video}
- Follow-up: What breaks when you generate at a resolution larger than training?
### A
**Sinusoidal and learned encodings add an absolute position vector to the input; RoPE rotates $q$ and $k$ by position-dependent angles so the dot product depends only on the relative offset, and it composes per axis for images and video.**

- Sinusoidal (Vaswani): $\mathrm{PE}(\text{pos}, 2i) = \sin\!\left(\text{pos} / 10000^{2i/D}\right)$, cos for odd dims. Fixed, extrapolates in principle, but the model never learns to use unseen positions well.
- Learned absolute (BERT, ViT, original DiT): a table of $N \times D$ parameters; no extrapolation at all — ViT interpolates the 2D grid when changing resolution.
- **RoPE** (Su et al.): split the head dim into $d/2$ pairs; for pair $i$ with frequency $\theta_i = 10000^{-2i/d}$, rotate $(q_{2i}, q_{2i+1})$ by angle $m \theta_i$ for token at position $m$:
  $q'_m = R(m\theta)\, q$, $k'_n = R(n\theta)\, k$.
  Because rotations compose, $q'^{\top}_m k'_n = q^\top R((n - m)\theta)\, k$ — **the score depends only on $n - m$**. Absolute position is applied, relative position is what attention sees. It's applied to $q$ and $k$ only (not $v$), inside every layer, and costs nothing in parameters. Low-frequency pairs give long-range smooth positional signal; high-frequency pairs resolve adjacent tokens.

2D / 3D extension (FLUX, WAN, most video DiTs): partition the head dim into axis groups — e.g., for 3D, split $d = 128$ into (t: 32, h: 48, w: 48) or FLUX's (idx: 16, h: 56, w: 56) — and apply 1D RoPE per group with that axis's coordinate. The score then decomposes as a sum of per-axis relative terms. Text tokens in MMDiT get position $(0,0,0)$ or their own 1D axis. Video latents from WAN at 480p × 81 frames: grid (21, 30, 52), each token gets $(t, h, w)$ integer coordinates.

What breaks beyond training resolution: unseen positions produce rotation angles the model never saw in the *low-frequency* pairs (the high-frequency ones wrap around and are fine). Attention gets diffuse or repeats structure (duplicated subjects). Fixes: NTK-aware / YaRN scaling of $\theta$ base (interpolate the low frequencies), position interpolation (scale coordinates back into the training range), or — the SPEED insight — spend most steps at native resolution and only touch high-res in the final stage.

## Q: How does ViT patchify work, and why does patch size matter so much? Give me the token counts. {diff=1 tags=vit,tokenization,efficiency}
- Follow-up: What's the patch size in latent diffusion transformers and why is it 2?
### A
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

## Q: DiT vs U-Net for diffusion: how does a DiT condition on timestep, class, and text? Explain adaLN-Zero and the MMDiT dual-stream design. {diff=2 tags=diffusion,dit,architecture,conditioning}
- Follow-up: Why zero-init the gate, and why is adaLN better than cross-attention or token concatenation for the timestep?
### A
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

## Q: Why do residual connections work, and what's the difference between pre-norm and post-norm? Which would you pick for a 40-layer DiT and why? {diff=2 tags=architecture,normalization,depth}
- Follow-up: What is the downside of pre-norm at scale, and what's "sandwich norm" / peri-LN?
### A
**Residuals turn a deep composition into a sum of shallow paths, keeping an identity gradient path so depth doesn't multiply Jacobians; pre-norm puts the normalization inside the branch, keeping that identity path clean, so it's the choice for deep models.**

Residuals: $x_{l+1} = x_l + f_l(x_l)$. Gradient $\partial L/\partial x_l = \partial L/\partial x_{l+1} \cdot (I + \partial f/\partial x)$ — the "$I$" term guarantees gradient reaches every layer at least unattenuated. Unrolled, the network is an ensemble of $2^L$ paths dominated by short ones (Veit et al.), which is why removing a single block barely hurts a trained ResNet. Also enables the "each block is a small update to a shared stream" view that makes zero-init and layer-drop work.

- **Post-norm** (original transformer): $x \leftarrow \mathrm{LN}(x + f(x))$. Normalization is *on* the residual path; gradient must pass through every LN, whose Jacobian scales like $1/\|x\|$ — at init this makes gradients in early layers large and the model needs warmup + small lr. Better final performance in some shallow settings (BERT-base) because the stream is re-normalized.
- **Pre-norm**: $x \leftarrow x + f(\mathrm{LN}(x))$. Identity path untouched; trains stably without warmup, allows larger lr; standard in GPT-2+, ViT, DiT, LLaMA, FLUX.

Pre-norm downsides at scale: the residual stream's norm grows with depth ($\propto \sqrt{L}$ or more), so later blocks' contributions $f(\mathrm{LN}(x))$ become relatively small — deep layers are under-utilized ("representation collapse" / effectively shallower net). Remedies: **sandwich norm** (norm both input and output of the branch — CogView, Gemma), peri-LN, DeepNorm (scale residual by $\alpha > 1$, init branch by $\beta < 1$), QK-norm to stop attention logit growth that pre-norm leaves unchecked.

For a 40-layer DiT: **pre-norm with adaLN-Zero gating, plus QK-norm and RMSNorm** — that's what FLUX, SD3, and WAN do; zero-gated branches make depth trainable from step 0, and QK-norm prevents the attention-logit divergence that shows up around 1–10B params in bf16. Post-norm at this depth without warmup would not train.

## Q: Give me the basics of mixture-of-experts: routing, why it's efficient, and how load balancing works. What breaks without it? {diff=2 tags=moe,architecture,scaling}
- Follow-up: How would you apply MoE to a diffusion transformer?
### A
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

## Q: What inductive biases does a convolution have that attention lacks, and when does that matter? Are ViTs "worse" at small data? {diff=2 tags=cnn,attention,inductive-bias}
- Follow-up: How does a ViT recover locality, and what does DINO show about learned attention?
### A
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

## Q: Attention is $O(n^2)$. Spell out what that means for a 480p, 81-frame WAN 2.1 generation — token count, attention FLOPs, memory — and what you'd do about it. {diff=3 tags=attention,efficiency,video,scaling}
- Follow-up: Does FlashAttention change the asymptotics? What about sparse/linear attention for video?
### A
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

## Q: Explain grouped-query attention and multi-query attention. Why do they exist, and what's the actual bottleneck they fix? {diff=2 tags=attention,inference,kv-cache,efficiency}
- Follow-up: Numbers for LLaMA-2 70B: KV-cache size per token with MHA vs GQA?
### A
**MQA shares one K/V head across all query heads; GQA shares K/V across groups of query heads. They cut the KV cache — the memory-bandwidth bottleneck of autoregressive decoding — by $H/G$ with almost no quality loss.**

The bottleneck: during decoding each new token attends to all cached $K, V$. Per token the cache is $2$ (K and V) $\times L$ layers $\times H_{kv}$ heads $\times d_{\text{head}} \times$ bytes. Decode is *bandwidth-bound* — every step reads the whole cache from HBM to produce one token, so latency $\propto$ cache size, and batch size is capped by cache memory.

Numbers for LLaMA-2 70B ($L = 80$, $d_{\text{head}} = 128$, bf16):
- MHA, $H = 64$ kv heads: $2 \times 80 \times 64 \times 128 \times 2\,\text{B}$ = **2.6 MB per token** — a 4k context is 10.7 GB per sequence.
- GQA, 8 kv groups: $2 \times 80 \times 8 \times 128 \times 2\,\text{B}$ = **0.33 MB per token** — 8× smaller. That's why LLaMA-2 70B and all LLaMA-3 models use GQA-8.
- MQA (1 kv head): 41 KB/token — 64× smaller, used by PaLM, Falcon; slight quality drop and less parallel-friendly under tensor parallelism (you'd replicate the single K/V head on every rank, wasting compute), which is exactly why GQA with $G$ = TP degree is the sweet spot.

Quality: GQA-8 matches MHA in perplexity within noise on the LLaMA scaling; MQA loses a little. Both can be *uptrained* from an MHA checkpoint by mean-pooling the K/V heads and fine-tuning for ~5% of pretraining compute.

Relevance beyond LLMs: in autoregressive world / video-action models (streaming rollouts with a KV cache over past latents), the cache is the dominant memory — the same lever applies, and it composes with token-level KV eviction (keeping only surprising or attended tokens, ~40–50% reduction in the candidate's lingbot-VA work). For bidirectional diffusion transformers there's no cache, so GQA only saves projection params and a bit of bandwidth; FLUX/WAN still use full MHA.

MLA (DeepSeek-V2/V3) is the next step: cache a low-rank latent (512-d) per token and up-project $K, V$ on the fly — ~93% smaller cache than MHA with *better* quality than GQA.

## Q: Compare ReLU, GELU, SiLU, and SwiGLU. Why do modern transformers use gated units, and what does that do to the MLP dimensions? {diff=1 tags=activation,mlp,architecture}
- Follow-up: What's the dead-ReLU problem and does GELU fix it?
### A
**ReLU is $\max(0, x)$; GELU and SiLU are smooth, non-monotone approximations that let small negative values pass; SwiGLU replaces the MLP's single nonlinearity with a gated product, which gives better loss per parameter at the cost of a third weight matrix.**

- $\mathrm{ReLU}(x) = \max(0, x)$. Cheap, sparse; gradient exactly 0 for $x < 0$ ⇒ **dead units** that never recover if their pre-activations go negative for the whole dataset (common with high lr / bad init). Doesn't scale as well in transformers.
- $\mathrm{GELU}(x) = x\, \Phi(x)$ (Gaussian CDF); $\approx x\, \sigma(1.702\, x)$. Used in BERT, GPT-2, ViT, DiT. Smooth, has a small negative dip (min $\approx -0.17$ at $x \approx -0.75$), non-zero gradient for moderately negative inputs so units can recover — mostly fixes dead-ReLU.
- $\mathrm{SiLU}/\mathrm{Swish}(x) = x\, \sigma(x)$. Nearly identical to GELU, slightly cheaper; used in EfficientNet, LLaMA, and as the nonlinearity inside adaLN modulation MLPs.
- **SwiGLU** (Shazeer, "GLU Variants Improve Transformer"):
  $\mathrm{FFN}(x) = (\mathrm{SiLU}(x W_1) \odot x W_3)\, W_2$.
  Three matrices instead of two; to keep parameter count equal to the $4D$ standard MLP, the hidden size is set to $\tfrac{8}{3} D$ (LLaMA rounds to a multiple of 256: 4096 → 11008; FLUX single-stream blocks and most 2024+ DiTs use gated MLPs too). Gives ~1–2% lower loss at equal params/FLOPs; the multiplicative gate gives the MLP a data-dependent "attention over features," which is the intuition for why it helps (same reason gating helps LSTMs and MoE routers).

Why not sigmoid/tanh: saturate, gradients $\le 0.25$, cause vanishing gradients in deep nets — the original reason ReLU won in 2012.

Trap: the choice of activation matters far less than normalization, init, and lr schedule; don't spend hyperparameter budget here. But do note that GELU's exact `erf` form vs. the tanh approximation matters for reproducing checkpoints (GPT-2 uses tanh-approx; most HF ViTs use exact).

## Q: Estimate the parameter count and training FLOPs for a transformer layer and a full model. Where does the $6ND$ rule come from, and what are the FLOPs per token at inference? {diff=3 tags=scaling,flops,transformer,systems}
- Follow-up: Apply it to training a 10B-param video DiT on 1T tokens on 512 H100s — how long?
### A
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

<!-- rapid-fire -->

## Q: What is an embedding? {diff=1 tags=architecture,embedding,representation,rapid-fire}
### A
**A learned map from a discrete or structured input — token, patch, timestep, class — to a dense vector in $\mathbb{R}^d$, such that geometry in that space (dot product, distance) reflects semantic similarity.**

It exists because networks compute on continuous vectors and one-hot inputs are huge, sparse, and carry no similarity: every pair of words is equally far apart. A lookup table $E \in \mathbb{R}^{V \times d}$ is just a linear layer on a one-hot vector, trained end-to-end. The same idea covers ViT patch embeddings (linear projection of pixels), diffusion timestep embeddings (sinusoidal features → MLP), and CLIP/T5 text embeddings as conditioning.

Trap: the word is overloaded — input lookup table vs. an encoder's output representation. And embeddings are only meaningful relative to their trained head: cosine similarity across two different models means nothing.

## Q: What is layer normalization? Give the formula. {diff=1 tags=normalization,architecture,rapid-fire}
### A
**LayerNorm normalizes each token's feature vector to zero mean and unit variance across the feature dimension, then rescales with learned $\gamma, \beta$: $y = \gamma \odot (x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta$, where $\mu, \sigma^2$ are computed over the $d$ features of that single token.**

It keeps activations at a fixed scale layer after layer so gradients neither vanish nor explode and the learning rate stays meaningful; unlike BatchNorm it has no batch dependence, so it works at batch size 1, with variable sequence lengths, and identically at train and test. RMSNorm drops the mean subtraction, $y = \gamma \odot x / \mathrm{RMS}(x)$, and is cheaper with no quality loss; LLaMA and most DiTs use it.

Trap: in DiTs $\gamma$ and $\beta$ are not static — adaLN generates them from the timestep/class embedding, so the norm *is* the conditioning mechanism.

## Q: What is a residual connection? {diff=1 tags=architecture,residual,depth,rapid-fire}
### A
**A residual connection adds a layer's input to its output, $y = x + F(x)$, so the layer learns a correction to the identity instead of a whole new mapping.**

Why: deep plain networks got *worse* with depth even on training loss — an optimization problem, not overfitting. With residuals the identity is the default, so adding layers never hurts at init, and the gradient has a direct path, $\partial y/\partial x = I + \partial F/\partial x$, that never vanishes through the stack. Each block makes a small update to a shared residual stream — the mental model behind transformers.

Trap: the residual stream's variance grows with depth as blocks add to it, which is why pre-norm and zero-init of each block's last layer (adaLN-Zero) matter for very deep DiTs.

## Q: What is the receptive field? {diff=1 tags=cnn,architecture,receptive-field,rapid-fire}
### A
**The receptive field of an output unit is the region of the input that can influence it; for stacked convolutions it grows with depth and kernel size, $\mathrm{RF}_l = \mathrm{RF}_{l-1} + (k - 1) \cdot (\text{product of strides so far})$.**

A conv layer only sees a $k \times k$ window, so relating distant pixels needs depth, dilation, striding, or attention; two 3×3 convs match one 5×5 with fewer parameters (the VGG argument). In a ViT or DiT every token attends to every token, so the receptive field is global from layer one — that's the inductive-bias trade: convs get locality for free and must earn global context; attention gets global context and must learn locality.

Trap: the *effective* receptive field is much smaller than the formula — gradient contributions are Gaussian-shaped around the center — so deep CNNs "see" less than claimed.

## Q: What's the difference between a U-Net skip connection and a residual connection? {diff=2 tags=unet,architecture,skip-connection,rapid-fire}
### A
**A residual connection adds a block's input to its output at the *same* depth, $y = x + F(x)$; a U-Net skip carries encoder features across the whole network to the decoder at the *same spatial resolution*, usually by channel concatenation.**

Different problems: residuals fix optimization depth. U-Net skips fix information loss — downsampling to a bottleneck destroys high-frequency detail, and the skip hands the decoder the fine encoder features so it can localize edges, essential for dense prediction and denoising. Concatenation lets the decoder learn how to weight the two sources.

Trap: in diffusion U-Nets the skips carry so much high-frequency content that the decoder can bypass the bottleneck; FreeU shows down-weighting them at inference improves samples. DiTs have no U-Net skips — the full-resolution residual stream does the job, which is why DiT quality leans so heavily on the VAE.

## Q: Define attention in one sentence, and give me the intuition for Q, K, and V. {diff=1 tags=attention,transformer,rapid-fire}
### A
**Attention is a data-dependent weighted average: each token builds a query, compares it against every token's key to get softmax weights, and sums those tokens' values with those weights — $\mathrm{Attn} = \operatorname{softmax}(Q K^\top / \sqrt{d})\, V$.**

Intuition: a soft dictionary lookup. Query is "what am I looking for", key is "what I advertise", value is "what I hand back if chosen". Separating K from V lets a token be found by one property and contribute another. Softmax makes a convex combination, so the output stays in the span of the values, and $\sqrt{d}$ keeps logits $O(1)$ so softmax doesn't saturate at init.

Trap: attention is a set operation with no notion of position — RoPE is what makes it a sequence model — and it's quadratic in tokens, the entire reason for efficient-attention and token-reduction work.

## Q: What is a causal mask? {diff=1 tags=attention,autoregressive,masking,rapid-fire}
### A
**A causal mask sets attention logits to $-\infty$ for every key at a later position than the query, so token $i$ attends only to tokens $\le i$; after softmax those entries are exactly zero.**

It exists so one forward pass over a whole sequence trains next-token prediction at every position in parallel while guaranteeing no position sees its own target. At inference it's what makes the KV cache valid: past keys and values never change when new tokens arrive, so each token is computed once.

Trap: you choose the granularity. In video models, full per-token causality is slow and unnecessary; block-causal per frame (bidirectional within a frame, causal across frames) is what CausVid / Self-Forcing use. And mask with $-\infty$ *before* softmax, not by zeroing after, or rows won't sum to one.

## Q: What is teacher forcing? {diff=1 tags=autoregressive,training,exposure-bias,rapid-fire}
### A
**Teacher forcing trains a sequence model by feeding it the ground-truth previous tokens as context at every step, rather than its own predictions, so the whole sequence trains in one parallel pass.**

It exists because sampling your own outputs during training would be sequential and slow and would route gradients through a stochastic decoder. With teacher forcing plus a causal mask, next-token loss at every position comes from a single forward pass — why transformers train efficiently.

The trap it creates is **exposure bias**: at inference the model conditions on its *own* imperfect outputs, a distribution it never saw in training, so errors compound over long rollouts — acute in autoregressive video / world models, where a small frame error drifts into garbage within seconds. Fixes: scheduled sampling, noise-augmented context, DAgger-style rollouts, Self-Forcing.
