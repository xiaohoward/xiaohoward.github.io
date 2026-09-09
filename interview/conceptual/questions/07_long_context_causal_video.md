# Long Context & Causal Video Generation
<!-- weight: 4 -->

## Q: Why can't a standard bidirectional video diffusion model like WAN 2.1 stream frames, and what has to change architecturally to make it causal / autoregressive? {diff=1 tags=causal-video,autoregressive,kv-cache}
- Follow-up: What is the difference between frame-by-frame and chunk-wise causal generation?
### A
A bidirectional video DiT denoises **all frames jointly**: every token attends to every other token in space and time, and every denoising step refines the whole clip. So frame 1 is not finished until frame 80 is finished — there is no point at which you can emit frame 1 and keep going. Latency is the full clip's 50 steps × full-clip attention, and the length is fixed at training time.

To stream, you need three changes:
- **Causal attention over time**: frame (or chunk) $k$ only attends to frames $\le k$. Then earlier frames' KV are fixed once generated and can be **cached**, exactly like an LLM decoder.
- **Per-frame (or per-chunk) denoising**: fully denoise chunk $k$ conditioned on the clean cache of chunks $< k$, then append its KV and move on. This is the CausVid / Self-Forcing / MAGI-1 recipe.
- **Few-step denoising** (DMD or consistency distillation), because you now pay the sampler cost once per chunk, so 50 steps × chunks is too slow for real time.

Frame-by-frame gives lowest latency but the weakest intra-chunk coherence and the most AR steps (more error accumulation); **chunk-wise** (e.g. 3–5 latent frames = 12–20 pixel frames) lets the chunk be denoised bidirectionally inside itself while remaining causal across chunks, which is the sweet spot most systems use. The common follow-up is "what does the cache hold?" — the K/V projections of clean context tokens at every layer, which is what makes memory grow linearly with rollout length.

## Q: Explain Diffusion Forcing. How does it unify autoregressive next-token prediction and full-sequence diffusion, and what does it buy you at sampling time? {diff=2 tags=diffusion-forcing,causal-video,training}
- Follow-up: Why does it need causal attention rather than full attention?
### A
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

## Q: What is exposure bias in autoregressive video generation, why does it hurt more for video than for text, and what are the main remedies? {diff=2 tags=exposure-bias,causal-video,error-accumulation}
- Follow-up: Which of these remedies does Self-Forcing actually implement?
### A
**Exposure bias** is the train/test mismatch: at training the model conditions on ground-truth past frames (teacher forcing); at inference it conditions on its **own** generated frames, which carry small errors. Those errors compound over steps, and for video the compounding is brutal because the conditioning is high-dimensional and continuous: a slight color shift or blur in frame $k$ is copied and amplified into frame $k+1$, so after a few hundred frames you get saturation, drift, frozen motion, or texture collapse — the classic "AR video degrades after 10 s".

Text tolerates it better because tokens are discrete (small logit errors don't accumulate as continuous drift) and language has strong local re-anchoring.

Remedies, roughly by strength:
- **Noise augmentation of context** (Diffusion Forcing, Cosmos-style): add noise to conditioning frames during training so the model sees "imperfect history"; simple and works, but only matches errors that look like Gaussian noise.
- **Rollout / self-forcing training**: actually **generate the history with the model itself** (few-step sampler, KV cache) during training and train on the resulting sequence — the training distribution equals the test distribution. Self-Forcing does this with a **DMD-style distribution-matching loss** against a bidirectional teacher so you don't need ground-truth continuations of your own rollouts.
- **Scheduled sampling / mixed teacher-student history** as a cheaper approximation.
- **Discrete-ish anchoring**: periodically re-condition on a clean keyframe or a compressed memory token so drift can't run away.

Self-Forcing implements rollout training + DMD distillation + KV-cached causal student, and reports that even a few-frame rollout horizon in training removes most drift. The follow-up trap: rollout training needs backprop through the sampler, so you truncate gradients to the last chunk or two and use a small number of denoising steps (4) for memory.

If pushed on the distillation itself: DMD minimizes $\mathrm{KL}(p_{\text{student}} \,\|\, p_{\text{teacher}})$ with gradient $\propto (s_{\text{fake}} - s_{\text{real}})$, where $s_{\text{real}}$ is the frozen bidirectional teacher's score and $s_{\text{fake}}$ is a copy fine-tuned online on the student's samples; applying it to the student's *rolled-out* clip is what makes the teacher judge whole-video coherence while the student stays causal. Gradients are truncated to the last chunk or two for memory.

## Q: Do the memory arithmetic: a causal video DiT with 30 layers, hidden size 3072, 3 latent frames per chunk, 1560 tokens per latent frame, bf16. How much KV cache per chunk, and how long a rollout fits in 80 GB alongside the model? {diff=1 tags=kv-cache,memory,systems}
- Follow-up: How does GQA/MQA change the picture?
### A
KV cache per token per layer $= 2\,(\text{K and V}) \times d_{\text{model}} \times 2\,\text{bytes (bf16)} = 2 \times 3072 \times 2 =$ **12,288 B ≈ 12 KB**.
Per token over 30 layers: $12\,\text{KB} \times 30 =$ **360 KB**.
Tokens per chunk: $3\ \text{frames} \times 1560 = 4680$ tokens → $4680 \times 360\,\text{KB} \approx$ **1.68 GB per chunk**.

If the model is ~14B params in bf16 that's ~28 GB, leaving ~50 GB. Without activations you'd fit ~30 chunks ≈ 90 latent frames ≈ 360 pixel frames (at 4× temporal VAE compression) ≈ **15 s at 24 fps**. Realistically activations and the fake-score model (if training) take half of that, so a **naive cache is good for 5–10 s of video** — which is exactly why long rollouts need a window or eviction.

Follow-up: with **GQA** (say 8 KV heads instead of 24), the cache drops 3× because only KV heads are stored; MQA drops it by $n_{\text{heads}}\times$. Most video DiTs (WAN, Cosmos) still use full MHA, so this is a free 3–4× on the table. Also mention that attention FLOPs scale with cache length: per new chunk you do $O(n_{\text{new}} \times n_{\text{cache}})$ attention, so a 30-chunk cache makes the last chunk 30× more expensive than the first — time per chunk grows linearly, total cost quadratically in rollout length.

## Q: Explain sliding-window attention with attention sinks (StreamingLLM) and heavy-hitter eviction (H2O). Why does naive sliding-window fail, and what carries over to video? {diff=2 tags=kv-cache,streaming,attention}
### A
**Naive sliding window** keeps only the last $W$ tokens' KV. It fails catastrophically the moment the very first tokens leave the window — perplexity explodes. StreamingLLM's finding: softmax needs somewhere to dump attention mass when nothing is relevant, and models learn to use the **first few tokens as "attention sinks"** (huge attention scores, near-zero information). Evict them and every attention distribution is mis-normalized. Fix: **keep the first ~4 tokens permanently + a sliding window**, and assign positions relative to the cache (not absolute) so RoPE doesn't extrapolate.

**H2O (Heavy-Hitter Oracle)** is content-aware: track the accumulated attention each cached token has received; a small set of "heavy hitters" gets most of the mass (power-law). Keep the top-$k$ heavy hitters + a recent window, evict the rest. It's a greedy, per-layer / per-head budget policy and gives ~5–10× cache reduction on LLMs with little loss.

**What carries over to video:**
- Sinks exist too — usually the first chunk's tokens and any conditioning (text) tokens. Keeping the **first chunk as a permanent anchor** also helps identity/scene persistence, so the sink and the semantic reason coincide.
- Heavy-hitter statistics are noisier per token because video has 1000s of tokens per frame and attention is spread spatially; you'd aggregate per patch across heads or over a chunk.
- Video has structure text doesn't: **static background tokens are redundant across frames** (predictable), so eviction should be by *information*, not just by attention — which motivates surprise-based criteria.
- A cheap baseline that's surprisingly hard to beat for video: keep chunk 1 + last $N$ chunks. Always report it.

## Q: You've built surprise-gated KV eviction for a world-action model. Describe the criterion, why you intersect it with action-expert attention, and then critique it — where does it fail? {diff=3 tags=kv-cache,world-models,project,eviction}
- Follow-up: Why running quantiles instead of z-score thresholds?
### A
**Criterion.** During an AR rollout the world model *imagines* the next latent $\hat{x}_{k+1}$ before it sees the ground-truth observation $x_{k+1}$. The per-token **surprise** is $\|x_{k+1} - \hat{x}_{k+1}\|$ in a **standardized VAE latent space** (per-channel mean/std so no channel dominates). Tokens the model predicted well carry no new information — the model can regenerate them from what it already has — so they're evicted; surprising tokens are kept.

**Why intersect with action-expert attention.** Surprise alone keeps tokens that are unpredictable but irrelevant (flickering background, shadows). The action expert's cross-attention over video tokens tells you which tokens **the policy actually reads**. Keeping only $\text{surprising} \cap \text{attended}$ tokens removes both redundant and irrelevant tokens: ~44–54% KV reduction with **task success unchanged** on LIBERO and only a 0.3–1.7 dB PSNR hit on imagined video.

**Critique (what an interviewer wants to hear me say):**
- **It needs ground truth.** Surprise requires the real next observation; in pure generation (no sensor) there's no surprise signal. It's a *world-action-model / robot* trick, not a video-generation trick; a proxy is denoiser residual or teacher-vs-student disagreement.
- **Attention is a lagging signal.** A token unattended now may be needed in 50 steps (object hidden behind the gripper). Eviction is irreversible — I'd want a small "cold" cache or compressed summary rather than deletion.
- **Confounded evaluation**: a robot that stalls produces a static scene that's trivially predictable → high PSNR and low surprise. I had to report on successful episodes and paired seeds.
- **Per-step overhead** of computing the criterion is non-trivial in a real-time loop; the intersection also needs the action expert's attention maps, which some fused-attention kernels don't expose.
- **Budget stability**: it's not a fixed-size cache; a novel scene floods it. A hard cap with heavy-hitter fallback is needed.

**Running quantiles vs $z$-score.** Surprise is heavy-tailed and non-stationary (scene changes shift the whole distribution). A $z$-score threshold $\mu + c\sigma$ assumes Gaussianity — the mean and std are dragged by outliers, so the threshold is either too loose in calm scenes or too tight in busy ones. A **running quantile** (e.g. keep top 40% by a streaming $P^2$ / reservoir estimate) is distribution-free and self-calibrates to "keep a fixed fraction," which is what you actually want for a memory budget. Median-based MAD is in between but still needs a scale assumption.

## Q: Compare ring attention, blockwise/sparse attention, sliding-window + global tokens, linear attention / SSMs (Mamba), and TTT layers for long video. What does each trade? {diff=2 tags=long-context,attention,ssm,ttt}
### A
Framing: full attention is $O(N^2)$ in compute and $O(N)$ in KV memory; every method attacks one of those, and each pays in **either exactness, expressivity, or parallelism**.

- **Ring attention**: exact full attention, distributed — shard the sequence across GPUs, rotate K/V blocks around a ring while overlapping communication with the local block compute. Trades **communication + hardware** for exactness; nothing is approximated. It's what you use to *train* on minute-long clips at all. Cost per device stays $O(N^2/P)$; comm volume per layer $\approx 2 \times N \times d \times \text{bytes}$.
- **Blockwise / sparse attention** (local blocks + strided/dilated, or learned sparse patterns like NSA/MoBA): reduces FLOPs to $O(NB)$. Trades **exactness for speed**; quality depends on whether the pattern matches the true dependency structure — good for video because dependencies are mostly local in space-time.
- **Sliding window + global tokens** (Longformer-style; for video: local temporal window + a few "memory"/register tokens): $O(NW)$. Trades **long-range recall for bounded cost**; the global tokens are a bottleneck everything long-range must pass through.
- **Linear attention / SSMs (Mamba, RWKV)**: $O(N)$ compute, **$O(1)$ state** — the state is a fixed-size matrix, so memory doesn't grow with rollout at all. Trades **exact retrieval for compression**: a fixed state cannot losslessly remember 10,000 frames; they're weak at "copy the object from 30 s ago" tasks. Great for a streaming backbone with hybrid attention layers for recall.
- **TTT layers**: the hidden state is a small neural net's weights, updated by gradient descent on a self-supervised loss over the incoming tokens; expressivity between SSM and attention. Trades **compute per token (an inner optimization) and implementation complexity** for a state that's more expressive than a linear RNN. The TTT-Video work showed minute-long Tom & Jerry episodes from a 5B backbone by inserting TTT layers into a local-attention DiT — the usual follow-up is "how is it different from fine-tuning at test time?": same principle, but scoped to one layer's state and with the update baked into training so the outer model learns to exploit it.

Rapid-fire answer: *train* with ring attention; *stream* with a hybrid of local attention + compressive state (SSM/TTT) + a few global anchor tokens.

## Q: Your model was trained on 81-frame clips with RoPE. A user asks for 300 frames at inference. Why does it break and what do RoPE base scaling, NTK-aware scaling, and YaRN each do about it? {diff=2 tags=rope,positional-encoding,length-generalization}
- Follow-up: Is the same fix right for the spatial axes at super-native resolution?
### A
**Why it breaks.** RoPE rotates each 2-D pair of $q/k$ dimensions by angle $p\theta_i$ with $\theta_i = \text{base}^{-2i/d}$. The low-frequency dimensions (large $i$) complete a small fraction of a period over 81 frames, so the model has only ever seen a narrow arc of those rotations. At position 300 those dimensions land on angles never seen in training → attention logits are out-of-distribution and you get repetition, motion freezing, or noise.

- **Position interpolation (linear)**: scale positions by $81/300$ so the maximum stays in range. Keeps low-freq dims in-distribution but **compresses the high-frequency dims** too, so neighboring frames become hard to distinguish — blurrier local motion.
- **NTK-aware base scaling**: instead of scaling positions, raise the base (e.g. $10{,}000 \to 10{,}000 \cdot s^{d/(d-2)}$). This scales rotation speed **non-uniformly**: high-freq dims barely change (local structure preserved), low-freq dims are compressed to stay in range. Training-free and usually the best zero-shot choice.
- **YaRN**: NTK-by-parts — leave dims with many periods in training untouched, interpolate dims with $< 1$ period, ramp in between; plus a **temperature on attention logits** ($\propto 1/\sqrt{\text{scale}}$ style) to counteract the entropy increase from a longer softmax. Best when you also do a short fine-tune at the target length.

The honest answer for video: any of these gives you maybe 2–3× length before quality falls off; beyond that you need training at length (ring attention) or a causal model where the *cache* positions are relative and never exceed the window.

**Spatial follow-up (from SPEED):** at 2× native resolution the same extrapolation problem appears on the spatial RoPE axes; NTK scaling helps but the sharper fix is to **spend most denoising steps at native resolution** and only expand late, which is why staged resolution beat full-res low-pass at 960p in my work.

## Q: How would you give a long video generator "memory" of scene state without keeping every frame? Compare latent memory tokens, recurrent state, and compressive transformers. {diff=3 tags=memory,world-state,compression,long-context}
### A
The problem: exact KV grows linearly and attention over it grows quadratically, but the *information* you need to keep — scene layout, object identities, camera pose, what's behind you — is roughly constant. So compress the history into a bounded **world-state**.

- **Latent memory / register tokens**: append $M$ learned tokens that attend to the current chunk and are carried to the next (Memorizing-Transformer / Perceiver style). Cheap, transformer-native, differentiable; the model learns what to write. Weakness: fixed $M$ means saturation, and without a write/erase rule old content decays uncontrollably; plus there's no explicit geometry, so "look back at the room" is learned, not guaranteed.
- **Recurrent state (SSM / GRU / TTT weights)**: state updated every chunk; $O(1)$ memory, streams forever. Weakness: lossy in a way you can't inspect; recall of specific past frames is poor. Good for *dynamics* (velocity, momentum), bad for *episodic* recall.
- **Compressive transformer**: keep a recent exact window, and when tokens age out, **compress them** (pool, conv, or a learned compressor) into a smaller set that stays attendable — a two-tier cache. Gives graceful degradation: recent = exact, old = coarse. Training needs an auxiliary reconstruction loss so the compressor keeps useful content. This is closest to what I'd build: exact last $N$ chunks + compressed older chunks + a permanent first-chunk anchor.

Two extra options worth naming: **explicit 3D memory** (a point map / Gaussian / voxel state updated from generated frames — WorldMem / Genie 3-style memory) which makes revisiting a location consistent by construction, and **surprise-based selective retention** (keep only tokens the model couldn't predict — my KV work), which is compressive in a content-adaptive way.

The design question an interviewer will push on: memory vs. **consistency vs. controllability**. Recurrent state gives smooth dynamics but forgets; explicit 3D memory gives revisitation consistency but assumes a static-ish world. A hybrid is the current frontier.

## Q: How do you evaluate a minute-long generated video? FVD on a 16-frame window doesn't tell you much. {diff=2 tags=evaluation,fvd,long-video,drift}
### A
Right — FVD on 16 frames measures per-clip realism at a single scale and is blind to the two failure modes of long video: **drift** (slow degradation) and **inconsistency** (subject/scene changes). I'd use a battery:

- **Windowed FVD over time**: FVD of window $k$ vs. real windows, plotted against $k$. A rising curve = drift; report the slope, not just the mean. Also FVD between the first and last window (self-consistency).
- **Subject persistence**: track the main subject (DINO / CLIP features or an identity embedding) and report cosine similarity to the first-window embedding over time; for humans, a face-ID model. Also object count / position consistency via a detector.
- **Scene consistency on revisits**: force a camera loop (turn 360°) and compare the returned view to the original — this is the honest test for memory.
- **Motion statistics drift**: optical-flow magnitude over time — most drifted models either freeze (flow $\to 0$) or explode. Also color histogram / saturation over time for the classic "AR videos turn orange" failure.
- **VBench-style decomposed scores** (subject consistency, background consistency, temporal flicker, imaging quality) reported per minute, not per clip.
- **Human preference** on pairs, at fixed timestamps (0 s, 30 s, 60 s) — still the deciding evidence.

Traps: (1) FVD is I3D-based and saturates/prefers static video; pair it with a motion score. (2) Always evaluate the *same* number of frames and windows for every model — window count silently changes FVD. (3) A model that freezes has excellent "consistency" — every consistency metric must be paired with a motion or task metric.

## Q: In your KV-compression evals you compared imagined-video PSNR between methods. Why must the comparison be paired-seed, and what confounds did you hit? {diff=3 tags=evaluation,determinism,project,world-models}
### A
Because a **stochastic rollout's PSNR variance across seeds is larger than the effect size**. With unpaired seeds, method A on seed 1 vs method B on seed 2 differ in initial noise, sampled actions, and physics outcomes; a 0.5 dB difference is pure noise. With **paired seeds** — same initial noise, same environment seed, same action sampling seed — the only difference is the KV policy, and you can do a paired t-test on per-episode deltas, which shrinks the CI by the between-seed variance.

Confounds I actually hit:
- **Flash-attention non-determinism**: even with identical seeds, the backward pass (and some forward reductions, atomics in split-K) is non-bitwise-deterministic, so trajectories diverge after a few steps and "paired" isn't exactly paired. Fixes: `torch.use_deterministic_algorithms`, deterministic attention kernels or SDPA math backend for eval, fixed batch composition, and — more honestly — report the *inherent* seed-to-seed spread of the same method as a noise floor, and only claim differences above it.
- **Failed episode = static scene = high PSNR**: when the policy stalls, the world barely changes, so imagined frames are trivially accurate. A KV policy that *hurts* the policy can score *better* PSNR. So PSNR must be reported **on the same episode outcome stratum** (success-only, or with success rate as the primary metric and PSNR secondary), and paired by episode.
- **PSNR is a bad video metric anyway** — it rewards blur. I added LPIPS / DINO distance and, most importantly, the downstream task success as the primary metric.
- **Ordering effects**: the running-quantile threshold is stateful, so the first episodes in a run behave differently; warm-up episodes are excluded.

The general principle: for stochastic generators, the comparison unit is the (seed, method) pair, and every "quality" metric needs a task metric next to it that exposes the "do nothing" degenerate solution.

## Q: Design question: build a minute-long, consistent, prompt-following video generator that runs at interactive speed on 8 GPUs. Walk me through the design and the tradeoffs. {diff=3 tags=design,long-video,causal-video,systems}
### A
I'd structure it as **a bidirectional teacher + a causal, few-step, memory-augmented student**, and be explicit about what I'm trading at each layer.

**1. Backbone and data.** Start from a strong bidirectional video DiT (WAN-class, ~14B) as the teacher — that's where quality and prompt-following come from. Curate long clips with scene-cut detection and per-shot captions; long-video data is the real bottleneck.

**2. Causal student.** Convert to chunk-wise causal attention (3–4 latent frames per chunk, bidirectional within chunk), KV-cached. Distill with **DMD against the teacher on self-generated rollouts** (Self-Forcing) so the student is trained on its own errors; 4 denoising steps per chunk. Tradeoff: some loss of diversity from DMD and a small quality gap vs. the teacher, in exchange for ~real-time.

**3. Memory.** Three tiers: permanent first-chunk anchor (identity + attention sink), exact KV for the last ~8 chunks, and **compressed older context** (learned pooling or selected surprising tokens) capped at a fixed budget so cost per chunk is constant. Positions are cache-relative so RoPE never extrapolates. Tradeoff: exact recall of the far past is lost; if the product needs "walk around and come back," add an explicit 3D memory.

**4. Speed on 8 GPUs.** Tensor/sequence parallel across the 8 GPUs per chunk, or — better for latency — pipeline chunks so the VAE decode of chunk $k$ overlaps denoising of chunk $k+1$. Add spatial efficiency: SPEED-style spectral progressive resolution (early denoising steps at low res) is training-free and gives ~2× on the few steps that dominate; foveated/mixed-resolution tokens if the app has a natural attention center. Target ≈ 1 chunk (0.5 s of video) in < 0.5 s.

**5. Prompt-following over time.** Support per-chunk prompt updates (the text KV is re-encoded per chunk), and use a lower CFG in the student than the teacher to avoid saturation drift.

**6. Evaluation** built in from day one: windowed FVD slope, subject-persistence curve, motion-magnitude drift, revisit consistency, paired-seed comparisons for every ablation.

**Where it fails:** compounding still happens over minutes (rollout training covers ~10 chunks, not 120), so I'd add periodic re-anchoring — re-denoise a "keyframe" bidirectionally every $N$ seconds with the teacher and re-seed the cache. That's the design lever I'd be least sure about and would ablate first.

<!-- rapid-fire -->

## Q: What is autoregressive video generation? {diff=1 tags=autoregressive,causal-video}
### A
**Generating a video in temporal order, each new frame or chunk conditioned only on what has already been generated**: $p(x) = \prod_k p(x_k \mid x_{<k})$. Contrast with bidirectional diffusion, which denoises the whole clip jointly and cannot emit frame 1 before frame 80 is done. It exists because it enables streaming, variable length, and interactive conditioning — you can inject an action or a prompt change at every step, which is what a world model needs. Each AR step can itself be a diffusion process (CausVid, Self-Forcing, MAGI-1), so "autoregressive" does not imply discrete tokens. The trap: errors compound over steps (exposure bias), so quality drifts after a few hundred frames unless you train on your own rollouts.

## Q: What is a KV cache in the video-diffusion setting? {diff=1 tags=kv-cache,causal-video}
### A
The **stored key and value projections, at every layer, of the already-generated context tokens**, so the next chunk can attend to its history without recomputing it — the LLM idea applied to video latents. Two differences from text: each latent frame is ~1500 tokens, so the cache grows by gigabytes per second of video, and the cache is filled from **clean** frames, i.e. after a chunk's denoising finishes, not from noisy intermediates. Trap: the cache is built at a single noise level ($t = 0$), so Self-Forcing runs an extra forward pass on the finished clean chunk to populate it — and cache length, not model size, is what bounds rollout length in memory.

## Q: Diffusion forcing in one sentence. {diff=1 tags=diffusion-forcing,training}
### A
**Train a causal sequence model where every frame gets its own independent noise level**, so the model learns to denoise frame $k$ given an arbitrarily noisy history — which makes teacher-forced next-frame prediction (clean past, fully noisy current) and full-sequence diffusion (one shared $t$) two corners of the same training distribution. It exists to make autoregressive rollout robust: because the model has seen noisy pasts in training, feeding it its own imperfect frames at test time is in-distribution. It also lets you choose any noise schedule along the time axis at sampling time (a staircase where far frames are noisier). The follow-up: it needs causal attention, otherwise a frame's denoising leaks information from independently noised future frames.

## Q: What is exposure bias? {diff=1 tags=exposure-bias,error-accumulation}
### A
**The train/test mismatch in autoregressive models: training conditions on ground-truth history (teacher forcing), inference conditions on the model's own outputs.** Small errors in generated frame $k$ enter the conditioning for $k+1$, get copied and amplified, and after enough steps the video saturates, blurs, or freezes. It hurts video more than text because the conditioning is continuous and high-dimensional — a slight color shift compounds, whereas a discrete token error is either corrected or not. Remedies: noise-augment the history in training (diffusion forcing), or **train on your own rollouts** (Self-Forcing, with a DMD loss against a bidirectional teacher) so the training distribution equals the test distribution. Trap: scheduled sampling is the cheap approximation, but it biases toward the mode.

## Q: What is sliding-window attention? {diff=1 tags=attention,streaming,kv-cache}
### A
**Each query attends only to the most recent $W$ keys, so compute per token is $O(W)$ instead of $O(N)$ and the KV cache is bounded at $W$ entries** — the simplest way to stream indefinitely with constant memory. Stacked over $L$ layers the receptive field is $LW$, so information can still propagate further than the window. Why it is not enough on its own: the moment the very first tokens leave the window, attention distributions collapse (the sink problem), so practical systems keep a few permanent tokens plus the window, and use cache-relative positions so RoPE never sees positions beyond $W$. Trap: exact long-range recall is gone — anything older than $W$ is forgotten unless you add memory tokens.

## Q: What is an attention sink? {diff=1 tags=attention-sink,streaming}
### A
**A token — usually the first one or two in the sequence — that receives a large share of attention mass regardless of content, acting as a "no-op" target the softmax can dump probability on when nothing is relevant.** It exists because softmax must sum to one: with no explicit null option, the model learns to use an always-visible, low-information token as the null. Consequence (StreamingLLM): naive sliding-window eviction removes the sink, every head's attention becomes mis-normalized, and perplexity explodes. Fix: keep the first ~4 tokens permanently alongside the window. For video the sink is typically the first chunk and the text tokens, and keeping the first chunk doubles as an identity anchor.

## Q: Ring attention in one sentence. {diff=2 tags=ring-attention,sequence-parallel,long-context}
### A
**Shard a long sequence across $P$ devices, and compute exact full attention by passing each device's K/V blocks around a ring while overlapping the communication with the local attention compute** — each device ends up having seen every K/V block. It exists because a minute-long video is millions of tokens, which does not fit one GPU's activation memory; ring attention makes context length scale linearly with device count with **no approximation**. Cost: per layer, each device sends and receives roughly $2Nd$ bytes of K/V, so it only pays off when compute per block exceeds transfer time. Trap: it is a training/prefill tool; it does not help streaming inference, where the bottleneck is cache length.

## Q: What is RoPE extrapolation, and why does length generalization fail? {diff=2 tags=rope,length-generalization,positional-encoding}
### A
**RoPE encodes position by rotating each 2-D pair of $q/k$ dimensions by angle $p\theta_i$, $\theta_i = \text{base}^{-2i/d}$; extrapolation is asking the model to attend at positions $p$ larger than anything seen in training.** It fails because the low-frequency dimensions rotate only a fraction of a period over the training length, so the model has only seen a narrow arc of those angles; at 4× the training length, those dimensions land on unseen angles, the dot products become out-of-distribution, and you get repetition, frozen motion, or noise. Fixes: position interpolation, NTK-aware base scaling, YaRN. Trap: the same effect appears on the spatial axes when you generate at super-native resolution, which is why SPEED spends most steps at native resolution.
