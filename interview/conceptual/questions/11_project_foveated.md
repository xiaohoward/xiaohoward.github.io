# Project Grill: Foveated Imaging & Foveated Diffusion
<!-- weight: 4 -->

## Q: Pitch the SIGGRAPH foveated imaging paper in two minutes. What is the problem, what is the policy, what is the headline number? {diff=1 tags=foveated,pitch,active-perception}
- Follow-up: Why is this a graphics/imaging paper and not a robotics paper?
### A
**Problem: a 200-megapixel sensor produces far more pixels than any interface, ISP, or downstream network can consume at frame rate, yet most of those pixels are irrelevant to the task at hand.** Reading the whole sensor at full resolution is bandwidth-bound; uniform downsampling throws away the small, distant, or text-bearing detail that the task needs.

**Solution: policy-based foveation.** The sensor is read in two streams — a coarse global view and one or more high-resolution regions of interest (ROIs) — and a learned policy decides, every frame, where the ROIs go, conditioned on the coarse view, the previous ROIs and the task. The policy learns saccade-like behaviour (jump to a new target) and pursuit-like behaviour (track a moving one) without being told to. It is trained in simulation against task reward and transferred to a real 200 MP sensor prototype with a readout that supports programmable ROIs.

Headline: **task performance matched to full-resolution readout while consuming < 1/8 of the pixel bandwidth**, across three tasks: object tracking, text recognition (OCR) and MuJoCo robotic manipulation, and demonstrated on hardware.

Why imaging/graphics rather than robotics: the contribution is a **sensing** primitive — closing the loop between a task and where the sensor spends its bandwidth — analogous to foveated rendering in VR (spend shading where the eye looks), just on the capture side. The robotics tasks are downstream consumers that make the benefit measurable.

## Q: Formulate it as a sequential decision problem. State, action, reward, horizon — and why is it sequential at all rather than a per-frame saliency map? {diff=2 tags=foveated,rl,formulation,pomdp}
- Follow-up: Is this a POMDP? What is the hidden state?
### A
It is a **POMDP**: the true scene at full resolution is the hidden state; the agent only observes what it chose to read.

- **Observation:** the coarse global frame (e.g. full sensor downsampled by a large factor), the previously read high-res ROI crops, and a recurrent/temporal memory of previous observations [fill in: exact memory — recurrent state, stack of past crops, or feature buffer]. Optionally the task context (target description, OCR query, robot proprioception).
- **Action:** the ROI parameters for the next frame — centre $(x, y)$ and, if supported, scale / number of ROIs — under a hard bandwidth budget (ROI pixels + global pixels $\le B$ per frame). Continuous in the simulator; quantized to the sensor's readout grid on hardware.
- **Reward:** downstream task metric — tracking IoU / centre error, OCR character accuracy, manipulation success or dense distance-to-goal — possibly minus a small penalty for large ROI jumps (readout latency) [fill in: exact shaping terms].
- **Horizon:** episodic (a tracking clip, a manipulation episode); the policy acts every frame.

Why sequential and not a saliency map: (1) **the information needed to place the ROI is itself only available at high resolution** — a small distant object or a line of text is invisible in the coarse view, so the agent must *search* (saccade) then *hold* (pursue); that is a memory-dependent strategy, not a function of the current frame. (2) The reward is delayed: reading the wrong region now costs you the next several frames of tracking. (3) Bandwidth budget across time is a constraint a per-frame heuristic cannot reason about. Saliency is one of our baselines and it loses precisely on small targets and text.

## Q: How is the policy trained, and why that choice? I'll push: why RL instead of supervised learning from an oracle that knows the full-res frame? {diff=3 tags=foveated,rl,imitation,training}
- Follow-up: How did you deal with credit assignment for a reward that only arrives at the end of a manipulation episode?
### A
[fill in: the exact algorithm and architecture from the paper — e.g. PPO / SAC / DAgger-style imitation, the policy backbone, and training budget. Adapt the reasoning below to what you actually did.]

The argument I'd make for **RL with task reward, warm-started or regularized by an oracle where one exists**:

- **An oracle exists for some tasks, not others.** For tracking, the simulator knows the target's location, so "put the ROI on the target" is a perfect supervised label — behaviour cloning from that oracle is fast and stable. For OCR the oracle is "wherever the text is", also known in sim. For manipulation there is **no oracle for where to look** — the policy must discover that looking at the gripper–object contact, then at the goal, is what helps; only task reward defines that.
- **Imitating an oracle teaches you to look where the answer is, not how to *find* it.** The oracle never needs to search; a cloned policy therefore never learns the saccade behaviour needed when the target is lost or initially unknown, and it fails on exactly the frames where foveation matters. RL (or DAgger-style on-policy correction) exposes the policy to its own mistakes.
- **Reward shaping:** tracking uses dense per-frame IoU / negative distance; OCR uses per-step character accuracy of a frozen recognizer on the ROI; manipulation uses the downstream controller's success plus dense progress terms; all plus a bandwidth penalty if the budget is soft. Because the *perception* policy and the *control* policy can be decoupled (the controller consumes the foveated observation), I could keep the controller fixed and train only the sensing policy, which massively simplifies credit assignment.
- **Credit assignment for sparse rewards:** short effective horizon via discounting and dense proxy rewards (e.g. does the downstream perception module's estimate improve after this ROI), plus the pre-training from the oracle for tasks that have one.

Honest weaknesses to volunteer: RL is sample-hungry (fine in MuJoCo, not on hardware), and rewards from a frozen downstream model make the policy overfit that model's failure modes. That is the "why not X" I would expect, and the answer is: the oracle/imitation baseline *is* in the paper's comparison [fill in: numbers], and RL wins where search is required.

## Q: Describe the MuJoCo manipulation setup and, honestly, how much of the sim result actually transferred to the real 200 MP prototype. {diff=2 tags=foveated,mujoco,sim-to-real,hardware}
- Follow-up: What is the largest sim-to-real gap for a *sensing* policy, as opposed to a control policy?
### A
Simulation: a MuJoCo scene rendered at very high resolution (emulating the 200 MP sensor's field of view), with a robot arm doing [fill in: the task — e.g. pick-and-place / peg insertion] where the relevant object is small in the global view. The foveated observation (coarse global + ROI crops) is what the manipulation controller sees; the sensing policy chooses the ROI per step. The controller is [fill in: a fixed pre-trained visuomotor policy or a scripted/IK controller with a learned perception head]. Metric: task success at a given pixel-bandwidth budget, compared to full-res and to uniform downsampling at the same budget.

Sim-to-real: the hardware prototype is a 200-megapixel sensor with programmable ROI readout [fill in: sensor / interface details], and we ran tracking and text recognition on it live [fill in: which tasks ran on hardware; whether manipulation was hardware or sim-only]. What transferred well: the *behaviour* — search-then-lock, pursuit on moving targets, revisiting text — because the policy operates on coarse images whose statistics we can match with randomization (blur, noise, exposure, colour) and the ROI is a geometric action that is exact on hardware.

The largest gap for a sensing policy is **timing, not appearance**: on hardware the ROI you request is applied to the *next* frame after readout and transport latency, so the target you pointed at has moved. In sim the action is instantaneous unless you model that. We had to [fill in: model a one-frame action delay in sim / predict the target's next position]. The second gap is the readout constraint set (ROI grid alignment, max number of windows, fixed row bandwidth) which the simulator must enforce exactly or the policy learns unrealizable actions.

## Q: What are the baselines, and how exactly is "< 1/8 of the bandwidth" measured? Convince me the comparison isn't rigged. {diff=2 tags=foveated,baselines,evaluation,bandwidth}
- Follow-up: What happens to the curve as you push the budget to 1/32 or 1/64?
### A
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

## Q: On real hardware, the policy's action affects the *next* frame. What closed-loop and latency problems did you hit and how did you handle them? {diff=3 tags=foveated,latency,hardware,control}
- Follow-up: What is the maximum target speed you can track, and what sets it?
### A
The loop per frame is: read global + ROI → policy inference → send ROI coordinates → sensor applies them to the next exposure → readout. Three latency sources: exposure/readout of a 200 MP sensor (even at reduced bandwidth, the row scan is slow), transport, and policy inference. Together they mean the policy is always acting on an observation that is at least one frame old.

What this did to a policy trained with an instantaneous simulator: on fast targets it **lags and oscillates** — it places the ROI where the target *was*, loses it, saccades, re-acquires, and the effective tracking accuracy collapses. Fixes, in order of how much they helped [fill in: which you actually used]:
1. **Model the delay in simulation:** the action is applied $k$ frames later, with $k$ matched to the measured hardware pipeline. The policy then learns to lead moving targets (predictive pursuit), which is what biological smooth pursuit does.
2. **Lightweight policy** so inference is a small fraction of the frame time; the global view is small, so this is easy.
3. **Slightly larger ROI than the target** as a margin — a bandwidth/robustness trade-off that is exactly the fovea/parafovea structure.
4. Asynchronous pipeline: policy runs on frame $n$ while frame $n+1$ is exposing.

Maximum trackable speed is set by $(\text{ROI margin}) / (\text{loop latency})$: a target must not leave the ROI within one loop delay. Increasing the ROI helps linearly but eats budget; reducing latency helps everywhere. On the prototype the binding constraint was sensor readout time, not the network.

## Q: Connect this to biological foveation and to the active-perception literature. What did you borrow, and what is different from the classic "active vision" work? {diff=2 tags=foveated,biology,active-perception,related-work}
### A
**Borrowed from biology:** the retina samples with a high-acuity fovea and a low-acuity periphery, and the brain compensates with eye movements — **saccades** (fast ballistic jumps to a new target selected from the periphery) and **smooth pursuit** (keep a moving target on the fovea). Our two-stream readout is the same architecture: coarse global view = periphery, ROI = fovea; and the learned policy reproduced saccade/pursuit behaviour without those being engineered, which is a nice sanity check that the objective (task reward under a bandwidth budget) is the right one — the visual system is solving the same optimization with a fixed optic-nerve bandwidth.

**Active perception / active vision** (Bajcsy, Aloimonos, Ballard's "animate vision", later Mnih's recurrent attention model, hard-attention / glimpse networks): the idea that the perceiver chooses its observations to reduce task uncertainty. Recurrent attention models are the closest ML ancestor: a glimpse policy trained with REINFORCE on classification. Also foveated rendering in graphics, which is the *display-side* dual.

What is different:
- The constraint is a **real hardware readout budget** on a real ultra-high-res sensor, not a synthetic glimpse on a 28×28 image; actions must be realizable on the sensor's ROI grid, and latency matters.
- **Multiple downstream tasks with a fixed controller**, including closed-loop manipulation, rather than static classification.
- The bandwidth-accuracy trade-off is quantified as a curve, with an oracle upper bound.
- Sim-to-real for a *sensing* policy, which brings the delay-modelling issues that classic active-vision papers did not face.

The honest open question: biological vision also has a smooth eccentricity-dependent resolution falloff and multiple scales; ours is two-level. That is where the foveated diffusion work's multi-scale tokenization ideas come back in.

## Q: Foveated Diffusion: how do you actually put mixed-resolution tokens into a pretrained DiT? Patchify, position embeddings, attention cost — and why does a LoRA suffice? {diff=3 tags=foveated-diffusion,dit,tokens,rope,lora}
- Follow-up: Show me the cost model that gives 2× on images and 4× on video.
### A
Setup: FLUX.2 and WAN 2.1, unchanged architecture, LoRA post-training only.

**Patchify at two scales.** The latent is split into a foveal region and a periphery (a mask). Foveal latents are patchified at the native patch size $p$; peripheral latents at $2p$ (image) or a larger spatiotemporal patch (video), so a peripheral token covers 4× (image) or up to 8×+ (video) the area of a foveal token. The patch-embedding linear layer for the coarse scale is [fill in: a new small projection trained with the LoRA / the native projection applied after average-pooling the latent]. Tokens from both scales are concatenated into one sequence; the transformer is agnostic to their origin.

**Position embeddings.** FLUX and WAN use 2D/3D RoPE indexed by latent coordinates. Each token gets the RoPE of its **patch centre in the full-resolution coordinate frame** — a coarse token at rows 8–11 gets position 9.5. That keeps relative distances between foveal and peripheral tokens geometrically correct, which is what RoPE encodes; the model only needs to learn that some tokens are "blurrier", which is a small distribution shift — hence LoRA. [fill in: whether you also add a scale indicator.]

**Cost model.** Let $N$ be the full-res token count and $a$ the foveal area fraction, periphery at $1/s$ tokens per area. Tokens $N' = N\,(a + (1-a)/s)$. Attention $\propto N'^2$, MLP $\propto N'$. Image, $a = 0.25$, $s = 4$: $N' = 0.44N \to$ attention $\approx 0.19\times$, MLP $\approx 0.44\times$; with attention $\approx$ half the FLOPs at 4k tokens, total $\approx 0.3\times$, minus overhead $\approx$ **2× wall-clock**. Video, $s = 8$ spatiotemporal and attention dominating at ~30k tokens: $N' \approx 0.34N$, attention $\approx 0.12\times \to$ **~4×**. Attention share is why video gains more.

**Why LoRA is enough.** The pretrained model already generates the periphery content; what changes is the *input/output statistics* of coarse tokens (they look like a low-passed latent) and their unusual density in the sequence. That is a low-rank adaptation of the projections, learned in hours from the model's own data with a random fovea. The final image is assembled by upsampling the periphery's coarse latent (it is genuinely lower resolution there) and blending at the boundary before VAE decoding.

## Q: Where does the fovea come from at generation time, what does the periphery look like, and where does foveated diffusion fail? Then: how would you combine it with SPEED? {diff=3 tags=foveated-diffusion,speed,failure-modes,combination}
### A
**Fovea source.** Three practical choices: (1) **user gaze** — the VR/AR case, the periphery is literally not looked at, so its lower quality is invisible; (2) **saliency / subject mask** from the prompt or a first low-res pass — put the fovea on faces, text, the main object; (3) **text-driven** — the prompt (or a VLM) names what deserves detail. In the paper we [fill in: which of these you evaluated and how the fovea was sampled during training — random rectangles / random positions and sizes].

**Periphery quality.** It is a genuinely lower-resolution generation: coarse tokens mean the model can only represent frequencies up to the periphery's Nyquist. Upsampled to full res it looks like a mild blur, which is fine for backgrounds, sky, out-of-focus regions, and for gaze-contingent display; it is not fine for a second face in the periphery. Boundary handling matters: without care you get a visible seam of sharpness change; blending tokens across scales at the border and having the LoRA see mixed boundaries during training removes it.

**Failure cases.** Small important content outside the fovea (a second subject, text) is rendered coarsely; semantic consistency across the boundary can slip (an object crossing the border is sharp on one side); a wrong fovea choice is unrecoverable within the sample; strong periphery downsampling on video can produce temporally flickering coarse regions.

**Combining with SPEED.** They are complementary axes: SPEED is *temporal* allocation (resolution as a function of denoising time), foveated diffusion is *spatial* allocation (resolution as a function of location). The natural combination: run SPEED's early stages uniformly at low resolution — at high noise nobody has high frequencies anyway, so foveation buys nothing there — and only in the last stage(s), when high frequencies activate, switch to the mixed-resolution token layout with the fovea at native res. In DCT terms, the transition embeds the low-res state and fills the new band with $t \cdot \text{noise}$ **only inside the fovea**; the periphery simply never gets a high band. Speedups should roughly multiply (SPEED cuts early steps, foveation cuts late steps), which is the experiment I would run first. The wrinkle is that the LoRA has then to handle both the low-res regime and the mixed-res layout — a joint LoRA with both augmentations.

## Q: You are on Cosmos / Isaac at NVIDIA. What is the follow-up to these two papers, and what should the sensor and the world model share? {diff=2 tags=foveated,nvidia,cosmos,isaac,future}
- Follow-up: Which of the two papers is more useful to a robot, and why?
### A
The thread connecting both papers is **allocate resolution where and when it carries information** — sensing side and generation side. The follow-up I would propose:

1. **Foveated world models.** A video world model (Cosmos-style) for a robot does not need uniform resolution: the gripper–object contact region needs detail, the rest of the scene does not. Combine foveated diffusion tokens with an *action-aware fovea* (predicted from the policy's attention or from proprioception), and SPEED-style schedules for the denoising. That gives faster imagination rollouts, which matters for model-based planning where you roll out many futures.
2. **Learned sensing + world model in the loop.** The foveated-imaging policy decides where to read; the world model predicts what will be there. A world model with an uncertainty estimate is an excellent reward signal for the sensing policy: look where prediction is worst — this is exactly the surprise idea I have used for KV-cache compression, and it does not need an oracle.
3. **Isaac Sim / Lab** as the simulator: physically-based rendering at high res, programmable virtual sensors with realistic readout and latency models, and thousands of parallel environments for RL — solving the sample-efficiency problem I had in MuJoCo. The sim-to-real recipe (delay modelling, readout constraints) carries over; the appearance gap is smaller with a proper renderer.
4. **Synthetic data**: a foveated generator can produce high-detail training data for exactly the regions a perception model is weak on, at a fraction of the cost.

Which is more useful to a robot: the sensing policy, immediately — bandwidth and latency are the binding constraints on any embodied system with high-res cameras, and it plugs in front of an existing VLA. Foveated/spectral generation matters once the robot *uses* a world model in the loop, where imagination cost dominates. I would argue NVIDIA is in the rare position to own both the sensor readout path and the model, and that is exactly where the two papers meet.

<!-- rapid-fire -->
## Q: What is foveated imaging, in one sentence? {diff=1 tags=foveated,rapid-fire,active-perception}
### A
**Foveated imaging reads a small region of the sensor at full resolution and the rest at low resolution, and moves that region over time, like the human eye's fovea and saccades.** The why: a 200-megapixel sensor cannot be read out, transmitted, or processed at full rate, but most of those pixels are irrelevant for any given task at any given moment; a task-driven fovea gets the useful pixels for a fraction of the bandwidth. Follow-up you will get: "how is that different from cropping a region of interest?" — the difference is that we also keep a coarse global view, and the ROI is chosen by a learned policy that acts sequentially rather than by a fixed heuristic.

## Q: What is the "policy" in your paper? {diff=1 tags=foveated,rapid-fire,rl,policy}
### A
The **policy is a learned network that, at each timestep, looks at the current coarse global view plus the previous foveal crop and outputs where to place the next high-resolution region**. It is the decision-maker in a sequential problem: state is what has been observed so far, action is the fovea location (and size), reward is downstream task performance under a bandwidth budget. It is trained with RL / policy learning rather than a per-frame saliency map because good behavior is temporal — pursuit to track a moving object, saccades to re-check something uncertain — and depends on what the task needs next. Follow-up: it is separate from the task network, so it plugs in front of an existing model.

## Q: What does "bandwidth" mean here, and what was the number? {diff=1 tags=foveated,rapid-fire,bandwidth,evaluation}
### A
**Bandwidth is the number of pixels read off the sensor per frame** (equivalently bits per second from readout to compute), the resource that limits ultra-high-resolution sensors: readout, transmission, and inference all scale with it. Our foveated policy keeps task performance on manipulation, tracking, and text recognition using **less than 1/8 of the full-resolution pixel bandwidth**; the budget counts the coarse global view plus the high-res fovea, so the comparison is against reading the whole 200-megapixel frame. Trap: the honest baseline is not just "full-res vs ours" but also uniform downsampling and a fixed-center or saliency fovea at the same budget, which is what we show.

## Q: What simulator did you use, and for which task? {diff=1 tags=foveated,rapid-fire,mujoco,simulation}
### A
**MuJoCo**, for robotic manipulation: a simulated arm with a camera where the policy chooses which region to sense at high resolution while a downstream controller does the task, so fine detail (a small object, a grasp point) is only paid for where it matters. We also validated on object tracking and text recognition, and on a real 200-megapixel sensor prototype. Why MuJoCo: fast, deterministic physics with a controllable renderer, so we can generate large amounts of policy rollouts cheaply and control the bandwidth accounting exactly. Follow-up: sim-to-real — the sensing policy transferred better than a control policy would, because it only has to decide where to look, and the rendering gap is smaller than the dynamics gap.

## Q: What is foveated diffusion, in one sentence? {diff=1 tags=foveated-diffusion,rapid-fire,dit,efficiency}
### A
**Foveated diffusion generates an image or video with a pretrained diffusion transformer using full-resolution tokens only in a foveal region and coarse tokens in the periphery, cutting the quadratic attention cost.** Since attention cost scales with the square of token count, replacing most tokens with a few coarse ones gives a large saving; the model is adapted with a LoRA post-training on FLUX.2 and WAN 2.1, giving about 2× image and 4× video wall-clock speedup. Follow-up: "where does the fovea come from at generation time?" — from a user or task-specified region of interest, or from attention/saliency, and the periphery is upsampled and blended.

## Q: What is a mixed-resolution token? {diff=2 tags=foveated-diffusion,rapid-fire,tokens,patchify}
### A
A diffusion transformer patchifies the latent into fixed-size tokens; a **mixed-resolution token sequence uses small patches (many tokens) in the fovea and large patches (few tokens covering more area) in the periphery**, so one sequence contains tokens at different spatial scales. Two things must be handled: position embeddings, since a coarse token covers a region rather than a point, so its RoPE position is the region's center at a scale consistent with the fine tokens; and patchify/unpatchify at multiple scales, so the periphery is decoded at low resolution and upsampled. Weights are shared; a LoRA teaches the model to accept the mixed sequence. Trap: this is a token-count reduction, so the speedup is on attention, not on the VAE.
