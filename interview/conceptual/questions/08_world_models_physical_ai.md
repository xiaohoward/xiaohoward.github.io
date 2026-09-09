# World Models, Physical AI & Robotics
<!-- weight: 4 -->

## Q: What is a world model? Contrast Ha & Schmidhuber's original, Dreamer's latent dynamics, and video-prediction world models like Cosmos, Genie, and GAIA. {diff=1 tags=world-models,dreamer,video-prediction}
- Follow-up: Which of these can you plan in, and which can you only sample from?
### A
A **world model** is a learned model of the environment's dynamics: $p(o_{t+1} \mid o_{\le t}, a_t)$ — given history and an action, predict what happens next — used either to **plan/learn a policy in imagination** or as a **simulator / data engine**.

- **Ha & Schmidhuber (2018)**: VAE compresses frames to $z$; an MDN-RNN predicts $z_{t+1}$ from $(z_t, a_t, h_t)$; a tiny linear controller is trained *inside the dream* with evolution strategies. Established the V-M-C decomposition and "train in imagination."
- **Dreamer (v1–v3)**: RSSM latent dynamics with a deterministic GRU state $h_t$ plus a stochastic $z_t$ (categorical in v2/v3), trained with reconstruction + KL; the policy and value are trained by **backprop through imagined latent rollouts** (actor-critic in latent space). Never decodes pixels at policy-training time; compact latents make long imagination cheap. Dreamer v3 solved Minecraft diamonds with fixed hyperparameters.
- **Video-prediction world models** (Cosmos, Genie, GAIA-1/2, Sora as "world simulator"): a large video diffusion / autoregressive transformer that predicts *pixels/latent-video* conditioned on actions (Genie: latent actions learned unsupervised from video; GAIA: ego-vehicle actions + text; Cosmos: diffusion & AR variants, camera/robot-action conditioned post-training). They're pretrained on internet-scale video so they carry visual and physical priors, and their output is human-viewable — useful for synthetic data, policy evaluation, and as a pretrained backbone. They're heavy (billions of params, seconds per rollout) so planning inside them is expensive.

Follow-up: Dreamer-style latent models are designed for planning (cheap rollouts, gradients through dynamics). Video-prediction models are mostly sampled from (data generation, evaluation) or used as a **frozen representation**; planning in them is emerging (MPC with a few candidate rollouts) but cost-limited. The trend is to close the gap: video-pretrained models with a compact action-aware latent for control.

## Q: Describe a world-action model that jointly predicts next video latents and actions. How does the action expert read the video context, and what's the difference between conditioning on imagined vs. ground-truth latents? {diff=2 tags=world-action-models,vla,world-models,project}
- Follow-up: Why not just predict actions from the current frame?
### A
A **world-action model** (lingbot-VA style, also Unified-Video-Action / WorldVLA) is one autoregressive transformer that at each step emits **both** the next observation latent $\hat{x}_{k+1}$ and the action chunk $a_k$, sharing a backbone. Two heads: a **video (dynamics) branch** — a diffusion or flow head that denoises the next video latent conditioned on the KV cache of past latents and actions — and an **action expert**, typically a smaller flow-matching head that **cross-attends (or attends via joint attention with its own token stream) to the video tokens** in the shared cache, including the just-imagined future latent.

Why joint prediction helps: the video objective forces the backbone to learn physics and object permanence from video alone (which is abundant), and the action head reads an explicit "what will happen" representation instead of inferring it from a single frame. It also gives you a **free evaluator**: imagined frames can be compared to what actually happened.

**Imagined vs. ground-truth latents.** In closed loop, the real camera gives you $x_{k+1}$ after acting, so you can insert the GT latent into the cache (**teacher-forced at test time**). Alternatively you condition the next step on the *imagined* latent. Tradeoffs:
- GT latents: accurate, no drift, but the action expert must handle the mismatch between the imagined latent it planned on and the GT it now sees; also requires a sensor at that frame rate.
- Imagined latents: needed when the world model runs ahead of the sensor (planning multiple steps, low-latency), but errors compound exactly like AR video.
- Hybrid (what I used): GT replaces imagined in the cache, and the **difference** $\|x - \hat{x}\|$ is a surprise signal — used both for diagnostics and for choosing which tokens to keep.

Follow-up: single-frame action prediction is a reactive policy; without a dynamics objective it never learns *why* an action works, is brittle to distribution shift, and gives no lookahead. The joint model's cost is inference latency (video head is expensive), which is why decoupled async or sparse video prediction is an active topic.

## Q: Explain the design of modern VLA models: OpenVLA vs. π0. What is action chunking and why does it help? Tokenized vs. continuous actions? {diff=2 tags=vla,pi0,flow-matching,action-chunking}
### A
**OpenVLA (2024)**: a 7B Llama-2 + fused DINOv2/SigLIP vision encoder, fine-tuned on Open X-Embodiment to emit **discretized actions as text tokens** (each of 7 DoF binned into 256 values, remapped to rarely-used vocab tokens). Simple, uses the LLM loss unchanged; but one action per forward pass, autoregressive over 7 tokens → slow (~5 Hz) and binning quantizes fine motion.

**π0 (Physical Intelligence, 2024)**: PaliGemma VLM backbone + a separate ~300M **action expert** connected via joint attention, trained with **flow matching** to output a **chunk of 50 continuous actions** in one denoising process (10 steps). Continuous, multimodal, 50 Hz-capable; the action expert is a separate set of weights so the VLM's language knowledge isn't overwritten by robot data.

**Action chunking** (ACT, Diffusion Policy, π0): predict the next $H$ actions (e.g. 16–50) and execute several before re-planning. Why it helps:
- **Temporal consistency**: humans demonstrate with pauses and idiosyncratic timing; per-step policies learn noisy, jittery actions and get stuck at "pause" states. Chunks commit to a coherent motion.
- Reduces the **effective horizon** of compounding errors by $H\times$ (fewer decisions).
- Handles **non-Markovian** demonstrations (the demonstrator's intent spans multiple steps).
- Amortizes inference cost of a big backbone. Downside: less reactive; fixed with temporal ensembling or receding-horizon re-planning.

**Tokenized vs. continuous**: tokens are plug-and-play with LLM infrastructure and cross-entropy handles multimodality natively, but quantize precision and need autoregressive decoding per dimension. Continuous with diffusion/flow gives high precision and multimodality via the generative head at the cost of iterative denoising. FAST (π0-FAST) tokenizes with a DCT + BPE over action chunks — a compression scheme very close in spirit to my spectral work — to get the best of both: ~5× shorter token sequences with continuous precision.

## Q: Imitation learning vs. RL for manipulation: when would you use behavior cloning, why does Diffusion Policy work better than a regression head, and how do you RL-fine-tune a diffusion policy? {diff=2 tags=imitation-learning,rl,diffusion-policy,manipulation}
### A
**BC** is the default when you have demonstrations and a stable environment: supervised, sample-efficient, safe. It fails on **covariate shift** (small errors lead to unseen states) and on **multimodal demonstrations**.

**Why Diffusion Policy beats regression.** Human demos are multimodal: go around the obstacle left or right; grasp from above or side. An MSE regression head predicts the **mean of the modes**, which is often an invalid action (straight into the obstacle). A diffusion / flow head models the **full conditional distribution** $p(a_{t:t+H} \mid o_t)$ and samples one coherent mode. Extra benefits: iterative denoising handles high-dimensional action chunks gracefully, training is stable (no GAN, no mixture-count choice as in MDNs), and classifier-free guidance / constraints can be applied at test time.

**RL after BC** addresses the covariate shift and lets the policy exceed the demonstrator. Options:
- **Advantage-weighted / filtered BC** (RWR, AWR, "ReinboT"): reweight demonstration or rollout samples by return — keeps the diffusion loss, cheapest and most stable.
- **Policy gradient through the denoising chain** (DPPO): treat the $K$ denoising steps as an MDP inside the environment MDP and run PPO on it; works but expensive.
- **Q-guided sampling / Q-score matching**: train a critic, then guide diffusion samples with $\nabla_a Q$ — decouples critic learning from the generative head.
- **Residual RL**: freeze the diffusion policy, learn a small residual action with RL.

What breaks: RL on real robots is data-starved and unsafe, so it's usually done in sim with a large domain-randomized wrapper, or as offline RL on logged data. The interviewer follow-up is "why not RL from scratch?" — sparse rewards and exploration in 7-DoF contact-rich tasks are hopeless without a demonstration prior; BC gives the prior, RL polishes.

## Q: Where is the sim-to-real gap for a vision-based manipulation policy — perception or dynamics? What do domain randomization, system ID, and real-to-sim digital twins each address? {diff=2 tags=sim2real,domain-randomization,digital-twin}
### A
Direct answer: for **vision-based manipulation** the gap is dominated by **perception** (rendering, lighting, textures, camera model), for **contact-rich or dynamic** tasks (insertion, in-hand, locomotion) it's dominated by **dynamics** (friction, compliance, actuator delay, contact solver artifacts). You need to know which one you're fighting.

- **Domain randomization** attacks both by making the policy invariant: randomize textures, lighting, camera pose, distractors (visual DR) and mass, friction, motor gains, latency (dynamics DR). It trades peak in-sim performance for robustness. Overdoing it produces a conservative policy; the modern practice is **automatic DR** (widen ranges as the policy succeeds) and **structured randomization** informed by real measurements.
- **System identification** narrows the dynamics gap: fit friction, damping, inertia, actuator model to real trajectories so the simulator is *centered* on reality, then randomize around it. Essential for locomotion and dexterous hands; less useful for vision.
- **Real-to-sim digital twins**: reconstruct the real scene (Gaussian splatting / mesh + physical params) so the simulator matches your specific lab, objects, and camera. Closes the visual gap by construction and lets you evaluate in a faithful replica; the risk is over-fitting to one twin. NVIDIA's "real-to-sim-to-real" pipelines (Isaac Lab + neural reconstruction) are exactly this.

Practical knobs I'd add: use **depth or segmentation as the policy input** to sidestep appearance; **pretrained visual encoders** (DINO/SigLIP) that already generalize; and **matching the sensor pipeline** — for my foveated-sensing project the sim rendered the same ROI-crop readout the 200 MP sensor performs, so the policy's *observation interface* was identical in sim and real, which mattered more than photorealism.

## Q: MuJoCo vs. Isaac Sim / Isaac Lab: how do the contact models and parallelism differ, and what did you actually do in MuJoCo for foveated perception with manipulation? {diff=2 tags=simulators,mujoco,isaac,project}
### A
**MuJoCo**: CPU, soft-contact model solved as a convex optimization (contacts are slightly compliant by design, tunable via solref/solimp), very stable and smooth gradients → the standard for RL research and system-ID; recent **MJX** runs it on GPU/TPU via JAX for thousands of parallel envs. Rendering is basic (OpenGL/OSMesa), no ray tracing.

**Isaac Sim / Lab**: PhysX GPU-rigid-body with rigid contacts (TGS solver), designed for **massively parallel** RL (tens of thousands of envs on one GPU), plus **photorealistic RTX rendering**, USD scene format, sensor simulation (cameras, lidar), and ready-made robot assets. Contact behavior can be more jittery and needs solver-parameter tuning; but for vision-based sim-to-real the rendering and asset ecosystem is the decisive advantage.

Rule of thumb: MuJoCo for algorithms and contact-accuracy research; Isaac Lab for scale and vision realism; both are fine for tabletop manipulation.

**What I did.** I built the foveated perception benchmark in MuJoCo: a manipulation task (pick-and-place style with a 7-DoF arm) observed by a simulated ultra-high-resolution camera. The full-res image is never given to the task policy; instead a **sensing policy** picks a small set of high-resolution ROIs (plus a low-res global view) each step, mimicking the ROI readout of the real 200 MP sensor. The manipulation policy then operates on that foveated observation. The sim let me (a) sweep the bandwidth budget (full res down to < 1/8 pixels) and measure task success, (b) render exactly the sensor's readout mode for sim-to-real transfer, and (c) get ground truth for tracking targets so I could also train and evaluate the sensing policy on object tracking and text recognition. The main sim gotcha was headless rendering (OSMesa/EGL) and keeping the high-res render cost from dominating RL throughput — I rendered at full res only for the selected ROIs.

## Q: Formulate foveated / active sensing as a policy. Why RL rather than supervised, how do you design the reward, and how do you evaluate on tracking, OCR, and manipulation? {diff=3 tags=active-perception,foveation,rl,project}
### A
**Formulation.** The sensor is a POMDP: state = full scene; observation = what you chose to read out ($K$ high-res ROIs + a low-res global view) within a pixel budget $B$; action = where to place the ROIs next (continuous centers/sizes or a discrete grid); reward = downstream task performance. The sensing policy $\pi_s(\text{ROIs}_{t+1} \mid \text{history of foveated observations})$ has to learn **saccades** (jump to a new salient region) and **pursuit** (track a moving target) as emergent behaviors.

**Why RL (or at least sequential decision learning) rather than supervised.** Supervised needs a label for "where should I look"; you can derive proxies (saliency, target position) but they don't capture the *sequential* value of looking — a glance now saves bandwidth later, and the task cost is non-differentiable through the crop operation. RL optimizes the actual objective under the budget. That said, I'd say honestly: with differentiable relaxations (soft attention over a grid, Gumbel-softmax over ROI choices) a **supervised / end-to-end** variant works for tasks with dense proxies, is far more sample-efficient, and I used it as a strong baseline; RL wins when the task reward is sparse (grasp success) or the horizon is long.

**Reward design.** Task reward (tracking IoU, OCR accuracy, manipulation success/shaped distance) minus a **bandwidth cost** $\lambda \cdot (\text{pixels read} / \text{budget})$, with the trade-off swept via $\lambda$ to draw the accuracy–bandwidth curve. Add a small penalty on ROI jitter to avoid oscillation. The trap: if the global low-res view alone solves the task, the policy learns to ignore ROIs — so the tasks were chosen to *need* high-res detail (small text, small objects, fine grasp alignment).

**Evaluation.** Report **task metric vs. pixel bandwidth** curves against baselines: full-res, uniform downsample, random ROIs, saliency-based ROIs, and a center-fixed fovea. Tracking: success/precision vs. budget on moving targets; OCR: character accuracy on text at varying scale; manipulation: success rate over ≥ 10 seeds × tasks with paired seeds across sensing policies. The headline: task performance held with < 1/8 of full-res bandwidth, and the real 200 MP prototype reproduced the sim trend.

## Q: Video generation models as data engines for robotics: how would you generate synthetic training data with them, and what are the risks? {diff=2 tags=synthetic-data,video-generation,robotics}
### A
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

## Q: LIBERO-style benchmarks: why do people run 10 seeds × N tasks, how do you get a confidence interval on success rate, and when does sim eval mislead you about real performance? {diff=1 tags=benchmarks,evaluation,libero}
### A
**Success rate is a Bernoulli estimate**, so its standard error is $\sqrt{p(1-p)/n}$. With $10\ \text{tasks} \times 50\ \text{episodes} = 500$ rollouts at $p = 0.8$, SE $\approx 1.8\%$, so two methods 3 points apart are barely distinguishable; with 10 episodes per task the SE is ~13% per task. **Seeds** matter because both the initial layout (object poses) and the policy's stochasticity vary; multiple seeds average over the layout distribution and let you report a CI across seeds (t-interval over per-seed means) rather than pooling episodes as if independent. Using **the same seeds for every method** (paired) removes layout variance from the comparison. Also report per-task results — averages hide that a method fixed one task and broke another.

Sim eval misleads when:
- The sim's **success predicate** is loose (object "in bowl" tolerance) or exploitable.
- The policy overfits sim rendering — LIBERO's fixed textures/lighting make vision trivially easy.
- The benchmark's initial-state distribution is narrow; a policy that memorizes 50 layouts looks great.
- **Failure modes differ**: in sim the robot stalls harmlessly; on real hardware the same policy collides.
- Timing: sim runs open-loop at fixed control rate; real inference latency changes the closed loop.

So: sim numbers for ablations and ranking, with paired seeds and CIs; real-robot trials (even 20–50) for the headline claim.

## Q: How would you measure and improve the physical consistency of a video world model? {diff=3 tags=physics,world-models,evaluation,latent-actions}
### A
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

## Q: Planning with a world model: how does MPC in imagination work, and how do you choose the rollout horizon when the model's errors compound? {diff=2 tags=planning,mpc,world-models}
### A
**MPC in imagination**: at each control step, sample $N$ candidate action sequences of length $H$, roll them through the world model from the current latent, score them with a learned reward/value (or a goal-image distance), pick the best (CEM / MPPI refinement over a few iterations), execute the first action(s), re-plan. PlaNet and TD-MPC2 are the canonical latent-space versions; Dreamer instead amortizes planning into a policy trained by imagined rollouts, which is cheaper at test time.

**Horizon choice.** The imagined return has two error sources: **model error** growing with horizon (compounding) and **value-bootstrap error** at the truncation point. The trade-off is like an n-step return: short $H$ leans on the value function; long $H$ leans on the model. Practical rule: $H \approx$ the horizon over which the model's rollout error stays below the reward's sensitivity — measure it directly by rolling the model on held-out trajectories and plotting prediction error vs. step; pick $H$ where the error curve knees. Typical $H$ is 5–15 latent steps in Dreamer/TD-MPC. Mitigations: **ensembles** to penalize disagreement (MBPO-style pessimism), **short rollouts branched from real states** rather than long ones from the start, and **receding horizon** so only the first action is ever executed.

With a **video world model** the same logic holds but rollouts cost seconds, so you use very few candidates ($N \approx 4\text{–}16$), coarse actions (subgoals), and a VLM/critic scorer; the latency makes it a high-level planner, with a fast reactive policy underneath. The follow-up: "why not backprop through the dynamics instead of sampling?" — you can (Dreamer's actor loss), but gradients through long imagined horizons are chaotic; sampling-based MPC is robust to a non-smooth model.

## Q: Using VLMs as the perception backbone of a robot policy — freeze or fine-tune? What does each choice break? {diff=2 tags=vlm,foundation-models,robot-learning}
### A
**Freeze** (train only the action head / adapter): preserves the VLM's open-vocabulary grounding and language understanding, avoids catastrophic forgetting from small, narrow robot datasets, cheap to train, and robust to distribution shift in *semantics*. Breaks: the frozen features are tuned for captioning/VQA, not for **precise spatial and metric information** (where exactly is the handle, how far); resolution is often low (224–448) and fine manipulation suffers; no adaptation to egocentric/wrist cameras.

**Full fine-tune**: best in-distribution task performance and spatial precision. Breaks: forgets language and generalization (OpenVLA's authors found fine-tuning the vision encoder was necessary for performance yet it hurt zero-shot semantics); needs large robot datasets (OXE-scale) or it overfits; expensive.

**The middle ground most systems use**: LoRA or partial fine-tuning of the VLM, a **separate action expert** (π0) so robot data flows through its own weights while joint attention reads the VLM features, **co-training** with VLM data (captioning/VQA) mixed into the robot batches to anchor the language ability, and adding **higher-resolution or multi-scale visual tokens** for precision (which is where foveated / mixed-resolution tokenization directly applies — spend tokens where the task needs detail).

Rapid follow-up: "what does the VLM give you that a DINO encoder doesn't?" — language grounding for instruction following and object generalization; for a single fixed task, a DINO + small transformer policy is often equal and 10× cheaper.

## Q: Quick take on humanoid loco-manipulation: how are today's systems built, and where does the data come from? {diff=1 tags=humanoid,locomotion,whole-body-control}
### A
Today's stack is usually **two-tier**:
- **Low-level whole-body controller** trained with **RL in simulation** (Isaac Lab, thousands of parallel envs, heavy domain randomization + system ID for actuators), taking proprioception and a command (target root velocity, upper-body joint targets or end-effector poses) and outputting joint targets at ~50 Hz through a PD loop. It handles balance, stepping, and disturbance rejection. Motion priors from human mocap (AMP / adversarial motion priors, or tracking retargeted motions) make the gait natural.
- **High-level policy** — a VLA or a diffusion policy on vision — outputs the commands to the low-level controller. Manipulation-heavy work often trains the high-level policy by **BC on teleoperation data** collected with VR / mocap suits / exoskeletons that retarget human motion to the humanoid (GR00T, the Unitree/Tesla-style setups).

**Where the data comes from**: (1) sim for locomotion (essentially unlimited), (2) **teleop** for manipulation (expensive, ~hours to hundreds of hours), (3) **human video** — egocentric datasets and retargeted human motion as a pretraining prior, with the embodiment gap bridged by latent actions or retargeting, (4) increasingly synthetic video from world models (GR00T-Dreams) for augmentation.

Key difficulties to mention: the whole-body controller must remain stable while the upper body does arbitrary things (coupled dynamics), latency between tiers, and the sim-to-real gap in contact with the environment during manipulation (not just feet). NVIDIA's positioning — Isaac Lab + GR00T + Cosmos — is exactly this stack.

## Q: Concrete pipeline: you have a video generation model and a mediocre manipulation policy with 200 demos. How would you use the video model to improve the policy, and how would you prove it worked? {diff=3 tags=design,video-generation,robot-learning,synthetic-data}
### A
I'd go in three stages of increasing risk, each with a measurable gate.

**Stage 1 — Visual augmentation (low risk, actions untouched).** Fine-tune a controllable video model (depth/segmentation-conditioned, Cosmos-Transfer style) on the 200 demos, then re-render each demo 10–50× with varied lighting, textures, backgrounds, distractors, and camera jitter while keeping the exact action trajectory. Train the policy on real + augmented. Gate: success rate on a **perturbed-appearance** eval set (new lighting/objects) with paired seeds; expect the biggest win here for a vision-brittle policy.

**Stage 2 — New trajectories with pseudo-labeled actions (medium risk).** Fine-tune the video model as an image+instruction → video predictor on the demos (plus any related web/robot video). Generate rollouts from *new* initial states and instructions. Label actions with an **inverse dynamics model** trained on the 200 demos + sim play data (the IDM is the crux; validate its accuracy on held-out real demos first). Filter generations with a physics/plausibility critic (VLM + object tracker: object moves only when in contact, gripper closes on the object). Train on filtered data. Gate: success on new initial-state distributions, plus ablate "no filter" to show the filter matters.

**Stage 3 — World model in the loop (highest risk, highest ceiling).** Action-condition the video model (or use the world-action-model form) and use it as an evaluator: run the current policy in imagination over many initial states, score with a VLM/success detector, and do **filtered BC / advantage-weighted fine-tuning** on the imagined successes, or short-horizon RL. Gate: correlation between imagined success rate and real success rate across policy checkpoints must be high before I trust it for training — if the world model can't rank policies it can't improve them.

**Proof.** Real robot: ≥ 3 conditions (in-distribution, new appearance, new layouts) × ≥ 30 trials, same seeds/layouts across policies, blind evaluator; report the whole curve of success vs. amount of synthetic data. Ablations: real-only, sim-DR-only (the cheap competitor everyone will ask about), each stage cumulatively. And report failure analysis — if the policy learned a physically impossible strategy from hallucinated video, that's the risk I'd flag up front and the reason the filter and the Stage-3 correlation gate exist.

<!-- rapid-fire -->

## Q: What is a world model, in one sentence? {diff=1 tags=world-models}
### A
**A learned model of environment dynamics, $p(o_{t+1} \mid o_{\le t}, a_t)$, that predicts what happens next given history and an action** — so an agent can plan, learn, or be evaluated in imagination instead of in the real world. It exists because real interaction is slow, expensive, and unsafe, whereas a model can roll out thousands of futures in parallel. Two flavors: compact latent dynamics (Dreamer, TD-MPC — cheap to plan in) and pixel/latent-video predictors (Cosmos, Genie — expensive but human-viewable, pretrained on internet video). The trap: a video model without action conditioning is a video generator, not a world model — the test is whether varying the action changes the predicted outcome in the right direction.

## Q: What is a VLA? {diff=1 tags=vla,robot-learning}
### A
**A vision-language-action model: a vision-language model fine-tuned to output robot actions, so a single network maps camera images plus a language instruction directly to motor commands.** It exists to reuse the VLM's open-vocabulary grounding and web-scale visual knowledge for robots, where task-specific data is tiny. Two design axes: how actions are emitted — as discretized text tokens (OpenVLA, RT-2) or as continuous chunks from a separate diffusion/flow "action expert" (π0) — and how much of the VLM is fine-tuned versus frozen. Trap: a VLA is still imitation learning — it inherits covariate shift from behavior cloning and only sees what the teleop data covered; the language ability is what makes it generalize across objects, not across skills.

## Q: What is action chunking? {diff=1 tags=action-chunking,imitation-learning}
### A
**Predicting a sequence of the next $H$ actions (e.g. 16–50 steps) in one forward pass and executing several before re-planning, instead of one action per step.** It exists because human demonstrations are non-Markovian and noisy in timing: a per-step policy learns jittery actions and gets stuck at pause states, whereas a chunk commits to a coherent motion. It also cuts the number of decisions by $H$, which shrinks compounding error, and amortizes the cost of a big backbone. The trade-off is reactivity — the robot is open-loop for $H$ steps — mitigated by receding-horizon execution (run only the first few actions) or temporal ensembling of overlapping chunks.

## Q: What is behavior cloning? {diff=1 tags=behavior-cloning,imitation-learning}
### A
**Supervised learning of a policy from demonstrations: fit $\pi(a \mid o)$ to expert $(o, a)$ pairs with a regression or likelihood loss.** It is the default for manipulation because it is sample-efficient, stable, and safe to train. Its two failure modes are the whole story: **covariate shift** — small errors put the robot in states the demonstrator never visited, where the policy has no idea what to do, so errors compound (DAgger fixes this by querying the expert in those states); and **multimodality** — an MSE head averages the demonstrator's modes into an invalid action, which is why diffusion policies replaced regression heads. Trap: BC cannot exceed the demonstrator; RL fine-tuning or filtered BC is needed for that.

## Q: What is the sim-to-real gap? {diff=1 tags=sim2real}
### A
**The drop in performance when a policy trained in simulation is deployed on real hardware, caused by every way the simulator differs from reality.** It splits into a **perception gap** (rendering, lighting, textures, camera and sensor noise) and a **dynamics gap** (friction, compliance, actuator delay, contact modeling). Which one dominates depends on the task: vision-based tabletop manipulation is mostly perception; locomotion and insertion are mostly dynamics. Tools: domain randomization (make the policy invariant), system identification (center the sim on measured reality), and real-to-sim digital twins (reconstruct the actual scene). Trap: matching the observation interface matters more than photorealism — in my foveated-sensing work the sim reproduced the sensor's ROI readout exactly, and that transferred.

## Q: What is domain randomization? {diff=1 tags=domain-randomization,sim2real}
### A
**Training in simulation while randomizing the parameters you cannot match to reality — textures, lighting, camera pose, distractors, mass, friction, motor gains, latency — so the policy learns to be invariant to them and the real world looks like one more sample.** It exists because closing the sim-to-real gap by making the simulator perfect is impossible; widening the training distribution is cheap. This can also reduce overfitting. Visual DR attacks the perception gap, dynamics DR the dynamics gap. Trap: too much randomization produces a conservative, low-performance policy that hedges against everything; the fix is automatic DR (widen ranges as the policy succeeds) and system ID so you randomize around the true values rather than across a huge box.

## Q: What is a diffusion policy? {diff=1 tags=diffusion-policy,imitation-learning}
### A
**A behavior-cloning policy whose action head is a conditional diffusion (or flow-matching) model: it samples an action chunk $a_{t:t+H}$ by iteratively denoising from noise, conditioned on the observation.** It exists because demonstrations are multimodal — go left or right around an obstacle — and a regression head predicts the invalid mean; a generative head models the full distribution and samples one coherent mode. Compared to mixture density networks or GANs it is stable to train and needs no choice of mixture count. Cost: 10–100 denoising steps per decision, so it pairs with action chunking. Trap: it is still supervised on demos, so covariate shift is unchanged; RL fine-tuning of diffusion policies (DPPO, Q-guided sampling) is the follow-up.

## Q: What is a POMDP? {diff=2 tags=pomdp,rl,active-perception}
### A
**A partially observable Markov decision process: an MDP (states, actions, transitions, rewards) where the agent does not see the state, only an observation drawn from $p(o \mid s)$.** The optimal policy must therefore act on a belief — a distribution over states given the history — rather than on the current observation, which is why memory is needed. Active perception makes the observation itself part of the action — my foveated-sensing policy chooses where to look, so the reward includes the information value of the next observation. Trap: an observation-only reactive policy can still work if the task is effectively Markov in the observation, which is why frame stacking is the first thing to try.
