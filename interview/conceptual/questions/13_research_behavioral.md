# Research Story & Behavioral
<!-- weight: 3 -->

## Q: Walk me through your research in about three minutes. {diff=1 tags=research-story,behavioral,pitch}
- Follow-up: what is the single thread connecting these projects?
### A
Structure it as **one thread, three chapters, one destination**. The thread: *spend sensing and compute only where the signal is.*

**Chapter 1 — Foveated sensing (SIGGRAPH 2026).** Cameras now have 200-megapixel sensors, but no downstream model can consume that bandwidth. We trained a **policy** that decides which regions of an ultra-high-res sensor to read at full resolution, like a saccading eye, for tasks such as MuJoCo manipulation, tracking, and text recognition. It keeps task performance with **under 1/8 of the pixel bandwidth**, and we validated it on a real 200 MP prototype. Lesson: adaptive allocation of a scarce resource, driven by the task.

**Chapter 2 — Efficient generation (SPEED + Foveated Diffusion, 2026).** Same idea applied to diffusion models. In **SPEED**, I measured that latents have a power-law spectrum $P(f) \propto f^{-\beta}$, so at high noise levels high frequencies are below the noise floor — the model literally cannot see them. So we run early denoising steps at low resolution and grow resolution in stages when each band's SNR crosses 1. Training-free or with a small LoRA; **up to 7× wall-clock speedup on FLUX and 2.5× on WAN 2.1**, quality preserved. **Foveated Diffusion** does the spatial analogue: full-res tokens in the fovea, coarse in the periphery, 2×/4× image/video speedups.

**Chapter 3 — World models for physical AI (ongoing).** Robots need world models that run long horizons cheaply. My current work compresses the KV cache of a world-action model by keeping only tokens that are *surprising* and *attended by the action head* — ~44–54% less KV with task success unchanged. Plus the CVPR 2025 oral on single-photon ToF, which is where my sensor-physics instincts come from.

**Destination:** efficient, sensor-aware generative world models that a robot can run in real time — which is why NVIDIA's Cosmos / Isaac stack is where I want to be. [fill in: one sentence on your intended thesis framing.]

Land within three minutes; the follow-up "what's the thread?" should already be answered.

## Q: Tell me about a result that didn't work, and what you learned. {diff=2 tags=behavioral,speed,debugging,research-story}
- Follow-up: how did you know it was the method and not a bug?
### A
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

## Q: What would you want to work on at NVIDIA, and why here rather than another lab? {diff=1 tags=behavioral,nvidia,motivation}
- Follow-up: which team, concretely?
### A
**Direct answer.** I want to work on **efficient world models for physical AI** — video/world foundation models that are fast enough and long-context enough to sit inside a robot's control loop — and NVIDIA is the one place where the model (Cosmos), the simulator (Isaac Sim / Isaac Lab), and the hardware co-design all live under one roof.

Three concrete threads I'd bring, matched to teams:
- **GenAI / Cosmos world foundation models.** My SPEED and foveated-diffusion work is about *where* a diffusion model spends compute — spectrally and spatially. World models are the regime where that matters most: long rollouts, high resolution, tight latency. I'd like to push resolution-progressive and token-adaptive generation into autoregressive/causal video, and into the tokenizer + diffusion-decoder stack.
- **Robotics / Isaac, world-action models.** My KV-cache compression for a world-action model (surprise-gated, ~50% less cache with unchanged task success) is a small step toward long-horizon rollouts on-device. Isaac's sim-to-real and GR00T-style VLA work is exactly where I'd test whether "predict only what surprises you" scales.
- **Diffusion fundamentals.** I care about the spectral / SNR view of flow matching — it explains resolution schedules, why super-native generation works when you stay in the native-resolution regime, and it hints at better noise schedules and solvers. That's basic research NVIDIA's GenAI group actually publishes.

**Why here.** (1) Scale: the questions I care about only get answered at 10B-parameter video models on hundreds of GPUs, and NVIDIA runs those. (2) Research-to-product path: my speedups are only interesting if they ship in an inference stack; NVIDIA turns papers into Cosmos releases and TensorRT paths. (3) The people: [fill in: 1–2 specific NVIDIA papers/people you've read — e.g., Cosmos WFM tech report, EDM/EDM2 lineage, the Isaac Lab / GR00T team].

Keep it honest: mention the second-choice lab only if asked, and then say why NVIDIA's combo is unique.

## Q: An experiment gives you an ambiguous result. How do you decide what to try next? {diff=2 tags=behavioral,methodology,experiments}
- Follow-up: give a concrete example from your work.
### A
**Principle: turn ambiguity into a factorial design before running more of the same.** Ambiguous usually means two variables changed at once or the noise floor is unknown.

My checklist, in order:
1. **Estimate the noise floor first.** Re-run the baseline with N seeds and report the spread. Half of "ambiguous" results are within seed variance. In the KV-compression work I used **paired seeds** — same seed for compressed and uncompressed rollouts — because LIBERO task success at 20–50 episodes has a wide confidence interval and pairing removes the shared variance.
2. **Isolate the axis.** Write down the ≤3 things that differ between the good and bad condition, then build the 2×2 (or 2×2×2) of controls. In SPEED the axes were {continuous vs staged masking} × {resolution reduction vs full-res low-pass} × {Euler vs UniPC}; running the grid told us quality came from staging and speed came from resolution — two separate findings.
3. **Look for a confound that flips the metric's meaning.** Example: in world-action rollouts, failure episodes had *higher* imagined-video PSNR because a stalled robot is trivial to predict. Split metrics by success/failure before believing any average.
4. **Check determinism.** Flash-attention kernels are non-deterministic across runs; I verify bitwise reproducibility of the baseline before attributing a 0.3 dB change to the method.
5. **Pick the experiment that would change my decision.** Not the one that's easiest. If neither outcome changes what I'd do next, skip it.
6. **Time-box.** If the ablation grid doesn't disambiguate in a day, step back and re-derive the expected effect size from the model — if theory says the effect should be tiny, the experiment isn't ambiguous, it's null.

Follow-up example: the per-step-masking blur (see the "result that didn't work" answer) was resolved entirely by step 2.

## Q: Describe a time you collaborated closely, or disagreed with a co-author, and how it resolved. {diff=1 tags=behavioral,collaboration}
- Follow-up: what would you do differently?
### A
Pick one real story and tell it in four beats: context → the disagreement → how you resolved it with evidence → what changed in you.

**Suggested story (SPEED / Foveated Diffusion are co-authored with the same group — Chao, Yariv, Wetzstein):** [fill in: the actual disagreement. Plausible shapes: (a) whether to ship SPEED as training-free only vs. add the LoRA variant — one side wanted a clean training-free story, the other wanted the best numbers; (b) staged vs continuous masking, where the "elegant" continuous version was someone's preference and the data said staged; (c) how to split credit/scope between SPEED and Foveated Diffusion, which share machinery.]

Model structure:
- **Context.** Two of us had different intuitions about [fill in]. Both were reasonable; neither had data.
- **Disagreement.** I made my position falsifiable: "if X, then we should see Y in control Z." I proposed the smallest experiment that would settle it — not a debate.
- **Resolution.** [fill in: who was right]. If I was wrong, say so plainly and name what convinced you — interviewers rate that higher than being right. If I was right, credit the co-author for the pressure that made the control exist.
- **What changed.** We adopted a habit: every design disagreement gets a control in the ablation table. It made the paper stronger because the reviewers' "why not X" was already answered.

**Collaboration texture to include:** shared codebase across FLUX / WAN / Qwen-Image ports meant agreeing on interfaces early (schedulers, DCT utilities); I owned [fill in] and reviewed [fill in]. Say what you did to make co-authors' lives easier (reproducible scripts, seed-pinned configs).

Follow-up: "differently" — raise the disagreement earlier and in writing, with the proposed experiment attached, rather than letting it sit in a meeting.

## Q: How do you read and keep up with papers, and what recent paper impressed you and why? {diff=2 tags=behavioral,literature,video-generation,vla,world-models}
- Follow-up: what's the weakness of the paper you just praised?
### A
**Process.** Daily arXiv skim by keyword (diffusion, flow matching, video, world model, VLA), a weekly deeper read of 2–3 papers where I reproduce the key equation on paper and note "what would break this." I keep a running notes file per topic and re-read it before starting a project. I also read *outside* the lane deliberately — neuroscience of foveation, signal processing — because that is where SPEED's spectral view came from.

**Three directions that impressed me (pick two, know them cold):**

- **Self-forcing / causal autoregressive video.** Training a causal video model by rolling out its *own* generations during training (rather than teacher-forcing on ground truth) closes the train-test gap that causes drift in long rollouts, and pairs with KV-cache-based streaming generation. Why it matters to me: it's the bridge from bidirectional video diffusion to a *world model* you can run interactively; and my KV-compression work is about what to keep in exactly that cache. Weakness: exposure to its own errors makes training expensive and sensitive to the rollout horizon; quality still lags bidirectional models.

- **VLAs with flow-matching action heads.** Predicting continuous action chunks with a flow/diffusion head on top of a VLM backbone (π0-style) instead of discretizing actions into tokens. Why it impressed me: it's the cleanest evidence that diffusion fundamentals transfer to control — multimodal action distributions, few-step sampling, and the same guidance tricks. Weakness: the vision backbone is still frozen-ish and low-res; sensing is not adaptive — which is where foveated perception should plug in.

- **World-action models** (jointly predicting future video latents and actions, one transformer). Why: it unifies the "imagine, then act" loop, and gives you a prediction error signal for free — the surprise signal I use for KV compression. Weakness: imagined video is expensive, and in practice failure cases have *easier* video (stalled robot), so the video loss can mislead the policy.

**The framing I'd give:** the field is converging on *causal, long-context, action-conditioned video models*, and the open problem is making them cheap enough to run in the loop — which is my research thread.

[fill in: exact titles/authors of the two you choose; be able to state each one's key equation or training objective in one line.]

## Q: What questions do you have for us? {diff=1 tags=behavioral,questions-for-interviewer}
### A
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

## Q: Tell me about your internship at Bell Canada, and what you learned about shipping software. {diff=1 tags=behavioral,bell,industry,shipping}
- Follow-up: how does that change how you do research code?
### A
**Setup.** Sixteen months as a software intern; I built internal **document-retrieval and code-generation tools** (Python, Ruby, SQL, LangChain) and **fine-tuned open-source LLMs** on internal data. Real users, real deadlines, a team that had to maintain it after I left.

**What I actually did** (be concrete): [fill in: the tool, who used it, one usage number or outcome]. The technical core was a RAG pipeline plus LoRA fine-tunes with an eval set I built.

**Three lessons about shipping:**
1. **The eval set is the product.** The first version of the retrieval tool "felt" good in demos and failed on real queries. Building a labeled query set and tracking recall@k turned arguments into numbers. I now build the metric before the method — the same discipline that made SPEED's ablation table exist.
2. **Boring reliability beats clever.** Hybrid BM25 + dense retrieval, explicit "not found" answers, and logging beat a fancier model. Users trust a tool that fails loudly over one that hallucinates quietly.
3. **Hand-off is part of the work.** Documentation, reproducible configs, tests, and a clean interface mattered more to the team than my last 5% of accuracy. I left with the tool still running.

**How it changed my research code.** Seed-pinned configs, one-command reproductions, and scripts a co-author can run on a different model family (the SPEED code was ported across FLUX, WAN, and Qwen-Image because of that). I also learned to time-box: a 16-month deadline horizon teaches you to pick the experiment that changes the decision.

Follow-up trap: don't oversell it as research. It was engineering, and the point is that you know the difference and can do both.

<!-- rapid-fire -->
## Q: Describe your research in one sentence. {diff=1 tags=behavioral,rapid-fire,pitch}
### A
**I make visual generation and visual sensing spend compute only where the signal is: spectral resolution schedules for diffusion models, learned foveation for sensors and diffusion transformers, and photon-level imaging that recovers depth from whatever light is available.** If they ask for the throughline: the same idea, that information in images is unevenly distributed over frequency, space, and time, applied to both the sensor and the generator, which is why my interests point at world models and physical AI, where the model and the camera sit in one loop. Keep it to one breath, then stop and let them pick which thread to pull.

## Q: What is the single most important result you have? {diff=1 tags=behavioral,rapid-fire,results}
### A
**SPEED: a training-free 7.09× wall-clock speedup on FLUX.1-dev, and 2.54× on WAN 2.1 video, with quality preserved, from a resolution schedule derived from the latent power spectrum $P(f) = A \cdot f^{-\beta}$ rather than tuned by hand.** Why I rank it first: it is principled (the schedule falls out of an $\mathrm{SNR} = 1$ criterion), it transfers across four model families without architecture changes, and the ablation that continuous frequency masking blurs while staged reveal is sharp taught us something about how diffusion models use steps. The follow-up I expect is "does this survive distillation?" and the honest answer is partially: for 4-step models the ceiling is about 4/3×, so the value is in the 20-50 step tier and in video.

## Q: What is your biggest weakness as a researcher? {diff=1 tags=behavioral,rapid-fire,weakness}
### A
**I tend to over-derive before I run the cheap experiment.** My math background pulls me toward getting the SNR-correction or the schedule exactly right on paper first; on SPEED I spent days on the transition scaling before a one-hour sweep would have shown which regime mattered. What I do about it now: I write the falsifying experiment first — the control that would kill the idea, such as the full-resolution box low-pass baseline that showed only real resolution reduction gives speedup — and run it before the derivation is polished. Related: I underestimate engineering time for video pipelines, so I now time-box and build the smallest end-to-end version first. Do not say "perfectionism"; give the incident and the fix.

## Q: Where do you see video generation and world models in three years? {diff=2 tags=behavioral,rapid-fire,world-models,video-generation,future}
### A
**Video generation becomes a component inside interactive world models rather than a standalone product: causal, streaming, action-conditioned, and evaluated on whether a robot or agent using it gets better, not on FVD.** Three concrete bets. First, the cost problem gets solved by spending compute unevenly, resolution schedules, foveation, and KV-cache compression that keeps only surprising tokens, since a world model in a control loop cannot afford 50 full-resolution steps per frame. Second, distillation plus long-context memory makes minute-scale consistent rollouts routine. Third, the data bottleneck shifts to physics correctness, so simulation and real-to-sim become the training signal. The open risk: we still lack an evaluation that measures physical plausibility, and that will decide who wins.
