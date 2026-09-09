# Context brief for interview-prep authors (read fully before writing)

## Candidate
Hanchen (Howard) Xiao — 2nd-year Stanford EE PhD (Stanford Physical and Spatial Intelligence Lab, advisor Gordon
Wetzstein). Anticipated graduation June 2029. UofT Honours BSc CS + Math. Stanford Graduate Fellowship.
Applying to NVIDIA 2027 PhD research internships: Generative AI, Computer Vision & Deep Learning, Robotics — and
similar PhD-intern roles elsewhere (research scientist intern style interviews: conceptual + coding + project grill).

Research interests (stated): image and video generation, fundamentals of diffusion / flow models, world models,
physical AI, robotics, long-context methods.

## CV projects (interviewers WILL grill these)
1. **SPEED — Spectral Progressive Diffusion for Efficient Image and Video Generation** (preprint 2026; Xiao, Chao,
   Yariv, Wetzstein). Progressively increases spatial resolution along the denoising trajectory using spectral
   (DCT) representations and power-spectrum-derived resolution schedules. Training-free inference OR LoRA fine-tune,
   no architecture change. Evaluated on latent image (FLUX.1-dev, Z-Image), pixel image (PixelGen), latent video
   (WAN 2.1). Up to 7.09x wall-clock speedup on FLUX, 2.54x on WAN 2.1, quality preserved.
   Mechanism details (real, from the candidate's work):
   - Measure radial power spectrum of latents: P(f) = A * f^(-beta) (FLUX beta~1.92, WAN 2.42, PixelGen 2.45,
     Qwen-Image 2.235). With ortho-normalized DCT/FFT, flow-matching noise has unit power per bin, so P(f)=1 is the
     SNR=1 line. A frequency f "activates" (signal exceeds noise) at a flow time t* where (1-t)^2 P(f) = t^2 (or
     similar, depending on the interpolant x_t = (1-t) x_0 + t eps). Stage transitions happen when the current
     stage's Nyquist frequency activates; delta is a small margin hyperparameter controlling how early.
   - Stage transition: DCT the low-res latent, embed into the low-frequency corner of the higher-res DCT grid, fill
     new high-frequency coefficients with t*noise, inverse DCT. A kappa scale / t-tilde alignment corrects SNR because
     embedding into a bigger grid spreads energy (1/r_eff attenuation, r_eff = sqrt(N_target/N_source)).
   - Findings: continuous per-step frequency masking (brick-wall radial reveal every step) BLURS (late frequencies
     starved of steps + ringing); staged reveal (whole band at a transition, then all remaining steps) is sharp.
     Staged box low-pass at full res matches quality but gives NO speedup; only actual resolution reduction speeds up.
     At super-native resolution (WAN 1.3B at 960p, 2x native), staged resolution beats full-res low-pass on both
     speed (3.8x) and quality because the model spends most steps at its native resolution (avoids RoPE
     extrapolation regime).
   - Video extension: joint spatiotemporal spectrum; temporal axis is ~50x more energetic than spatial at equal
     Nyquist-normalized frequency (beta_s~2.65, beta_t~1.57); 3-stage schedules (spatial-first 240p40f->480p40f->
     480p80f = 1.85x; temporal-first 1.56x); 3D DCT expansion; Euler solver with step-index reset at transitions
     (multistep solvers like UniPC keep stale history across a resolution change — a correctness bug).
   - Solvers used: FlowUniPCMultistep (shift=3), Euler; 50 steps; CFG 6 for WAN. Qwen-Image uses true CFG (2 passes).
2. **Policy-based Foveated Imaging and Perception** (SIGGRAPH 2026; Xiao, Ackermann, Deng, Wetzstein). A learned
   policy adaptively allocates sensing bandwidth (which regions to read at high resolution) from an ultra-high-res
   sensor for downstream tasks. Validated in simulation (MuJoCo robotic manipulation, object tracking, text
   recognition) and on a real 200-megapixel sensor prototype. Keeps task performance with < 1/8 of full-resolution
   pixel bandwidth. Involves: RL/policy learning for active perception, saccade/pursuit-style behaviors, foveated
   ROI vs global view, bandwidth-accuracy tradeoff, sim-to-real for a sensing policy, MuJoCo simulation.
3. **Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation** (preprint 2026; Chao*, Yariv*,
   Xiao, Wetzstein). Mixed-resolution generation for pretrained diffusion transformers: high-res tokens in foveal
   regions, coarse tokens in periphery, cutting attention cost. LoRA post-training on FLUX.2 and WAN 2.1: 2x image,
   4x video wall-clock speedup. Topics: token count vs attention cost (quadratic), position embeddings for
   mixed-res tokens, patchify at multiple scales, training with LoRA only, how to blend/upsample periphery.
4. **Opportunistic Single-Photon Time of Flight** (CVPR 2025 oral; Nousias*, Wei*, Xiao, ... Lindell, Kutulakos).
   Single-photon avalanche diode (SPAD) imaging; using ambient/opportunistic light sources for time-of-flight depth;
   ultra-wideband single-photon 3D imaging (USRA award). Topics: photon timestamps, histogramming, pile-up, Poisson
   statistics, correlation / cross-correlation for ToF, computational imaging.
5. **Bell Canada software intern (16 months)**: document retrieval + code generation tools (Python, Ruby, SQL,
   LangChain), fine-tuned open-source LLMs. Topics: RAG, embeddings, LoRA/SFT of LLMs, eval.
6. Math: "Virasoro Extensions for Diffeomorphisms with Breaks" (preprint) — strong math background; TA for advanced
   linear algebra. Interviewers may probe linear algebra / probability rigor.

## Unlisted but real work (fair game if the candidate mentions it; useful for deep questions)
- **KV-cache compression for a world-action model** (lingbot-VA on LIBERO): during autoregressive video/action
  rollouts, keep only latent tokens that are "surprising" (||GT latent - imagined latent|| in standardized VAE latent
  space) AND attended by the action expert; running-quantile thresholds (distribution-free, self-calibrating) beat
  z-score/median; ~44-54% KV reduction with task success unchanged and a small imagined-video PSNR hit (~0.3-1.7 dB).
  Paired-seed evaluation, determinism issues with flash-attention, failure-episode PSNR confound (stalled robot =
  easy to predict).
- Ported SPEED to Qwen-Image (true CFG, dynamic-shift FlowMatchEuler scheduler, 3D-latent single frame).
- Works with WAN 2.1 T2V, FLUX, DiT-style models, diffusers, flash-attn on B200s, multi-GPU (torch.distributed.run).

## Claimed skills (every one must be tested somewhere)
Research: computer vision; generative AI; diffusion models; flow matching; image & video generation; efficient deep
learning; long-context modeling; causal video generation; world models; vision-language-action (VLA) models; world
action models; physical AI.
ML systems: distributed and multi-GPU training; model fine-tuning; efficient attention; inference optimization.
Programming: Python; PyTorch.

## Job description focus (NVIDIA, all three postings)
Common: Python, C++, CUDA, PyTorch/JAX; "experience with large-scale model training is a plus"; publications;
transfer research to product.
- GenAI: multimodal foundation models, diffusion models, world models, image/video/audio generation, LLMs, VLMs,
  action-based transformers, long-context methods, physics-based simulation, flow-based generative models,
  synthetic data generation, AI for science (PDEs, weather).
- CV & DL: 3D vision (optical/scene flow, SLAM, depth, digital twins), human-centric (motion, avatars, Gaussian
  avatars, physics sim of clothing), neural rendering & generative 3D (diffusion, world models, NeRF, novel view
  synthesis, 3D content), model efficiency (pruning, compression, efficient ViTs, NAS), multimodal/VLM, synthetic
  data, weather sim.
- Robotics: manipulation & control (dexterous, humanoid loco-manipulation, kinematics/dynamics/sensors), perception
  & world understanding (robot perception, VLA), robot learning (foundation models for robotics, imitation & RL),
  simulation / sim-to-real / real-to-sim (Isaac Sim/Lab, MuJoCo), motion planning & navigation, synthetic data.

## Interview style to mimic
Research-scientist intern loops: (a) 45-60 min technical deep dive on the candidate's papers with "why not X",
"what breaks if", "how would you scale this"; (b) ML fundamentals rapid-fire; (c) generative-modeling specifics
(derivations, sampler math, guidance, parameterizations); (d) systems / large-scale training (how would you train
a 10B video DiT on 512 GPUs; what's the comm volume; what breaks); (e) coding (implement X in 20-30 minutes in a
shared editor, PyTorch/numpy, then discuss complexity and numerical stability). Answers should be what a strong
candidate would SAY: crisp, structured, with the key equation or number, and the common follow-up.
