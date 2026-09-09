# Distributed & Large-Scale Training
<!-- weight: 5 -->

## Q: Your CV says "distributed and multi-GPU training". Start simple: what does PyTorch DDP actually do during a step, and how does it hide the communication? {diff=1 tags=distributed,ddp,data-parallel}
- Follow-up: what happens if one rank has a parameter that received no gradient this step?
- Follow-up: why is `find_unused_parameters=True` slow?
### A
**DDP = data parallelism: every rank holds a full replica, sees a different micro-batch, and the gradients are averaged with an all-reduce before the (identical, replicated) optimizer step.** Because all ranks start from the same weights and apply the same averaged gradient, replicas stay bit-identical without ever communicating parameters.

The trick that makes it fast is **overlap with backward**. DDP registers autograd hooks on every parameter and groups parameters into **buckets** (default 25 MB, in reverse registration order because backward produces gradients for the last layers first). As soon as every gradient in a bucket is ready, DDP launches an asynchronous all-reduce on that bucket on a separate CUDA stream while autograd keeps computing earlier layers. By the time backward finishes, most of the communication is done; only the last bucket is exposed. Gradient averaging is done by all-reducing the sum and dividing by world size.

Traps an interviewer looks for:
- **Unused parameters**: a bucket never becomes "ready", the all-reduce never fires, and the run hangs at the end of backward. `find_unused_parameters=True` fixes it by traversing the autograd graph each iteration to mark those params ready — that traversal is the overhead, and it disables some static-graph optimizations.
- All ranks must execute the **same collectives in the same order**; a data-dependent branch on one rank is a classic hang.
- With gradient accumulation you must wrap the non-final micro-steps in `model.no_sync()` or you all-reduce every micro-batch and waste bandwidth.

## Q: Derive the cost of a ring all-reduce. Why is it built from a reduce-scatter and an all-gather, and what does "all-reduce is bandwidth-bound" mean? {diff=2 tags=distributed,nccl,all-reduce}
- Follow-up: put numbers on it — 10B parameters, bf16 gradients, 8 GPUs on NVLink versus 64 nodes on InfiniBand.
- Follow-up: when is a tree all-reduce better than a ring?
### A
**A ring all-reduce over $N$ GPUs on $M$ bytes moves $2\frac{N-1}{N} M$ bytes per GPU, and the time is $\approx 2\frac{N-1}{N} \cdot \frac{M}{B} + 2(N-1)\alpha$**, where $B$ is per-GPU link bandwidth and $\alpha$ is per-step latency. For large $N$ that's $\approx 2M/B$ — independent of $N$, which is why data parallelism scales.

Decomposition:
- **Reduce-scatter**: split the buffer into $N$ chunks. In $N-1$ steps each GPU sends one chunk to its right neighbour and adds the chunk it receives. After $N-1$ steps every GPU owns one fully reduced chunk. Traffic per GPU: $\frac{N-1}{N} M$.
- **All-gather**: another $N-1$ steps circulating the reduced chunks so everyone has all of them. Traffic: $\frac{N-1}{N} M$.
- Total $2\frac{N-1}{N} M$, and the reduce-scatter/all-gather split is exactly what ZeRO/FSDP exploit: they stop halfway (each rank keeps only its shard).

**"Bandwidth-bound"** means for the sizes we care about (tens of GB of gradients) the $M/B$ term dominates the latency term, so the wall-clock is set by the slowest link's bytes/second, not by how many messages are sent. That's why you bucket small tensors (amortize $\alpha$) and why interconnect bandwidth, not GPU FLOPs, decides DP scaling.

**Bandwidths to hold in your head** (per GPU, per direction): NVLink 4 on H100 ≈ 450 GB/s (900 bidirectional; 1.8 TB/s on B200), NDR InfiniBand 400 Gb/s ≈ 50 GB/s per NIC (one NIC per GPU on a DGX), PCIe Gen5 x16 ≈ 64 GB/s, HBM3 ≈ 3.35 TB/s. NCCL builds every collective (all-reduce, reduce-scatter, all-gather, broadcast, all-to-all, send/recv) from rings and trees over that discovered topology, hierarchically: reduce inside the node on NVLink, then across nodes on IB, so IB carries 1/8 of the naïve traffic. `NCCL_DEBUG=INFO` prints the rings/trees and confirms GPUDirect RDMA is on.

Numbers: 10B params in bf16 = 20 GB of gradients. On an 8-GPU H100 node (NVLink ≈ 450 GB/s per direction per GPU): $2 \cdot \frac{7}{8} \cdot 20/450 \approx 78$ ms. Across 64 nodes over NDR InfiniBand at 400 Gb/s ≈ 50 GB/s per GPU: $2 \cdot 20/50 \approx 0.8$ s. A forward+backward step on a 10B model with a decent per-GPU batch is a few seconds, so DDP still works, but the exposed fraction grows fast if you can't overlap. Hierarchical/tree all-reduce wins when $N$ is large and latency matters (small messages); NCCL picks automatically and uses NVLink for the intra-node stage and IB for the inter-node stage.

## Q: Explain ZeRO stages 1, 2, 3 and FSDP. Do the memory arithmetic for a mixed-precision Adam run. {diff=2 tags=distributed,zero,fsdp,memory}
- Follow-up: which one is PyTorch FSDP equivalent to?
- Follow-up: what does ZeRO-3 NOT shard?
### A
**Mixed-precision Adam costs 16 bytes per parameter**: $2\,(\text{bf16 weight}) + 2\,(\text{bf16 gradient}) + 4\,(\text{fp32 master weight}) + 4\,(\text{Adam } m) + 4\,(\text{Adam } v)$. A 10B model is 160 GB of state before a single activation — it does not fit on one 80 GB H100, so plain DDP is out.

ZeRO observes that in DDP all of that state is **replicated** on every rank, which is pure waste. With $N$ data-parallel ranks:
- **ZeRO-1** shards the optimizer states (12 of the 16 bytes): per-GPU $4 + 12/N$ bytes/param. Each rank updates only its $1/N$ slice of parameters, then all-gathers the updated weights.
- **ZeRO-2** also shards gradients: $2 + 14/N$. Gradients are reduce-scattered instead of all-reduced, so each rank receives only the slice it will update.
- **ZeRO-3** also shards the bf16 weights: $16/N$. Parameters are all-gathered just-in-time before each layer's forward and backward, then freed.

For 10B on 64 GPUs: DDP 160 GB (impossible), ZeRO-1 ≈ 52 GB, ZeRO-2 ≈ 34 GB, ZeRO-3 ≈ 2.5 GB per GPU for model state. The rest of the memory budget is **activations**, which ZeRO does not touch — that is what tensor/sequence parallelism and activation checkpointing are for.

**PyTorch FSDP is ZeRO-3** (with `SHARD_GRAD_OP` giving ZeRO-2 behaviour and HSDP giving ZeRO-3 within a node and replication across nodes). FSDP wraps modules into "units" (typically one transformer block each); each unit is a flat sharded parameter that is all-gathered on entry and reduce-scattered on backward, with prefetching of the next unit to overlap. ZeRO-3 also does not shard activations, and in its basic form it does not reduce the per-layer temporary: you still need the full unsharded weights of one block resident at a time.

## Q: What is the communication cost of ZeRO-3 relative to plain data parallelism, and when would you pick ZeRO-3/FSDP over tensor parallelism? {diff=3 tags=distributed,zero,fsdp,communication}
- Follow-up: why does ZeRO-3 scale badly across many nodes while ZeRO-1 does not?
### A
Count bytes per parameter per step, with $M$ = model size in bf16:
- **DDP**: one all-reduce of gradients = reduce-scatter + all-gather = **$2M$**.
- **ZeRO-1/2**: reduce-scatter of gradients ($M$) + all-gather of updated weights ($M$) = **$2M$ — same as DP**. Free memory savings.
- **ZeRO-3**: all-gather weights for forward ($M$) + all-gather weights again for backward ($M$) + reduce-scatter gradients ($M$) = **$3M$, i.e. 1.5× DP**.

The 1.5× is per-parameter volume; the real problem is granularity and latency: ZeRO-3 issues one all-gather per layer on the critical path, and every layer's compute must wait for its own parameters. Within a node on NVLink that is easily overlapped by prefetching the next block; across InfiniBand the all-gathers become exposed unless the per-GPU batch is large enough that the compute per block ($\approx 6 \cdot \text{params}_{\text{block}} \cdot \text{tokens}$ FLOPs) exceeds all-gather time. That is why ZeRO-1 scales to thousands of GPUs without much thought, while ZeRO-3 is usually confined to a node or a few nodes (HSDP: shard within the node, replicate across nodes).

**When ZeRO-3/FSDP vs tensor parallelism**:
- FSDP is a memory trick that keeps compute local — every GEMM is full-size and efficient; no code changes to the model. Use it when the model fits in aggregate node memory and per-GPU batch is decent.
- TP splits each GEMM so every layer needs two all-reduces on activations (forward) and two more (backward). It reduces **activation** memory as well and shortens per-GPU time per step, but every GEMM gets smaller (worse utilisation), so it only pays inside NVLink.
- Rule of thumb from Megatron/Meta: TP up to 8 inside the node, then FSDP/ZeRO-1 across nodes, add PP only when TP=8 + FSDP still doesn't fit or the all-gather volume dominates (models $\gg$ 70B).

For a 10B video DiT the honest answer is: FSDP within a node plus context parallelism handles it; TP is optional.

## Q: Walk me through Megatron-style tensor parallelism for one transformer layer. How many collectives per layer, and why does TP stay inside the node? Contrast with expert parallelism. {diff=2 tags=distributed,tensor-parallel,megatron,moe}
- Follow-up: what does Megatron "sequence parallelism" add on top of TP?
- Follow-up: why column-then-row and not row-then-column?
### A
**Tensor parallelism splits the weight matrices of a layer across $t$ GPUs so that each GPU does $1/t$ of every GEMM, at the cost of one all-reduce on the activations per sub-block.**

MLP: $Y = \operatorname{GeLU}(XA)\,B$.
- **Column-split $A$** into $[A_1 \dots A_t]$: each GPU computes $\operatorname{GeLU}(XA_i)$ independently — the nonlinearity is elementwise so no sync is needed. This is why column comes first; row-splitting $A$ would require a sum before the GeLU.
- **Row-split $B$**: each GPU computes its partial product $Y_i = \operatorname{GeLU}(XA_i)\,B_i$ and the outputs are summed with an **all-reduce**.

Attention: split heads across GPUs ($Q, K, V$ column-parallel — each GPU owns $h/t$ heads, so attention is local), output projection row-parallel → one all-reduce. Layer total: **2 all-reduces in forward, 2 in backward** (the identity/all-reduce pair f/g are conjugate). Each all-reduce is on a tensor of shape $[b, s, h]$ in bf16 — for $s = 32\text{k}$, $h = 4096$, $b = 1$ that is 256 MB per all-reduce, four times per layer, ~50 layers: ~50 GB per step per GPU of activation traffic. Only NVLink (~450 GB/s) makes that cheap; over IB (~50 GB/s) it would take a second per step. Hence **TP $\le 8$, within the NVLink domain**.

Megatron **sequence parallelism** notices that LayerNorm and dropout between the TP regions are replicated on every GPU; it shards those along the sequence and replaces each all-reduce with a reduce-scatter + all-gather (same bytes), cutting activation memory by another factor $t$.

**Expert parallelism** places different MoE experts on different GPUs. The router decides per token which expert it goes to, so the communication is an **all-to-all** (dispatch tokens, compute, all-to-all back) — two all-to-alls per MoE layer. Volume scales with tokens × hidden, like TP, but the pattern is irregular and load-imbalanced (hot experts), which is why capacity factors, token dropping and load-balancing losses exist. EP is usually paired with DP inside the expert groups and can span nodes better than TP because the all-to-all is one-shot rather than per-GEMM.

## Q: Pipeline parallelism: what is the bubble, how do GPipe and 1F1B differ, and what do interleaved schedules buy you? {diff=2 tags=distributed,pipeline-parallel,scheduling}
- Follow-up: why is PP awkward for a DiT with adaLN conditioning or for encoder-decoder models?
### A
**Pipeline parallelism puts different layers on different GPUs; the price is the bubble — time when stages idle waiting for the first micro-batches to arrive or the last to drain.**

With $p$ stages and a global batch split into $m$ micro-batches, each stage is busy for $m$ forward + $m$ backward slots but the schedule spans $(m + p - 1)$ of each, so **bubble fraction $= \frac{p-1}{m}$** of the ideal time (equivalently, efficiency $= \frac{m}{m+p-1}$). To keep the bubble under ~5% you need $m \gtrsim 20p$ micro-batches, which means large global batches — often not available with 32k-token video sequences.

- **GPipe**: run all $m$ forwards, then all $m$ backwards. Simple, but every stage must hold activations for all $m$ micro-batches → activation memory $\propto m$.
- **1F1B** (PipeDream-flush, Megatron): after warm-up, each stage alternates one forward and one backward, so at most $p$ micro-batches are in flight per stage → activation memory $\propto p$ instead of $m$. Same bubble fraction, much less memory.
- **Interleaved 1F1B**: give each GPU $v$ non-contiguous chunks of layers (e.g. GPU0 holds layers 0-3 and 16-19). The "virtual" pipeline is $pv$ deep with smaller stages, so the bubble drops to **$\frac{p-1}{vm}$** at the cost of $v\times$ more point-to-point sends per micro-batch. Zero-bubble schedules (ZB-H1/H2) go further by splitting backward into input-gradient and weight-gradient parts and reordering.

Traps: PP needs the model to be a clean chain of stages with equal compute; DiT blocks are uniform (good), but the shared timestep/text conditioning must be broadcast to every stage, and anything with cross-stage skip connections (U-Nets) or unbalanced first/last stages (embedding, VAE, loss) kills balance. PP traffic is tiny (one activation tensor per micro-batch boundary), so it is the parallelism you put across nodes when you must.

## Q: You are training a video DiT on 32k-token sequences. Explain sequence/context parallelism: DeepSpeed-Ulysses versus ring attention. When would you use each, and how do you fix load imbalance with a causal mask? {diff=3 tags=distributed,context-parallel,ring-attention,ulysses,video}
- Follow-up: how does the KV-heads count of GQA constrain Ulysses?
- Follow-up: can you combine Ulysses and ring?
### A
**Context parallelism (CP) shards the sequence axis across $P$ GPUs so activations shrink by $P$; the only operation that needs the whole sequence is attention, and the two schemes differ in how they get it.**

**Ulysses (all-to-all on heads)**: every GPU holds $s/P$ tokens of all heads after the MLP. Before attention, an all-to-all re-partitions $Q, K, V$ so each GPU holds all $s$ tokens of $h/P$ heads; attention is then completely local (regular flash-attention). After attention, another all-to-all goes back to sequence-sharded. Four all-to-alls per layer ($Q, K, V, O$), each moving $\frac{P-1}{P}$ of a $[s/P, h]$ tensor per GPU — traffic per GPU $\propto sh/P$, so it *decreases* with $P$. Limitation: **$P \le$ number of heads** (with GQA, $\le$ KV heads unless you replicate KV), and all-to-all is latency-heavy across nodes.

**Ring attention (pass KV blocks)**: each GPU keeps its $s/P$ queries and its own KV block. In $P-1$ steps it sends its KV block to the next GPU in the ring and receives the previous one, computing a partial attention block each step and merging with the running online-softmax (log-sum-exp) state, exactly like flash-attention across GPUs. Communication per step is a $[s/P, h_{\text{kv}}]$ KV block, overlapped with the compute of the current block. No head limit, scales to millions of tokens, but $P-1$ sequential rounds and overlap only works if block compute time $\ge$ KV transfer time — which needs long chunks per GPU.

**Causal imbalance**: with a causal mask and contiguous chunks, GPU 0 does 1 block of work while GPU $P-1$ does $P$ blocks. The fix is **zig-zag assignment**: split into $2P$ chunks and give GPU $i$ chunks $i$ and $2P-1-i$, so every GPU has one early and one late chunk and does the same total work. Video DiTs are usually bidirectional in space and time so this only matters for autoregressive/causal video models.

When to use: Ulysses for $P \le 8$ within a node (cheap, simple, no kernel changes); ring for cross-node or very long context; **2D hybrid** (Ulysses inside the node × ring across nodes) is what USP/Megatron-CP do for 100k+ token video. For our 32k tokens and a ~48-head model, CP = 4–8 with Ulysses is the natural pick.

## Q: Give me a concrete layout: a 10B-parameter DiT, 32k tokens per sample, 512 H100s. Choose DP/TP/PP/CP and justify with memory and communication numbers. {diff=3 tags=distributed,parallelism,planning,dit,video}
- Follow-up: what changes if it's a B200 cluster?
- Follow-up: what if the model were a 10B MoE with 64 experts?
### A
Start from the constraints, then place the parallelism.

**Model state**: $10\text{B} \times 16\,\text{B} = 160$ GB → must be sharded across $\ge 3$ GPUs; ZeRO-1/FSDP across the DP group handles this.

**Activations**: for a transformer block with flash attention, $\approx 34\,sbh$ bytes per layer (Megatron's formula minus the $s^2 a$ term). With $h = 4096$, $s = 32768$, $b = 1$: $34 \times 32768 \times 4096 \approx 4.6$ GB per layer; ~48 layers → ≈ 220 GB per sample. Even with full checkpointing (keep only the block input, $2sh = 256$ MB per layer) you still need to hold one block's internals (~4.6 GB) during recompute — fine — but the un-checkpointed case demands sharding the sequence. So **CP (or TP+SP) is mandatory**, and activation memory, not weights, is the binding constraint.

**Compute**: $6N$ per token $= 60$ GFLOP, plus attention $12 L s h$ per token $\approx 12 \times 48 \times 32768 \times 4096 \approx 77$ GFLOP (the $s/(6h) \approx 1.33$ factor — at 32k tokens attention exceeds the GEMMs). $\approx 140$ GFLOP/token. Global batch 512 samples = 16.8M tokens/step → 2.35 EFLOP per step. At 40% MFU, $512 \times 989\,\text{TFLOPS} \times 0.4 \approx 200$ PFLOP/s → ~12 s/step.

**Layout** (64 nodes × 8 GPUs):
- **TP = 1 or 2**: the 4096-wide GEMMs are already small per token; TP would shrink them further. I'd rather spend the NVLink on CP.
- **CP = 8 within the node** (Ulysses on heads; $48\ \text{heads} / 8 = 6$ heads per GPU): activation memory per GPU $\approx 220/8 \approx 28$ GB unchecked, ~10 GB with selective checkpointing. All-to-all traffic per layer ~$4 \times 256\,\text{MB} / 8$ per GPU on NVLink — negligible.
- **PP = 1**: uniform blocks make PP possible, but with $b = 1$ per GPU the micro-batch count is too small to hide the bubble ($\frac{p-1}{m}$ with $m \approx 8$ is 12%+). Not worth it at 10B.
- **DP = 64 across nodes with FSDP/HSDP**: shard weights within the node, replicate across → per-node all-gathers on NVLink; the cross-node traffic is one reduce-scatter of 20 GB of grads per step over $8 \times 400\,\text{Gb/s} \approx 400\,\text{GB/s}$ per node $\approx 0.1$ s — under 1% of the 12 s step and fully overlappable.

Result: (CP 8) × (DP 64), FSDP inside the node, ZeRO-1 optimizer sharding across, selective activation checkpointing, VAE latents pre-encoded. On B200s (192 GB, 1.8 TB/s NVLink, ~2.2 PFLOPS bf16) I'd drop checkpointing first, then reduce CP to 4 to raise per-GPU sequence length. For an MoE variant, add EP = 8 across the same NVLink group and keep the all-to-all inside the node, with DP over experts across nodes.

## Q: Where does activation memory come from in a transformer, and how does activation checkpointing trade it for compute? What is "selective" recomputation? {diff=2 tags=distributed,memory,activation-checkpointing}
- Follow-up: the "33% extra compute" number — where does it come from exactly?
### A
Activations are every intermediate autograd needs for backward: attention inputs, Q/K/V, softmax output (if not fused), MLP pre-activations, LayerNorm inputs, dropout masks. Megatron's count for one layer in bf16 is **$sbh\,(34 + 5as/h)$ bytes**: the $34\,sbh$ is the linear-size stuff, the $5 a s^2 b$ term is the attention probabilities — quadratic in sequence length and the first thing that explodes at 32k tokens. **Flash attention removes the $s^2$ term** by recomputing the softmax in backward from stored log-sum-exp statistics, which is itself a form of selective checkpointing.

**Activation checkpointing** stores only the input of each transformer block ($2sbh$ bytes) and re-runs the block's forward during backward to regenerate the intermediates. Memory drops from ~$34\,sbh$ per layer to ~$2sbh$ plus one live block; the cost is one extra forward. Since forward is ~1/3 of the step's FLOPs (forward $2ND$, backward $4ND$), full recomputation adds **≈ 33% compute** — MFU looks fine but HFU rises to $8ND$ per token.

**Selective recomputation** (Megatron "selective") only checkpoints the expensive-to-store but cheap-to-recompute parts — historically the attention softmax/dropout, which is ~70% of the memory and ~few % of FLOPs. With flash attention that part is already gone, so modern selective schemes checkpoint every $k$-th block, or offload the block inputs to CPU (asynchronously over PCIe) instead of recomputing. The practical decision rule: checkpoint just enough that you can raise per-GPU batch or sequence until the GEMMs saturate; beyond that the 33% is wasted.

## Q: bf16 versus fp16 versus fp8: what breaks with each, why do we keep fp32 master weights, and what is loss scaling? {diff=1 tags=mixed-precision,bf16,fp8,numerics}
- Follow-up: what exactly does Transformer Engine do to make fp8 work?
### A
**fp16** has 5 exponent bits: max 65504, smallest normal $\approx 6 \times 10^{-5}$. Gradients routinely underflow to zero, so you use **loss scaling**: multiply the loss by $S$ (e.g. $2^{16}$) before backward, unscale the gradients before the optimizer step, and dynamically halve $S$ when you see inf/NaN and double it after a few thousand clean steps. That is `torch.cuda.amp.GradScaler`.

**bf16** keeps fp32's 8 exponent bits and drops mantissa to 7 bits (≈ 3 significant decimal digits). Same dynamic range as fp32 → no loss scaling, no overflow drama, and it is the default on A100/H100. The price is precision: a weight update $\text{lr} \cdot g$ with $\text{lr} \cdot g / w < 2^{-8} \approx 0.4\%$ is **rounded away entirely** if the weight itself is bf16. That's why we keep **fp32 master weights** (and fp32 Adam moments, and usually an fp32 all-reduce or at least fp32 accumulation in the reduction): the update is applied in fp32, then cast to bf16 for the next forward. Also keep sensitive reductions in fp32: softmax, LayerNorm statistics, loss, and the attention logits if they grow large.

**fp8** (H100+): two formats, **E4M3** (more mantissa, range $\pm 448$, for weights and activations in forward) and **E5M2** (more range, for gradients). Both are useless without **per-tensor scaling**: each tensor is multiplied by a scale so its max lands in range, and Transformer Engine uses *delayed scaling* — the scale for this step comes from the amax history of previous steps so the cast can be done without a separate reduction. Only the GEMM inputs are fp8; accumulation is fp32, master weights and optimizer states stay in higher precision, and LayerNorm/softmax stay bf16. Payoff: 2× the peak of bf16 (≈ 1979 vs 989 dense TFLOPS on H100) and half the activation bytes; risk: outlier channels and instability late in training, which is why people often fall back to bf16 for the last few percent.

## Q: Gradient accumulation, effective batch size, and learning-rate scaling: how do they interact, and why warmup? {diff=1 tags=optimization,batch-size,learning-rate}
- Follow-up: does gradient accumulation give the same result as a bigger batch with BatchNorm?
### A
**Effective batch = micro-batch × gradient-accumulation steps × data-parallel world size.** Accumulation runs $k$ forward/backward passes summing gradients before one optimizer step; mathematically identical to a $k\times$ larger batch (for losses that are means, divide by $k$), except for anything with batch statistics — BatchNorm sees the micro-batch, which is one reason large models use LayerNorm/RMSNorm. In DDP wrap the first $k-1$ micro-steps in `no_sync()` so you all-reduce once.

**LR scaling** when you grow the batch by $k$:
- **Linear** (Goyal et al., SGD): $\text{lr} \propto k$, justified because $k\times$ batch with $\text{lr} \cdot k \approx k$ small steps of $\text{lr}$ in the small-lr limit. Works until the "critical batch size", beyond which extra samples stop reducing gradient noise and you get diminishing returns (McCandlish gradient-noise scale).
- **Square-root** ($\text{lr} \propto \sqrt{k}$) is the common heuristic for Adam because Adam normalizes by gradient magnitude; the noise standard deviation shrinks as $1/\sqrt{k}$.
- Neither is a law; sweep at small scale and, ideally, use μP so the optimal LR transfers across width.

**Warmup** exists because early on the Adam second-moment estimate $v$ is tiny and noisy (bias-corrected but from few samples), the network's activations are far from their stationary statistics, and a full-size LR step kicks the weights into a bad region you never recover from. Linear warmup over ~1–5% of training (or a few thousand steps) lets $v$ stabilise and the LayerNorm gains settle. Follow with cosine or WSD (warmup-stable-decay) decay. Practical DiT setting: batch 256–1024 videos, AdamW lr $10^{-4}$, $\beta = (0.9, 0.95)$, warmup 1–5k steps, no weight decay on norms/biases, EMA of weights for sampling.

## Q: State the Chinchilla scaling-law result and how you would use scaling laws to plan a video DiT training run. {diff=2 tags=scaling-laws,chinchilla,dit}
- Follow-up: why do practitioners "overtrain" past Chinchilla-optimal?
- Follow-up: what's the y-axis for a diffusion model — loss or FID?
### A
**Chinchilla**: for a fixed compute budget $C \approx 6ND$ FLOPs, the loss is minimised when parameters $N$ and training tokens $D$ grow together at roughly **$D \approx 20N$**. Fit form: $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$ with $\alpha \approx 0.34$, $\beta \approx 0.28$. GPT-3 (175B on 300B tokens) was badly under-trained; Chinchilla (70B on 1.4T) beat it with the same compute.

**Overtraining**: the compute-optimal point ignores inference cost. If you will serve the model a lot, a smaller model trained on 5–20× more tokens than optimal is only slightly worse in loss but much cheaper to run (LLaMA-3 8B on 15T tokens is ~100× past Chinchilla). For a research intern project the calculus flips — you want the best model for a fixed training budget.

**Applying it to DiTs**: the same functional form fits (Li et al. 2024 "Scalability of Diffusion Transformers", Liang et al. 2024): loss vs $N$ and $D$ follows a power law and the optimum is also roughly balanced. Caveats specific to diffusion:
- "Tokens" = latent patches × frames; a 32k-token video is one sample. Data $D$ is counted in tokens seen, including repeated epochs, and video datasets are small in unique tokens so repetition curves matter.
- The **training loss is a noisy proxy**: the flow-matching MSE at random $t$ is dominated by high-noise timesteps and correlates only loosely with FID/FVD or human preference. Fit the law on the loss (smooth, cheap), then verify the top few points on FVD/quality.
- Attention makes FLOPs non-linear in $N$ at long sequence (the $s/(6h)$ factor), so use measured FLOPs, not $6ND$.
- Procedure: train 5–8 models from 100M to 2B for several token budgets, fit the frontier, extrapolate to the 10B budget, and derive the target tokens and batch/LR from μP-style transfer. Expect the DiT optimum to be more data-hungry than LLMs because the loss is a regression target with high irreducible variance.

## Q: How do you compute MFU for a training run, and what number would make you happy on H100s? What's the difference between MFU and HFU? {diff=2 tags=mfu,throughput,flops}
- Follow-up: 10B DiT, 16.8M tokens/step, 12 s/step on 512 H100s — what's the MFU?
### A
**MFU = model FLOPs actually needed per second ÷ hardware peak FLOPs per second.** The model FLOPs come from a formula, not a profiler: for a dense transformer **$\approx 6N$ per token** ($2N$ forward: one multiply-add per parameter per token; $4N$ backward: gradient w.r.t. inputs and w.r.t. weights) plus the attention term **$12 L s h$ per token** (flash attention's recomputation is *not* counted — that's the point of "model" FLOPs). So

$$
\text{MFU} = \frac{(6N + 12 L s h) \cdot \text{tokens per second}}{n_{\text{GPU}} \cdot \text{peak}}.
$$

Peak: H100 SXM bf16 dense **989 TFLOPS** (the 1979 number is with 2:4 sparsity — never use it), A100 312, B200 ≈ 2.25 PFLOPS dense bf16.

Example: $N = 10\text{B}$, $L = 48$, $s = 32768$, $h = 4096$ → $60 + 77 \approx 137$ GFLOP/token; $16.8\text{M tokens} / 12\,\text{s} = 1.4\text{M}$ tokens/s → $1.92 \times 10^{17}$ FLOP/s; peak $512 \times 989 \times 10^{12} = 5.06 \times 10^{17}$ → **MFU $\approx 38\%$**.

**HFU** (hardware FLOPs utilisation) counts every FLOP the hardware actually executed, including activation recomputation ($+2N$ per token → $8N$ with full checkpointing) and flash-attention recompute. HFU $\ge$ MFU; the gap is your recompute overhead. Report MFU — it is what you'd pay for.

What's good: LLM pretraining on H100 with well-tuned Megatron reaches **40–50%** (PaLM on TPU reported 46%, LLaMA-3 ~38–43%); anything above 35% for a long-sequence video DiT with CP is solid, under 25% means something is exposed (comm, dataloader, small GEMMs from too much TP, python overhead). Roofline: the GEMMs are compute-bound at these shapes; MFU loss comes from non-GEMM ops (norms, activations, RoPE, adaLN modulation — memory-bound, fused kernels help), communication that failed to overlap, and pipeline bubbles.

## Q: Data loading for a 512-GPU video run: how do you shard the dataset, why pre-encode with the VAE, and how do you make resume deterministic? {diff=1 tags=data-loading,webdataset,video,vae}
- Follow-up: how do you know the dataloader is the bottleneck?
### A
**Sharded tar archives (WebDataset-style)**: pack samples into ~1 GB shards, assign shards to ranks (rank $r$ takes shards $r, r + W, r + 2W, \dots$, $W$ = number of dataloader workers × world size), stream them sequentially from object storage or a parallel FS. Sequential reads of large files are what storage is good at; millions of small mp4s are what it is terrible at. Shuffle at two levels: shuffle the shard list per epoch, then a per-worker in-memory shuffle buffer of a few thousand samples.

**Pre-encode with the VAE**: decoding mp4 → frames → 3D-VAE encode is far more expensive than the DiT step for a 1.3B–10B model (the WAN VAE compresses 4× temporally, 8×8 spatially; an 81-frame 480p clip is ~$21 \times 60 \times 104 \times 16$ latents). Doing it online wastes GPU time and makes the dataloader the bottleneck. Instead encode once offline (with several random crops/aspect buckets and the text embeddings from T5/UMT5), store latents in bf16 or fp16 plus the text-encoder outputs, and train from tensors. Cost: storage (latents are ~1/100 of pixels, fine) and losing on-the-fly augmentation. Keep aspect-ratio buckets so each batch has uniform token count and no padding.

**Deterministic resume**: the sampler state must be a pure function of (seed, epoch, step). Derive the shard order from `seed + epoch`, record the global sample index in the checkpoint, and on resume fast-forward the iterator (skip $N$ samples, or store per-shard offsets) rather than restarting the epoch — otherwise you retrain on seen data and the loss curve gets a visible kink. Checkpoint the RNG states of every rank too.

Diagnosis: profile with the torch profiler or simply time `next(loader)` — if the GPU idles between steps or `nvidia-smi` shows utilisation sawtoothing, raise num_workers, use pinned memory + non_blocking copies, prefetch 2–4 batches, and move CPU decode to DALI or GPU.

## Q: Fault tolerance at scale: how often do you checkpoint, what is elastic training, and how do you handle NaNs and silent data corruption? {diff=2 tags=fault-tolerance,checkpointing,elastic,reliability}
- Follow-up: how would you even detect a GPU that computes wrong answers?
### A
At 512 GPUs a hardware failure every few hours is normal (Meta reported ~1 failure per 3 hours on 16k H100s), so the question is how much work you lose and how fast you restart.

**Checkpoint cadence**: the Young/Daly rule gives the optimal interval $\tau \approx \sqrt{2 \delta \cdot \text{MTBF}}$, where $\delta$ is the time to write a checkpoint. 160 GB of state for 10B params written asynchronously (copy to host, then to storage on a background thread, distributed/sharded save so every rank writes its shard) takes $\delta \sim 1$ min; MTBF 3 h → $\tau \approx 20$ min. Save sharded (FSDP `state_dict_type=SHARDED`, torch.distributed.checkpoint) and consolidate offline; keep the last few plus periodic permanent ones.

**Elastic training**: `torchrun --max-restarts` with a c10d rendezvous: when a rank dies, the agent kills all workers, re-rendezvouses (possibly with fewer nodes if `--nnodes=min:max`), re-assigns ranks, and every worker reloads the last checkpoint. Automated node health checks pull bad nodes before the job restarts on them. Restart time is dominated by loading the checkpoint and re-initialising NCCL, so keep checkpoints close (local NVMe or a burst buffer).

**NaN handling**: check the global gradient norm every step (you already compute it for clipping). On inf/NaN: skip the optimizer step (what `GradScaler` does for fp16) and count; a few per thousand steps is tolerable, a burst means a bad batch, an LR problem or a broken GPU. Never let a NaN reach the weights — one corrupted parameter propagates to all ranks through the all-reduce.

**Silent data corruption (SDC)**: a GPU that produces wrong numbers with no error. Detection is by redundancy: periodically run the same micro-batch on two ranks and compare, or compare per-rank gradient norms before the all-reduce — an outlier rank is either seeing a strange batch or is broken. Log Xid errors from the driver, run DCGM diagnostics on nodes with anomalies, and keep the training loss/grad-norm curves so a divergence can be traced back to the exact checkpoint to roll back to.

## Q: War story: your 512-GPU run hangs at step 500, no error, GPUs at 100% utilisation. Walk me through how you debug it, and how you would have profiled the run beforehand. {diff=3 tags=debugging,nccl,profiling,hang}
- Follow-up: why do hung NCCL kernels show 100% GPU utilisation?
- Follow-up: how do you find a straggler rank?
### A
Step 500 is not random — it is almost certainly the first time something periodic runs: evaluation, checkpoint save, logging with a gathered metric, or an LR-schedule boundary. A hang with 100% utilisation is the signature of a **collective mismatch**: some ranks are spinning in an NCCL kernel waiting for peers that never arrive (NCCL kernels busy-wait, hence the 100%).

Procedure:
1. **Get stack traces from every rank** without killing the job: `py-spy dump --pid` on each worker (a wrapper script over all nodes), or the built-in `TORCH_NCCL_TRACE_BUFFER_SIZE` flight recorder + `TORCH_NCCL_DUMP_ON_TIMEOUT`, which dumps the last $N$ collectives per rank with their sequence numbers and sizes. Set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and a sane timeout so hangs become exceptions instead of eternity.
2. **Diff the traces**: the rank whose stack differs (e.g. sitting in `save_checkpoint` or `evaluate` while the others are in `all_reduce`) is the culprit. Classic causes: rank 0 does `if rank == 0: save()` which internally calls a collective (FSDP state dict all-gathers!) that the others never join; uneven data — one rank's dataloader ran out of samples and it exited the loop; a data-dependent branch (`if loss.isnan(): continue`) taken on one rank; `find_unused_parameters` mismatch; a `torch.distributed.barrier()` in a rank-conditional block.
3. If all stacks are in the same collective, check the hardware: `NCCL_DEBUG=INFO` for link errors, `dmesg`/Xid for a fallen-off NVLink or NIC, `ibstat`, and a standalone nccl-tests all-reduce on the suspect node.

**Profiling beforehand**: `torch.profiler` with CUPTI for one rank for a few steps → look for gaps between kernels (dataloader, python overhead), NCCL kernels not overlapped with compute (comm exposed), small GEMMs (TP too high). `nsys profile` across ranks with `--capture-range` for the multi-GPU timeline. **Stragglers**: log per-rank step time and the wait time inside the first collective of each step (`torch.cuda.Event` timing before/after all-reduce); a rank whose compute is consistently slow (thermal throttling, a bad HBM stack, a noisy neighbour on the network) will show up as everyone else waiting on it. Meta's practice is to run this straggler detector continuously and evict the node.

## Q: The loss spikes and sometimes diverges once you scale to 10B. List the likely causes and the fixes you would try, in order. {diff=3 tags=stability,loss-spikes,optimization,dit}
- Follow-up: what does QK-norm fix mechanically?
- Follow-up: what is μP and why does it help here?
### A
Spikes at scale come from a few well-known mechanisms; the order below is the order I'd check them.

1. **Learning rate too high for the width**: the optimal LR shrinks roughly as $1/\text{width}$ for Adam under standard parameterisation. Fix: lower LR, longer warmup, or adopt **μP** (maximal update parameterisation): scale init and per-layer LRs so that activations and updates stay $O(1)$ as width grows, which makes the small-model LR sweep transfer to the big model. Also lower Adam $\beta_2$ from 0.999 to 0.95 — with $\beta_2 = 0.999$ a rare large gradient inflates $v$ slowly and the following updates over-shoot.
2. **Attention logit growth**: as training proceeds, $\|q\| \cdot \|k\|$ grows, the softmax saturates, gradients through it vanish for most tokens and explode for a few — the entropy collapse that precedes a spike. **QK-norm** (LayerNorm/RMSNorm on $q$ and $k$ before the dot product, standard in ViT-22B, SD3, WAN) bounds the logits to a learnable temperature. Related: bf16 rounding on large logits — compute attention scores and softmax in fp32 (flash attention already does).
3. **Output-logit / modulation drift**: in LLMs the $z$-loss ($10^{-4} \log^2 Z$) keeps the softmax normaliser near 1; in DiTs the analogue is the adaLN modulation vectors (scale/shift/gate) blowing up — zero-init the gate, RMSNorm the modulation input, and apply weight decay to them.
4. **Gradient clipping** at global norm 1.0 is a must; watch the pre-clip norm — a rising trend predicts the spike a few hundred steps early.
5. **Data**: a batch of corrupted latents or an aspect bucket with an extreme token count. Log per-batch loss with sample ids; when a spike is data-driven, PaLM's fix works — roll back to the previous checkpoint and **skip ~100–200 batches**.
6. **Numerics**: fp16 without loss scaling, fp8 outliers, or a LayerNorm $\epsilon$ that is too small relative to bf16 activations. Check for inf in the gradient norm before the clip.
7. Diffusion-specific: the loss at low-noise timesteps is tiny and at high-noise huge; use logit-normal or SD3-style timestep sampling and loss weighting so one $t$-range does not dominate the gradient, and keep an EMA of the weights so sampling quality is not hostage to the spike.

If it still diverges: reduce LR by 2×, increase warmup, or init the residual branches with $1/\sqrt{2L}$ scaling (or zero) so the residual stream does not grow with depth.

## Q: Get concrete at the API level: what does `torch.distributed.run` do, what are process groups, and sketch how you'd implement a ring-attention step with send/recv. {diff=3 tags=torch-distributed,api,ring-attention,implementation}
- Follow-up: why `batch_isend_irecv` rather than blocking `send`/`recv`?
- Follow-up: how do you merge two partial attention outputs correctly?
### A
**`torch.distributed.run` (torchrun)** is the elastic launcher: on each node it starts an agent that spawns `--nproc_per_node` worker processes, performs a rendezvous through a c10d TCP store at `MASTER_ADDR:MASTER_PORT`, and sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE` in each worker's environment. The worker then calls `dist.init_process_group("nccl")`, which reads those variables, exchanges NCCL unique ids through the store, and builds the default (world) communicator. `--max-restarts` and `--nnodes=min:max` give the elastic behaviour: on a worker failure the agent tears everything down and re-rendezvouses. Always set `torch.cuda.set_device(LOCAL_RANK)`.

**Process groups** are subsets of ranks with their own NCCL communicator. Every parallelism dimension is a process group: for CP=8 × DP=64 the DP group of a rank is the 64 ranks with the same position inside the node, the CP group its 8 node-mates. `dist.new_group(ranks)` must be called by *all* ranks in the same order (it is collective), and the modern way is `init_device_mesh("cuda", (64, 8), mesh_dim_names=("dp", "cp"))`, whose `mesh["cp"].get_group()` hands you the group; FSDP and the CP ops take it as an argument. Each rank belongs to one group per dimension.

**Ring attention with send/recv** (per layer; $q, k, v$ are $[b, s/P, h_{\text{local}}, d]$ on each rank of the cp group):

```python
def ring_attn(q, k, v, group):
    P, r = group.size(), group.rank()
    nxt, prv = (r + 1) % P, (r - 1) % P
    out, lse = None, None
    k_recv, v_recv = torch.empty_like(k), torch.empty_like(v)
    for step in range(P):
        if step < P - 1:  # post next KV exchange before computing
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, k, nxt, group), dist.P2POp(dist.irecv, k_recv, prv, group),
                dist.P2POp(dist.isend, v, nxt, group), dist.P2POp(dist.irecv, v_recv, prv, group)])
        o_blk, lse_blk = flash_attn_fwd(q, k, v, return_lse=True)  # local block
        out, lse = merge(out, lse, o_blk, lse_blk)              # online softmax
        if step < P - 1:
            for rq in reqs: rq.wait()
            k, v, k_recv, v_recv = k_recv, v_recv, k, v
    return out
```

The merge is the flash-attention rescale: with running $(o, l)$ and new block $(o', l')$, $l_{\text{new}} = \operatorname{logaddexp}(l, l')$, $o_{\text{new}} = o \exp(l - l_{\text{new}}) + o' \exp(l' - l_{\text{new}})$. Backward runs the ring in the same direction, passing $\mathrm{d}K/\mathrm{d}V$ accumulators along.

Why `batch_isend_irecv`: blocking `send` on every rank at once deadlocks (all wait to send), and posting the send and receive as one batch lets NCCL group them into a single kernel and lets the transfer run on the comm stream while the local flash-attention block computes — the overlap that makes ring attention viable. Use double buffers so the tensor being sent is not overwritten, and `torch.cuda.stream` events (or `.wait()` before the swap) to keep the compute stream from reading a buffer still in flight.


<!-- rapid-fire -->

## Q: One minute: what is data parallelism? {diff=1 tags=data-parallel,distributed,rapid-fire}
### A
**Data parallelism** replicates the full model on every GPU, gives each GPU a different slice of the batch, runs forward/backward independently, then **all-reduces the gradients** so every replica applies the same averaged update and stays in sync. It exists because it is the simplest way to scale throughput: no model code changes, and the communication (one gradient-sized all-reduce per step, overlapped with backward in DDP) is cheap relative to compute for large per-GPU batches. Limits: every GPU must hold the full model, gradients, and optimizer state, so it caps out at a few billion parameters in mixed-precision Adam (~16 bytes/param) — which is where ZeRO/FSDP take over. Trap: global batch size grows with GPU count, so you must rescale LR and warm up.

## Q: What is an all-reduce? {diff=1 tags=all-reduce,nccl,collectives,rapid-fire}
### A
An **all-reduce** is a collective where every rank starts with a tensor and ends with the element-wise reduction (usually sum) of all ranks' tensors — every rank gets the full result. It is the primitive under data-parallel gradient averaging. The efficient implementation is **ring all-reduce** = reduce-scatter (each rank ends up owning the reduced $1/N$ chunk) followed by all-gather (everyone collects the chunks): each rank sends and receives $2\frac{N-1}{N} \times$ tensor bytes, roughly 2× the tensor size regardless of $N$, so it is bandwidth-bound, not latency-bound, for large tensors. Follow-up: on 8 GPUs with NVLink, $2 \times$ (gradient bytes) at ~450 GB/s per direction tells you the floor for one DDP step's communication; DDP hides it by bucketing and overlapping with backward.

## Q: One minute: what is tensor parallelism? {diff=1 tags=tensor-parallel,megatron,rapid-fire}
### A
**Tensor parallelism** splits individual weight matrices across GPUs so one layer's matmul runs on several devices at once. Megatron's recipe for a transformer block: split the first MLP matrix by columns and the second by rows, so each GPU computes a slice and a single **all-reduce** recombines the output; attention splits by heads the same way. That gives two all-reduces per block in forward and two in backward, on activation-sized tensors every layer, so it needs very high bandwidth — which is why TP stays **inside a node on NVLink** (TP $\le 8$). It exists because a single layer of a 70B–400B model does not fit, or is too slow, on one GPU. Trap: TP reduces per-GPU memory for weights and activations but does not touch optimizer-state replication; combine with ZeRO/FSDP for that.

## Q: One minute: what is pipeline parallelism? {diff=1 tags=pipeline-parallel,distributed,rapid-fire}
### A
**Pipeline parallelism** cuts the model into stages by depth — layers 0–15 on GPU 0, 16–31 on GPU 1, etc. — and streams micro-batches through, passing only the activations at stage boundaries (point-to-point send/recv, small compared with TP's all-reduces, so it works across nodes). It exists to fit very deep models across many GPUs with low communication. The cost is the **bubble**: stages idle while the pipeline fills and drains, with fraction $\approx \frac{p-1}{m+p-1}$ for $p$ stages and $m$ micro-batches, so you need many micro-batches per step. 1F1B interleaves forward and backward to bound activation memory; interleaved/virtual stages shrink the bubble further. Trap: the layers must be balanced in compute and the last stage often carries the extra loss-head work.

## Q: One minute: what is ZeRO / FSDP? {diff=1 tags=zero,fsdp,memory,rapid-fire}
### A
**ZeRO** (DeepSpeed) and **FSDP** (PyTorch) are data parallelism without the replication: instead of every GPU holding a full copy of parameters, gradients, and optimizer state, each GPU owns $1/N$ of them. Stage 1 shards optimizer state, stage 2 also gradients, stage 3 (= FSDP full shard) also parameters. During forward and backward each layer's weights are **all-gathered** just before use and freed after, and gradients are **reduce-scattered** instead of all-reduced. Memory per GPU drops from ~16 bytes/param to ~$16/N$, so a 10B model fits on 8 GPUs with no model-parallel code. The cost is ~1.5× the communication of plain DP (an extra all-gather in backward) and that it is latency-sensitive across nodes. Trap: activations are not sharded — you still need checkpointing or sequence parallelism for long-token video models.

## Q: What is gradient checkpointing, briefly? {diff=1 tags=activation-checkpointing,memory,rapid-fire}
### A
**Gradient (activation) checkpointing** trades compute for memory: instead of storing every intermediate activation from the forward pass for backward, you keep only the inputs to selected blocks and **recompute** the rest during backward. Applied per transformer block, memory drops from $O(\text{layers} \times \text{per-layer activations})$ to $O(\text{layers} \times \text{block inputs})$ plus one block's worth of live activations, at the cost of roughly one extra forward pass (~33% more compute). It exists because for long-token models (video DiTs at 32k+ tokens) activations, not weights, dominate memory. Practical points: wrap it at the transformer-block level (`torch.utils.checkpoint`); **selective** recomputation recomputes only the cheap, memory-heavy parts (attention softmax, norms, GELU) and keeps matmul outputs, cutting the overhead to a few percent. Trap: RNG state must be saved for dropout to recompute identically.

## Q: One minute: what is mixed-precision training? {diff=1 tags=mixed-precision,bf16,rapid-fire}
### A
**Mixed precision** runs the forward and backward matmuls in 16-bit (bf16 today, fp16 previously) for tensor-core speed and halved activation memory, while keeping an **fp32 master copy** of the weights and fp32 optimizer state so small updates ($\text{lr} \times \text{grad} \ll \text{weight}$) are not rounded away. bf16 has fp32's exponent range with 8 bits of mantissa, so it needs no loss scaling; fp16 has more mantissa but a tiny range, so gradients underflow and you need dynamic loss scaling. Per-parameter memory in mixed Adam is ~16 bytes: $2\,(\text{bf16 weight}) + 2\,(\text{bf16 grad}) + 4 + 4 + 4\,(\text{fp32 master}, m, v)$. Trap: reductions and softmax should still accumulate in fp32, and the loss/norms stay fp32 — `autocast` handles the op list. Follow-up: fp8 on Hopper adds per-tensor scaling and delayed scaling to keep the range usable.

## Q: What are a process group, rank, and world size? {diff=1 tags=torch-distributed,api,rapid-fire}
### A
In `torch.distributed`, **world size** is the total number of processes in the job (typically one per GPU), **rank** is a process's unique integer id in $0 \dots \text{world\_size} - 1$, and **local rank** is its index within its node (used to pick the CUDA device). A **process group** is a named subset of ranks that a collective runs over; the default group is all ranks, and you build sub-groups for each parallelism axis — e.g. with 512 GPUs, TP=8, PP=4, DP=16, every rank belongs to one TP group of 8, one PP group of 4, and one DP group of 16, and an all-reduce on the DP group only involves those 16 ranks. `torchrun` sets the env vars (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR) and `init_process_group("nccl")` does the rendezvous. Trap: every rank must call `new_group` for every group, even ones it is not in.

## Q: What is NCCL? {diff=1 tags=nccl,collectives,rapid-fire}
### A
**NCCL** (NVIDIA Collective Communications Library) is the GPU-native library that implements the collectives — all-reduce, all-gather, reduce-scatter, broadcast, send/recv — directly between GPU memories over NVLink, PCIe, and InfiniBand/RoCE, without staging through the host. It is the backend PyTorch uses for `init_process_group("nccl")`, and DDP, FSDP, and Megatron all sit on top of it. It exists because MPI's CPU-centric collectives could not saturate GPU interconnects; NCCL builds topology-aware ring and tree algorithms and runs them as CUDA kernels on a communication stream so they overlap with compute. Practical knowledge: NCCL_DEBUG=INFO shows the topology it picked, collectives are blocking on the GPU stream (hangs when one rank misses a call), and NCCL_ALGO/PROTO and NVLink SHARP affect large all-reduce bandwidth.

## Q: What is MFU? {diff=1 tags=mfu,throughput,rapid-fire}
### A
**MFU (Model FLOPs Utilization)** is the fraction of the hardware's peak throughput your training run actually converts into useful model FLOPs: $\text{MFU} = (\text{model FLOPs per second achieved}) / (\text{peak FLOP/s} \times \text{GPUs})$. Model FLOPs per token $\approx 6 \times \text{parameters}$ (2 forward, 4 backward) plus the attention term $12 \times \text{layers} \times \text{hidden} \times \text{sequence}$; multiply by tokens/s. It exists because "GPU utilisation 100%" says nothing about whether tensor cores are busy; MFU is the honest efficiency metric across systems. Good numbers: ~40–50% on H100 for dense LLMs; video DiTs with long sequences and heavy attention often land lower. Trap: **HFU** counts recomputation from activation checkpointing as real work, so it is higher than MFU — quote MFU when comparing, and remember bf16 peak (~990 TFLOP/s dense for H100) is the denominator, not the sparse marketing number.
