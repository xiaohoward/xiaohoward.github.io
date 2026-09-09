"""
13 — LLM decoding: temperature, top-k, top-p (nucleus) sampling

Implement the standard logits post-processing pipeline used by every LLM decoder, and the final sampling step
with a torch.Generator so it is reproducible.

Signatures:
    def apply_temperature(logits, temperature) -> Tensor
    def top_k_filter(logits, k) -> Tensor              # disallowed entries set to -inf
    def top_p_filter(logits, p) -> Tensor              # disallowed entries set to -inf
    def sample(logits, temperature=1.0, top_k=None, top_p=None, generator=None) -> LongTensor (B,)

Constraints: torch (CPU) only; logits are (B, V) float. Filtering must be batched (no Python loop over B).
Order of operations in sample(): temperature -> top-k -> top-p -> softmax -> multinomial (HF convention).
Use torch.multinomial(probs, 1, generator=generator).

Interview budget: 20 min

Discussion follow-ups:
  * Why do top-k and top-p differ on flat vs peaked distributions? When does top-p reduce to greedy?
  * Temperature -> 0 gives argmax; how do you implement that without dividing by zero?
  * Repetition penalty, min-p, typical sampling, beam search: where would each plug into this pipeline?
  * Speculative decoding: the accept/reject rule needs the draft and target probs AFTER this filtering — why
    must both models use the same filtering to keep the output distribution exact?
"""
import torch

__implement__ = ["apply_temperature", "top_k_filter", "top_p_filter", "sample"]


def apply_temperature(logits, temperature):
    """
    logits : (B, V) float.  temperature : float > 0.
    Returns logits / temperature (new tensor). Raise ValueError if temperature <= 0 (temperature 0 is handled
    by the caller as greedy).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def top_k_filter(logits, k):
    """
    logits : (B, V) float.  k : int >= 1.
    Returns a copy of logits where every entry that is NOT among the k largest of its row is set to -inf.
    Ties at the k-th largest value are all kept (threshold rule: keep logit >= k-th largest value).
    k >= V -> unchanged copy. Raise ValueError if k < 1.
    """
    if k < 1:
        raise ValueError("top_k must be >= 1")
    k = min(k, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[:, -1:]            # (B, 1)
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    """
    logits : (B, V) float (may already contain -inf).  p : float in (0, 1].
    Nucleus filtering: sort each row descending, take probs = softmax(logits), and keep the smallest prefix whose
    cumulative probability reaches p. Precisely: with cum_before[j] = sum of probs of tokens ranked above j,
    keep token j iff cum_before[j] < p. This always keeps the top-1 token, keeps the token that crosses p, and
    p == 1.0 keeps everything with non-zero probability. Everything else -> -inf. Returns a new (B, V) tensor.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_before = probs.cumsum(dim=-1) - probs
    remove_sorted = cum_before >= p
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def sample(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """
    logits : (B, V) float.  temperature : float >= 0 (0 -> greedy argmax, no filtering needed).
    top_k : optional int;  top_p : optional float;  generator : optional torch.Generator.
    Pipeline: temperature -> top_k_filter (if given) -> top_p_filter (if given) -> softmax -> multinomial(1).
    Returns LongTensor of shape (B,) with the sampled token ids (always an index whose filtered logit is finite).
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    x = apply_temperature(logits, temperature)
    if top_k is not None:
        x = top_k_filter(x, top_k)
    if top_p is not None:
        x = top_p_filter(x, top_p)
    probs = torch.softmax(x, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
