"""KV-cache attention and incremental decoding for a tiny causal transformer

A 2-layer causal transformer LM (embeddings, pre-norm blocks, LM head) is given. Implement the parts that make
autoregressive inference efficient:
  (1) causal_mask(T_new, T_past, window): boolean mask for T_new new queries against T_past + T_new keys,
      optionally restricted to a sliding window (each query sees at most `window` keys, itself included);
  (2) CausalSelfAttention.forward(x, cache, window): attend over cached keys/values plus the new ones, return the
      output AND the updated cache (append along the sequence axis, then evict all but the last `window`);
  (3) decode_incremental(model, tokens, window): feed the sequence ONE token at a time through the model with the
      cache and return logits identical to a single full forward pass (up to float error).
Absolute positions are needed for the learned positional embedding, so the model takes start_pos explicitly
(the cache length is NOT the position once eviction kicks in).

Signatures:
    causal_mask(T_new, T_past, window=None) -> bool tensor (T_new, T_past + T_new), True = may attend
    CausalSelfAttention.forward(x, cache=None, window=None) -> (out (B, T_new, D), (k, v) each (B, H, T_keep, hd))
    decode_incremental(model, tokens, window=None) -> (logits (B, T, V), caches: list of per-layer (k, v))

Constraints: torch CPU float32. tokens is int64 (B, T) with T <= model.max_len. Use F.scaled_dot_product_attention
with an explicit boolean attn_mask (or your own softmax). Never recompute K/V for cached positions.

Interview budget: 35 min

Discussion follow-ups:
  - Memory of the KV cache for a 10B model at 32k context (per layer: 2 * T * H * hd * bytes); why decode is
    memory-bandwidth bound and how GQA / MQA / MLA reduce it.
  - Sliding-window eviction loses information: what do StreamingLLM's attention sinks fix? How would you decide
    which tokens to evict adaptively (e.g. keep "surprising" or highly attended tokens)?
  - Positional encodings under eviction: absolute vs RoPE (RoPE is relative, so evicted prefixes do not shift
    the remaining tokens).
  - Causal VIDEO generation / world models: what is the cache (latent frames), why chunked prefill matters, and
    how does teacher forcing with a cache differ from inference (exposure bias, error accumulation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__implement__ = ["causal_mask", "CausalSelfAttention.forward", "decode_incremental"]


# ----------------------------------------------------------------------------- given helpers
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cache=None, window=None):
        """Causal multi-head self-attention with an optional KV cache and sliding window.

        Steps: qkv = self.qkv(x) split into (q, k, v) (in that order), each reshaped to (B, H, T_new, hd).
        If cache is not None, k = cat(cache_k, k) and v = cat(cache_v, v) along the sequence axis (dim 2).
        Attend with mask causal_mask(T_new, T_past, window) so that new query i (absolute position T_past + i)
        sees keys j with j <= T_past + i and, if window is given, j > T_past + i - window. Scale 1/sqrt(hd).
        Merge heads, apply self.proj. The returned cache holds the FULL concatenated k, v truncated to the last
        `window` positions (if window is given); no truncation otherwise.

        Args:
            x: (B, T_new, D). cache: None or (k, v) each (B, H, T_past, hd). window: None or int >= 1.
        Returns:
            (out (B, T_new, D), (k_all, v_all)) with k_all, v_all of shape (B, H, min(T_past + T_new, window), hd).
        """
        # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
        raise NotImplementedError


class Block(nn.Module):
    """Given helper: pre-norm transformer block that threads the cache through the attention."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, cache=None, window=None):
        a, cache = self.attn(self.ln1(x), cache, window)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, cache


class TinyCausalLM(nn.Module):
    """Given helper: token + learned absolute position embeddings, n_layers Blocks, final LN, LM head.

    forward(tokens, start_pos=0, caches=None, window=None) -> (logits (B, T_new, V), caches)
    `tokens` are the NEW tokens only (B, T_new) occupying absolute positions start_pos .. start_pos + T_new - 1;
    `caches` is None or a list with one (k, v) per layer.
    """

    def __init__(self, vocab=64, dim=32, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens, start_pos=0, caches=None, window=None):
        B, T = tokens.shape
        pos = torch.arange(start_pos, start_pos + T)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None]
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, None if caches is None else caches[i], window)
            new_caches.append(c)
        return self.head(self.ln_f(x)), new_caches


def make_model(seed=0, **kw):
    """Given helper: deterministic randomly-initialised TinyCausalLM in eval mode."""
    torch.manual_seed(seed)
    return TinyCausalLM(**kw).eval()


# ----------------------------------------------------------------------------- to implement
def causal_mask(T_new, T_past, window=None):
    """Boolean attention mask for T_new new queries over T_past cached keys followed by the T_new new keys.

    Query i has absolute position p = T_past + i; key j (0 <= j < T_past + T_new) is visible iff j <= p and,
    when window is not None, j > p - window (so at most `window` keys per query, including itself).

    Args:
        T_new: int >= 1. T_past: int >= 0. window: None or int >= 1.
    Returns:
        torch.bool tensor of shape (T_new, T_past + T_new); True = attend.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def decode_incremental(model, tokens, window=None):
    """Feed `tokens` one position at a time through `model` using the KV cache.

    For t = 0..T-1: logits_t, caches = model(tokens[:, t:t+1], start_pos=t, caches=caches, window=window).
    Concatenate the per-step logits along the sequence axis. Run under torch.no_grad().

    Args:
        model: TinyCausalLM. tokens: (B, T) int64. window: None or int.
    Returns:
        (logits (B, T, V) float32, caches: list of per-layer (k, v) after the last step).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
