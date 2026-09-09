"""
31 — Causal LM next-token loss, padding / ignore_index, perplexity, label smoothing, sequence packing

The training objective of every decoder-only LLM (and of the text encoders / captioners in multimodal stacks).
Given logits (B, L, V) and token ids (B, L), the model at position t predicts token t+1: shift, mask out padding
and ignored labels, reduce per sequence or per batch, report perplexity. Then the data-side helper: pack
variable-length documents into fixed blocks and build the block-diagonal causal mask that stops attention leaking
across documents.

Signatures:
    def next_token_nll(logits, labels, attention_mask=None, ignore_index=-100, label_smoothing=0.0) -> (nll, valid)
    def sequence_loss(nll, valid) -> Tensor                          (B,) mean over valid tokens of each sequence
    def batch_loss(nll, valid, reduction="token") -> Tensor           0-dim, "token" or "sequence" weighting
    def perplexity(nll, valid) -> float                               exp(token-weighted mean nll)
    def pack_sequences(seqs, block_size, eos_id, pad_id) -> (tokens, sequence_ids)   both (num_blocks, block_size)
    def packed_causal_mask(sequence_ids) -> BoolTensor                (num_blocks, block_size, block_size)

Constraints: torch (CPU) only. Use F.cross_entropy(reduction="none") (or log_softmax + gather) — do not loop over
tokens. Masks: attention_mask is 1 for real tokens and 0 for padding (either side). Position t of the shifted
output is valid iff labels[t+1] != ignore_index AND attention_mask[t] == 1 AND attention_mask[t+1] == 1 (context
and target both real). Invalid positions carry nll = 0.

Interview budget: 25 min

Discussion follow-ups:
  * Token-weighted vs sequence-weighted batch loss: which one does HF Trainer use with gradient accumulation, and
    why did that produce a loss mismatch across accumulation steps (the 2024 "grad accum bug")?
  * Perplexity is only comparable at the same tokenizer. Why? What is bits-per-byte and when do you prefer it?
  * Packing without the block-diagonal mask (the "naive packing" most pretraining code used): what does the model
    learn to do at document boundaries, and what does FlashAttention's varlen API give you instead of a mask?
  * Label smoothing hurts calibration of the next-token distribution (and thus sampling temperature behavior).
    Why is it common in MT but rare in LLM pretraining?
"""
import torch
import torch.nn.functional as F

__implement__ = ["next_token_nll", "sequence_loss", "batch_loss", "perplexity", "pack_sequences", "packed_causal_mask"]


def next_token_nll(logits, labels, attention_mask=None, ignore_index=-100, label_smoothing=0.0):
    """
    logits : (B, L, V) float, labels : (B, L) int64 token ids (same tensor as the input ids for a causal LM; may
    contain ignore_index, e.g. on prompt tokens in SFT), attention_mask : (B, L) 0/1 (int or bool) or None.
    Shift: prediction at position t (logits[:, t]) is scored against labels[:, t+1], for t = 0..L-2.
    Returns (nll, valid), both (B, L-1):
        valid[b, t] = labels[b, t+1] != ignore_index  AND  attention_mask[b, t] == 1  AND  attention_mask[b, t+1] == 1
        nll[b, t]   = cross-entropy of logits[b, t] against labels[b, t+1] where valid, else 0.0.
    label_smoothing eps: nll = (1 - eps) * (-log p_y) + eps * mean_v(-log p_v)   (the F.cross_entropy convention).
    Must not index the logits with ignore_index (replace ignored labels by 0 before gathering, then mask).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def sequence_loss(nll, valid):
    """
    nll, valid : (B, T). Returns (B,) = sum_t nll * valid / max(sum_t valid, 1)  (a sequence with no valid token
    gets 0, not nan).
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def batch_loss(nll, valid, reduction="token"):
    """
    nll, valid : (B, T). reduction:
        "token"    -> sum of all valid nll / number of valid tokens in the batch (every token weighs the same; this
                      is what F.cross_entropy(reduction="mean", ignore_index=...) on the flattened batch gives)
        "sequence" -> mean over sequences of sequence_loss (every sequence weighs the same regardless of length)
    Returns a 0-dim tensor. Raises ValueError for any other reduction string.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def perplexity(nll, valid):
    """
    nll, valid : (B, T). Returns exp(token-weighted mean nll) as a python float. A uniform model over V tokens has
    perplexity exactly V.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def pack_sequences(seqs, block_size, eos_id, pad_id):
    """
    seqs : list of python lists of token ids (variable length, none contains eos/pad). Each document becomes
    seq + [eos_id] (len(seq) + 1 <= block_size, else raise ValueError). Greedy in-order packing: append the document
    to the current block if it fits entirely, otherwise pad the current block with pad_id and start a new block.
    Documents are never split across blocks.
    Returns (tokens, sequence_ids), both int64 (num_blocks, block_size):
        tokens[b, i]       = token id (pad_id in padding)
        sequence_ids[b, i] = global index of the document occupying position i (0, 1, 2, ... in input order),
                             or -1 for padding.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def packed_causal_mask(sequence_ids):
    """
    sequence_ids : (num_blocks, block_size) int64 from pack_sequences (-1 = padding).
    Returns allowed : (num_blocks, block_size, block_size) bool, True where query position i may attend key j:
        allowed[b, i, j] = (j <= i) AND sequence_ids[b, i] == sequence_ids[b, j] AND sequence_ids[b, i] != -1
    i.e. block-diagonal within each document, causal inside the block, padding queries attend to nothing (the
    caller must not take a loss there). Vectorized (broadcast compare), no loops.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def make_batch(B=3, L=10, V=50, seed=0, pad_id=0):
    """Given helper: random logits + right-padded labels with an attention mask (lengths L, L-3, L-6)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, L, V, generator=g)
    labels = torch.randint(1, V, (B, L), generator=g)
    mask = torch.ones(B, L, dtype=torch.long)
    for b in range(B):
        n = L - 3 * b
        labels[b, n:] = pad_id
        mask[b, n:] = 0
    return logits, labels, mask
