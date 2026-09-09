"""
17 — Pairwise IoU, greedy NMS, class-aware (batched) NMS

Detection post-processing from scratch: the IoU matrix between two box sets, greedy non-maximum suppression, and
the "coordinate offset" trick that runs NMS independently per class with a single call.

Signatures:
    def box_iou(a, b) -> Tensor                       (N,4), (M,4) -> (N,M)
    def nms(boxes, scores, iou_thresh) -> LongTensor  kept indices, in descending score order
    def batched_nms(boxes, scores, class_ids, iou_thresh) -> LongTensor

Constraints: torch (CPU) only; no torchvision. Boxes are float (x1, y1, x2, y2) with x2 >= x1, y2 >= y1 (area may
be 0). box_iou must be fully vectorised (no Python loop). nms may loop over the (at most N) kept boxes but must
use box_iou (vectorised) to suppress — no O(N^2) pure-Python loops. Ties in score: the lower index wins (use a
stable sort). IoU of two zero-area boxes is 0 (no NaN).

Interview budget: 20 min

Discussion follow-ups:
  * Complexity of greedy NMS: O(N^2) worst case; how would you make it fast on GPU (bitmask NMS, blocked IoU)?
  * Soft-NMS / DIoU-NMS: what changes and why? When does hard NMS hurt (crowded scenes)?
  * Set-based detectors (DETR) don't need NMS — what replaces it (Hungarian matching at training time) and what is
    the cost?
  * Why does the offset trick work, and what could go wrong if the offset is too small or coordinates are huge
    (float32 precision)?
"""
import torch

__implement__ = ["box_iou", "nms", "batched_nms"]


def box_area(boxes):
    """Given helper. boxes : (N, 4) (x1, y1, x2, y2) -> (N,) areas."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(a, b):
    """
    a : (N, 4), b : (M, 4) float boxes (x1, y1, x2, y2). Returns (N, M) IoU matrix in the dtype of a:
    iou[i, j] = |a_i ∩ b_j| / (|a_i| + |b_j| - |a_i ∩ b_j|). Intersection width/height are clamped at 0.
    If the union is 0 (both boxes zero-area) return 0, not NaN. Fully vectorised; N or M may be 0.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def nms(boxes, scores, iou_thresh):
    """
    Greedy NMS. boxes : (N, 4) float, scores : (N,) float, iou_thresh : float in [0, 1].
    Repeatedly take the highest-scoring remaining box, keep it, and discard every remaining box whose IoU with it
    is > iou_thresh (strictly greater: boxes with IoU exactly equal to the threshold survive).
    Returns a LongTensor of kept indices into `boxes`, sorted by descending score. Ties: lower index first
    (torch.sort(..., stable=True) / argsort(descending=True, stable=True)). N == 0 -> empty LongTensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError


def batched_nms(boxes, scores, class_ids, iou_thresh):
    """
    Class-aware NMS: boxes of different classes never suppress each other.
    boxes : (N, 4), scores : (N,), class_ids : (N,) int tensor, iou_thresh : float.
    Implement with the offset trick: shift each box by class_id * (max_coord + 1) so boxes of different classes
    are disjoint, then run a single `nms`. Returns kept indices into the ORIGINAL boxes, sorted by descending score
    (across all classes). N == 0 -> empty LongTensor.
    """
    # TODO(candidate): implement. Read the docstring above; run the tests with ../run.py
    raise NotImplementedError
