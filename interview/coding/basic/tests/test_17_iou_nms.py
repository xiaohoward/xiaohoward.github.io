import torch


def _iou_scalar(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms_ref(boxes, scores, thr):
    """Plain Python greedy NMS (stable descending order, IoU > thr suppresses)."""
    b = boxes.tolist()
    order = sorted(range(len(b)), key=lambda i: -scores[i].item())  # sorted() is stable
    keep = []
    for i in order:
        if all(_iou_scalar(b[i], b[j]) <= thr for j in keep):
            keep.append(i)
    return keep


def _rand_boxes(g, n, scale=100.0):
    xy = torch.rand(n, 2, generator=g) * scale
    wh = torch.rand(n, 2, generator=g) * scale * 0.4
    return torch.cat([xy, xy + wh], dim=1)


def check_iou(mod):
    g = torch.Generator().manual_seed(0)
    a, b = _rand_boxes(g, 7), _rand_boxes(g, 5)
    iou = mod.box_iou(a, b)
    assert iou.shape == (7, 5)
    for i in range(7):
        for j in range(5):
            assert abs(iou[i, j].item() - _iou_scalar(a[i].tolist(), b[j].tolist())) < 1e-5, f"IoU[{i},{j}] wrong"
    # self-IoU is 1 on the diagonal
    assert torch.allclose(torch.diag(mod.box_iou(a, a)), torch.ones(7), atol=1e-6)
    # hand cases: identical, disjoint, touching edges, contained, zero-area
    boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10], [10, 0, 20, 10], [2, 2, 4, 4], [5, 5, 5, 5], [0, 0, 5, 10]],
                         dtype=torch.float32)
    m = mod.box_iou(boxes, boxes)
    assert m[0, 1] == 1.0
    assert m[0, 2] == 0.0, "touching edges must give IoU 0"
    assert abs(m[0, 3].item() - 4 / 100) < 1e-6, "contained box IoU = small area / big area"
    assert abs(m[0, 5].item() - 0.5) < 1e-6
    assert m[0, 4] == 0.0 and m[4, 4] == 0.0, "zero-area boxes must give 0 (no NaN)"
    assert torch.isfinite(m).all()
    assert mod.box_iou(torch.zeros(0, 4), boxes).shape == (0, 6)


def check_nms_hand_cases(mod):
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10], [52, 52, 58, 58]],
                         dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.9, 0.95])
    keep = mod.nms(boxes, scores, 0.5)
    assert keep.dtype == torch.long
    # ties: index 0 beats index 3 (same box, same score); 4 first (highest score); 4 suppresses 2 (IoU 36/100 < .5 -> no!)
    # IoU(2, 4) = 36/100 = 0.36 <= 0.5 so 2 survives; IoU(0,1)=81/119=0.68 -> 1 suppressed; 3 identical to 0 -> suppressed
    assert keep.tolist() == [4, 0, 2], f"expected [4, 0, 2], got {keep.tolist()}"
    assert mod.nms(boxes, scores, 0.3).tolist() == [4, 0], "with thr 0.3, box 2 (IoU .36 with 4) must be suppressed"
    # threshold equality survives
    two = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 5]], dtype=torch.float32)  # IoU exactly 0.5
    assert mod.nms(two, torch.tensor([1.0, 0.5]), 0.5).tolist() == [0, 1], "IoU == thr must NOT suppress"
    assert mod.nms(two, torch.tensor([1.0, 0.5]), 0.49).tolist() == [0]
    # thr=1 keeps everything (ordered by score), thr=0 kills every overlap
    assert mod.nms(boxes, scores, 1.0).tolist() == [4, 0, 3, 1, 2]
    assert mod.nms(boxes, scores, 0.0).tolist() == [4, 0]
    assert mod.nms(torch.zeros(0, 4), torch.zeros(0), 0.5).numel() == 0


def check_nms_random(mod):
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        n = 60 + 40 * seed
        boxes = _rand_boxes(g, n)
        scores = torch.rand(n, generator=g)
        if seed % 2:  # inject ties
            scores = (scores * 5).round() / 5
        for thr in (0.3, 0.5, 0.7):
            keep = mod.nms(boxes, scores, thr).tolist()
            ref = _nms_ref(boxes, scores, thr)
            assert keep == ref, f"seed {seed} thr {thr}: nms differs from loop reference (len {len(keep)} vs {len(ref)})"
            # sanity: kept boxes are mutually below threshold
            kb = boxes[keep]
            m = mod.box_iou(kb, kb) - torch.eye(len(keep)) * 2
            assert (m <= thr + 1e-6).all()


def check_batched_nms(mod):
    g = torch.Generator().manual_seed(7)
    n = 120
    boxes = _rand_boxes(g, n)
    scores = torch.rand(n, generator=g)
    cls = torch.randint(0, 4, (n,), generator=g)
    keep = mod.batched_nms(boxes, scores, cls, 0.5)
    assert keep.dtype == torch.long
    # equals per-class NMS merged and sorted by score
    ref = []
    for c in range(4):
        idx = (cls == c).nonzero().flatten()
        ref += idx[mod.nms(boxes[idx], scores[idx], 0.5)].tolist()
    ref = sorted(ref, key=lambda i: (-scores[i].item(), i))
    assert keep.tolist() == ref, "batched_nms must equal per-class NMS merged in descending-score order"
    # identical boxes with different classes are all kept
    same = torch.tensor([[0, 0, 10, 10]] * 3, dtype=torch.float32)
    k = mod.batched_nms(same, torch.tensor([0.3, 0.9, 0.6]), torch.tensor([0, 1, 2]), 0.5)
    assert sorted(k.tolist()) == [0, 1, 2], "boxes of different classes must never suppress each other"
    assert k.tolist() == [1, 2, 0], "kept indices must be in descending score order"
    k2 = mod.batched_nms(same, torch.tensor([0.3, 0.9, 0.6]), torch.tensor([1, 1, 1]), 0.5)
    assert k2.tolist() == [1], "same class -> normal NMS"
    assert mod.batched_nms(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long), 0.5).numel() == 0


def run(mod):
    check_iou(mod);           print("  ok  pairwise IoU vs scalar reference (incl. zero-area, touching, contained)")
    check_nms_hand_cases(mod); print("  ok  NMS hand cases (ties, threshold equality, thr 0/1)")
    check_nms_random(mod);    print("  ok  NMS matches Python loop reference on random boxes (6 seeds x 3 thr)")
    check_batched_nms(mod);   print("  ok  class-aware NMS never suppresses across classes")
