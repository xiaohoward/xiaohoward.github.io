import copy

import torch
import torch.nn as nn

# tiny tensors: intra-op threading is pure overhead (and can be 10x slower on many-core boxes)
torch.set_num_threads(min(torch.get_num_threads(), 8))


def check_init_equals_base(mod):
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = mod.LoRALinear(copy.deepcopy(base), r=4, alpha=8.0)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6), "at init the LoRA layer must equal the base layer"
    assert lora.lora_A.shape == (4, 16) and lora.lora_B.shape == (24, 4), "A is (r, in), B is (out, r)"
    assert torch.all(lora.lora_B == 0), "B must be zero-init"
    assert lora.lora_A.abs().sum() > 0, "A must not be zero-init"
    assert not lora.base.weight.requires_grad and not lora.base.bias.requires_grad, "base must be frozen"
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad
    assert abs(lora.scaling - 2.0) < 1e-9, "scaling = alpha / r"
    # forward formula with nonzero B
    with torch.no_grad():
        lora.lora_B.normal_()
    ref = base(x) + 2.0 * (x @ lora.lora_A.T) @ lora.lora_B.T
    assert torch.allclose(lora(x), ref, atol=1e-5), "forward must be base(x) + (alpha/r) B A x"
    # 3-D inputs (tokens)
    assert lora(torch.randn(2, 7, 16)).shape == (2, 7, 24)


def check_inject_and_filter(mod):
    model = mod.TinyModel(d=32, n_layers=2)
    names = mod.inject_lora(model, r=4, alpha=4.0, name_filter=lambda n: n.endswith(".q") or n.endswith(".v"))
    assert names == ["blocks.0.attn.q", "blocks.0.attn.v", "blocks.1.attn.q", "blocks.1.attn.v"], names
    assert isinstance(model.blocks[0].attn.q, mod.LoRALinear) and isinstance(model.blocks[0].attn.k, nn.Linear)
    assert not isinstance(model.blocks[0].attn.k, mod.LoRALinear) and not isinstance(model.head, mod.LoRALinear)
    # every linear when no filter; already-wrapped layers are not re-wrapped
    model2 = mod.TinyModel(d=32, n_layers=2)
    names2 = mod.inject_lora(model2, r=2, alpha=2.0)
    assert len(names2) == 2 * 6 + 1, f"expected 13 linears, got {len(names2)}"
    names3 = mod.inject_lora(model2, r=2, alpha=2.0)
    assert names3 == [] and all(not isinstance(m.base, mod.LoRALinear) for m in model2.modules()
                                if isinstance(m, mod.LoRALinear)), "must not wrap a LoRALinear twice"
    # model output unchanged by injection
    ref = mod.TinyModel(d=32, n_layers=2)
    x = torch.randn(3, 5, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), ref(x), atol=1e-6), "injection must not change the function at init"


def check_trainable_count(mod):
    model = mod.TinyModel(d=32, n_layers=2)
    total = sum(p.numel() for p in model.parameters())
    assert mod.count_trainable_params(model) == total
    r = 4
    names = mod.inject_lora(model, r=r, alpha=8.0, name_filter=lambda n: ".attn." in n or ".mlp." in n)
    # everything not wrapped must be frozen by the caller for a real fine-tune; here only wrapped layers matter:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)
    expected = 0
    for n in names:
        lin = model.get_submodule(n).base
        expected += r * (lin.in_features + lin.out_features)
    assert mod.count_trainable_params(model) == expected, \
        f"trainable params {mod.count_trainable_params(model)} != sum r(in+out) = {expected}"
    assert expected == 2 * (4 * r * 64 + r * (32 + 128) * 2)
    assert expected < 0.2 * total, "LoRA should be a small fraction of the parameters"


def check_training_touches_only_lora(mod):
    torch.manual_seed(0)
    model = mod.TinyModel(d=32, n_layers=2)
    mod.inject_lora(model, r=4, alpha=8.0, name_filter=lambda n: n.endswith(".q") or n.endswith(".fc1"))
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(8, 5, 32)
    target = torch.randn(8, 5, 32)
    losses = []
    for _ in range(30):
        loss = ((model(x) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1] < losses[0], "LoRA training should reduce the loss"
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert not torch.allclose(p, before[n]), f"{n} should have changed"
        else:
            assert torch.equal(p, before[n]), f"base param {n} must not change"


def check_merge(mod):
    torch.manual_seed(0)
    model = mod.TinyModel(d=32, n_layers=2)
    mod.inject_lora(model, r=4, alpha=8.0)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, mod.LoRALinear):
                m.lora_B.normal_(0, 0.1)
    x = torch.randn(4, 6, 32)
    with torch.no_grad():
        y_unmerged = model(x)
        for m in model.modules():
            if isinstance(m, mod.LoRALinear):
                m.merge_weights()
        y_merged = model(x)
        assert torch.allclose(y_unmerged, y_merged, atol=1e-5), "merged output must equal unmerged"
        # merging twice must not double-apply
        w = model.blocks[0].attn.q.base.weight.clone()
        model.blocks[0].attn.q.merge_weights()
        assert torch.equal(model.blocks[0].attn.q.base.weight, w), "merge must be idempotent"
        assert torch.allclose(model(x), y_merged, atol=1e-6)
        # after merge the adapter path is not applied on top (zeroing A must not change the output)
        for m in model.modules():
            if isinstance(m, mod.LoRALinear):
                m.lora_A.zero_()
        assert torch.allclose(model(x), y_merged, atol=1e-6), "after merge, forward must use base only"
    # merged weight equals W + scaling * B A (checked on a standalone layer)
    base = nn.Linear(8, 6)
    W0 = base.weight.detach().clone()
    l = mod.LoRALinear(base, r=2, alpha=4.0)
    with torch.no_grad():
        l.lora_B.normal_()
        l.merge_weights()
    assert torch.allclose(l.base.weight, W0 + 2.0 * l.lora_B @ l.lora_A, atol=1e-6), "merge formula"


def check_dropout(mod):
    torch.manual_seed(0)
    l = mod.LoRALinear(nn.Linear(8, 8), r=2, alpha=2.0, dropout=0.5)
    with torch.no_grad():
        l.lora_B.normal_()
    x = torch.randn(64, 8)
    l.train()
    a, b = l(x), l(x)
    assert not torch.allclose(a, b), "dropout must be stochastic in train mode"
    l.eval()
    assert torch.allclose(l(x), l(x)), "eval mode must be deterministic"
    with torch.no_grad():
        assert torch.allclose(l(x), l.base(x) + 1.0 * (x @ l.lora_A.T) @ l.lora_B.T, atol=1e-5), "dropout only on the LoRA path"


def run(mod):
    check_init_equals_base(mod); print("  ok  LoRALinear init == base, forward formula")
    check_inject_and_filter(mod); print("  ok  inject_lora with name filter (in place, no double wrap)")
    check_trainable_count(mod); print("  ok  trainable params = r (in + out) per layer")
    check_training_touches_only_lora(mod); print("  ok  training changes only LoRA params")
    check_merge(mod); print("  ok  merge_weights (exact, idempotent)")
    check_dropout(mod); print("  ok  dropout on the adapter path")
