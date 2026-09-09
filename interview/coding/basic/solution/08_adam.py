"""
08 — Adam / AdamW optimizer from scratch

Implement the Adam optimizer (Kingma & Ba 2015) with bias correction, operating on a list of
torch parameter tensors that already hold `.grad`. Support two weight-decay modes:
  * "l2"        : classic Adam — decay is folded into the gradient, g <- g + wd * p   (torch.optim.Adam)
  * "decoupled" : AdamW (Loshchilov & Hutter) — p <- p * (1 - lr * wd) BEFORE the Adam update
                  (torch.optim.AdamW)

Signatures:
    class Adam:
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                     decoupled=False)
        def step(self)         # one update of every param in place, using p.grad
        def zero_grad(self)    # given

Constraints: torch (CPU) only. Do the update under torch.no_grad(). Your numbers must match
torch.optim.Adam / AdamW to 1e-6 after several steps, so follow the exact update order:
    m_t = b1 m_{t-1} + (1-b1) g ;  v_t = b2 v_{t-1} + (1-b2) g^2
    m_hat = m_t / (1-b1^t) ;  denom = sqrt(v_t) / sqrt(1-b2^t) + eps
    p <- p - lr * m_hat / denom
(eps is added AFTER the sqrt and the bias correction of v.)

Interview budget: 15 min

Discussion follow-ups:
  * Why bias correction? What happens to the first step without it (m ~ 0.1 g, v ~ 0.001 g^2 -> step ~ 3 lr)?
  * L2 vs decoupled weight decay: why does L2 decay get "divided by sqrt(v)" and therefore barely regularise
    weights with large gradients? Why does AdamW generalise better for transformers?
  * Memory: Adam keeps 2 fp32 states per parameter (m, v). For a 10B-param model with mixed precision, how many
    bytes per parameter live on each GPU (weights fp16 + master fp32 + m + v + grad)? Where does ZeRO help?
  * eps placement: inside vs outside the sqrt (Adam vs "epsilon-hat" variants); why eps=1e-8 can be a problem in
    fp16 / with tiny gradients.
"""
import torch

__implement__ = ["Adam.__init__", "Adam.step"]


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        """
        params       : iterable of torch.Tensor (leaf tensors, requires_grad=True). Store them as a list.
        lr, betas, eps, weight_decay : standard Adam hyper-parameters.
        decoupled    : False -> classic Adam with L2 decay folded into the gradient (torch.optim.Adam)
                       True  -> AdamW, decay applied multiplicatively to the weights before the update.

        State to keep per parameter: first moment m (zeros_like p), second moment v (zeros_like p), and a
        shared integer step counter t (starts at 0, incremented at the start of each step()).
        """
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.decoupled = decoupled
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        """
        Perform one Adam / AdamW update IN PLACE on every parameter whose .grad is not None.
        Parameters with .grad is None are skipped (their m, v and the shared counter still advance the counter
        only once per call). Must be decorated / wrapped with torch.no_grad(). Returns None.
        """
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd != 0.0:
                if self.decoupled:
                    p.mul_(1.0 - self.lr * self.wd)
                else:
                    g = g + self.wd * p
            m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            denom = (v.sqrt() / (bc2 ** 0.5)).add_(self.eps)
            p.addcdiv_(m, denom, value=-self.lr / bc1)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
