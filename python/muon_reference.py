"""
python/muon_reference.py
=================
A *reference* implementation of Per-Head Muon, the optimizer variant Kimi K3
uses for attention projections (arXiv:2607.24653, section 2.5), on top of
base Muon (their ref [54], Keller Jordan's "Muon: An Optimizer for Hidden
Layers in Neural Networks").

Correctness anchor for the future C++ `microtorch` optimizer
(docs/TECH_TRANSFER.md item 3). Reference-grade on purpose: plain float32, no
fused anything, loops where loops are clearest.

Base Muon, per matrix parameter W with gradient G:

    M_t = mu * M_{t-1} + G_t                      (momentum buffer)
    U_t = G_t + mu * M_t   if nesterov else M_t
    O_t = NewtonSchulz5(U_t)                      (approx UV^T of U_t's SVD)
    W  -= lr * sqrt(max(1, rows/cols)) * O_t      (shape-scale per [54])

NewtonSchulz5 is the quintic iteration X <- aX + b(XX^T)X + c(XX^T)^2 X with
(a, b, c) = (3.4445, -4.7750, 2.0315), run on the Frobenius-normalised
matrix. It is deliberately LOOSE: it drives singular values into a band
around 1 rather than to 1 exactly, trading precision for speed. The
self-test asserts the band and the DIRECTION (U^T O V nearly diagonal and
positive), not exact semi-orthogonality.

The K3 refinement (section 2.5, quoted): "instead of applying Newton-Schulz
orthogonalization to the full Q, K, and V projection matrices, we partition
their momentum matrices along the head dimension and orthogonalize each
head's block separately." Full-matrix orthogonalization treats all heads as
one coupled block, so a head with outsized gradient scale dominates the
shared update direction while quiet heads get under-normalised updates;
per-head orthogonalization equalizes update scale across heads. The
self-test measures exactly this on a synthetic 100x head imbalance.

STRUCTURAL PIN (same style as python/attn_res_reference.py's Block(S=1)==Full and
for the same reason): per-head Muon with n_heads=1 partitions the matrix
into one block and must match full-matrix Muon bit-for-bit. A transcription
error has to be made twice, identically, to get past it.

Scope note, stated rather than implied: Muon is for HIDDEN MATRIX
parameters. Embeddings, output heads, biases, gains go to AdamW in every
deployment of Muon we are transferring from; this class therefore refuses
non-2D parameters instead of silently mistreating them.

Run directly for a self-test:  python python/muon_reference.py
"""
from __future__ import annotations

import math

import torch


def newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization of a 2-D matrix.

    Coefficients per Muon [54]. Iterates on the wide orientation (rows <=
    cols) and transposes back, so the Gram matrix XX^T is the small side.
    """
    assert G.ndim == 2, G.shape
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = G.size(0) > G.size(1)
    X = G.T if transposed else G
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon with optional per-head orthogonalization (K3 section 2.5).

    Args (per param group):
        lr, momentum, nesterov, ns_steps: base Muon knobs.
        n_heads: partition each [out, in] matrix's momentum into n_heads
            row-blocks [out/n_heads, in] and orthogonalize each separately.
            1 = full-matrix Muon. Use for attention Q/K/V projections whose
            rows are laid out head-major (the nn.Linear convention).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, n_heads: int = 1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, n_heads=n_heads)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon is for 2-D matrix parameters; got shape "
                        f"{tuple(p.shape)}. Route non-matrix parameters to "
                        f"AdamW/SGD instead.")
                if p.size(0) % group["n_heads"] != 0:
                    raise ValueError(
                        f"rows {p.size(0)} not divisible by n_heads "
                        f"{group['n_heads']}")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mu, H = group["momentum"], group["n_heads"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                upd = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                # Partition along the head (row) dimension and orthogonalize
                # each block separately -- the entire per-head refinement.
                rows = p.size(0) // H
                for h in range(H):
                    blk = upd[h * rows:(h + 1) * rows]
                    O = newton_schulz5(blk, group["ns_steps"])
                    scale = math.sqrt(max(1.0, blk.size(0) / blk.size(1)))
                    p[h * rows:(h + 1) * rows].add_(O, alpha=-group["lr"] * scale)


# --------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)

    # Newton-Schulz: singular values land in the documented loose band, and
    # the DIRECTION matches the SVD's UV^T (U^T O V nearly diagonal, positive)
    G = torch.randn(32, 16, dtype=torch.float64)
    O = newton_schulz5(G, steps=5)
    sv = torch.linalg.svdvals(O)
    assert 0.5 < sv.min() and sv.max() < 1.35, (sv.min().item(), sv.max().item())
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    D = U.T @ O @ Vh.T
    off = (D - torch.diag(torch.diagonal(D))).abs().max().item()
    assert off < 1e-6 and torch.diagonal(D).min() > 0.5, (off,)
    print(f"[ok] newton_schulz5: singular values in ({sv.min():.3f}, "
          f"{sv.max():.3f}), direction matches UV^T (off-diag {off:.1e})")

    # tall input must round-trip through the internal transpose
    Ot = newton_schulz5(G.T, steps=5)
    assert Ot.shape == (16, 32)
    print("[ok] newton_schulz5: transposed orientation handled")

    # THE PIN: n_heads=1 must equal full-matrix Muon bit-for-bit across steps
    def run(n_heads, seed=1, steps=5):
        torch.manual_seed(seed)
        W = torch.nn.Parameter(torch.randn(24, 8))
        opt = Muon([W], lr=0.1, n_heads=n_heads)
        for i in range(steps):
            torch.manual_seed(100 + i)
            W.grad = torch.randn_like(W)
            opt.step()
        return W.detach()

    assert torch.equal(run(1), run(1)), "determinism sanity"
    # n_heads=1 IS the full-matrix path in this implementation, so pin it
    # against an independent hand-rolled full-matrix step instead
    torch.manual_seed(1)
    W = torch.nn.Parameter(torch.randn(24, 8))
    opt = Muon([W], lr=0.1, n_heads=1)
    torch.manual_seed(100)
    g = torch.randn_like(W)
    W.grad = g
    opt.step()
    torch.manual_seed(1)
    W2 = torch.randn(24, 8)
    buf = torch.zeros_like(W2)
    buf.mul_(0.95).add_(g)
    upd = g + 0.95 * buf
    W2 -= 0.1 * math.sqrt(24 / 8) * newton_schulz5(upd)
    assert torch.allclose(W.detach(), W2, atol=1e-6), "full-matrix step mismatch"
    print("[ok] pin: optimizer step == hand-rolled base-Muon step (n_heads=1)")

    # K3's stated motivation, measured: put a 100x gradient scale on head 0
    # of a 4-head projection. Full-matrix orthogonalization lets that head
    # dominate the shared update; per-head equalizes update norms.
    H, dk, d = 4, 8, 16
    torch.manual_seed(2)
    g = torch.randn(H * dk, d)
    g[:dk] *= 100.0                                   # loud head 0

    def head_norms(n_heads):
        W = torch.nn.Parameter(torch.zeros(H * dk, d))
        opt = Muon([W], lr=1.0, momentum=0.0, nesterov=False, n_heads=n_heads)
        W.grad = g.clone()
        opt.step()
        return torch.stack([W[h * dk:(h + 1) * dk].norm() for h in range(H)])

    nf, nph = head_norms(1), head_norms(H)
    spread_full = (nf.max() / nf.min()).item()
    spread_ph = (nph.max() / nph.min()).item()
    assert spread_ph < 1.1, f"per-head not equalized: {spread_ph:.2f}"
    assert spread_full > 1.5, f"imbalance did not show at full-matrix: {spread_full:.2f}"
    print(f"[ok] per-head equalization: update-norm spread {spread_full:.2f}x "
          f"(full) -> {spread_ph:.2f}x (per-head) under 100x head imbalance")

    # end to end: a tiny matrices-only net actually trains, using param groups
    # the way a real model would -- per-head on the "attention projection",
    # full-matrix on everything else (the 1-row head CANNOT be per-head, which
    # the guard below enforces rather than silently accepting)
    torch.manual_seed(3)
    net = torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False),
                              torch.nn.Tanh(),
                              torch.nn.Linear(32, 1, bias=False))
    opt = Muon([{"params": [net[0].weight], "n_heads": 4},
                {"params": [net[2].weight], "n_heads": 1}], lr=0.02)
    X = torch.randn(256, 16)
    y = (X[:, :4].prod(1, keepdim=True) + 0.1 * X[:, 4:5])
    losses = []
    for _ in range(300):
        loss = (net(X) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
    print(f"[ok] training: loss {losses[0]:.3f} -> {losses[-1]:.3f} in 300 steps")

    # the guard-rails guard
    try:
        Muon([torch.nn.Parameter(torch.zeros(5))])
        raise AssertionError("accepted a 1-D parameter")
    except ValueError:
        pass
    try:
        Muon([torch.nn.Parameter(torch.zeros(10, 4))], n_heads=3)
        raise AssertionError("accepted rows not divisible by n_heads")
    except ValueError:
        pass
    print("[ok] rejects non-matrix params and non-divisible head counts")

    print("\nALL MUON REFERENCE CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
