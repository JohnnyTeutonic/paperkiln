"""
python/attn_res_reference.py
=====================
A faithful *reference* implementation of Attention Residuals (AttnRes), the
depth-attention mechanism used by Kimi K3 (arXiv:2607.24653, section 2.2,
Eq. 8-10; original method is their ref [58]).

This is the correctness anchor for the future C++ `microtorch` module and the
`dit` backbone (see docs/TECH_TRANSFER.md item 1). Like kda_reference.py it
implements the definitional form and treats efficiency (online-softmax
merging of inter/intra block terms) as out of scope.

The idea. A standard residual stream compresses every earlier layer into one
running sum -- an RNN over depth. AttnRes replaces that accumulation with
attention over depth: layer l's INPUT is an attention-weighted mixture of the
token embedding and all preceding layer outputs, under a layer-specific
learnable pseudo-query.

Full form (Eq. 8-9), sources indexed i = 0..l-1:

    k_i = v_i = h1           if i = 0        (token embedding)
                f_i(h_i)     if 1 <= i < l   (output of layer i)

    phi(q, k)  = exp(q^T RMSNorm(k))         (RMSNorm on KEYS ONLY --
                                              values are mixed raw)
    alpha_i->l = phi(w_l, k_i) / sum_j phi(w_l, k_j)
    h_l        = sum_i alpha_i->l * v_i

Block form (Eq. 10). Layers are partitioned into blocks of S layers; within
a block, outputs are reduced by SUMMATION into b_n (b_n^i = partial sum over
the block's first i layers, b_0 = h1); across blocks, attention runs over the
block representations only. For the i-th layer of block n the source list is

    V = [b_0, ..., b_{n-1}]              if i = 1
        [b_0, ..., b_{n-1}, b_n^{i-1}]   if i >= 2

and the final output aggregates all block representations. Memory drops from
O(L d) to O(N d); the paper reports N ~ 8 blocks recovers most of the benefit.

STRUCTURAL FACT the self-test leans on: with S = 1 every block IS one layer,
each b_n = f_n(h_n), the partial-sum branch never fires, and Eq. 10 reduces
token-for-token to Eq. 8. Block(S=1) must therefore equal Full exactly; this
pins the two implementations against each other, so a transcription error
must be made twice, identically, to pass.

DOCUMENTED CHOICE -- pseudo-query init. The report does not give w_l's
initialisation (it lives in their ref [58], which we are not reading from
here; standing rule: no training-corpus citations). We init w_l = 0, which
makes every depth-attention uniform at step 0 -- the stack starts as "average
all sources", the closest AttnRes analogue of a plain residual stream -- and
the gradient through softmax is nonzero there, which the self-test asserts.

FINDING FROM THE C++ PORT (2026-07-31, test_attn_res.cpp): the FIRST
layer's pseudo-query row (w[0]) is structurally gradient-dead — its source
list is always the single embedding, and softmax over a singleton is
constant regardless of the query. This implementation cannot see that:
the dead-check below sums |grad| over the WHOLE [L+1, d] parameter, and
rows 1..L carry gradient, so the dead row hides inside a live tensor.
The C++ port registers queries individually, caught it, and does not
allocate the dead parameter at all (layer 0 reads the embedding
directly, which is what the math already does).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMSNorm, applied to depth-attention KEYS (Eq. 9)."""
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _depth_attend(w: torch.Tensor, sources: list[torch.Tensor]) -> torch.Tensor:
    """Eq. 9 over an explicit source list.

    w: [d] pseudo-query; sources: list of [..., d] tensors (embedding first).
    Softmax over the LIST dimension; keys RMSNormed, values raw.
    """
    keys = torch.stack([_rmsnorm(s) for s in sources], dim=0)   # [S, ..., d]
    vals = torch.stack(sources, dim=0)                          # [S, ..., d]
    logits = (keys * w).sum(-1)                                 # [S, ...]
    alpha = torch.softmax(logits, dim=0)
    return (alpha.unsqueeze(-1) * vals).sum(0)                  # [..., d]


class FullAttnRes(nn.Module):
    """Eq. 8-9: every layer attends over the embedding + all prior outputs.

    `layers` are arbitrary d->d modules (the f_i). A final pseudo-query
    aggregates all sources into the stack's output representation, per "the
    final output layer then aggregates" in the report.
    """

    def __init__(self, layers: list[nn.Module], d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        # one pseudo-query per layer + one for the final aggregation
        self.w = nn.Parameter(torch.zeros(len(layers) + 1, d_model))

    def forward(self, h1: torch.Tensor) -> torch.Tensor:
        sources = [h1]
        for l, f in enumerate(self.layers):
            h_l = _depth_attend(self.w[l], sources)
            sources.append(f(h_l))
        return _depth_attend(self.w[-1], sources)


class BlockAttnRes(nn.Module):
    """Eq. 10: sum within blocks of `block_size` layers, attend across blocks.

    The source list for the i-th layer of block n is the banked block
    representations [b_0..b_{n-1}] plus, from the block's second layer on,
    the running partial sum b_n^{i-1}. The final aggregation runs over all
    banked blocks (including the last block's completed sum).
    """

    def __init__(self, layers: list[nn.Module], d_model: int, block_size: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.block_size = block_size
        self.w = nn.Parameter(torch.zeros(len(layers) + 1, d_model))

    def forward(self, h1: torch.Tensor) -> torch.Tensor:
        banked = [h1]                       # b_0 = h1: embedding always a source
        partial = None                      # b_n^{i-1}, None at a block boundary
        for l, f in enumerate(self.layers):
            sources = banked if partial is None else banked + [partial]
            h_l = _depth_attend(self.w[l], sources)
            out = f(h_l)
            partial = out if partial is None else partial + out
            if (l + 1) % self.block_size == 0 or l == len(self.layers) - 1:
                banked.append(partial)      # block complete (or partial final
                partial = None              # block, as in K3's 9-block layout)
        return _depth_attend(self.w[-1], banked)


# --------------------------------------------------------------------------
def _mlp(d: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))


def _stacks(L: int, d: int, block_size: int):
    """A Full and a Block stack over IDENTICAL layer weights."""
    full = FullAttnRes([_mlp(d, 100 + i) for i in range(L)], d)
    blk = BlockAttnRes([_mlp(d, 100 + i) for i in range(L)], d, block_size)
    blk.load_state_dict(full.state_dict())
    return full, blk


def _selftest():
    torch.manual_seed(0)
    B, T, d, L = 2, 5, 16, 6
    x = torch.randn(B, T, d)

    # shape + uniform-at-init: with w = 0 every depth-attention is uniform,
    # so h_1 (one source) must equal the embedding exactly
    full, blk = _stacks(L, d, block_size=2)
    y = full(x)
    assert y.shape == (B, T, d), y.shape
    a = torch.softmax((torch.stack([_rmsnorm(x)]) * full.w[0]).sum(-1), 0)
    assert torch.allclose(a, torch.ones_like(a)), "not uniform at init"
    print(f"[ok] forward shape {tuple(y.shape)}; depth-attention uniform at init")

    # THE EQUIVALENCE PIN: Block with S=1 is definitionally Full (every block
    # is one layer, the partial-sum branch never fires)
    full1, blk1 = _stacks(L, d, block_size=1)
    diff = (full1(x) - blk1(x)).abs().max().item()
    assert diff < 1e-6, f"Block(S=1) != Full: {diff}"
    print(f"[ok] Block(S=1) == Full exactly (max diff {diff:.2e})")

    # and they must DISAGREE for S>1 (summing is not attending); if they ever
    # agree the block path is silently running the full path
    diff2 = (full(x) - blk(x)).abs().max().item()
    assert diff2 > 1e-6, "Block(S=2) identical to Full: block path inert?"
    print(f"[ok] Block(S=2) differs from Full as it must (max diff {diff2:.2e})")

    # partial final block (K3: 8x12 layers -> partial 9th block): L=5, S=2
    full5, blk5 = _stacks(5, d, block_size=2)
    assert blk5(x).shape == (B, T, d)
    print("[ok] partial final block handled (L=5, S=2)")

    # every parameter -- pseudo-queries included -- gets nonzero grad at init.
    # w = 0 is exactly where a softmax gradient COULD die; assert it does not.
    for name, stack in (("full", full), ("block", blk)):
        stack.zero_grad()
        stack(x).pow(2).mean().backward()
        dead = [n for n, p in stack.named_parameters()
                if p.grad is None or not torch.isfinite(p.grad).all()
                or p.grad.abs().sum() == 0]
        assert not dead, f"{name}: gradient-dead at init: {dead}"
    print("[ok] backward: every parameter (incl. all pseudo-queries) got "
          "nonzero grad at zero-init")

    # finite-difference gradcheck through both wirings, float64
    for name, cls, kw in (("full", FullAttnRes, {}),
                          ("block", BlockAttnRes, {"block_size": 2})):
        torch.manual_seed(1)
        stack = cls([_mlp(8, 200 + i) for i in range(3)], 8, **kw).double()
        with torch.no_grad():               # move off the uniform point so the
            stack.w.normal_(0, 0.5)         # softmax jacobian is generic
        xs = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: stack(t), (xs,),
                                        eps=1e-6, atol=1e-7), f"{name} gradcheck"
    print("[ok] gradcheck: full and block forms match finite differences (float64)")

    print("\nALL ATTNRES REFERENCE CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
