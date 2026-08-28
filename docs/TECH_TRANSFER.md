# Tech-transfer backlog — techniques from open-weight frontier models
Source reading: **Kimi K3** tech report (arXiv 2607.24653; weights on HF), with
cross-references to **DeepSeek-V4** (2606.19348), **DeepSeek-V3** (2412.19437),
GLM-5.2, Qwen3.5, FlashMemory-DeepSeek-V4 (2606.09079).

Reality check: these are validated at trillion-parameter scale. We can't
reproduce the models; the value is **technique transfer** into our own small
libraries (`microtorch`, `dit`, `transformer_cpp`, `tinyllama.cpp`) for learning,
demos, and — in one case — a paper extension.

Effort key: **S** ≤ a day · **M** a few days · **L** a week+.

| # | Technique | Target lib(s) | Effort | Value | Notes |
|---|---|---|---|---|---|
| 1 | **Attention Residuals (AttnRes)** | microtorch, transformer_cpp, dit | S–M | high | **microtorch DONE 2026-07-31**: nn::AttnResStack (full + block forms, equivalence-pinned, FD-checked, dead-query finding recorded); dit + transformer_cpp remain. |
| 2 | **KDA (Kimi Delta Attention)** op | microtorch, chimera | M (recurrent) / L (chunkwise+CUDA) | high | Reference impl first (this backlog). Feeds the Chimera paper extension. |
| 3 | **Muon optimizer (per-head)** | microtorch | S | med–high | **DONE 2026-07-31**: nn::Muon + newton_schulz5, golden-pinned to the reference, per-head receipts, mtstudio hybrid routing + studio dropdown. |
| 4 | **SiTU-GLU + Quantile Balancing** | transformer_cpp (moe/router) | S–M | med | Bounded GLU stops activation explosion; QB improves load balance. |
| 5 | **Gated MLA + NoPE** | transformer_cpp, tinyllama (kv_cache) | M | med | KV-cache compression; NoPE removes RoPE-retuning at long ctx. |
| 6 | **MXFP4 wt / MXFP8 act + QAT** | quantization.cpp, tinyllama | L | med | Frontier quant format; bigger lift. |
| 7 | **mHC (Manifold-Constrained Hyper-Connections)** [DeepSeek-V4] | microtorch | M | med | Compare head-to-head with AttnRes (#1) — both are "better residuals". |
| 8 | **Multi-token prediction (MTP)** [DeepSeek-V3] | transformer_cpp (training) | S–M | med | Cheap auxiliary training objective; well documented. |

---

## Item 2 — KDA, the equations (so the reference impl is faithful)

Single head. Query/key `q_t,k_t ∈ R^{d_k}`, value `v_t ∈ R^{d_v}`, state
`S_t ∈ R^{d_k×d_v}`.

**Recurrence (Eq. 1):**
```
S_t = (I − β_t k_t k_tᵀ)·Diag(α_t)·S_{t−1} + β_t k_t v_tᵀ
õ_t = S_tᵀ q_t
```
- `α_t ∈ (0,1)^{d_k}` : channel-wise one-step retention (forget gate).
- `β_t ∈ (0,1)` : delta-rule write strength.

**Parameterisation (Eq. 2):**
```
q_t,k_t = L2Norm(Swish(ShortConv(W_{q/k} x_t)))
v_t     = Swish(ShortConv(W_v x_t))
β_t     = Sigmoid(W_β x_t)
z_t     = W↑ W↓ x_t + b     (low-rank decay logit)
```

**Lower-bounded decay (Eq. 5) — K3's numerical-stability fix over Kimi Linear:**
```
g_t = g_min · Sigmoid(e^A · z_t),   α_t = exp(g_t),   g_min = −5,
A learnable per-head LOG-scale (init 0 ⇒ e^A = 1)
```
*(Corrected 2026-07-29: an earlier transcription here read `Sigmoid(A·z)`. With
A init 0 that makes the gate a constant and kills the gradient to the whole
decay projection — the exact bug the reference impl's self-test now asserts
against. The scale is `e^A`; the report calls A a log-scale.)*
Keeps 1/Γ (reciprocal cumulative decay) inside BF16 range so every tile is a
dense matmul (no position-pair diagonal path).

**Full-rank output gate (Eq. 6):**
```
y_t = W_o [ Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t) ]
```

**Chunkwise parallel form (Eq. 3–4):** recurrent across chunks, parallel within
(UT transform → U,W; `Ṽ = U − W·S`; masked intra-chunk + inter-chunk terms). This
is the *efficiency* form and the L-effort part; the recurrent form above is the
correctness anchor and what the reference implements.

---

## Item 1 — AttnRes, the mechanism (for the S–M implement)
Per layer `l`: a learnable pseudo-query `w_l`; keys/values are the outputs of all
preceding layers (`h_1 = token embedding`). Softmax over RMSNorm'd keys:
`α_{i→l} = ϕ(w_l, k_i)/Σ_j ϕ(w_l, k_j)`, `h_l = Σ_i α_{i→l} v_i`.
- **Full**: O(L²d) compute, O(Ld) memory. Fine for L<100.
- **Block**: partition L into N≈8 blocks, sum within block, attend across N block
  reps → O(Nd). Use online-softmax to merge inter/intra at inference.

---

## Suggested order
1. **KDA reference** (recurrent, gradient-checked) — done as the first code.
2. **AttnRes** — smallest high-value module; add to microtorch + DiT.
3. **Muon** — small optimizer.
4. The Chimera KDA experiment (see `AI_ML/chimera/KDA_CHIMERA_EXTENSION.md`).
5. Everything else as microtorch matures.
