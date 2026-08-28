# Phase 0 — transformer_core kernel audit
*(docs/DESIGN.md section 1's "phase-0 risk to check first": which primitives are
shape-general enough to be microtorch's autograd op set, and which are
transformer-bound and stay behind.)*

**Method and its limit, stated up front.** Every kernel signature in
`transformer_cpp/src/cuda/` was enumerated (26 files, ~70 `__global__`
kernels) and the bodies of the load-bearing ones read (tiled matmul, fused
attention, layer-norm stats/apply/backward, the elementwise family). The
classifications below are from signatures plus those spot-reads, not a
line-by-line read of all 70 bodies; anything promoted into `primitives.hpp`
gets a numerical parity test in phase 1a anyway (the tape's gradient checks
double as the kernel audit's verification), so a misclassification here is
caught one phase later, not shipped.

**The CMake seam exists as docs/DESIGN.md hoped:** `transformer_core` is a real
STATIC library target ([CMakeLists.txt:168](../transformer_cpp/CMakeLists.txt#L168)),
already consumed by four in-repo executables via
`target_link_libraries(... transformer_core)`. `add_subdirectory` +
`primitives.hpp` is viable with no build surgery.

---

## 1. Verdict in one table

| Op | Kernel(s) | File | Shape-general? | microtorch op set? |
|---|---|---|---|---|
| matmul (GPU) | `matrix_multiply_kernel` (32x32 tiled) + cuBLAS `Sgemm` paths | `matrix_ops.cu` | **yes** — plain (M,N,K) | **yes** (forward + both grad matmuls) |
| matmul (CPU) | AVX2 micro-kernel + blocked tiling | `matmul_optimized.cpp` | **yes** | **yes** — this is the CPU-first tape's workhorse |
| GELU fwd/bwd | `gelu_forward_kernel`, `gelu_backward_kernel` | `matrix_ops.cu` (+2 dupes, see §3) | **yes** — elementwise over `size` | **yes** |
| Swish/SwiGLU | `swish_kernel`, `swish_backward_kernel`, `elementwise_mul_kernel` | `swiglu_kernels.cu` | **yes** — elementwise | **yes** |
| bias add / bias grad | `add_bias_kernel`, `compute_bias_gradients_kernel` | `matrix_ops.cu`, `cuda_utils.cu` | **yes** — (rows, cols) broadcast / column-reduce | **yes** |
| row softmax | `softmax_kernel_rowwise` | `matrix_ops.cu` | **yes** — (rows, cols) | **yes** |
| layernorm fwd | `layer_norm_stats_kernel` + `layer_norm_kernel` | `layernorm_kernels.cu` | **yes** — (batch, hidden) | **yes** |
| layernorm bwd | `layer_norm_backward_kernel` | `backward_ops.cu` | **yes** — (batch, hidden) | **yes** |
| CE loss + grad | `softmax_cross_entropy_grad_kernel`, `reduce_losses_kernel` | `loss_kernels.cu` | **yes** — (rows, vocab) | **yes** |
| Adam update | `lmhead_adam_update_kernel` | `loss_kernels.cu` | **yes** — elementwise; full m/v/bias-correction math | **yes** — the reuse DESIGN §3 hoped for ("reuse transformer_cpp's optimizer math if present": present) |
| embedding gather/scatter | `embedding_forward_kernel`, `embedding_project_kernel` | `token_embedding_cuda.cu` | mostly — index-gather over a table | **yes**, as `nn.Embedding`'s kernel |
| fp16<->fp32 | `convert_fp32_to_fp16_kernel` + pair | `half_precision_kernels.cu` | **yes** | yes (mixed-precision later, not 1a) |
| fused attention | `batched_attention_scores/softmax/output_kernel` | `fused_attention_kernels.cu` | **NO** — indexes `[batch*seq, hidden]` with baked head-major `head_offset = head*head_dim` packing | no — stays behind; microtorch's `MultiheadAttention` *composes* matmul+softmax ops instead |
| causal softmax fwd/bwd | `batched_softmax_causal_kernel`, `batched_softmax_backward_kernel` | `attention_ops.cu` | **NO** — causal mask + `[heads, seq, seq]` layout baked in | no (compose; revisit as a fused op post-1c) |
| naive attention | `attention_scores_kernel`, `softmax_kernel(seq_len)`, `attention_kernel` | `attention_kernels.cu`, `cuda_kernels.cu`, `cuda_utils.cu` | **NO** — square `seq_len` assumptions | no; also superseded in-repo by the batched path |
| GQA | `gqa_forward_kernel` | `gqa_kernels.cu` | **NO** — grouped-KV layout | no |
| MoE / router | `moe_kernel`, `router_kernel` | `moe_kernel.cu`, `router_kernel.cu` | **NO** | no (SiTU-GLU/QB backlog item lives here, in transformer_cpp) |
| vocab-expand lm-head | `convert_and_expand_vocab_kernel` | `lm_head_kernels.cu` | **NO** — lm-head-specific | no |
| quantization, beam search, tokenizer | various | `quantization_kernels.cu` etc. | n/a | no — inference tooling, out of tape scope |

The split lands exactly where DESIGN §1 predicted: matmul, elementwise,
softmax, layernorm are clean; everything attention-shaped is bound to the
`[batch*seq, hidden]` head-major flattening and stays behind the seam.

## 2. What the tape gets for free (grad_fn bodies)

Hand-written backwards exist and are live in training today:

- `backward_ops.cu` — layernorm backward, GELU backward (GPU).
- `attention_ops.cu` — softmax backward `dS = S .* (dP − rowsum(dP .* S))`,
  batched, plus its cuBLAS score/value grad matmuls.
- `MultiHeadAttention::backward` ([attention.cpp:690](../transformer_cpp/src/attention.cpp#L690)),
  `feed_forward.cpp`, `layer_norm.cpp`, `lm_head.cpp`, `embeddings.cpp` —
  full manual module-level backwards on CPU, with the exact-backward branch
  and cached activations already worked out.

So phase 1a is genuinely "wrap, don't derive" — with one exception found
when the tape's gradcheck ran (2026-07-29, exactly the verification pass §0
promised): **`Matrix::apply_gelu_derivative` is mathematically wrong.** It
evaluates tanh at `x` instead of at the inner argument
`u = √(2/π)(x + 0.044715x³)` and drops the `0.5·sech²` factor
([components.cpp:220-224](../transformer_cpp/src/components.cpp#L220-L224)),
~13% error at x=1. The CUDA `gelu_backward_kernel`
([matrix_ops.cu:442](../transformer_cpp/src/cuda/matrix_ops.cu#L442)) has
the *correct* formula, so transformer_cpp's CPU and GPU training paths
disagree with each other today. microtorch implements its own correct CPU
derivative (matching the CUDA kernel), and `test_gradcheck.cpp` keeps a
canary that measures the legacy formula's disagreement with finite
differences — if the upstream bug is ever fixed, the canary trips and says
so. Whether to fix transformer_cpp itself is a separate decision: it
changes the numerics of a trainer with verified behaviour, so it belongs to
its owner, not to this audit.

## 3. Landmine found: duplicate kernels with diverging signatures

The same op is defined in multiple files, with *different* contracts:

- `gelu_backward_kernel` — **three** definitions: in-place on `grad_output`
  (`matrix_ops.cu`), in-place variant (`backward_kernels.cu`/`backward_ops.cu`),
  and out-of-place writing `d_input` (`feed_forward_kernels.cu`).
- `softmax_kernel` — three: `(rows, cols)` general (`cuda_kernels.cu`) and two
  square `(seq_len)` variants (`cuda_utils.cu`, `attention_kernels.cu`).
- `add_bias_kernel` — two (`matrix_ops.cu`, `cuda_utils.cu`).
- `layer_norm_backward_kernel` — declared+defined in `backward_ops.cu` *and*
  defined in `backward_kernels.cu`.

These coexist only because they live in separate translation units.
`primitives.hpp` must therefore do more than re-export: it **names one
canonical entry point per op** (recommendation: the `matrix_ops.cu` family
for elementwise/matmul/softmax-rowwise, `backward_ops.cu` for backwards,
`layernorm_kernels.cu` for layernorm) and the others are left un-exported.
Exporting "whatever the header finds" would make microtorch's numerics
depend on link order, which is the kind of bug that costs a week.

## 4. Data-structure seam

`Matrix` is row-major float32, `std::vector<float>` on CPU with an optional
`float* gpu_data_` mirror and view semantics (`owns_data_`); `Tensor` is a
4-D-shape veneer over the same storage. Consequence for DESIGN §6.2: wrap
`Tensor`/`Matrix` inside a `Variable` that *owns* one — do not graft
`grad`/`grad_fn` onto `Matrix` itself, because half the codebase constructs
`Matrix` temporaries in hot loops and would pay the autograd bookkeeping tax
everywhere. This settles open decision 2: **`Variable` owns a `Tensor`;
transformer_cpp is untouched.**

## 5. Phase-0 exit criteria — status

| Criterion | Status |
|---|---|
| Kernel shape-generality audit | **done** (this document) |
| `primitives.hpp` op list | **done** — §1's "yes" column, canonical entry points per §3 |
| CMake link works end-to-end | **done** (2026-07-29) — microtorch's own CMakeLists consumes transformer_cpp via `add_subdirectory(... EXCLUDE_FROM_ALL)` and links `transformer_core`; configure + build + gradcheck run clean in WSL, CPU-only. The predicted failure mode (CUDA symbols leaking into the CPU link) did not occur; the actual landmine was transformer_cpp using `${CMAKE_SOURCE_DIR}` for include paths, which points at the *consumer's* root under add_subdirectory. Fixed in place to `${CMAKE_CURRENT_SOURCE_DIR}` (9 occurrences) — behaviour-identical for standalone builds. |

## 6. Decisions this audit settles (DESIGN §6)

1. **Granularity:** op-level, as recommended — the audit confirms the op set
   is small (11 forward ops, 6 with existing backwards) and the attention
   fusions can't be ops anyway.
2. **Wrapper:** `Variable` owns a `Tensor`; `Matrix` untouched (§4).
3. **CPU-first:** yes — `matmul_optimized.cpp` + the module-level CPU
   backwards mean the whole tape can be gradient-checked without a GPU in
   the loop; CUDA parity comes after, kernel by kernel, against the same
   checks.
4. Repo name: still open, still a placeholder, still not blocking anything.
