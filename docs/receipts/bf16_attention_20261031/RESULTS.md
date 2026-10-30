# Standalone BF16 attention primitive — 31 October 2026

The fixed **32-case L4 matrix passed**, and an independent local replay of every raw
input, output and metric passed with Python 3.11 / NumPy 1.26.4. The replay used one CPU
thread, made no GPU calls, and left all extracted evidence bytes unchanged.

This is a standalone primitive result. Production training dispatch and autograd wiring
are unchanged. It does not establish model-level parity, training loss, quality or an
end-to-end speed gain.

## Fixed scope and numerical result

Eight `[batch, heads, sequence, head_dim]` shapes cover sequence lengths
1, 7, 16, 33, 64, 128, 256 and 512, with head dimensions 8–64 and odd sequence tails.
Each shape runs dense and causal masks in FP32 and BF16:16 cases per precision.
The six attention matrix products use cuBLAS; softmax and its backward remain FP32.
BF16 operands are rounded before each product. Backward is the declared straight-through
surrogate, not the mathematical derivative of discontinuous rounding.

| Precision | Largest per-stage error | Frozen stage bound | Largest end-to-end error | Frozen end bound |
|---|---:|---:|---:|---:|
| FP32 | 9.518974e-7 | 3e-5 | 1.038433e-6 | 3e-5 |
| BF16 | 2.246476e-6 | 3e-5 | 1.297146e-3 | 5e-3 |

Error is maximum absolute array error divided by `max(1,max(abs(reference)))`.
Per-stage references independently reduce the product's **actual** FP32/BF16 operands
in float64. End-to-end references independently propagate the specified precision path.
The BF16 end bound compares with that BF16-operand surrogate oracle; it is not a
BF16-versus-FP32 model-fidelity measurement. No bounds or fixtures changed after the run.

All 32 cases retain scores, probabilities, their backward intermediates, output and Q/K/V
gradients. Probability normalization and causal masked zeros also pass. GQA, dropout,
attention bias and stateful/production integration are outside this fixed matrix.
Retained GPU-event timings are 20-iteration averages, descriptive only; they do not support
an adoption or end-to-end performance verdict.

## Evidence

- Frozen source: `557322bac6a8dafd5c3479c3616031ee175e835a`.
- Actual device: NVIDIA L4, SM89; CUDA 12.4 build; remote NumPy 1.26.4.
- [Independent audit and exact hashes](independent_audit.json).
- [Canonical raw archive](C:/ml_artifacts/bf16_attention_20261031/bf16_attention_primitive/out.zip)
  on the author's workstation, not committed to Git: 123,495,491 bytes,
  SHA256 `e5cb03a1753ee8d9d6cf4910ce5c150b8102bf7adfbca5bb65345aef354db107`.
  It retains the native binary, frozen sources, manifest, operational/version logs,
  full 32-case raw inputs/outputs, timing records and numerical reports.

For read-only replay, use the archived checker, reference and binary with NumPy 1.26.4,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, and `--replay`.
The exact independently executed command is recorded in the audit. Preserve any future
strict replay discrepancy for review; do not alter dispatch, bounds or fixtures to remove it.

Production wiring, model-level forward/backward and loss checks, and incremental
end-to-end benefit remain open in ROADMAP. This completed primitive matrix leaves that list.
