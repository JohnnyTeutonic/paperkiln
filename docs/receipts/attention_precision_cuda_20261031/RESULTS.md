# Explicit autograd attention precision: actual CUDA gate

The frozen 44-case L4 CUDA 12.4 + DEVCHECK run passed. The gate exercised explicit FP32 and BF16 dense/causal autograd paths, unequal feature widths, stacked sequences and all gradient masks. CUDA profile markers proved the expected GEMM dispatch; independent NumPy reconstruction checked the raw operands, outputs and requested gradients with the prospectively frozen bounds. The parent replay independently verified the complete case inventory, fixture values and raw log hashes; worst normalized error was 3.293902099352775e-8.

The existing FP32 entry point retains its dispatch and CPU output/gradient bits. BF16 remains an explicit C++ op with FP32 softmax/backward and a declared straight-through BF16 operand surrogate. This result does not establish model-training quality, inference export semantics, throughput benefit or adoption. Train-only model integration is separate work.

The existing watchdog brought all artifacts home and disabled its own entry. An independent provider query confirmed that the owned pod and volume were absent after a transient cleanup network error. Raw logs and archive are retained unchanged.
