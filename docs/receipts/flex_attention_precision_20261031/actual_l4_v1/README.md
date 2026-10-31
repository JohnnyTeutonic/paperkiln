Actual L4 model-only correctness gate passed. Independent source-bound replay verified four CPU and four CUDA cases, including every one of 116 gradient tensors per backend. The registered bounds are unchanged. CUDA BF16 GEMM counts are 12 and 36 for the one- and three-layer BF16 cases, and zero for FP32 controls.

`out.zip` is the sole raw payload here: full parameter/gradient arrays, original logs, hardware identity and gate summary. Its CRC and every declared artifact hash were verified; the landed and archived summaries match exactly. `independent_cpu_replay.json`, `independent_cuda_replay.json` and `review.json` record the independent landed verification.

This establishes the two tiny model forward/backward cases at both precisions. It does not establish a CUDA CLI training lifecycle, quality, speed, broader architectures or adoption. Gradients follow the explicit six-GEMM quantized-input surrogate, not a derivative of bit rounding.
