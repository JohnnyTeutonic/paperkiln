# Gemma-3-270m-it on 40 UltraChat prompts (the long-prompt run)

`tools/thirdparty_parity.py` on the official google/gemma-3-270m-it, converted to F32 by llama.cpp b6000
(C:/ml_artifacts/thirdparty/gemma-3-270m-it/gemma-3-270m-it-f32.gguf, sha256 0a174d42...): the gemma3 IR on
the numpy reference against HF transformers reading the same GGUF, the Gemma chat template, 32 new tokens,
on `prompts_ultrachat_100_139.jsonl` (21-1,230 prompt tokens).

- Prompts 0-35: 22 Oct 2026 at paperkiln 24df1ebc (`gate_n40.jsonl.ir.json` is that run's IR, ir_version 0).
- Prompts 36-39: 26 Oct 2026 at paperkiln 3395b16c (keramos v1.1.4 vendored), CPU in WSL, resumed after the
  last row. That run's IR equals the stored one in every field but `ir_version` (1 since 5349ee19).

Result (`gate_n40.jsonl.summary.json`): first-token top-1 40/40, greedy 40/40, worst max|dlogit| / range
4.55e-6 (prompt 0, 1,230 tokens, the one prompt past the 512-token local window); prompts 36-39 at
140, 23, 108 and 147 tokens, 2.2e-6, 1.4e-6, 1.6e-6 and 2.3e-6.
