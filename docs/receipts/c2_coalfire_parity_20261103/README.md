# C2: paperkiln/coalfire cross-engine logit pin (3 Nov 2026)

Claude Opus 5.5, CPU only. C2 in `docs/ECOSYSTEM.md`: load the same tiny checkpoint into both trainers and
assert logit parity on a fixed input. The pin found a coalfire defect on its first run. With the defect fixed,
both engines agree with an independent float64 forward to 5-6e-7.

## Method (`run.sh`)

1. **Corpus:** coalfire's own `data/training_pairs.txt` (split in two, `|` to a space) and `validation_pairs.txt`.
2. **Training:** coalfire trains its `--tiny` llama-family model on the CPU: 2 layers, hidden 128, 4 heads, FFN
   512, vocabulary 2,275, 40 steps. The final state goes to a format-3 checkpoint (`--pol-save`) and to an HF
   `LlamaForCausalLM` directory (`--export-hf`), from the same in-memory model.
3. **coalfire's logits:** `coalfire_dump.cpp` loads the checkpoint with the config rebuilt as coalfire's
   `golden_batch_test` does, and runs `forward_batch` on 48 fixed ids.
4. **paperkiln's logits:** `tools/c2_parity` loads the HF export into `nn::Llama`, strictly: every tensor lands
   and every parameter is loaded. HF's `[out, in]` weights are transposed to paperkiln's `[in, out]`, and HF's
   rotate-half RoPE is paperkiln's `rope_pairing "neox"`. It runs one forward over the same ids.
5. **Referee:** `compare.py` compares both against an independent float64 forward of the HF export, written from
   HF Llama semantics. Pass: every row within 1e-4 (`max |a - ref| / max(1, max |ref row|)`), identical argmax.

## Before the fix: coalfire's CPU model was not the model it exports

Coalfire `9c0b023`, `before_fix/`.

| | max rel. error vs float64 | argmax |
|---|---:|---|
| paperkiln, on the HF export | 5.4e-7 | identical |
| coalfire, on its checkpoint | 0.284 (median 0.161) | differs |

Diagnosis, each step in `before_fix/`:
- **Not the weights.** Re-exporting HF from the loaded checkpoint gives byte-identical `model.safetensors`, and
  coalfire's logits are reproducible.
- **Not any single convention** (`diagnosis.json`, `diagnosis_pos0.json`): RoPE pairing, positions, attention
  scale, eps, LayerNorm, embedding scale, `(1 + w)` norms, activation, gate/up order.
- **Not the head or final norm** (`diagnosis_head.json`). Coalfire's logits lie in the exported `lm_head`'s
  column space to 1e-7, and its final hidden rows are normalised, but point in a slightly different direction
  (cosine 0.981-0.990).
- **The layer-0 FFN** (`diagnosis_subops.json`, `subop_trace.txt`, from coalfire's `TCPP_LAYER_TRACE`).
  Embedding, attention norm, attention, residual and FFN norm agree to 5 significant figures; the FFN output is
  9.834 in coalfire against 9.666.
- **Cause.** The CPU `FeedForward::forward` adds its gate/up/down biases. In llama mode those biases should be
  exactly zero and frozen, and `FeedForward`'s constructor does set them so. But `Transformer`'s weight
  initialiser then overwrote them with 0.01 unconditionally ("small positive values for ReLU"). Llama mode freezes
  biases, so they stayed at 0.01. The GGUF and HF exports write no FFN biases, and the CUDA SwiGLU kernel adds
  none. So a CPU-trained llama-family model's export, and its CUDA forward, were a different model from the one
  trained.
- **Confirmation.** With the biases zeroed (`C2_ZERO_FFN_BIAS=1`), coalfire matches the referee to 6.1e-7 and
  paperkiln to 7.2e-7 (`result_zeroed_bias.json`).

## The fix (coalfire branch `claude/c2-ffn-bias`)

- `src/transformer.cpp`: the initialiser uses `llama_no_bias`; in llama mode the FFN biases stay at 0.
- `include/export_guard.hpp`, called by the GGUF, safetensors and HF exporters: refuse to export a llama-mode
  model whose FFN biases are not zero, instead of writing a different model. Checkpoints trained on the CPU
  before the fix are refused, with a message saying why. Tested on the before-fix checkpoint, max |bias| 0.01.

## After the fix (`after_fix/`)

| | max rel. error vs float64 | argmax |
|---|---:|---|
| coalfire, on its checkpoint | 5.6e-7 | identical |
| paperkiln, on the HF export | 5.2e-7 | identical |
| coalfire against paperkiln | 7.9e-7 | identical |

The FFN biases are 0 after construction and after loading. **C2 passes.**

## Scope

- One tiny model (2 layers, hidden 128), CPU only, 48 positions, synthetic ids.
- Multi-head attention without GQA, untied head, coalfire's word-level vocabulary.
- Coalfire's CUDA path was not run. It adds no FFN bias, so it agrees with the export by construction, and the
  fix only removes the CPU/CUDA difference.
- Artifacts (checkpoints, exports, logits): `/mnt/c/ml_artifacts/c2_coalfire_parity_20261103/`, hashes in
  `artifacts.sha256`.

Rerun: `bash run.sh COALFIRE_CHECKOUT COALFIRE_BUILD PAPERKILN_BUILD OUT`, CPU, under `scripts/cpu_budget.py`
(about 2 minutes).
