# E9 quantised-cache gate: registration (approved by Jonathan, 1 Nov 2026)

Frozen at this commit, before S is measured and before any device run. Changes after this point are dated
amendments below, never edits. Design rationale: `GATE_PROPOSAL.md`.

## Fixed inputs
- Models and hashes: as frozen on 30 Oct in `ember.cpp/experiments/kv_quant_r2_2026-10-28/` (Qwen3-0.6B Q8_0,
  Qwen2.5-0.5B-Instruct f16; hashes in `C:/ml_artifacts/kvq/ember-r2c-3/models.sha256`).
- Prompts: `ir_device_gate`'s own generator, `--n 12 --len 24 --gen 32 --seed 0` (std::mt19937 seed 0,
  uniform over [0, tokenizer size); libstdc++/GCC on both CPU and VM). Prefill rows at all 24 positions;
  device decode rows against the CPU prefill, as in `ir_device_gate`.
- Measure: |dlogit| / CPU row range (`rel_row` in `ir_device_gate.cpp`).
- Perturbation: every newly written K and V element moved by one binary32 ulp, direction from a seeded hash,
  before the cache's format round trip (hook `EMBER_KV_PERTURB`, CPU only). Seeds 1, 2, 3.

## Quantities (per model)
- S_prefill = max over seeds 1–3 of max over prompts and positions of rel(CPU-f16 perturbed prefill,
  CPU-f16 prefill). S_decode = the same for CPU-f16 perturbed token-by-token decode against CPU-f16 prefill.
- Q_cpu = max rel(CPU-f16 prefill, CPU-f32 prefill) over the same prompts and positions.

## Pass criteria for a future device run (all required)
1. max rel(CUDA-f16 prefill, CPU-f16 prefill) ≤ 5 · S_prefill and max rel(CUDA-f16 decode, CPU-f16 prefill)
   ≤ 5 · S_decode (k = 5 = the specification's CUDA/CPU tolerance ratio).
2. CUDA-f32 vs CPU-f32 within 1e-4, prefill and decode (unchanged).
3. max rel(CUDA-f16, CUDA-f32) ≤ 1.5 · Q_cpu; top-1 at every position and all 12 greedy continuations of CUDA-f16
   equal CPU-f16's (unchanged counts).
4. Unchanged legs as frozen on 30 Oct: g1 default-mode byte identity, g2 f16 deterministic repeats, both models'
   decode teacher forcing against stored prefill observations, Qwen2.5 8,064-position bench.

## Not blind
Qwen3's 30 Oct device numbers (3.4e-4 prefill, 3.6e-4 decode) are known. Qwen2.5's device leg, decode teacher
forcing and the 8,064 bench have never run.

## Measured (1 Nov 2026, CPU, `e9_sensitivity.cpp`, `run_sensitivity.sh`; raw `qwen3.json`, `qwen25.json`)
| model | S (prefill = decode) | per seed | limit 5·S | Q_cpu | limit 1.5·Q_cpu |
|---|---|---|---|---|---|
| Qwen3-0.6B | 5.29e-4 | 3.67e-4, 4.44e-4, 5.29e-4 | 2.64e-3 | 8.36e-4 | 1.25e-3 |
| Qwen2.5-0.5B | 1.04e-3 | 7.26e-4, 5.33e-4, 1.04e-3 | 5.22e-3 | 8.13e-3 | 1.22e-2 |

Observation: on the gate's own prompts the CPU's one-ulp sensitivity alone (Qwen3 5.3e-4) exceeds the CUDA-vs-CPU
difference measured on 30 Oct (3.4e-4 prefill, 3.6e-4 decode). Concern for Jonathan before any device run: with
k = 5, criterion 1's limit (2.6e-3 on Qwen3) is looser than the whole f16-vs-f32 difference (8.4e-4), so criterion 1
alone would not discriminate a real f16 defect of that size; criteria 2 and 3 still would. Any change to k is to be
recorded below as a dated amendment before the device run.

## Amendments
- **A1, 1 Nov 2026 (Jonathan's decision, before any device run):** k = 2, replacing k = 5 in criterion 1. Limits:
  Qwen3-0.6B 1.06e-3 (prefill and decode); Qwen2.5-0.5B 2.09e-3. Reason: with k = 5 the criterion was looser than
  the whole f16-vs-f32 difference and could not discriminate a real f16 defect. Disclosure: Qwen3's 30 Oct device
  values (3.4e-4, 3.6e-4) were known when k was chosen; Qwen2.5's device leg has never run. Criteria 2–4 unchanged.
