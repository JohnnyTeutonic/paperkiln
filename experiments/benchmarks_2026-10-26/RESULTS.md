# Standard small benchmarks and IF-tiny (26 Oct 2026)

LOCAL_LLM_STACK.md section 11, items 5.2 (standard small benchmarks) and 3 (IF-tiny; instruction compliance
by turn distance). One RunPod L4 campaign (`pk-bench`, root `scripts/colab_driver.json`; paperkiln 0e6b299c
on branch `bench-2026-10-26`; RunPod ledger $1.85), plus CPU checks at home. The tiny-chat picks' numbers are
descriptive only and do not enter the registered tiny-chat analyses (`experiments/tiny_chat_2026-10/`).

## Benchmarks

Zero-shot log-prob forced choice in lm-evaluation-harness's formats (`tools/bench_eval.py`). Every prompt starts
with the model's end-of-text id; context and continuation are tokenised separately. 95% Wilson intervals on the
item count. Items: `tools/bench_build.py` from pinned revisions (`sources.json`: URLs, licences, sha256 of all
74 files and of the item file, `978548d1...`, rebuilt and matched on the pod). BLiMP: the first 100 pairs of
each of the 67 paradigms (6,700 pairs).

| model | params | HellaSwag acc_norm | ARC-Easy acc | ARC-Challenge acc_norm | PIQA acc | WinoGrande acc | LAMBADA acc | BLiMP acc |
|---|---|---|---|---|---|---|---|---|
| n | | 10,042 | 2,376 | 1,172 | 1,838 | 1,267 | 5,153 | 6,700 |
| chance | | 0.250 | 0.250 | 0.250 | 0.500 | 0.500 | 0 | 0.500 |
| pk_base_v1 | 97.5M | 0.292 [0.283, 0.301] | 0.437 [0.417, 0.457] | 0.253 [0.229, 0.278] | 0.612 [0.590, 0.634] | 0.521 [0.493, 0.548] | 0.252 [0.240, 0.264] | 0.755 [0.745, 0.765] |
| pk_sft_v1 | 97.5M | 0.287 [0.278, 0.296] | 0.436 [0.416, 0.456] | 0.243 [0.220, 0.269] | 0.605 [0.582, 0.627] | 0.511 [0.484, 0.539] | 0.243 [0.232, 0.256] | 0.740 [0.729, 0.750] |
| live (chat_sftfinal_v3) | 14.6M | 0.264 [0.256, 0.273] | 0.272 [0.255, 0.291] | 0.251 [0.227, 0.277] | 0.532 [0.509, 0.554] | 0.501 [0.474, 0.529] | 0.005 [0.004, 0.008] | 0.587 [0.575, 0.599] |
| pick_e1_L16 (descriptive) | 25.1M | 0.262 [0.253, 0.270] | 0.358 [0.339, 0.377] | 0.197 [0.175, 0.221] | 0.558 [0.535, 0.580] | 0.495 [0.467, 0.522] | 0.136 [0.127, 0.146] | 0.703 [0.692, 0.714] |
| pick_e1_L32 (descriptive) | 23.2M | 0.263 [0.254, 0.271] | 0.345 [0.326, 0.364] | 0.226 [0.203, 0.251] | 0.565 [0.542, 0.588] | 0.500 [0.472, 0.527] | 0.144 [0.134, 0.153] | 0.698 [0.687, 0.709] |
| pick_pilot_s35 (descriptive) | 27.2M | 0.262 [0.253, 0.270] | 0.360 [0.341, 0.379] | 0.210 [0.188, 0.234] | 0.563 [0.540, 0.585] | 0.511 [0.484, 0.539] | 0.153 [0.144, 0.163] | 0.698 [0.686, 0.709] |
| Pythia-160m (reference) | 162M | 0.298 [0.289, 0.307] | 0.447 [0.427, 0.467] | 0.244 [0.220, 0.269] | 0.618 [0.595, 0.639] | 0.525 [0.497, 0.552] | 0.357 [0.344, 0.371] | 0.783 [0.773, 0.792] |
| SmolLM2-135M (reference) | 135M | 0.425 [0.415, 0.435] | 0.644 [0.625, 0.663] | 0.291 [0.266, 0.318] | 0.679 [0.658, 0.701] | 0.534 [0.507, 0.562] | 0.436 [0.422, 0.449] | 0.803 [0.793, 0.812] |

LAMBADA perplexity of the target word: pk_base_v1 89.3, pk_sft_v1 109.3, the picks 947-972, Pythia-160m 35.9,
SmolLM2-135M 17.8; the live model's is 5.0e9 (it saw TinyChat only). acc and acc_norm for every set, and BLiMP per
paradigm, are in `receipts/bench/<model>/summary.json`; per-item records in `items.jsonl.gz` beside it.

- **What the table says.** pk_base_v1 is level with Pythia-160m on HellaSwag, ARC, PIQA and WinoGrande, below it
  on LAMBADA and BLiMP, and well below SmolLM2-135M everywhere. The SFT costs it 0.001-0.015 on every set,
  inside the intervals. WinoGrande is at chance for every model but SmolLM2-135M (0.534), and ARC-Challenge is
  at or below chance for all but SmolLM2-135M. The tiny-chat picks sit just above chance on HellaSwag and PIQA
  and clearly above it on ARC-Easy, LAMBADA and BLiMP. The live model is near chance (HellaSwag 0.264,
  ARC-Easy 0.272) except on PIQA and BLiMP.
- **The SmolLM2 reference matches its published numbers.** Its model card at the pinned revision gives
  HellaSwag 42.1, PIQA 68.4 and WinoGrande 51.3; this harness gives 0.425, 0.679 and 0.534. That checks the
  formats and the item files.
- **Contexts.** The models with 512 positions truncated nothing. The live model has 256: 4 ARC-Challenge and
  10 PIQA requests lost context tokens from the left. 4 PIQA items have a solution longer than the context on
  its own, so they get chance credit (counted in `summary.json`, `unfit`).

### Engines

- **paperkiln models with the 16k vocabulary** (pk_base_v1, pk_sft_v1, the picks): their GGUFs through
  transformers' GGUF loader, float32, batched (`bench_eval.py --scorer hfgguf`). Each GGUF was exported on the
  pod from the uploaded safetensors with `mtstudio export`; its tensors are bit-identical to the home GGUF's
  (checked at home for all six models; the files differ in metadata only; sha256s in
  `receipts/bench/xcheck/staged.jsonl`).
- **The live model** (8k paperkiln BPE, which transformers' loader refuses): `mtstudio score` in full on the pod.
- **References**: HF transformers 4.46.3, torch 2.4.1, float32, TF32 off, at pinned revisions (Pythia-160m
  50f5173d, SmolLM2-135M 93efa2f0).
- **Cross-checks**:
  - hfgguf against `mtstudio score`, pk_sft_v1, the first 1,371 HellaSwag requests (pod): max |dlogprob|
    3.2e-5, argmax 20,581/20,581 (`receipts/engine_checks/cmp_pksft_mtstudio_cuda_vs_hfgguf_pod.json`).
  - On every 400th item (199 requests per model), the pod's CUDA `mtstudio score` disagreed with hfgguf on the
    four requests of one HellaSwag item (17234) in all five models, by up to 12 nats, and agreed within 2e-5 on
    the other 195. On that item alone, the CPU mtstudio and the numpy IR both give hfgguf's numbers to 4
    decimals. So the CUDA `mtstudio score` returned wrong log-probabilities for that item
    (`receipts/bench/xcheck/compare_*.json`; see Open items).
  - mtstudio (CPU) against the numpy IR on samples: live 501 requests, max |dlogprob| 1.8e-5, argmax
    8,518/8,518; pk_sft_v1 81 requests, 1.4e-5, 1,467/1,467 (`receipts/engine_checks/`).
- **ember's POST /score** (agreed with ember, being built) is wired in as `--scorer ember` and is not used here.

### Tiny-chat picks' weights

The picks' `model.safetensors` hold the SFT run's AdamW state, not weights: `tiny_chat_run.py pick` at
dc29b5e6 copies `optim.safetensors`, and best.json's `weights_sha256` is that file's hash. The weights used here
come from each pick's `model.gguf`:

| pick | model.gguf sha256 |
|---|---|
| pick_e1_L16 | 96f6afdcbbcf1a4cf53ad6c0e451e75301dc4187f726f3c2218600fbe72a8cc9 |
| pick_e1_L32 | 40d4f0c448d191b093e2a6299f54da631dfd7994163caccc687b730e78c32e18 |
| pick_pilot_s35 (Colab pilot) | ba809ab18200455abe34581640c44c9244bd0dc86a148b763d08e8aa4e000c30 |

`rebuild_weights.py` writes weights-only safetensors from them (in `C:/ml_artifacts/benchmarks/weights/`).

- **The inverse is exact.** It inverts `export_gguf_llama`. Run on pk_sft_v1's GGUF, it reproduces that model's
  real safetensors bit for bit (110 tensors).
- **The round trip is exact.** `mtstudio export` of each rebuilt file gives a GGUF whose tensors are
  bit-identical to the pick's (147/147, 291/291, 75/75).

## IF-tiny v1

574 single-turn items in 15 families whose compliance code decides (`corpus/eval/iftiny_v1.jsonl`, sha256
f2300622...; `tools/iftiny_build.py`, `tools/iftiny_checks.py`, 64 unit tests in `tests/test_iftiny_checks.py`).
Each item also has a bare prompt, the same content without the instruction, which gives the format's base rate.

Generation is greedy, max 64 new tokens:

- **paperkiln models**: ember fe01b35 with `EMBER_DETERMINISTIC=1` and TF32 off, through `/chat` on the pod's
  L4 (GPU f5500088).
- **Instruct references**: HF generate with each model's own chat template, at pinned revisions
  (SmolLM2-135M-Instruct 12fd25f7, Qwen2.5-0.5B-Instruct 7ae55760).

Compliance (bare base rate in brackets); macro is the mean over the 15 families; micro is over the 574 items,
with a 95% interval:

| model | macro | micro [95% CI] | include word | start with | repeat phrase | end with full stop | avoid word | one word | numbered list | lower case | capitals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| live | 0.115 (0.103) | 0.120 [0.096, 0.149] | 0.03 (0) | 0 (0) | 0 (0) | 0.93 (1.00) | 0.78 (0.55) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| pick_e1_L16 | 0.152 (0.090) | 0.159 [0.131, 0.191] | 0.45 (0) | 0.33 (0) | 0.20 (0.03) | 1.00 (0.85) | 0.23 (0.42) | 0.03 (0) | 0 (0.03) | 0 (0) | 0 (0) |
| pick_e1_L32 | 0.135 (0.092) | 0.141 [0.115, 0.172] | 0.28 (0) | 0.15 (0) | 0.05 (0) | 0.97 (0.88) | 0.57 (0.45) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| pick_pilot_s35 | 0.151 (0.072) | 0.155 [0.128, 0.187] | 0.25 (0) | 0.15 (0) | 0.23 (0) | 0.95 (0.75) | 0.45 (0.30) | 0.03 (0) | 0.03 (0) | 0 (0) | 0 (0) |
| pk_sft_v1 | 0.123 (0.062) | 0.129 [0.104, 0.159] | 0.42 (0) | 0.10 (0) | 0.47 (0.30) | 0.70 (0.38) | 0 (0.25) | 0.03 (0) | 0.03 (0) | 0 (0) | 0 (0) |
| SmolLM2-135M-Instruct | 0.276 (0.073) | 0.287 [0.252, 0.326] | 0.33 (0) | 0.25 (0) | 0.97 (0.47) | 0.85 (0.38) | 0.30 (0.20) | 0.03 (0) | 0.93 (0.05) | 0 (0) | 0 (0) |
| Qwen2.5-0.5B-Instruct | 0.515 (0.045) | 0.500 [0.459, 0.541] | 0.20 (0) | 0.42 (0) | 0.90 (0) | 0.57 (0.42) | 0.40 (0.25) | 0.68 (0) | 0.90 (0) | 0.72 (0) | 0 (0) |

All 15 families per model, and content accuracy where an item has an answer, are in
`receipts/iftiny_*/<model>/summary.json`.

- **Our models follow almost nothing.** Their macro compliance is at most 0.08 above the bare base rate. What
  they do comply with is lexical: using a given word, opening with a given word, repeating a phrase.
- **Format constraints are almost never met.** Our models produce one word, three words or fewer, lower case,
  capitals, quotation marks, yes or no, a number only, a numbered list or a repeated word on request in at
  most 8% of items (most cells 0).
- **The set separates the references.** Qwen2.5-0.5B-Instruct meets most of the format constraints, and
  SmolLM2-135M-Instruct meets the numbered list and the repeated phrase.

## Compliance by turn distance v1

1,344 conversations (`corpus/eval/ifdist_v1.jsonl`, sha256 9299d5f6...). Each has a persistent instruction at
turn 1 (7 families) and is checked at turn 1 + d, for d in {0, 1, 2, 4}. The earlier turns are teacher-forced
with canned replies, and each sequence's target question is the same at every distance; there are 84
conversations per cell. Four conditions:

- **user**: the instruction opens the first user message.
- **system**: the instruction is the system message, which conversation schema v1 folds into the first user
  message (contract point 3).
- **nohint**: the same compliant history, with no instruction.
- **bare**: plain history replies, with no instruction.

Every server prompt for the five paperkiln models equals `chat_template.prompt(..., system=...)` tokenised at home
(1,474/1,474 per model, 224 with a system message; `receipts/iftiny_cuda/prompt_check.json`).

Compliance at turn 1 + d (user placement; nohint in brackets):

| model | d = 0 | d = 1 | d = 2 | d = 4 |
|---|---|---|---|---|
| live | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| pick_e1_L16 | 0.083 (0.000) | 0.036 (0.059) | 0.107 (0.155) | 0.357 (0.250) |
| pick_e1_L32 | 0.071 (0.000) | 0.131 (0.107) | 0.143 (0.131) | 0.262 (0.214) |
| pick_pilot_s35 | 0.012 (0.000) | 0.238 (0.107) | 0.298 (0.202) | 0.452 (0.464) |
| pk_sft_v1 | 0.000 (0.000) | 0.191 (0.167) | 0.309 (0.250) | 0.548 (0.476) |
| SmolLM2-135M-Instruct | 0.071 (0.000) | 0.643 (0.452) | 0.869 (0.738) | 0.952 (0.905) |
| Qwen2.5-0.5B-Instruct | 0.691 (0.000) | 0.714 (0.083) | 0.762 (0.298) | 0.845 (0.631) |

The paired differences over the 84 (family, sequence) pairs, with 95% intervals, are in `receipts/*/paired.json`
(`analyze_if.py`).

- **Our models copy the format; the instruction adds little.** Compliance rises with d for every paperkiln model
  except live. The rise comes from the compliant replies in the history, not the instruction: user minus nohint
  is at most +0.13, and for pk_sft_v1 +0.00, +0.02, +0.06 and +0.07 at d = 0, 1, 2 and 4.
- **The bare base rate is near zero** for every model (at most 0.024 in any cell).
- **The instruction carries in Qwen2.5-0.5B-Instruct.** It follows the instruction itself (0.69 at d = 0) and
  holds it four turns later, still +0.21 over nohint at d = 4. SmolLM2-135M-Instruct gains +0.05 to +0.19 from
  the instruction over the history alone.
- **System versus user placement.** System minus user is within ±0.10 for the paperkiln models. Qwen at d = 0 is
  the one case where placement clearly matters: -0.21, so the folded system message helps less than the same
  text in the user turn.

### Determinism and parity

- **Replay.** Two passes in fresh server processes (ember fe01b35, `EMBER_DETERMINISTIC=1`, one L4) gave
  bit-identical replies and ids for all 1,474 distinct prompts of each of the five models. The identity holds on
  one GPU model and one build, not between a T4 and an L4; the GPU's UUID is in `receipts/iftiny_cuda/gpu.txt`.
- **Parity.** Every reply that ended on end-of-text was re-scored with `mtstudio score` on the pod. Its argmax
  equals the reply at every position for every model: live 1,474/1,474, pick_e1_L16 1,271/1,271, pick_e1_L32
  1,271/1,271, pick_pilot_s35 1,305/1,305, pk_sft_v1 1,214/1,214. The rest stopped at 64 tokens and were not
  checked.
- **A CPU leg** on ember cbdcb915 (`/home/jonat/tc_ember`) gives the same IF-tiny numbers for live and
  pick_e1_L16, with parity against the CPU mtstudio 1,474/1,474 and 1,294/1,294. That commit ignores `/chat`'s
  `system` field: its instr_system prompts equal nohint's. So its distance cells are not used here.
  `tools/iftiny_eval.py` now checks every server prompt against schema v1 and reports a mismatch.

## Files

- **Sets**: `corpus/eval/iftiny_v1.jsonl`, `corpus/eval/ifdist_v1.jsonl`. The benchmark items are rebuilt by
  `tools/bench_build.py` (data in `C:/ml_artifacts/benchmarks/raw/`, untrusted, read under `python -I`).
- **Tools**:
  - `tools/bench_build.py`, `tools/bench_eval.py` (`run`, `report`, `compare`);
  - `tools/iftiny_build.py`, `tools/iftiny_checks.py`, `tools/iftiny_eval.py`;
  - `tools/colab_chat/bench_v1.json`, `bench_vm.sh`, `bench_stage.py`;
  - here: `rebuild_weights.py`, `analyze_if.py`, `specs/`.
- **Receipts**:
  - `receipts/bench/` (summary, meta and per-item records per model; `xcheck/`);
  - `receipts/iftiny_cuda/`, `receipts/iftiny_hf/` (summary, per-item records and both passes' replies per
    model);
  - `receipts/engine_checks/`.
- **Full score chunks** (per-token log-probabilities) and the run folders: `C:/ml_artifacts/benchmarks/runs/`.
- **SmolLM2-135M** as an F32 GGUF for ember's /score gate: `C:/ml_artifacts/thirdparty/smollm2-135m/`.

## Open items

- **CUDA `mtstudio score`.** It returned wrong log-probabilities for one HellaSwag item (17234) in all five
  models, consistently in one request sequence. The CPU build and the numpy IR are right on it. Not yet
  diagnosed.
- **mtstudio score is slow.** It scores one request at a time on the host: ~75 ms each for the 14.6M live model
  on the pod, with the GPU near idle. Batching it would let it carry the full sets.
- **ember's POST /score** (being built by ember): once gated, cross-check it against hfgguf and mtstudio with
  `bench_eval.py compare`, including the live model and SmolLM2-135M.
