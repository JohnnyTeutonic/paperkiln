# Standard small benchmarks and IF-tiny (26 Oct 2026)

LOCAL_LLM_STACK.md section 11, items 5.2 (standard small benchmarks) and 3 (IF-tiny; instruction compliance
by turn distance). Two RunPod L4 campaigns (root `scripts/colab_driver.json`), plus CPU checks at home:

- **pk-bench**: paperkiln 0e6b299c on branch `bench-2026-10-26`, `tools/colab_chat/bench_v1.json`; ledger $1.85.
- **pk-bench2** (26 Oct 2026): paperkiln 5e00cd40 (master, with `tools/bench_eval.py` at fafc4750),
  `tools/colab_chat/bench_v2.json`; ledger $0.87. It scored the three picks that came home after pk-bench
  (pick_e1_L4, pick_e1_60m_L32, pick_e5_s0), ran IF-tiny on pick_e1_L4, and ran ember's POST /score on the
  full sets for pk_sft_v1, the live model and those picks.
- **pk-bench3** (27 Oct 2026): paperkiln ed614ee8, `tools/colab_chat/bench_v3.json`, ember e57c0fb; ledger
  $0.32. IF-tiny for pick_e1_60m_L32 and pick_e5_s0, and pick_e1_L4 again.

The tiny-chat picks' numbers are descriptive only and do not enter the registered tiny-chat analyses
(`experiments/tiny_chat_2026-10/`).

## Benchmarks

Zero-shot log-prob forced choice in lm-evaluation-harness's formats (`tools/bench_eval.py`). Every prompt starts
with the model's end-of-text id; context and continuation are tokenised separately. 95% Wilson intervals on the
item count. Items: `tools/bench_build.py` from pinned revisions (`sources.json`: URLs, licences, sha256 of all
74 files and of the item file, `978548d1...`, rebuilt and matched on each pod). BLiMP: the first 100 pairs of
each of the 67 paradigms (6,700 pairs). HellaSwag validation's `ind` names two items 433 times; every item is
scored by its own requests (`bench_eval.unique_ids`, below).

| model | params | HellaSwag acc_norm | ARC-Easy acc | ARC-Challenge acc_norm | PIQA acc | WinoGrande acc | LAMBADA acc | BLiMP acc |
|---|---|---|---|---|---|---|---|---|
| n | | 10,042 | 2,376 | 1,172 | 1,838 | 1,267 | 5,153 | 6,700 |
| chance | | 0.250 | 0.250 | 0.250 | 0.500 | 0.500 | 0 | 0.500 |
| pk_base_v1 | 97.5M | 0.299 [0.290, 0.308] | 0.437 [0.417, 0.457] | 0.253 [0.229, 0.278] | 0.612 [0.590, 0.634] | 0.521 [0.493, 0.548] | 0.252 [0.240, 0.264] | 0.755 [0.745, 0.765] |
| pk_sft_v1 | 97.5M | 0.295 [0.286, 0.304] | 0.436 [0.416, 0.456] | 0.243 [0.220, 0.269] | 0.605 [0.582, 0.627] | 0.511 [0.484, 0.539] | 0.243 [0.232, 0.256] | 0.740 [0.729, 0.750] |
| live (chat_sftfinal_v3) | 14.6M | 0.270 [0.261, 0.278] | 0.272 [0.255, 0.291] | 0.251 [0.227, 0.277] | 0.532 [0.509, 0.554] | 0.501 [0.474, 0.529] | 0.005 [0.004, 0.008] | 0.587 [0.575, 0.599] |
| pick_e1_L16 (descriptive) | 25.1M | 0.266 [0.258, 0.275] | 0.358 [0.339, 0.377] | 0.197 [0.175, 0.221] | 0.558 [0.535, 0.580] | 0.495 [0.467, 0.522] | 0.136 [0.127, 0.146] | 0.703 [0.692, 0.714] |
| pick_e1_L32 (descriptive) | 23.2M | 0.267 [0.258, 0.276] | 0.345 [0.326, 0.364] | 0.226 [0.203, 0.251] | 0.565 [0.542, 0.588] | 0.500 [0.472, 0.527] | 0.144 [0.134, 0.153] | 0.698 [0.687, 0.709] |
| pick_pilot_s35 (descriptive) | 27.2M | 0.266 [0.258, 0.275] | 0.360 [0.341, 0.379] | 0.210 [0.188, 0.234] | 0.563 [0.540, 0.585] | 0.511 [0.484, 0.539] | 0.153 [0.144, 0.163] | 0.698 [0.686, 0.709] |
| pick_e1_L4 (descriptive) | 30.6M | 0.261 [0.252, 0.269] | 0.339 [0.321, 0.358] | 0.222 [0.199, 0.246] | 0.567 [0.544, 0.589] | 0.507 [0.479, 0.534] | 0.136 [0.127, 0.146] | 0.698 [0.687, 0.709] |
| pick_e1_60m_L32 (descriptive) | 66.5M | 0.264 [0.256, 0.273] | 0.330 [0.312, 0.350] | 0.229 [0.205, 0.254] | 0.562 [0.539, 0.585] | 0.515 [0.487, 0.542] | 0.165 [0.155, 0.175] | 0.700 [0.689, 0.711] |
| pick_e5_s0 (descriptive) | 27.2M | 0.271 [0.262, 0.280] | 0.372 [0.353, 0.392] | 0.221 [0.198, 0.246] | 0.558 [0.535, 0.580] | 0.494 [0.467, 0.522] | 0.142 [0.133, 0.152] | 0.683 [0.672, 0.694] |
| Pythia-160m (reference) | 162M | 0.306 [0.297, 0.315] | 0.447 [0.427, 0.467] | 0.244 [0.220, 0.269] | 0.618 [0.595, 0.639] | 0.525 [0.497, 0.552] | 0.357 [0.344, 0.371] | 0.783 [0.773, 0.792] |
| SmolLM2-135M (reference) | 135M | 0.436 [0.426, 0.445] | 0.644 [0.625, 0.663] | 0.291 [0.266, 0.318] | 0.679 [0.658, 0.701] | 0.534 [0.507, 0.562] | 0.436 [0.422, 0.449] | 0.803 [0.793, 0.812] |

pk_base_v1, pk_sft_v1, the live model, the first three picks and the references were scored in pk-bench, the
last three picks in pk-bench2. LAMBADA perplexity of the target word: pk_base_v1 89.3, pk_sft_v1 109.3, the picks
785-1,187 (pick_e1_60m_L32 785, pick_e1_L32 947, pick_pilot_s35 961, pick_e1_L16 972, pick_e5_s0 1,081,
pick_e1_L4 1,187), Pythia-160m 35.9, SmolLM2-135M 17.8; the live model's is 5.0e9 (it saw TinyChat only). acc and
acc_norm for every set, and BLiMP per paradigm, are in `receipts/bench/<model>/summary.json`; per-item records in
`items.jsonl.gz` beside it.

- **What the table says.** pk_base_v1 is level with Pythia-160m on HellaSwag, ARC, PIQA and WinoGrande, below it
  on LAMBADA and BLiMP, and well below SmolLM2-135M everywhere. The SFT costs it 0.001-0.015 on every set,
  inside the intervals. WinoGrande is at chance for every model but SmolLM2-135M (0.534), and ARC-Challenge is
  at or below chance for all but SmolLM2-135M. The six tiny-chat picks sit just above chance on HellaSwag
  (0.261-0.271) and PIQA (0.558-0.567) and clearly above it on ARC-Easy (0.330-0.372), LAMBADA (0.136-0.165) and
  BLiMP (0.683-0.703). pick_e1_60m_L32, at 66.5M the largest pick, has the highest LAMBADA of the picks (0.165,
  perplexity 785; its interval clears every other pick's but pick_pilot_s35's) and the lowest ARC-Easy (0.330;
  its interval clears pick_e5_s0's 0.372). The live model is near chance (HellaSwag 0.270, ARC-Easy 0.272) except
  on PIQA and BLiMP.
- **The SmolLM2 reference is within 2.1 points of its published numbers.** Its model card at the pinned
  revision gives HellaSwag 42.1, PIQA 68.4 and WinoGrande 51.3; this harness gives 0.436, 0.679 and 0.534.
- **Contexts.** The models with 512 positions truncated nothing. The live model has 256: 4 ARC-Challenge and
  10 PIQA requests lost context tokens from the left. 4 PIQA items have a solution longer than the context on
  its own, so they get chance credit (counted in `summary.json`, `unfit`).

### Engines

- **paperkiln models with the 16k vocabulary** (pk_base_v1, pk_sft_v1, the picks): their GGUFs through
  transformers' GGUF loader, float32, batched (`bench_eval.py --scorer hfgguf`). Each GGUF was exported on the
  pod from the uploaded safetensors with `mtstudio export`. For pk-bench's six models its tensors were checked
  at home bit-identical to the home GGUF's (the files differ in metadata only); for pk-bench2's three picks a home
  `mtstudio export` of the same uploaded files gives tensors bit-identical to each pick's GGUF (below). sha256s in
  `receipts/bench/xcheck/staged.jsonl` and `bench2_staged.jsonl`.
- **The live model** (8k paperkiln BPE, which transformers' loader refuses): `mtstudio score` in full on the pod.
- **References**: HF transformers 4.46.3, torch 2.4.1, float32, TF32 off, at pinned revisions (Pythia-160m
  50f5173d, SmolLM2-135M 93efa2f0).
- **Cross-checks**:
  - hfgguf against `mtstudio score`, pk_sft_v1, the first 1,371 HellaSwag requests (pod): max |dlogprob|
    3.2e-5, argmax 20,581/20,581 (`receipts/engine_checks/cmp_pksft_mtstudio_cuda_vs_hfgguf_pod.json`).
  - On every 400th item (199 requests per model), the pod's CUDA `mtstudio score` against hfgguf. pk-bench2's
    three picks: within 2.6e-5 on all 199 requests, argmax 3,827/3,827, choices 59/59 each
    (`receipts/bench/xcheck/compare_{pick_e1_L4,pick_e1_60m_L32,pick_e5_s0}_mtstudio_vs_hfgguf.json`). The
    sample holds the first of the two items under `hellaswag/17234`. In pk-bench, 195 requests per model agreed
    within 2e-5; the other four were that item's, which the harness of that run compared with the second item's
    scores, stored under the same key (the other five `compare_*.json` there). On the right item, CUDA
    `mtstudio score` agrees with the numpy IR within 1.2e-5 (`../tiny_chat_2026-10/SCORE_AUDIT_2026-10-26.md`).
  - mtstudio (CPU) against the numpy IR on samples: live 501 requests, max |dlogprob| 1.8e-5, argmax
    8,518/8,518; pk_sft_v1 81 requests, 1.4e-5, 1,467/1,467 (`receipts/engine_checks/`).
- **HellaSwag's repeated ids.** HellaSwag validation's `ind` names two items 433 times. `tools/bench_eval.py`
  (fafc4750) keys the n-th item under an id `id#n`, so every request key names one request; pk-bench2 ran with
  it. The pk-bench runs' HellaSwag is aggregated the same way from their own chunks: each first item's scores sit
  in a different 2,048-request chunk from its second's, so `bench2.py hellaswag` reads each chunk on its own,
  gives each request its fafc4750 key and runs `bench_eval.summarise`
  (`receipts/bench/hellaswag_one_request_per_key.json`). ember's full-set pass, a separate scoring of every item,
  gives the same HellaSwag acc and acc_norm for pk_sft_v1 and the live model (next section).

### ember's POST /score

ember 580e108's POST /score (CUDA, FP32 GEMMs, `EMBER_DETERMINISTIC=1`; `--scorer ember`) scored the full sets
in pk-bench2 for pk_sft_v1, the live model and the three new picks, on the GGUFs the pod exported. At home,
`bench2.py ember` compares it per request and per item with hfgguf on the same model (the full sets), with the
pod's CUDA `mtstudio score` (the every-400th sample for pk_sft_v1 and the picks; the full sets for the live
model), and gives each item's choice (acc and acc_norm), LAMBADA's argmax list and every token's log-prob:

| model | against | requests | max \|dlogprob\| | mean \|dlogprob\| | max \|dsum\| | argmax positions | items, choices equal |
|---|---|---|---|---|---|---|---|
| pk_sft_v1 | hfgguf | 79,119 | 1.0e-4 | 3.4e-6 | 1.9e-4 | 1,611,349/1,611,353 | 28,548/28,548 |
| pk_sft_v1 | mtstudio (every 400th) | 199 | 2.3e-5 | 2.0e-6 | 5.9e-5 | 3,827/3,827 | 72/72 |
| live | mtstudio | 79,119 | 7.0e-5 | 1.5e-6 | 8.7e-5 | 1,863,689/1,863,690 | 28,544/28,544 (4 PIQA items unfit in both) |
| pick_e1_L4 | hfgguf | 79,119 | 5.2e-5 | 3.3e-6 | 1.4e-4 | 1,611,351/1,611,353 | 28,548/28,548 |
| pick_e1_L4 | mtstudio (every 400th) | 199 | 2.5e-5 | 2.0e-6 | 4.4e-5 | 3,827/3,827 | 72/72 |
| pick_e1_60m_L32 | hfgguf | 79,119 | 4.3e-5 | 2.8e-6 | 2.1e-4 | 1,611,350/1,611,353 | 28,548/28,548 |
| pick_e1_60m_L32 | mtstudio (every 400th) | 199 | 1.4e-5 | 2.2e-6 | 5.2e-5 | 3,827/3,827 | 72/72 |
| pick_e5_s0 | hfgguf | 79,119 | 7.6e-5 | 2.9e-6 | 1.9e-4 | 1,611,349/1,611,353 | 28,548/28,548 |
| pick_e5_s0 | mtstudio (every 400th) | 199 | 1.6e-5 | 2.2e-6 | 6.0e-5 | 3,827/3,827 | 72/72 |

- **No item's choice differs** between ember and the reference, on any set or model (acc and acc_norm picks;
  every LAMBADA argmax list is equal). ember's summaries equal the reference runs' in every acc and acc_norm,
  BLiMP per paradigm included; LAMBADA perplexity agrees within 1e-6 relative (pick_e5_s0 1,081.325 against
  1,081.324, live 5.019052e9 in both).
- **The 14 argmax positions that differ are ties.** All are inside HellaSwag continuations, where only the
  summed log-prob counts. At each, the numpy IR's top two logits are 9.5e-7 to 1.5e-5 apart, and they are the
  two tokens the engines chose: the IR sides with ember at 11 and with the reference at 3
  (`receipts/ember_score/argmax_flips_ir.json`). pk_sft_v1: `hellaswag/1140|1`, `10641|0`, `21102|3`,
  `22178|1`; live: `11242|1`; pick_e1_L4: `4992|2`, `4590|0`; pick_e1_60m_L32: `24129|0`, `35163|1`, `50313|0`;
  pick_e5_s0: `12222|3`, `2116|1`, `36177|2`, `39312|2`.
- Receipts: `receipts/ember_score/<model>/` (summary, meta, the server's /health) and
  `receipts/ember_score/compare_*.json` (per model and reference: the per-set agreement counts and every
  disagreeing item, none here). ember's per-item records and score chunks are in
  `C:/ml_artifacts/benchmarks/runs2/ember_score/x/bench/`.

### Tiny-chat picks' weights

The picks' `model.safetensors` hold the SFT run's AdamW state, not weights: `tiny_chat_run.py pick` at
dc29b5e6 copies `optim.safetensors`, and best.json's `weights_sha256` is that file's hash. The weights used here
come from each pick's `model.gguf`:

| pick | model.gguf sha256 |
|---|---|
| pick_e1_L16 | 96f6afdcbbcf1a4cf53ad6c0e451e75301dc4187f726f3c2218600fbe72a8cc9 |
| pick_e1_L32 | 40d4f0c448d191b093e2a6299f54da631dfd7994163caccc687b730e78c32e18 |
| pick_pilot_s35 (Colab pilot) | ba809ab18200455abe34581640c44c9244bd0dc86a148b763d08e8aa4e000c30 |
| pick_e1_L4 | 363416da0e4b7482b061b7c717900c2ffc2b1bde80b4b24eebbeb02297fa16c9 |
| pick_e1_60m_L32 | 54a4180eb14cec41f50a0f4ef569b0033b6445bfd5c3d632134f402fbc40166c |
| pick_e5_s0 | 1b14eb2997bc1caca01a7111d4a17434948039ccbce0483586d90a6f465db064 |

`rebuild_weights.py` writes weights-only safetensors from them (in `C:/ml_artifacts/benchmarks/weights/`; the
sha256s the pods staged match these files).

- **The inverse is exact.** It inverts `export_gguf_llama`. Run on pk_sft_v1's GGUF, it reproduces that model's
  real safetensors bit for bit (110 tensors).
- **The round trip is exact.** `mtstudio export` of each rebuilt file gives a GGUF whose tensors are
  bit-identical to the pick's: 147/147, 291/291 and 75/75 for pick_e1_L16, pick_e1_L32 and pick_pilot_s35;
  39/39, 291/291 and 75/75 for pick_e1_L4, pick_e1_60m_L32 and pick_e5_s0.

## IF-tiny v1

574 single-turn items in 15 families whose compliance code decides (`corpus/eval/iftiny_v1.jsonl`, sha256
f2300622...; `tools/iftiny_build.py`, `tools/iftiny_checks.py`, 64 unit tests in `tests/test_iftiny_checks.py`).
Each item also has a bare prompt, the same content without the instruction, which gives the format's base rate.

Generation is greedy, max 64 new tokens:

- **paperkiln models**: ember fe01b35 with `EMBER_DETERMINISTIC=1` and TF32 off, through `/chat` on an L4:
  pk-bench's pod (GPU f5500088) for live, pk_sft_v1 and the first three picks, pk-bench2's (GPU 15ee501a) for
  pick_e1_L4. pick_e1_60m_L32 and pick_e5_s0 ran in pk-bench3 (GPU 8c5ce6af) at ember e57c0fb: fe01b35 with the
  reply text made valid UTF-8, generation unchanged (pk-bench2's IF-tiny job, below).
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
| pick_e1_L4 | 0.195 (0.077) | 0.204 [0.173, 0.239] | 0.45 (0) | 0.50 (0) | 0.47 (0) | 0.93 (0.78) | 0.45 (0.38) | 0.03 (0) | 0.03 (0) | 0 (0) | 0 (0) |
| pick_e1_60m_L32 | 0.138 (0.085) | 0.145 [0.118, 0.176] | 0.70 (0) | 0 (0) | 0.05 (0) | 0.97 (0.62) | 0.17 (0.47) | 0 (0) | 0.07 (0) | 0.03 (0) | 0 (0) |
| pick_e5_s0 | 0.110 (0.087) | 0.115 [0.091, 0.144] | 0.40 (0) | 0.07 (0) | 0 (0.05) | 1.00 (0.80) | 0.12 (0.42) | 0 (0) | 0.05 (0) | 0 (0) | 0 (0) |
| pk_sft_v1 | 0.123 (0.062) | 0.129 [0.104, 0.159] | 0.42 (0) | 0.10 (0) | 0.47 (0.30) | 0.70 (0.38) | 0 (0.25) | 0.03 (0) | 0.03 (0) | 0 (0) | 0 (0) |
| SmolLM2-135M-Instruct | 0.276 (0.073) | 0.287 [0.252, 0.326] | 0.33 (0) | 0.25 (0) | 0.97 (0.47) | 0.85 (0.38) | 0.30 (0.20) | 0.03 (0) | 0.93 (0.05) | 0 (0) | 0 (0) |
| Qwen2.5-0.5B-Instruct | 0.515 (0.045) | 0.500 [0.459, 0.541] | 0.20 (0) | 0.42 (0) | 0.90 (0) | 0.57 (0.42) | 0.40 (0.25) | 0.68 (0) | 0.90 (0) | 0.72 (0) | 0 (0) |

All 15 families per model, and content accuracy where an item has an answer, are in
`receipts/iftiny_*/<model>/summary.json`.

- **Our models follow almost nothing.** Their macro compliance is at most 0.12 above the bare base rate
  (pick_e1_L4 0.118, pick_e1_60m_L32 0.053, pick_e5_s0 0.023; the others at most 0.079). What they do comply with is lexical: using a given word, opening
  with a given word, repeating a phrase.
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

Every server prompt for the eight paperkiln models equals `chat_template.prompt(..., system=...)` tokenised at home
(1,474/1,474 per model, 224 with a system message; `receipts/iftiny_cuda/prompt_check.json`, and
`prompt_check` in `receipts/iftiny_cuda_bench2/*/summary.json` and `receipts/iftiny_cuda_bench3/*/summary.json`).

Compliance at turn 1 + d (user placement; nohint in brackets):

| model | d = 0 | d = 1 | d = 2 | d = 4 |
|---|---|---|---|---|
| live | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| pick_e1_L16 | 0.083 (0.000) | 0.036 (0.059) | 0.107 (0.155) | 0.357 (0.250) |
| pick_e1_L32 | 0.071 (0.000) | 0.131 (0.107) | 0.143 (0.131) | 0.262 (0.214) |
| pick_pilot_s35 | 0.012 (0.000) | 0.238 (0.107) | 0.298 (0.202) | 0.452 (0.464) |
| pick_e1_L4 | 0.083 (0.000) | 0.202 (0.095) | 0.238 (0.238) | 0.369 (0.393) |
| pick_e1_60m_L32 | 0.083 (0.000) | 0.131 (0.071) | 0.214 (0.202) | 0.321 (0.214) |
| pick_e5_s0 | 0.048 (0.000) | 0.131 (0.119) | 0.190 (0.238) | 0.298 (0.286) |
| pk_sft_v1 | 0.000 (0.000) | 0.191 (0.167) | 0.309 (0.250) | 0.548 (0.476) |
| SmolLM2-135M-Instruct | 0.071 (0.000) | 0.643 (0.452) | 0.869 (0.738) | 0.952 (0.905) |
| Qwen2.5-0.5B-Instruct | 0.691 (0.000) | 0.714 (0.083) | 0.762 (0.298) | 0.845 (0.631) |

The paired differences over the 84 (family, sequence) pairs, with 95% intervals, are in `receipts/*/paired.json`
(`analyze_if.py`).

- **Our models copy the format; the instruction adds little.** Compliance rises with d for every paperkiln model
  except live. The rise comes from the compliant replies in the history, not the instruction: user minus nohint
  is at most +0.13, and for pk_sft_v1 +0.00, +0.02, +0.06 and +0.07 at d = 0, 1, 2 and 4.
- **The bare base rate is near zero** for every model: at most 0.060 in any cell (pick_e1_L4 at d = 4;
  pick_e1_60m_L32 and pick_e5_s0 at most 0.036; the others at most 0.024).
- **The instruction carries in Qwen2.5-0.5B-Instruct.** It follows the instruction itself (0.69 at d = 0) and
  holds it four turns later, still +0.21 over nohint at d = 4. SmolLM2-135M-Instruct gains +0.05 to +0.19 from
  the instruction over the history alone.
- **System versus user placement.** System minus user is within ±0.10 for the paperkiln models. Qwen at d = 0 is
  the one case where placement clearly matters: -0.21, so the folded system message helps less than the same
  text in the user turn.

### Determinism and parity

- **Replay.** Two passes in fresh server processes (`EMBER_DETERMINISTIC=1`, one L4) gave bit-identical
  replies and ids for all 1,474 distinct prompts of each of the eight models (ember fe01b35; e57c0fb for
  pick_e1_60m_L32 and pick_e5_s0). Across pods and builds: pick_e1_L4 at e57c0fb on pk-bench3's L4 gives its
  fe01b35 replies from pk-bench2's L4 bit for bit, ids and text, in both passes (1,474/1,474;
  `receipts/iftiny_cuda_bench3/pick_e1_L4_rerun_check.json`), and pick_e1_60m_L32's 374 replies from pk-bench2
  likewise. The identity holds on one GPU model, not between a T4 and an L4; the GPUs' UUIDs are in each
  receipts folder's `gpu.txt`.
- **Parity.** Every reply that ended on end-of-text was re-scored with `mtstudio score` on the pod. Its argmax
  equals the reply at every position for every model: live 1,474/1,474, pick_e1_L16 1,271/1,271, pick_e1_L32
  1,271/1,271, pick_pilot_s35 1,305/1,305, pick_e1_L4 1,222/1,222, pick_e1_60m_L32 1,234/1,234, pick_e5_s0
  1,314/1,314, pk_sft_v1 1,214/1,214. The rest stopped at 64 tokens and were not checked.
- **A CPU leg** on ember cbdcb915 (`/home/jonat/tc_ember`) gives the same IF-tiny numbers for live and
  pick_e1_L16, with parity against the CPU mtstudio 1,474/1,474 and 1,294/1,294. That commit ignores `/chat`'s
  `system` field: its instr_system prompts equal nohint's. So its distance cells are not used here.
  `tools/iftiny_eval.py` now checks every server prompt against schema v1 and reports a mismatch.

### pk-bench2's IF-tiny job

It finished pick_e1_L4 and stopped at prompt 375 of 1,474 of pick_e1_60m_L32 ('Do not use the word "the" in
your reply. Which animal barks?'), so pick_e5_s0 did not run (`C:/ml_artifacts/benchmarks/runs2/iftiny/`).
ember fe01b35's /chat returned HTTP 500; the job's log holds only that, and the server's stderr was left on the
pod. On CPU, ember cbdcb915 gives the same 500 on the same prompt, with the reason in the body: the greedy reply
stops at 64 tokens inside a Hebrew character (lead byte 0xD7), and nlohmann::json refuses to serialise a string
that is not valid UTF-8 (`[json.exception.type_error.316] incomplete UTF-8 string; last byte: 0xD7`). ember on
CPU and the pod's CUDA give identical ids on the 374 prompts before it (next bullet).

- **ember e57c0fb** (branch `iftiny-utf8`, pushed; fe01b35 plus one commit) writes the reply text through
  `utf8_valid`, which replaces each ill-formed subsequence with U+FFFD as Python's `decode("utf-8", "replace")`
  does; generation and the returned ids are unchanged. Checked on CPU: its unit test (18 cases) and 20,000
  random byte strings against Python; the prompts around 375 answer 200; `tools/iftiny_eval.py` on the first
  400 prompts of pick_e1_60m_L32 gives ids identical to the pod's CUDA replies at fe01b35 on all 374 the pod has.
- **paperkiln**: `tools/iftiny_eval.py` raises with the server's message on an HTTP error
  (`tests/test_iftiny_eval.py`), and `bench_vm.sh` brings each ember server's `ember_stderr.log` home.
- **pk-bench3** reran IF-tiny at e57c0fb (`tools/colab_chat/bench_v3.json`): all three models finished, with
  empty `ember_stderr.log`s. One pick_e1_60m_L32 reply holds U+FFFD, prompt 375's. Its numbers are in the
  tables above.

## Files

- **Sets**: `corpus/eval/iftiny_v1.jsonl`, `corpus/eval/ifdist_v1.jsonl`. The benchmark items are rebuilt by
  `tools/bench_build.py` (data in `C:/ml_artifacts/benchmarks/raw/`, untrusted, read under `python -I`).
- **Tools**:
  - `tools/bench_build.py`, `tools/bench_eval.py` (`run`, `report`, `compare`);
  - `tools/iftiny_build.py`, `tools/iftiny_checks.py`, `tools/iftiny_eval.py`;
  - `tools/colab_chat/bench_v1.json`, `bench_v2.json`, `bench_v3.json`, `bench_vm.sh`, `bench_stage.py`;
  - here: `rebuild_weights.py`, `analyze_if.py`, `bench2.py` (`hellaswag`, `collect`, `ember`, `table`), `specs/`.
- **Receipts**:
  - `receipts/bench/` (summary, meta and per-item records per model; `hellaswag_one_request_per_key.json`;
    `xcheck/`, pk-bench2's files prefixed `bench2_`);
  - `receipts/ember_score/` (ember's summary, meta and /health per model; `compare_*.json`;
    `argmax_flips_ir.json`);
  - `receipts/iftiny_cuda/`, `receipts/iftiny_cuda_bench2/`, `receipts/iftiny_cuda_bench3/`, `receipts/iftiny_hf/`
    (summary, per-item records and both passes' replies per model; `paired.json` per folder);
  - `receipts/engine_checks/`.
- **Full score chunks** (per-token log-probabilities) and the run folders: `C:/ml_artifacts/benchmarks/runs/`
  (pk-bench), `runs2/` (pk-bench2) and `runs3/` (pk-bench3), each job's `results.tar.gz` unpacked in its `x/`.
- **SmolLM2-135M** as an F32 GGUF for ember's /score gate: `C:/ml_artifacts/thirdparty/smollm2-135m/`.

## Open items

- **ember main** still serialises the raw reply (bace837): e57c0fb's change is for ember's owner to take
  there.
- **mtstudio score's speed.** By default it scores one request per forward (~75 ms each for the 14.6M live model
  on the pod, the GPU near idle). `--batch-tokens N` (28 Oct 2026, opt-in) packs requests into one forward and
  runs 7.5-8.1x faster on the 60M pick's Carry-5 and Carry-12 requests on an L4, within ROUNDING of the
  unbatched scores (LOCAL_LLM_STACK.md section 10, 28 Oct); `bench_eval.py` does not pass it yet.


## Attention Residuals CPU diagnostic (30 Oct 2026)

The historical Stage L0 score slowdown (~4.5 versus ~0.8 min per chunk) was not reproduced by
controlled raw-ID scoring on the existing `~/pk_state_build/mtstudio`. Four identical requests,
each 480 prompt + 8 continuation tokens, took mean 25.81 s for Attention Residuals versus
23.69 s for the plain control at two threads (1.09x); one-thread means were 27.41 versus
28.05 s. The control uses the same trained shared weights with only the depth-query tensors
and Attention Residuals setting removed, so these are timing measurements, not a quality test.
The independently recovered original plain pick took 23.31 s at two threads. Both original
pick artifacts were preserved. A short 240-sample profile points mainly to matrix multiplication
and OpenMP work; it does not establish a depth-mixture bottleneck. Laptop contention and the
full original request distribution were not controlled. No numerical or inference code was
changed, and the original workload discrepancy remains on ROADMAP.

[Receipt](receipts/attnres_cpu_2026-10-30.json) records requests, specs, alternating repeated
timings, binary/weight hashes, the recovered plain control and the small profile. No paid
compute, training, benchmark labels or registered analysis inputs changed.

The historical comparison has now been traced to **different scoring backends**. The original
plain `pick_e1_L16/l0.json` records `--scorer ir`, whereas X5 records `--scorer mtstudio`.
Both commands include `--mtstudio /home/jonat/pk_state_build/mtstudio`, but that argument does
not select the backend: the IR branch scores through `IrBackend` and does not invoke that binary.
All 91 chunk key sequences are identical between the two runs. Their median adjacent file-mtime
gaps are 42.74 s and 367.93 s respectively; these are filesystem observations, with different
run dates and uncontrolled contention/resumption, rather than instrumented timings.

This resolves the basis for the roadmap's claimed same-backend regression. The observed latency
difference remains real as a historical observation, but cannot isolate Attention Residuals.
The controlled matched-`mtstudio` result above supports only its tested request/thread settings;
it does not establish general Attention Residuals performance. The
[backend audit](receipts/attnres_cpu_backend_audit_2026-10-30.json) preserves exact historical
arguments, receipt/model/spec hashes and every chunk hash, plus current source hashes explicitly
distinguished from an unavailable historical checkout.
