# Score audit: CUDA `mtstudio score` and the registered Carry-5 Y (26 Oct 2026)

**No registered number moves.** The CUDA `mtstudio score` was right. The 12-nat disagreement on HellaSwag 17234
came from the benchmark harness: two different HellaSwag items share the id `hellaswag/17234`, so the
cross-check compared the CUDA score on one item with hfgguf's score on the other. No Y moves by more than
0.005; checked against the numpy IR, no Y moves at all. Carry-5's ids are unique (1,032 of 1,032), so the
collision cannot reach Y.

The fix is in `tools/bench_eval.py` (paperkiln fafc4750). The defect it fixes does change the benchmarks
campaign's HellaSwag numbers: they are low by 0.003-0.005 acc and 0.004-0.011 acc_norm (section 3).

## 1. What happened on HellaSwag 17234

The HellaSwag validation split's `ind` field is not unique. `tools/bench_build.py` uses it as the item id, so
`bench_v1.jsonl` (sha256 978548d1...) holds 433 ids that each name two items. `bench_eval.py` stores results under
`id|choice`, so the second item's scores replace the first's.

- The full hfgguf run (every item) therefore holds, under `hellaswag/17234|0..3`, the scores of the second item
  (WikiHow, "Choose jeans...", continuations of 30, 34, 56 and 45 tokens).
- The every-400th cross-check sampled the first item (ActivityNet, "Clean and jerk...", continuations of 15, 8,
  12 and 6 tokens); it is the only sampled item whose id repeats. The pod's CUDA `mtstudio score` scored that item.
- `bench_eval.py compare` zipped the two log-prob lists, which have different lengths, so it compared the first
  item's positions with the first positions of the second item and reported differences of up to 12 nats. The
  argmax lists in the two records are 15 and 30 long.
- The numpy IR at home on pick_e1_L16's GGUF (`receipts/ir17234.txt`):

| item under `hellaswag/17234` | IR vs the pod's CUDA mtstudio | IR vs hfgguf |
|---|---|---|
| first (what the CUDA cross-check scored) | max \|dlogprob\| 1.2e-5 | different requests |
| second (what the hfgguf record holds) | different requests | max \|dlogprob\| 1.3e-5 |

On the whole cross-check chunk 0 (128 requests in the pod's order, 17234 at rows 12-15), the pod's CUDA output
agrees with the IR to 1.9e-5 and on argmax at 3,176 of 3,176 positions (`receipts/xcheck_chunk0_cuda_vs_ir.txt`).
The CPU and IR checks that "gave hfgguf's numbers on that item" (RESULTS.md) agree with the second item to the
same precision, so they most likely scored the second item.

Reading the code turned up no state that crosses requests in `score_cmd` either. Each request builds its own
context and calls `lm.forward` (no KV cache by default). Outside a step window, the device pool's buffers
(`pooled_matmul`) and the value cache carry no values. The same CUDA scorer gave the same Carry-5 results at two
commits: E1 at dc29b5e6 (before pooled_matmul) and the control re-run at d53c6939 (after it) have identical
`carry5` and `carry5_base` summaries for all three seeds.

## 2. Carry-5 Y, per run

Y (amendment 2) is forced-choice acc with self history minus twin acc, over the 888 items of recall, correction,
topic and knowledge. The VM runs scored every request with the pod's CUDA `mtstudio score`. Their per-item raw
sums (`carry5/m/items.jsonl`) were checked at home against the numpy IR on each run's GGUF, rebuilding the
requests from the run's `gen.jsonl` with `carry5_eval.requests`. "Gap" is the stored margin between a choice's
top two candidates. A choice can flip only if a scoring error reaches about half its gap.

| run | stored Y | weights home | IR spot check (requests, max \|dsum\|, choice flips) | choices with gap < 1e-3 / < 1e-2 | every gap < 1e-2 rescored by the IR | Y from the IR | change |
|---|---|---|---|---|---|---|---|
| arm X control, seed 1 (= E1 L16 r1, pick_e1_L16) | -0.0248 | yes (pick GGUF) | 192, 2.8e-5, 0 | 2 / 4 | 4 of 4 agree (smallest gap 1.4e-4) | -0.0248 | 0 |
| arm X control, seed 2 | -0.0687 | no | - | 0 / 1 | - | - | bound 0.0011 |
| arm X control, seed 3 | -0.0394 | no | - | 1 / 8 | - | - | bound 0.0090 at 1e-2, 0.0011 at 1e-3 |
| X5, seed 1 | -0.0529 | yes | 64, 1.8e-5, 0 | 0 / 9 | 9 of 9 agree | -0.0529 | 0 |
| X5, seed 2 | -0.0597 | yes | 64, 1.7e-5, 0 | 0 / 3 | 3 of 3 agree | -0.0597 | 0 |
| X5, seed 3 | -0.0507 | yes | 64, 1.3e-5, 0 | 1 / 5 | 5 of 5 agree | -0.0507 | 0 |

How to read it:

- **Runs whose weights are home.** Every choice whose stored gap is below 0.01 was rescored by the IR, and none
  flips. Every other choice has a gap of at least 0.01, against observed errors of at most 2.8e-5. So the CUDA
  choices are the IR's choices, and Y is unchanged to the last item.
- **Control seeds 2 and 3.** Their SFT weights are not home. E1's post dirs keep only the summary and the log;
  the sha256s in the summaries are cf38b327... and d8570ce3.... In the stored sums, no choice of seed 2 has a gap
  below 1e-3, and one of seed 3 does. At the error seen everywhere else (< 3e-5), no choice can flip in either
  seed. They also equal E1 r2 and r3 bit for bit in every Carry-5 summary field, scored on another pod at another
  commit.
- **The IR is an independent engine:** keramos's numpy reference on the exported GGUF, not paperkiln's C++.

The receipts are in `score_audit_2026-10-26/receipts/`: `spot_*.json`, `close_*.json` and `margin.json`.

The brief asked for a CPU `mtstudio score` pass over all 7,104 Y requests per run. It has not run: paperkiln's 4
budget slots are held by Stage L0's pick_e1_60m_L32 (about 10 minutes per chunk, 91 chunks). The IR checks above
were each under a minute on one thread. Everything that pass needs is in place, so it runs when slots free (from
the root):

    python scripts/cpu_budget.py run --session paperkiln --job "score audit: CPU mtstudio (d53c6939) Carry-5 rescoring" --cores 2 \
      --resume "rerun the same command (the build and every chunk resume)" -- wsl.exe --cd ~ -- bash -c \
      "bash /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/experiments/tiny_chat_2026-10/score_audit_2026-10-26/build.sh && \
       bash /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/experiments/tiny_chat_2026-10/score_audit_2026-10-26/score_all.sh"
    python microtorch/experiments/tiny_chat_2026-10/score_audit_2026-10-26/compare.py   # receipts/cpu_rescore.json

`build.sh` builds the CPU mtstudio at d53c6939 (the X5 and control post jobs' commit) from
`C:/ml_artifacts/score_audit/work/sa_src_d53c.tar` into `~/sa_build_d53c` on ext4. `score_all.sh` scores the
rebuilt chunks (`work/sa_{ctl_r1,x5_r1,x5_r2,x5_r3}/`), one chunk at a time. `compare.py` reports, per request, the
CPU sum against the stored CUDA sum, and Y from both. Delete `~/sa_build_d53c`, `~/sa_src_d53c`, `~/sa_models` and
`~/sa_out` afterwards.

## 3. The benchmarks campaign's HellaSwag

Every engine in the benchmarks campaign (hfgguf, the CUDA mtstudio, HF transformers) scored all 10,042 items, but
each first item under a repeated id was credited with the second item's sums. Every first item sits in a different
2,048-request chunk from its second, so its own scores are still on disk. With them restored
(`receipts/hellaswag_duplicate_ids.txt`):

| run | reported acc | corrected acc | reported acc_norm | corrected acc_norm |
|---|---|---|---|---|
| pick_e1_L16 (hfgguf) | 0.2618 | 0.2645 | 0.2617 | 0.2662 |
| pick_e1_L32 (hfgguf) | 0.2603 | 0.2635 | 0.2626 | 0.2669 |
| pick_pilot_s35 (hfgguf) | 0.2577 | 0.2610 | 0.2617 | 0.2661 |
| pk_base_v1 (hfgguf) | 0.2798 | 0.2845 | 0.2922 | 0.2986 |
| pk_sft_v1 (hfgguf) | 0.2799 | 0.2853 | 0.2872 | 0.2949 |
| live (CUDA mtstudio) | 0.2579 | 0.2609 | 0.2642 | 0.2695 |
| Pythia-160m (HF) | 0.2837 | 0.2887 | 0.2979 | 0.3056 |
| SmolLM2-135M (HF) | 0.3514 | 0.3565 | 0.4251 | 0.4356 |

The other sets have unique ids. `experiments/benchmarks_2026-10-26/RESULTS.md` (its HellaSwag column and the open
item on CUDA `mtstudio score`) is for its owner to update. pk-bench2 is pinned at a7333919, which does not have the
fix. Until it is repinned to a commit that does, its every-400th cross-check will flag 17234 again and its
HellaSwag will carry the same offset.

## 4. Fix and tests

- **`tools/bench_eval.py`** (fafc4750):
  - `unique_ids` renames the n-th item under an id (n >= 2) to `id#n`; the first item keeps its id.
  - `requests()` refuses a key that names two requests.
  - `compare` refuses shared keys whose continuations differ in length.
  - `tests/test_bench_eval.py` covers each of these (offline, a fake tokeniser), and checks that `bench_v1.jsonl`'s
    ids are unique after `unique_ids`. That check is skipped without the file.
- **The invariant**, whatever the cause: a request scored within a sequence equals the same request scored alone.
  - `tools/score_sequence_check.py` scores an input file three ways, each in its own process: in order, in reverse,
    and one request per process. It gates on sums, log-probs and argmax. `--ref` also compares the log-probs with
    another engine's output.
  - `tests/test_score_sequence.py` runs it on a CPU build with a trained model (`MTSTUDIO_BIN`, `SCORE_SEQ_SPEC`).
    Its requests have lengths that rise past every earlier one, fall, repeat, jump, and one runs past T + 1.
  - It passed bit for bit on 26 Oct 2026 (`~/pk_state_build/mtstudio`, pick_e1_L16).
- **CUDA leg, waiting for the RunPod top-up.** The root `scripts/colab_driver.json` entry `pk-score-audit` is
  disabled, with a $3 cap. It runs `tools/colab_chat/score_audit_v1.json` and `score_audit_vm.sh` on one L4, in the
  post jobs' `MICROTORCH_*` environment, on:
  - the cross-check's chunk 0 plus the second 17234 item (pick_e1_L16);
  - Carry-5 chunk 0 of control seed 1 and of X5 seed 1.

  Each runs in order, reversed and one request per process, against the IR references in
  `C:/ml_artifacts/score_audit/inputs/cpu_*.out.jsonl`. Expected: every gate passes; the pod and IR differ by
  about 2e-5.
