# End-to-end Mamba-1 prefill, ordered against chunked scan: protocol

Claude Opus 5.5, 3 Nov 2026. Fixed before any GPU result. Branch `claude/mamba-prefill-bench`, cut from master
`da5febf2` (both chunked kernels merged). It runs only after independent review, from the reviewed commit.

## Question and scope

The chunked selective-scan forward is 4.0-7.4x faster than the ordered one as a kernel for single sequences of
512-4096 tokens, and 1.14-1.47x at batch 4-16 (run `r1-20261103`). The question is how much of that reaches a whole
Mamba-1 model's prefill on paperkiln's CUDA path, where the scan is one op among projections, the convolution,
norms and the head.

The claim is limited to **prefill latency**: one forward over a prompt, no backward and no generation. It is
measured on one L4, for two model shapes, with paperkiln's engine as it stands at `da5febf2`, in FP32, with two
GEMM backends. It is not a claim about training speed, decode speed, other GPUs, model quality or other engines,
and it changes no default: `MICROTORCH_SCAN_CHUNK` stays opt-in.

## What is measured

- **Model.** `parity::MambaLM` with `Mamba1Config` on (`tools/parity_model.hpp`), the composed Mamba-1 that the
  31 Oct CUDA lifecycle receipts ran (`docs/receipts/mamba1_cli_cuda_20261031/`). It has:
  - a token embedding;
  - L blocks, each a pre-RMSNorm, `in_proj` d -> 2 Di, the causal depthwise convolution (K = 4) and SiLU,
    `x_proj` Di -> R + 2N, `dt_proj` R -> Di with a bias, the selective scan (softplus(dt), A = -exp(A_log),
    the D skip), a product with SiLU(z), `out_proj` Di -> d and the residual;
  - a final RMSNorm and an untied head over every position.
- **Weights** are the model's own initialisation from seed 1103, and token ids are uniform over the vocabulary
  from seed 7. Both are deterministic.
- **Shapes.** Both use d_state N = 16, expand 2 (Di = 2d), d_conv 4, dt_rank ceil(d / 16), and Mamba's GPT-NeoX
  vocabulary padded to 50280.

  | Shape | d | layers | Di | dt_rank | parameters |
  |---|---:|---:|---:|---:|---:|
  | `d768x24` | 768 | 24 | 1536 | 48 | 167.8M (Mamba-130m's width, depth and vocabulary; Mamba-130m ties the head, 129M) |
  | `d384x12` | 384 | 12 | 768 | 24 | 50.2M |

  **Why these two.** `d768x24` is the smallest released Mamba, and its Di = 1536 is the r1 kernel shape. The
  second shape is chosen to bracket the scan's share. Per token, the scan's work is proportional to Di N, while the
  projections' is proportional to d Di, so halving d doubles the scan's share of the block FLOPs. The ordered
  kernel's parallelism is batch x Di threads, so the narrower model is also where r1 found the chunked kernel's
  largest gains (its Di = 768 is r1's (1024, 8, 768) shape). The vocabulary is the same for both shapes, so the
  head is the same op, and its GEMM is reported separately (below).
- **Cases** (sequence length, batch): (512, 1), (1024, 1), (2048, 1), (4096, 1), (2048, 4).
- **Variants:** ordered (`MICROTORCH_SCAN_CHUNK` unset), L = 64 and L = 128. The tool sets the switch per arm;
  the device path reads it on every call.
- **Engine configuration.** This is mtstudio's on-device evaluation (25 Oct 2026): device ops, step residency and
  deferred downloads, with one step window per case. The GEMM weights upload once per window, and the warm-ups
  absorb that. Activations stay on the device.
- **GEMM backends**, in separate processes: the engine's default tiled FP32 kernel, and cuBLAS FP32
  (`MICROTORCH_CUBLAS=1`, the README's GPU quick start). TF32 is off in both.
- **Host work included in every arm alike.** These are paperkiln's Mamba-1 forward today, and they are measured,
  not removed:
  - `dt_proj`'s bias add has no device kernel (`ops::add_bias`), so the [rows, Di] projection downloads, the bias
    is added on the host, and the scan uploads the result. That is about 50 MB each way per layer at 8192 rows.
  - A = -exp(A_log) is computed on the host in every layer (`ops::exp` and `ops::scale`, Di x N each).
  - The embedding table is uploaded on every forward (`embed_gather`'s operand; 154 MB at d768), as are the small
    norm, convolution and skip parameters.
  - The scan op writes every state, (rows + sequences) x Di x N floats (805 MB per layer at 8192 rows of d768),
    which the backward reads, even under NoGrad. This is device work inside the scan's own time.

  These costs dilute the end-to-end ratio. The profile does not measure host time. The wall time minus the
  device ops' summed span time is a **residual**, which mixes:
  - host work and transfers;
  - allocation;
  - idle device gaps;
  - instrumentation gaps.

  Reading the residual as time outside the device ops assumes that the spans do not overlap and cover the device
  work. They are sequential on one stream, but uploads made before a span opens fall outside it.

## Gates (all required)

1. **Host test** (`tests/test_mamba_prefill_bench.cpp`), locally (`receipts/prefill_host_test_local.json`) and again
   on the VM at the run's commit. A prerequisite. It checks:
   - the tool's argument handling, the comparison metric, the variant rotation and the switch;
   - on the CPU path at the tiny shape (d 64, 2 layers, vocabulary 512, two sequences of 300 rows), that the tool's
     replicas `prefill()` and `block_forward()` equal `MambaLM::forward` and `Mamba1Block::forward` bit for bit;
   - that the chunked scan's host emulation inside the model, at L = 64, 128 and 7, all with short tails, stays
     within the tiny shape's bound of the ordered op;
   - that a scan that drops its chunk starts, and one that carries state across the sequence boundary, both
     exceed that bound.

   ctest also runs the tool's four modes on the CPU path as dry runs (bookkeeping only).
2. **Provenance.** Exactly one NVIDIA L4. The image's own CUDA toolkit, recorded (`nvcc.txt`, `cuda_dirs.txt`), as
   run b1 did (CUDA 13.0.88). No toolkit is installed. A prerequisite.

   **2b. Resources** (`resources`). An operational stop before any heavyweight step, checked twice in the same
   session: at the start, and again just before the correctness gate.
   - **Host.** MemAvailable from `/proc/meminfo`, or the cgroup's headroom (limit minus current) when it has a
     limit, if that is smaller, must be at least 16 GiB. The gate's batched d768x24 case holds about 7.4 GB of live
     result objects (two recorded runs of 2.86 GB and a live 1.65 GB logits matrix). With the models, parameters
     and allocator retention it is estimated at about 9 GB, unmeasured; 16 GiB leaves headroom.
   - **Device.** At least 12 GiB free (`nvidia-smi`), against the about 5 GB estimate.
   - **Records.** `/proc/meminfo`, `free -m`, the device's memory query and a summary are kept for each check
     (`resources_LABEL.json`), and once more at the end.
   - **Build jobs** are min(nproc, host headroom / 3 GiB), at least 1.
   - **Per step.** The build, the host test and every bench step run under `measure.py`, which records the elapsed
     time, CPU time and peak RSS (`STEP.time.json`).
3. **Engine unchanged.** The commit descends from `da5febf2`, and `src/`, `include/` and `tools/parity_model.hpp`
   are byte-identical to it. The kernels measured are then the ones that r1 and b1 gated, and the model is the one
   the lifecycle receipts ran. A prerequisite.
4. **Correctness before any timing**, once per GEMM backend (`prefill_gate`, then `prefill_gate_cublas`). Each
   is a prerequisite: a failure stops all timing. The checks:
   - **Tiny shape against the CPU path.** On the device, ordered, L64 and L128 are each within the bound of the
     CPU path, for the logits and every block's scan output. The 300 rows span 4 chunks and a 44-row tail at
     L = 64, and 2 chunks and a tail at L = 128. The replicas equal `MambaLM::forward` bit for bit on the CPU and on
     the device. A malformed switch value (`" 2"`) is refused with the parser's message.
   - **Each real shape against the CPU path.** At (512, 1), ordered, L64 and L128 on the device are within the
     bound of the CPU path.
   - **Each real shape and timed case on the device:**
     - L64 and L128 are within the bound of ordered;
     - an ordered repeat is bit-identical;
     - one chunk per sequence (L = sl) is bit-identical to ordered;
     - an L128 repeat is bit-identical (64-bit fingerprint);
     - the timed function's logits at L128 equal the gated ones (fingerprint);
     - every value is finite.
5. **The timings and the profile complete**, each a gate (`prefill_time`, `prefill_time_cublas`,
   `prefill_profile`, `prefill_profile_cublas`). Every timing sample must be finite and positive. The profile
   checks each repetition on its own and reports every failure:
   - the wall time, the scan's device time and the device-op sum are finite and positive;
   - every block's scan and convolution are counted on the device once per forward (`selective_scan_fwd` and
     `conv1d_causal_fwd` spans equal to the layer count), which shows that no arm fell back to the host scan;
   - the medians are finite and positive.

   Each repetition's counts are kept.

### The metric and the bound

err(a, ref) = max_i |a_i - ref_i| / rms(ref). It is computed over all logits, and separately over each block's
scan output y, taking the worst block. Any nonfinite value, or a size mismatch, fails.

The bound is **1e-4 for the 2-layer tiny shape and 1e-3 for the 12- and 24-layer shapes.** It must do two things:
- **Admit float32 reordering** propagated through the depth. That reordering is the chunked scan's different
  rounding of the chunk starts, and the device's arithmetic against the host's (CUDA `expf`/`log1pf`, another GEMM
  summation order).
- **Refuse structural errors**: a lost chunk carry, state crossing a sequence boundary, a wrong tail or a stale
  operand.

The logits alone are a weak witness at these weights. The D skip carries most of y, and each block adds a small
update to the residual stream, so every block's y is compared as well.

Host evidence (`receipts/prefill_calibration_local.json`; the chunked forward's host emulation inside the model,
on the CPU, against the ordered op):

| Shape | bound | reordering (chunked emulation), worst: logits / scan | chunk starts dropped, least: logits / scan |
|---|---:|---|---|
| tiny (host test: 2x300 rows at L 7, 64, 128; dropped carry at 2x256) | 1e-4 | 7.6e-7 / 3.9e-7 | 4.9e-4 / 2.5e-3 (merged sequences: 0.14) |
| `d768x24` (512, 1024 and 2x512 rows, L 64 and 128) | 1e-3 | 7.3e-6 / 7.6e-6 | 0.26 / 0.35 |
| `d384x12` (same) | 1e-3 | 4.9e-6 / 3.3e-6 | 0.054 / 0.061 |

The reordering error grows with depth, from about 1e-6 at 2 layers to about 1e-5 at 24. At every shape, the bound
sits more than 130 times above the measured reordering.
- **Deep shapes.** A dropped carry exceeds the bound at least 54 times over, on the logits alone.
- **Tiny shape.** A dropped carry exceeds the bound 25 times over on the scan outputs, but only 4.9 times on the
  logits.

In every case the one-chunk emulation equals the ordered op bit for bit, and the replicas equal the model's
forward. The calibration took 45 minutes on 4 local cores.
- **Assumption.** The device's own reordering is in the same class as the chunk-start rounding: a few ulps per op,
  propagated through the same depth. The CPU-agreement checks test this assumption directly.
- **Fixed in advance.** If they exceed the bound, the gate fails, the failure is preserved, and the bound is not
  relaxed after the result. The kernel-level contract (3e-5 against binary64) stays the scan's own. This bound is a
  pass/fail criterion for these model fixtures, not a general error bound.

## Timing schedule (reported, not a gate beyond completing)

- **Timed region.** Each sample is one whole prefill: `device::synchronize()`, start the clock, `prefill()` (the
  model's forward, logits for every position, left on the device), `device::synchronize()`, stop the clock. It is
  wall time from a steady clock, host work and transfers included. Freeing the outputs falls outside the span.
- **Schedule.** Within a case's step window, 2 warm-up rounds are run, then 15 timed repetitions. The kernel
  protocols used 20 after 3; 15 here keeps both GEMM backends' timings inside the polling window. Each round runs
  the three variants once, in an order rotated by the round number, so that over every three rounds each variant
  takes each position once. The cases and shapes run in the fixed order above.
- **Reported** for every shape, case and GEMM backend: the median, minimum, maximum and all samples per variant;
  ordered / variant at the median; the pool's peak and cached bytes; and the device's bytes in use. No cell is
  dropped or rerun.
- **The profile** (`MICROTORCH_GPU_PROFILE=1`, its own processes) runs 2 warm-up rounds and 5 profiled
  repetitions per variant, with the same rotation. It uses CUDA events around every device op and GEMM, summed per
  op name. The head's GEMM is tagged `gemm_head`. Medians are reported for:
  - the wall time;
  - each op's device time and the sum over ops;
  - the scan's time (`selective_scan_fwd`) and the head's time;
  - the scan's share of the wall time and of the device-op time.

  The events add a little time to the profiled forwards, so the profile gives the decomposition, and the timing
  run gives the latency.
- **Derived in the result, not gates.** The Amdahl check compares the wall-time saving with the scan's
  device-time saving. The share of the r1 kernel gain that reaches the prefill is also derived.

**Outcomes.** A ratio near 1 at some shape or case is a valid result. It means that the scan is not where that
prefill's time goes, and the profile says where it goes. A default change, a training-speed claim or a decode
claim needs its own protocol.

## Budget and procedure

- **Command.** Run from a paperkiln checkout's root, in a foreground WSL process, after the reviewed commit is
  pushed (the VM fetches `vm_run.sh` at that commit):
  `python3 experiments/mamba_parallel_scan_2026-11-03/drive.py FULL_COMMIT_SHA RUN_ID prefill`, with RUN_ID
  `p1-YYYYMMDD`.
- **Cap.** One Colab L4 session, AGENTS.md path C, named `claude-scan-RUN_ID`. At most 60 minutes from the
  driver's start, within Claude's Colab lane (stack page: one L4 at a time, at most 100 compute units a week).
- **Expected time:** about 30-38 minutes from the driver's start, against a polling end at 43. The parts:
  - session creation and the launch, 3-5 minutes;
  - one CUDA build of the library with 2 targets, its jobs bounded by memory, 5-8 minutes (r1's whole run took
    6 minutes).
  - the two gates, 3-5 minutes each. Each gate runs:
    - per real case, 7 device forwards, so 70 over the 10 cells: ordered, an ordered repeat, one chunk, L64, L128,
      an L128 repeat and the timed-path L128, each recorded with its logits and 24 or 12 scan outputs downloaded,
      compared and fingerprinted;
    - 6 device forwards for the real-shape CPU agreement and 2 CPU reference forwards (d768x24 at 512 rows is the
      largest, about 20-40 s on the VM's CPU);
    - at the tiny shape, 3 recorded device forwards, the replica checks and the malformed-switch forward.
  - the two timing runs, 4-6 minutes each: 17 rounds of 3 variants over 10 cells. The sum of one variant's forwards
    over the cells is estimated at about 3.8 s at d768x24 and 1.2 s at d384x12; a d768x24 forward of 8192 rows is
    estimated at 1-1.5 s.
  - the two profiles, 1.5-2 minutes each: 7 rounds of 3 variants.

  These are estimates; `STEP.time.json` records each step's actual time. The steps run in that order, so if
  polling ends first, the later steps are the ones lost. The schedule is not extended to obtain completion.
- **Memory.** These are estimates, unmeasured until this run, which records them:
  - **Device peak:** about 5 GB, at d768x24 (2048, 4): logits 1.65 GB, one layer's 805 MB of states and the
    activations. Each timing row records the pool's peak and cached bytes and the whole device's use
    (`cudaMemGetInfo`), which includes allocations outside the pool.
  - **Host peak:** about 9 GB, in the gate's batched case: about 7.4 GB of live result objects plus the models.
    `measure.py` records each step's peak RSS.

  Gate 2b stops the run if either is short.
- **Deadlines** from the driver's start:
  - polling ends at 43 minutes;
  - attempts to stop the VM workload end at 47;
  - retrieval ends at 51;
  - shutdown attempts end at 56;
  - the final inventory ends at 60.

  Stopping has its own window before retrieval's. A stop whose replies hang therefore uses at most 4 minutes (one
  full attempt, which needs about 3), and packing and download keep their 4 minutes. Without that, until retrieval,
  the VM holds the only copy of the evidence, and cleanup would destroy it. Every external call has a timeout
  capped by its deadline, timed-out children are killed, and a call past its deadline is not started.
  - **If polling ends without DONE:** the run is stopped through the VM script's PID, and the receipts so far are
    retrieved before shutdown. An unconfirmed stop keeps the exit at least 2.
  - **Forward and backward** keep their schedule (polling 60; stop and retrieval 75; shutdown 85; inventory 90).
    Their two stop attempts, at most 2 x 260 s, leave retrieval at least 6 of its 15 minutes.
- **Session ownership, shutdown and the launch and stop hardening** are the backward protocol's, unchanged:
  - preflight listings;
  - creation reconciled against the listing;
  - shutdown certified only by an authoritative listing;
  - the atomic script publication and run-directory claim;
  - the reconciled launch;
  - idempotent retries;
  - per-attempt extraction;
  - the PID-based stop with process groups.
- **The VM:**
  - checks out the full commit SHA and verifies HEAD and a clean tree before and after the build;
  - checks gate 3, configures as r1 and b1 did (`-DMICROTORCH_CUDA=ON -DMICROTORCH_DEVCHECK=ON
    -DCMAKE_CUDA_ARCHITECTURES=89`), and builds `mamba_prefill_bench` and `test_mamba_prefill_bench`;
  - runs every bench step with `MICROTORCH_SCAN_CHUNK`, `MICROTORCH_TF32`, `MICROTORCH_GPU_PROFILE` and
    `MICROTORCH_CUBLAS` unset, except where the step sets one. The tool refuses the scan switch or TF32 set from
    outside, refuses a GPU mode on a build without CUDA and DEVCHECK, and refuses the profile switch in the wrong
    mode.
  - writes `status.json` atomically, then `DONE`, on every exit path once the run directory exists. A failed
    prerequisite (provenance, resources, checkout, engine unchanged, configure, build, host test, either gate)
    stops further paid work. A failed timing or profile step is recorded, and the remaining steps still run.
- **Validation.** The driver validates `status.json`:
  - protocol `prefill`, run, commit and HEAD;
  - exactly the 13 required gates (`provenance`, `resources`, `checkout`, `engine_unchanged`, `configure`, `build`,
    `host_test`, `prefill_gate`, `prefill_gate_cublas`, `prefill_time`, `prefill_time_cublas`, `prefill_profile`,
    `prefill_profile_cublas`), all integers equal to 0;
  - aggregate exit 0, and the matching `DONE` marker seen.

  Exit codes: 0 validated, 1 experiment failed (receipts kept), 2 transport, receipt, status-identity or
  stop-confirmation failure, 3 cleanup unresolved.
- **Outputs** come home to `/mnt/c/ml_artifacts/mamba_parallel_scan_20261103/RUN_ID/`. After the VM is gone, the
  top-level files of `out/` are copied to `receipts/RUN_ID/`, with `archive.sha256` hashing every retrieved file,
  and then the driver log. The JSONL files are `prefill_gate[_cublas]`, `prefill_time[_cublas]` and
  `prefill_profile[_cublas]`.
- **Tests.** `test_drive.py` (mock Colab CLI, 23 tests) and `test_vm_run.py` (stub tools, 25 tests)
  test that bookkeeping locally for all three protocols. For prefill they cover:
  - the gate set and the deadlines;
  - a hung stop that still leaves retrieval before shutdown, with the exit at least 2;
  - the per-step environment and the resources stop (host, cgroup and device);
  - build jobs bounded by memory;
  - `measure.py`'s record and exit status;
  - the prerequisite stops before timing;
  - a timing failure that still collects the rest;
  - a stop during timing.
- **Compiled, not run, locally:** a CUDA 12.6 build of both targets (`-DMICROTORCH_CUDA=ON -DMICROTORCH_DEVCHECK=ON`,
  sm_89). No GPU was used.
- **No reruns.** The run is not repeated to improve a result. An operational failure (session lost, build failure)
  is recorded as such and may be rerun once, unchanged, under a new RUN_ID.
