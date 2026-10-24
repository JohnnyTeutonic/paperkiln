# Floor pilot (amendment 3)

`verdict.json`: the floor pilot's verdict (20m_L8, 35% dialogue, seed 99, lr 1e-3; Colab L4, paperkiln 68eea803),
written by `analyze.py --pilot-verdict`. PASS. Seed 99 is excluded from every analysis.

## The pilot repeated on a RunPod L4 (implementation note of 24 Oct 2026)

`runpod_comparison.json`, written by `compare_providers.py`: the same jobs file (`tools/colab_chat/tiny_chat_pilot.json`)
run with `runner_multi.py --backend runpod` at paperkiln 08caeac1 on RunPod L4s (EU-RO-1; driver 595.91, nvcc 12.4,
image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, digest in the file), artefacts in
`C:/ml_artifacts/tiny_chat/pilot_runpod`.

- Verdict PASS on both providers, by `analyze.py`'s own gate.
- Pretraining is bitwise identical: the final weights' sha256, and all 10,020 step and eval records (loss, gradient
  norm, validation loss; best validation 3.427698850631714 on both). The RunPod run was killed twice on purpose
  (SIGTERM at step 3000, SIGKILL at step 5500) and resumed from its network volume on new pods, optimiser state
  included; the result is still identical.
- The four rungs (500, 1000, 2500, 5000) are identical in every field.
- The post job's SFT differs by code, not by provider: the Colab pilot ran the host masked cross-entropy, the
  RunPod repeat the device kernel that the E runs use (4d2d1427, one fp32 sum in a different order). SFT validation
  differs by 6e-5 to 1e-4 nats. Twin accuracy (0.5315) and Y (-0.0507) are identical; twin minus question-alone on
  the context families is 0.3031 against 0.3019 (one item of 828, 0.04 standard errors).
- Wall-clock is a host property and is reported, not compared: pretraining at 25.1k (Colab) and 24.7k (RunPod) tokens/s
  with the rung jobs beside it; the campaign took 7.5 h on Colab (its SFT on the slow host path took 1.96 h) and
  4.3 h on RunPod (SFT 0.29 h). RunPod spend for the pilot: $2.50 by the runner's ledger.
