# bf16_gate_v2: the pass rule, stated before any result (25 Oct 2026)

Written and committed before `tools/colab_chat/bf16_gate_v2.json` runs; no v2 number exists yet.

## Why a noise floor

bf16_gate_v1 (RunPod L4, `C:/ml_artifacts/bf16_gate/`) compared one run per config with one FP32
run. FP32 flash attention equals the tiled attention to float rounding (device legs: y 7.2e-7,
dqkv 5.7e-7 relative), yet it moved the final validation loss by 0.66% over 300 steps: a single
paired run measures how far two trajectories drift apart, not the precision error. v1's
tolerances (flash 0.5%, bf16 2% / 1%) had no noise reference, so this gate replaces them.

## Design

- Shape 20m_L8 (d 448, L8, H7, d_ff 1248), batch 32, accum 1, T 512, 300 steps, lr 1e-3 (its
  registered Stage 0 rate), warmup 30, cosine to 0.1, clip 1, AdamW, one validation at step 300
  (16 windows), FineWeb-Edu sample text (first 20 MB), the 16k Llama 3-scheme vocabulary, the
  production device environment (jobs_vm.py ENV), one NVIDIA L4.
- Seeds 1, 2, 3 (train.seed: initialisation and data order) for FP32 tiled, and for flash, bf16
  and bf16_flash at the same seeds.
- Plus 60m_L8 (d 768, L8, H12, d_ff 2240), lr 5e-4, one seed, all four configs: reported only.
  One seed has no spread, so no verdict; it shows whether the BF16 gap grows with width.

## The rule

V_c,s is the final validation loss of config c at seed s, MA_c,s(t) the 20-step moving average of
the training loss at step t, n = 3.

- Primary: |mean_s (V_c,s - V_fp32,s)| <= R_V, where R_V = max_s V_fp32,s - min_s V_fp32,s (the
  FP32 seed-to-seed range of the final validation loss).
- Secondary: mean over t = 50..300 of |mean_s (MA_c,s(t) - MA_fp32,s(t))| <= mean over the same
  steps of (max_s MA_fp32,s(t) - min_s MA_fp32,s(t)).
- A config passes when both hold, every run exits 0, and every run records its setting (model
  event and result.json; the GGUF key paperkiln.train.precision for bf16).

Why the multiplier is 1 (the range itself). Under the null hypothesis (the config changes only
rounding, so the matched-seed runs decorrelate like independent seeds, as v1's FP32 flash run
shows they do within 300 steps), each matched difference has standard deviation sqrt(2) sigma,
with sigma the seed-to-seed standard deviation, and the mean of three has sqrt(2/3) sigma = 0.82
sigma. The expected range of three normal draws is 1.69 sigma, about 2.1 times that standard
deviation. So the primary rule is close to a two-sided test at about two standard deviations of
the null: a config whose final loss is shifted by more than the FP32 seed range fails. The
secondary rule compares a typical curve offset with the typical FP32 spread over the run; under
the null its left side is about 0.4 of its right, so it fails only a systematic offset over most
of the run. The range of three draws is itself a noisy estimate; that is the price of three seeds
and is stated rather than hidden.

The verdicts are computed by `tools/precision_gate.py noise` (function noise_verdict), committed
with this file.
