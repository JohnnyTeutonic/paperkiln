# Arm X5: Attention Residuals on the llama base, on Carry-5 (results, 26 Oct 2026)

`PREREGISTRATION.md` amendment 5. Cell X5 = 20m_L16 + `llama_attnres` (full form at sublayer level, 33 depth
sources, zero-initialised pseudo-queries, scale "sources"), seeds 1-3. Everything else is E1's L16 cell's,
including its learning rate (1e-3). The control is E1's three home L16 runs, paired by seed. The runs were
trained at paperkiln d53c6939 on RunPod L4s, one machine per seed. The numbers below come from `analysis_x5.json`
(`analyze_x5.py`).

**The scorer, audited.** Carry-5 forced choice is scored by CUDA `mtstudio score` on the VMs. A suspected
request-sequence fault in that path was traced to a duplicate item id in the benchmark harness, not the scorer
(`../SCORE_AUDIT_2026-10-26.md`).
- Every X5 seed and control seed 1, checked against keramos's numpy interpreter on each run's GGUF: Y unchanged
  to the last item.
- Control seeds 2 and 3 (SFT weights not home): no choice can flip at the observed error of under 3e-5.
- The audit's CPU `mtstudio` pass over all 7,104 requests per run waits for CPU slots.

## HX5-main (registered): EQUIVALENT

D = Y(X5) - Y(L16) over Y's 888 items x 3 seeds (2,664 item-level paired differences):

| | value |
|---|---|
| mean D | -0.0101 |
| 95% interval | [-0.0337, +0.0134] |
| 90% interval | [-0.0299, +0.0096], inside +-0.03 |
| verdict (Stage L0's rule) | **EQUIVALENT** |
| validity gate | both cells pass (twin over the primary families 0.543 and 0.515; twin minus question alone on the context families 0.300 and 0.269) |
| seeds with D > 0 | 1/3 (seed 1 -0.028, seed 2 +0.009, seed 3 -0.011) |
| the programme's seed-level t (descriptive) | mean -0.010, t -0.94, NOT_RESOLVED |

Jonathan's registered prediction (D > 0 beyond 3 points) is not supported. The registered verdict is
EQUIVALENT: no main effect on Carry-5 carry-over larger than the SESOI. The 90% interval's lower end, -0.0299,
lies 0.0001 inside the bound.

**Robustness of the verdict (descriptive; `cluster_bootstrap.py`, `cluster_bootstrap.json`).** The registered
test treats the 2,664 item-seed pairs as independent, and its verdict stands as registered. Carry-5's items share
answer frames and slots, so the pairs were also resampled by cluster (10,000 resamples, seed 0; whole clusters
with replacement), with Stage L0's rule applied to each interval:

| items | clusters | 90% interval | 95% interval | rule applied |
|---|---|---|---|---|
| Y's four families | item (888; its three seeds together) | [-0.0300, +0.0098] | [-0.0338, +0.0139] | NOT_RESOLVED |
| | answer frame (76) | [-0.0354, +0.0104] | [-0.0408, +0.0144] | NOT_RESOLVED |
| | slot (13) | [-0.0351, +0.0163] | [-0.0393, +0.0214] | NOT_RESOLVED |
| | family x slot (19) | [-0.0361, +0.0151] | [-0.0414, +0.0195] | NOT_RESOLVED |
| the planted-fact families (recall, correction, topic return; mean -0.0036) | item (828) | [-0.0242, +0.0173] | [-0.0278, +0.0213] | EQUIVALENT |
| | answer frame (19) | [-0.0268, +0.0170] | [-0.0308, +0.0212] | EQUIVALENT |
| | slot (12) | [-0.0276, +0.0227] | [-0.0318, +0.0274] | EQUIVALENT |
| | family x slot (18) | [-0.0290, +0.0212] | [-0.0349, +0.0251] | EQUIVALENT |
| knowledge (mean -0.100) | item (60) | [-0.178, -0.028] | [-0.189, -0.011] | CONTRADICTED |
| | answer frame (57) | [-0.175, -0.023] | [-0.189, -0.011] | CONTRADICTED |

- Resampled by cluster, the 90% interval over Y's items reaches below -0.03 and the equivalence no longer
  holds. The knowledge family's negative D drives this.
- The three planted-fact families are equivalent within 3 points under every resampling.
- For these three trained models, no analysis leaves room for a 3-point gain: every 95% upper bound is at most
  +0.021 (Y's items) or +0.027 (planted facts).
- Knowledge has a single slot, so it has no slot-level resampling.

**The control's item records reproduce E1 exactly.** E1's three L16 post jobs were re-run at the arm's commit
from E1's home pretrained weights. Each re-run equals its home summary in the pretrained and SFT weights'
sha256, every Carry-5 pool, Y, Y_all and the panels. Every data ledger (Stage 0, the re-run, the three seeds)
equals E1's build. Stage 0's 500-step replay of E1's pt_e1_L16_r1 at the arm's commit was bit-identical to E1's
log (`plain_bitid.json`).

## The counter-hypothesis's reading (descriptive)

X5 is a better language model at this budget and carries facts across turns no better.

| (means over seeds) | X5 | L16 | X5 - L16 |
|---|---:|---:|---:|
| pretraining best validation loss | 3.390 | 3.431 | -0.042 |
| FineWeb-Edu panel (nats/token) | 3.790 | 3.838 | -0.048 (every seed: -0.046, -0.040, -0.057) |
| dialogue panel | 3.206 | 3.243 | -0.037 |
| SFT held-out loss | 2.980 | 3.014 | -0.034 |
| multi-turn accuracy, self history (Y's items) | 0.489 | 0.471 | +0.018 |
| twin accuracy (Y's items) | 0.543 | 0.515 | +0.028 |
| Y (self) | -0.054 | -0.044 | -0.010 |
| Y with gold history | -0.037 | -0.045 | +0.008 (item-level 95% [-0.013, +0.030]) |

**The isoplot is weak evidence** (`isoplot_robustness.py`, `isoplot_robustness.json`).
- **Through all 22 home plain 20M runs** (E5's four mixes and E1's L2, L4, L16 and L32): the line of Y on the
  FineWeb panel loss has slope -0.019 Y per nat, with residual SD 0.021. X5's mean residual at its own panel loss
  is +0.002.
- **The slope is not stable.** Leaving out one cell at a time moves it from -0.082 to +0.051. A bootstrap over
  runs (10,000 resamples) gives a 95% interval of [-0.149, +0.091]. X5's residual ranges from -0.003 to +0.012
  across the leave-one-out fits, and its bootstrap interval is [-0.009, +0.016].
- **Through the 14 runs of the 35% mix alone:** the slope is +0.023 (leave-one-cell-out -0.16 to +0.15) and X5's
  residual is +0.012. X5's panel losses (3.781-3.798) lie below every plain 35% run's (3.828-3.986), so this is
  an extrapolation.
- **Reading.** Among the plain runs, Y barely depends on the FineWeb panel loss. That X5 sits near the line
  says only that its lower loss did not come with a higher Y; the line itself does not separate a general gain
  from a carry-over gain.
- The pattern the literature's default predicts is there in the losses and the twin accuracy: they improve
  while Y does not.

**By family and distance** (item-level paired D, descriptive):

| family | D | 95% |
|---|---:|---|
| recall | -0.010 | [-0.043, +0.023] |
| correction | +0.002 | [-0.050, +0.053] |
| topic return | +0.006 | [-0.043, +0.054] |
| knowledge | -0.100 | [-0.198, -0.002] |

- Knowledge (60 items x 3) is the one family whose interval excludes zero. Its twin is its question alone, and
  X5's twin there rose more (0.550 against 0.461) than its multi-turn accuracy (0.444 against 0.456).
- By distance: D is +0.005, +0.001, -0.027 and +0.025 at distances 1-4 (432, 972, 756 and 324 item-seeds), with
  no trend.

**Cost.**
- Probes on one RunPod L4: X5 17,378 tokens/s against plain 20m_L16's 26,007, so 0.668x, with +15% peak memory.
- In the runs (RunPod L4s): 15,122-15,376 tokens/s, pretraining 2.96-3.01 h. E1's L16 ran at 22,079-22,264 on
  Colab L4s, so this cross-provider comparison is flagged.
- Y per L4-hour: -0.015 to -0.018 (pretraining + SFT, 3.39-3.44 h per run).

**Generation.** Every X5 run and every re-run generated with ember e115fd3: spot-check 40/40, and 5,160/5,160
replies equal to `mtstudio score`'s argmax.

## Depth-attention weights (A5.7, descriptive)

`tools/carry5_attnres_trace.py` on each seed's SFT GGUF, gold history, Carry-5's 828 context-family items
(`traces_x5.json`). The weights are read where keramos's interpreter computes them. Source 0 is the
embedding; source 2i + 1 is layer i's attention output and 2i + 2 its FFN output.

| seed | 1 | 2 | 3 |
|---|---:|---:|---:|
| mean normalised routing entropy over the 32 mixtures that see 2+ sources | 0.878 | 0.887 | 0.875 |
| the same over layers 0-3's 7 mixtures / the 25 from layer 4 on (head included) | 0.716 / 0.923 | 0.735 / 0.929 | 0.707 / 0.922 |
| mixtures near uniform (entropy > 0.95) | 3 | 1 | 3 |
| embedding weight at answer positions, mean over mixtures (uniform would be 0.097) | 0.164 | 0.161 | 0.160 |
| embedding weight, answer minus other assistant positions | +0.010 | +0.008 | +0.003 |
| sources 0-2 (embedding, layer 0's outputs), answer minus other | +0.014 | +0.013 | +0.0003 |

- **From layer 4 on, the routing is close to uniform.** The 25 mixtures from layer 4 on (the head included)
  have normalised entropy 0.88-0.96, and each one's largest mean weight is 0.05-0.28 (uniform over
  their 9-33 sources would be 0.03-0.11). This is the near-uniform deep routing reported for AttnRes in
  2605.18855.
- **The decided mixtures are early, and they make the 0.88 mean.** Layer 0's FFN input chooses between the
  embedding and layer 0's attention output (entropy 0.02-0.05). Layer 1's two inputs sit at 0.58-0.77, and
  layer 2's FFN input at 0.71-0.92.
- **"Selective strengthening" of early activations at the answer is small.** The answer positions give the
  embedding 0.3-1.0 points more weight than the other assistant positions do, and the early sources 0-1.4
  points more. By fact distance (1-4) the embedding's and the early sources' answer-position weights are flat
  to within 0.003.
- **The head's mixture reads one mid-depth attention output more at the answer.** At answer positions its
  largest weight goes to layer 5's attention output in seeds 1 and 3 (0.10 and 0.11, against 0.02 at the other
  assistant positions) and to layer 8's in seed 2 (0.11 against 0.05). The embedding's head weight moves
  0.002-0.005.
- Read beside HX5-main: the early layers route depth decisively and the deep ones mix nearly uniformly.
  Neither improves carry-over on Carry-5.

## Stage L0 on X5's pick (descriptive)

The pick is seed 3 (Y -0.0507; the home pick over the three seeds' post folders, `ART/e1_x5/pick_e1_x5_X5`).

Not yet scored: Stage L0 (`tools/carry12_l0.py run --gen none`, into `ART/l0/pick_e1_x5_X5`) holds 13 of its 91
score chunks and resumes from them. CPU `mtstudio score` on this model runs at about 4.5 min per chunk, against
about 0.8 for plain 20m_L16's pick (ROADMAP section 3), so the run waits behind the main session's queue.

## What this does and does not show

- At 20M parameters, 163.84M tokens and 16 layers, sublayer-level Attention Residuals lower every loss by
  0.03-0.05 nats and leave Carry-5's carry-over cost where it was.
- The registered verdict is EQUIVALENT within 3 points. Under cluster resampling the equivalence holds for the
  planted-fact families and not for Y's four families, whose interval reaches below -0.03 through the knowledge
  family. No analysis leaves room for a 3-point gain.
- The gap hypothesis (a benefit that grows with distance) is not tested here. Carry-5 has no long gaps, and
  Carry-12's pre-registration keeps it (HX).
- One budget, one shape, E1 L16's learning rate (not tuned for X5), three seeds; item-level intervals treat items
  as independent, though templates repeat (`../../tiny_chat_long_2026-10/STAGE_L0.md` section 5).
