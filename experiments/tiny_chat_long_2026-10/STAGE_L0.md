# Stage L0: registration (frozen)

*Frozen 24 Oct 2026 by the commit that adds this file, before any Stage L0 scoring. Stage L0 of the
long-carry programme (`PREREGISTRATION_DRAFT.md` sections 5 and 11 point 3): E5's and E1's SFT models scored
on Carry-12 v1, no training. A change after that commit is an amendment, appended below with its date and
reason, before the scoring it affects.*

## 1. Models and set

- **Models:** every tiny-chat SFT model from E5 and E1 that comes home as a pick folder
  (`tools/colab_chat/tiny_chat_run.py pick`: `model.safetensors`, `model.gguf`, `spec.json`), each scored
  once as it arrives: E5's `pick_e5_s0`, `pick_e5_s17`, `pick_e5_s35`, `pick_e5_s70`
  (`C:/ml_artifacts/tiny_chat/e5/`), E1's `pick_e1_L2`, `pick_e1_L4`, `pick_e1_L16`, `pick_e1_L32`
  (`C:/ml_artifacts/tiny_chat/e1/`) and, if run, the 60M picks (`pick_e1_60m_*`). A pick is one seed per
  cell (the best by Carry-5's primary metric). Other SFT models of those programmes that come home are
  scored the same way and are descriptive.
- **Not L0 models:** the floor pilot's pick (`pick_pilot_s35`) and any toy model. They may be scored only
  to exercise the tooling, with `--smoke` or `--every N > 1`; such receipts say `"smoke": true` and are
  never read for HL0 or the gate.
- **Set:** `corpus/eval/carry12_v1.jsonl`, sha256
  `de24f2c726a1006e8e61542dd3b2dfbf8677b9e6fe689277cca035305bd32797` (`corpus/hashes/carry12_v1.sha256`),
  all 1,452 items, at each model's own context (T 512 for E5/E1).

## 2. Command

On the laptop CPU (forced choice only), through the budget wrapper:

    python3 tools/carry12_l0.py run --model-dir DIR --out C:/ml_artifacts/tiny_chat/l0 --mtstudio BIN --gen none

On Colab (adds self and near_self history; prepared with `tools/carry12_l0.py colab-jobs`, launched by the
main session): the same with `--gen ember --ember-bin auto --gpu-layers -1 --spot 20`. Both run
`tools/carry12_eval.py` at the commit that adds this file or a later one that leaves its scoring
unchanged: forced choice by `mtstudio score` (raw summed log-prob, `acc`; a tie at the top earns
1/tied) in the modes gold, near, twin and noctx on every item; with `--gen ember`, greedy self and near_self
history played to the target turn (`--max-new 24`) by ember's server after its 20-item spot-check against
`mtstudio chat` (ember only on 20/20 identical items, otherwise mtstudio for the whole run), every
generated reply passing the parity gate. The receipt is `ROOT/TAG/l0.json`; `carry12_l0.py table ROOT`
lists them.

## 3. What is read, and nothing else

1. **Validity gate (per model):** twin accuracy over the primary families (recall, correction, topic,
   load, knowledge) at least 0.35, and twin minus question-alone accuracy over the context families
   (recall, correction, topic, load) at least 0.10. A model failing either is **FLOOR**: its HL0 is not
   interpreted.
2. **HL0, the distance cost (prediction: accuracy falls with distance).** D = acc(gold) - acc(near) over
   the 1,320 context-family items, gold history, raw log-prob forced choice: the same facts and question
   with and without the small talk between them. **Predicted: D < 0 beyond the SESOI of 3 points.** Test:
   the item-level paired differences, their mean and normal-approximation 95% and 90% intervals.
   Verdicts: SUPPORTED (95% upper bound < 0 and D <= -0.03), SIGNIFICANT_BELOW_SESOI (95% upper bound < 0,
   D > -0.03), CONTRADICTED (95% lower bound > 0), EQUIVALENT (90% interval inside +-0.03), otherwise
   NOT_RESOLVED. **The registered test** is on the pick of E5's data-bar winner and the pick of E1's shape
   winner (20M; 60M if run), as their own analyses name them; every other model's verdict is descriptive.
   If a winner is not yet named when its programme's picks are scored, all are computed and the reading
   waits for the name.
3. **Descriptive only** (reported, not tested, used for no decision): the slope of gold accuracy, and of
   gold minus near, on distance (least squares over the context items); accuracy by distance, family x
   distance, target turn, interference and distractor window; twin minus near (the cost of splitting into
   turns); Y12 with gold history; the instruction family; with generation, self and near_self accuracy,
   generation verdicts and the per-turn curves.

## 4. Use

The long programme's pre-registration freezes after L0 (`PREREGISTRATION_DRAFT.md` section 11 point 3) and
may cite only section 3's items 1 and 2 from L0 (and the descriptive items as context, labelled so). No
cell, threshold or hypothesis of the long programme is chosen by looking at any other L0 number.

## 5. Threats

- Items share templates and values, so item-level intervals are approximate (too narrow if templates
  cluster); the family x distance table is the check on a single family driving D.
- Picks were chosen on Carry-5; HL0's direction does not depend on that choice, its size may.
- Near differs from gold in length as well as distance (fewer tokens of context); D is the cost of the
  intervening small talk as a whole, not of position alone.
