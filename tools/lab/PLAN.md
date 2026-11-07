# Plan: from these foundations to the first pilots

Claude Opus 5.5, 7 Nov 2026. Nothing below is authorised by this file. Costs come from `costs.py`, printed by `lab.py pilots`. Readings are Claude Opus 5.5's.

**Verdict: AMBER.** The strongest objection: nothing yet shows that the signal the controller steers by tracks what the pilot measures. On the only clean pair on record, S51 gained 12.6 nats summed across families against S52 and lost 1.48, and on the same models J1 read 2.00 to 2.08, task on analytic rubric v2 33 to 50, small talk 39 to 51. Over the five models, lower mean excess goes with lower J1 and small talk (Spearman -0.9 each) and only weakly with task (0.5) and told recall (0.2). This is observational, five models, one seed each, on S-full's base, so it is conditional on that base and not a failure of anything. It is enough to require a signal read before any adaptive arm.

## Order

1. **CPU, no L4.** T_E stage A and its shortcut reads are done: `reviews/te_stage_a_result.md`. Still open here: S54's J1 read, which the error pilot's parent needs and the signal read does not.
2. **CPU, engineering.** T_M and T_C hold contrast and stance candidates: `reviews/step2_contrast_stance.md`. Still open: T_E stage B and the stage A top-up through a blind author, then `te_merge.py`; the map from packet family to controller family, `mtstudio score` as a per-family read on the VM, the packet writer with its pre-launch recount, the independent target tracker by another hand, and one timed leg boundary.
3. **CPU, the signal read,** once, below. It waits for T_E stage B: S54 was dosed on contrast and on nothing mixed, so mixed families alone give these three models too little range to test the signal.
4. **Registration, an independent review, Jonathan's dated line,** for each L4 step below.
5. **L4, `pk_sft_v1`:** the noise pair, two fixed seeds excluded from every later comparison, and one mixed-row specialist. This gives `pair_sd_points` and design A's gate.
6. **L4, `pk_sft_v1`, only if the gate opens:** the comparison, three seed pairs, registered with numbers after step 5.
7. **L4, `pk_sft_v1`:** the error pilot, as a screen at one seed.
8. **After the race:** the same on the selected base, skipping what steps 5 to 7 showed to be pointless.

## The signal read, frozen before any model is read

Fixed here on 7 Nov 2026, before any T_E row exists, and in code as `verdict.signal_gate`.

- **Models:** `pk_sft_v1`, S54 and S55. Read once. No other model enters it, and no read is repeated.
- **Development signal:** family loss on T_M, nats per target, chosen before the read. Cells are model by family, for families present in both T_M and T_E with at least 20 rows.
- **Primary:** task points on T_E by family, greedy, gold history.
- **Statistic:** both reads centred within each family, so family difficulty cannot carry the rank; then Spearman of the negated loss against the points.
- **Bar:** Spearman at least 0.5, over at least 10 cells.
- **Range:** at least 5 families whose development loss varies by at least 0.30 nats across the three models. Short of that the read is not informative, and it does not pass.
- **If it does not pass:** the pilot stops at this read. A new metric is a new registration, with a new read on new models. The bar does not move.

The shortcut baselines are fixed in `shortcuts.py`, with a bar of 25 task points overall. A shortcut reaching it sends the instrument back for revision before any model is read on it.

## Caps relaxed by the lead's ruling of 7 Nov 2026

For two families of T_C only, the smallest cap on a 0.5-point grid from 2% that leaves at least 8 distinct reply frames, and no higher than 12.5%. The 2% rule exists to stop stock replies dominating; a family at zero teaches nothing, thin tables cannot meet the rule, and the family's dose is still bounded by the controller's share. Report these at registration.

| Family | Cap | Rows kept of filtered | Distinct frames | Why it was at zero |
|---|---:|---:|---:|---|
| `nameless_role` | 4.0% | 384 of 2,270 | 193 | No row set meets 3.5% or under |
| `affirm_in_role` | 2.0% | 105 of 764 | 90 | The first pool held 1.95%, a margin under the rule. The rule itself leaves rows, so nothing above 2% is used |

The lead's second ruling of 7 Nov 2026 extends the same rule to the ten contrast and stance families the 2% caps took to zero, because a pilot cannot allocate between families that are not in T_C. Twin pairs stay whole. **Frame definition, Claude Opus 5.5's, 7 Nov 2026:** these generators record no frame, so a frame here is the reply with the row's values masked and digits folded. It is not the generator's own frame tag that the mixed families carry. Report these at registration.

| Family | Cap | Rows kept of filtered | Distinct frames |
|---|---:|---:|---:|
| `correction` contrast | 2.5% | 902 of 3,052 | 214 |
| `followup` contrast | 5.0% | 200 of 4,796 | 142 |
| `recall_abstain` contrast | 2.5% | 186 of 3,110 | 152 |
| `roleplay` contrast | 3.5% | 576 of 2,108 | 340 |
| `whose_name` contrast | 2.5% | 2,264 of 3,994 | 123 |
| `correction` stance | 2.0% | 202 of 3,048 | 151 |
| `followup` stance | 5.0% | 450 of 4,796 | 222 |
| `recall_abstain` stance | 2.5% | 250 of 3,066 | 164 |
| `roleplay` stance | 3.5% | 632 of 2,490 | 597 |
| `whose_name` stance | 2.5% | 2,408 of 4,046 | 123 |

**Thin pools limit dose.** The trainer samples with replacement, so a family with few rows and a large share repeats them. `exposure.plan_leg` reports draws per row for every family, and refuses a leg past `max_draws_per_row`, 4 unless the config says otherwise. That default is Claude Opus 5.5's, set against S34's roughly 2 draws per row on these families, and is frozen at registration. The refusal is per leg, and the audit's distinct-scenario check covers the whole run.

## The two starts

| | `pk_sft_v1` line | Selected stronger base |
|---|---|---|
| State | Exists. 12×768, k = 0, T 512 pretraining, 97.5M. LAMBADA 1.014 against S-full's 1.493 | The race has not read. 8×768 or 12×768, k = 4, T 1,024 |
| Adaptive start | `pk_sft_v1`: every family dosed from one level, so the arms differ in how allocation answers real differences in learning | Its SFT1500 |
| Error parent | S54, the line's fixed recipe. `pk_sft_v1` alone would repeat S34's finding: failures that restate an absent dose | The fixed recipe on it |
| Specialist floors | S54 and S55 on file, one arm to train | Three arms to train |
| First L4 possible | After steps 1 to 4 | After the race, its base evaluation and its fixed SFT |

## L4 cost

Rates: 1.83 s a step at 12×768, 8×2 (S54, S55 receipts); 1.63 at 8×768. 19 minutes' overhead a session; 13 estimated for a pair's leg boundaries, which no timed boundary backs. $0.59 an L4-hour on RunPod.

| Step | `pk_sft_v1`, L4-hours | $ | Stronger, 8×768, L4-hours | $ |
|---|---:|---:|---:|---:|
| Specialists | 0.89 (one) | 0.53 | 2.98 (three) | 1.76 |
| Noise pair | 3.04 | 1.79 | 2.81 | 1.66 |
| **To the stage gate** | **3.93** | **2.32** | **5.79** | **3.42** |
| Comparison, three pairs | 9.12 | 5.38 | 8.43 | 4.97 |
| Error pilot, three arms at 1,024, one seed | 2.35 | 1.39 | 1.85 | 1.09 |
| All | 15.40 | 9.09 | 16.07 | 9.48 |

A two-arm block is 3.46 and 3.23 L4-hours at the worst overhead on record. At 12×768 on the stronger base a block is 3.06. Design B's "about 2.7" is one block, without inputs and retrieval; it does not buy a claim, because a claim now needs three pairs and a noise pair. Error pilot at 512 steps: 1.24 and 1.15, but A19 found 512 underdosed. Mining adds about 1.5 to 2 core-hours on CPU; T_E evaluation is CPU.

The cumulative cap is $80 (`b1_ledger.md`). The race's worst re-cost is $39.69, which leaves about $40 against about $19 for both lines in full. Money is not the limit. PROTOCOL section 8's weekly L4-hours are, and the sequence needs his dated line.

## What each start would and would not show

**`pk_sft_v1`, first.** It would show whether allocation moves a frozen read on a base that is not weak; the spread of two same-allocation seeds on T_E; whether any family stays above its floor after a balanced mixture; whether the development signal tracks the primary; and the whole path's timing. It would not show any effect on the selected base: shape, k, pretraining length, corpus and SFT stage all differ, and 62.3% of S32's packet targets sit past the positions this base trained on, so part of any dose repairs positions. A result there transfers as method, not as a result.

**Stronger base.** The real test, on the model the programme will use. It costs the wait for the race and its fixed SFT, and three specialists spent before anyone knows whether any family has excess to chase.

## Power

On assumed worlds (`replay-sim`, no evidence about any model), with a seed spread of 4.2 points between arms and three pairs: where adaptation should win at about 6 points, ADAPTIVE_BETTER 43 of 100 and UNRESOLVED 57; on a null world 2 of 100 false claims; where the development signal misleads, FIXED_BETTER 98 of 100. So the likely real outcome is UNRESOLVED unless the effect is large. One seed of the error pilot, against the record's noise, has a half-width near 13 points, so it can only screen.

## Stop rules

No comparison if no family stays above 0.15 nats beyond noise after the noise pair. The signal read not passing stops the pilot. No second round of the error pilot under this plan.

## Not built, and why

- Packet writer, VM read, T_E rows, the contrast and stance cut: engineering for steps 1 and 2. T_E needs a blind author, not this one.
- An independent target tracker: `rowcheck.py` still reads targets from the generator's metas, so generator and verifier can share a mistake. Another hand writes it.
- Level 3 learning: the proposer's priors are set by hand. Nothing yet exists to learn from.
- A plain fixed reference with no leg boundaries: both arms pay for reads, as the brief requires, so a verdict speaks of the matched envelope only. A read share over 10% prints a warning.
- A keyed ledger or signed authorisations: both are procedural here, and said to be.
