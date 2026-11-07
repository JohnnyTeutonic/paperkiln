# Plan: from these foundations to the first pilots

Claude Opus 5.5, 7 Nov 2026. Nothing below is authorised by this file. Costs come from `costs.py`, printed by `lab.py pilots`. Readings are Claude Opus 5.5's.

**Verdict: AMBER.** The strongest objection: nothing yet shows that the signal the controller steers by tracks what the pilot measures. On the only clean pair on record, S51 gained 12.6 nats summed across families against S52 and lost 1.48, and on the same models J1 read 2.00 to 2.08, task on analytic rubric v2 33 to 50, small talk 39 to 51. Over the five models, lower mean excess goes with lower J1 and small talk (Spearman -0.9 each) and only weakly with task (0.5) and told recall (0.2). This is observational, five models, one seed each, on S-full's base, so it is conditional on that base and not a failure of anything. It is enough to require a signal read before any adaptive arm.

## Order

1. **CPU, no L4.** T_E's rows through a blind data author, the shortcut reads published before any model is read, S54's J1 read completed, then T_M and T_E on `pk_sft_v1`, S54 and S55: the signal read, with its bar frozen first.
2. **CPU, engineering.** Cut contrast and stance families into T_M, T_C and T_E, recover `nameless_role` and `affirm_in_role` from the caps, freeze the map from packet family to controller family, wrap `mtstudio score` as a per-family read on the VM, write the packet writer with its pre-launch recount, write the independent target tracker by another hand, and time one leg boundary.
3. **Registration, an independent review, Jonathan's dated line,** for each L4 step below.
4. **L4, `pk_sft_v1`:** the noise pair, two fixed seeds excluded from every later comparison, and one mixed-row specialist. This gives `pair_sd_points` and design A's gate.
5. **L4, `pk_sft_v1`, only if the gate opens:** the comparison, three seed pairs, registered with numbers after step 4.
6. **L4, `pk_sft_v1`:** the error pilot, as a screen at one seed.
7. **After the race:** the same on the selected base, skipping what steps 4 to 6 showed to be pointless.

## The two starts

| | `pk_sft_v1` line | Selected stronger base |
|---|---|---|
| State | Exists. 12×768, k = 0, T 512 pretraining, 97.5M. LAMBADA 1.014 against S-full's 1.493 | The race has not read. 8×768 or 12×768, k = 4, T 1,024 |
| Adaptive start | `pk_sft_v1`: every family dosed from one level, so the arms differ in how allocation answers real differences in learning | Its SFT1500 |
| Error parent | S54, the line's fixed recipe. `pk_sft_v1` alone would repeat S34's finding: failures that restate an absent dose | The fixed recipe on it |
| Specialist floors | S54 and S55 on file, one arm to train | Three arms to train |
| First L4 possible | After steps 1 and 2 | After the race, its base evaluation and its fixed SFT |

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

No comparison if no family stays above 0.15 nats beyond noise after the noise pair. The signal read failing stops the pilot; a new metric is a new registration. No second round of the error pilot under this plan.

## Not built, and why

- Packet writer, VM read, T_E rows, the contrast and stance cut: engineering for steps 1 and 2. T_E needs a blind author, not this one.
- An independent target tracker: `rowcheck.py` still reads targets from the generator's metas, so generator and verifier can share a mistake. Another hand writes it.
- Level 3 learning: the proposer's priors are set by hand. Nothing yet exists to learn from.
- A plain fixed reference with no leg boundaries: both arms pay for reads, as the brief requires, so a verdict speaks of the matched envelope only. A read share over 10% prints a warning.
- A keyed ledger or signed authorisations: both are procedural here, and said to be.
