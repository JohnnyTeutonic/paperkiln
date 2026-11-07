# tools/lab: a bounded curriculum laboratory

Claude Opus 5.5, 7 Nov 2026. Standard library only, offline. **Nothing here trains, generates, spends or reaches a network.** It tests one claim: that allocation informed by earlier outcomes yields more independently measured conversational competence per unit of compute than a fixed mixture. It is allowed to conclude that the fixed mixture is better, and every verdict but one keeps it. This is adaptive sampling between bounded SFT legs. It is not a learner, not Group-DRO and not self-improvement of any larger kind. **Status: AMBER.** See `PLAN.md`.

## Data flow

```
packets, receipts ─► family_exposure (MT19937 recount, via adapters) ─► leg record ─┐
T_M rows ─► mtstudio score or mine_round/mine_score ─► miner ─► dev_read record ────┤
registration (config, rule hash, the one allowed read) ─────────────────────────────┤
                                                                                    ▼
                         ledger.py: append-only, hash-chained, anchored, validated
                                                    │ firewall: the registered read only, a cap on looks
                                                    ▼
                  controller.decide (fixed | adaptive): pure function of ledger, config, seed
                                                    │ executed: false
                                                    ▼
                  exposure.plan_leg: rows per family; both token totals met; family shares untouched
                                                    │ leg_plan record; a packet writer and launcher do not exist here
                                                    ▼
             gate.consume: denied unless registered, reviewed by another, and Jonathan's dated line names this action
T_E, frozen instruments ─► evaluation record ─► verdict.verdict_from_ledger ─► outcome record (one, never superseded)
history.py: stored receipts, observational, replay harness only. proposer.py: ranks, launches nothing.
```

## Modules

| Part | File | Note |
|---|---|---|
| Ledger | `ledger.py`, `schema.py` | Twelve record kinds. No update or delete. One registration and one outcome per experiment, one decision per arm and leg. Anchored beside the file; copy the anchor elsewhere, since the chain is unkeyed. |
| Taxonomy | `taxonomy.py` | Twelve categories. Tiers `verified_rule`, `heuristic`, `llm_judged` and clean or model-induced histories are never pooled. |
| Generator | wraps `mixed_gen.py`, `correction_split.py` | Not rewritten. `rowcheck.py` imports neither, but its targets still come from the metas: see `PLAN.md`. |
| Miner | `miner.py` | Wraps `mine_round.py`, `mine_score.py`. Each row's own provenance must say T_M. One round. One channel to training: a capped table of cell weights. |
| Controller | `controller.py` | Design B of `multi_objective_scoping.md`, plus the changes listed in its header. |
| Exposure | `exposure.py` | Shares are of decision tokens. At equal steps, tokens depend on the mixture: S51's leg sampled 1.32M to S52's 2.36M. Replay row length takes up the gap, and a leg that cannot match fails closed. |
| Verdict | `verdict.py` | Frozen rules, read from the ledger alone. Seed variance is in the interval. Three seed pairs for any claim. |
| Firewall | `firewall.py` | Four disjoint splits, a pinned sealed hash, an allowlisted read, a cap on looks, shortcut baselines, a prefix test for leakage. |
| Proposer | `proposer.py` | An opinion list with its arithmetic shown. Every entry declares hypothesis to stop rule. Ranks within one base and scale. |
| Gate | `gate.py` | The door any future launcher must use. Denies by default; no override exists. Procedural: agents never write authorisation records. |
| Reviews | `reviews/` | The adversarial review and an answer to every finding. |

## The priority formula

The brief suggests marginal gain × confidence / cost. It is not used. One leg's gain is a difference of two noisy reads, so a bad read at t-1 shows as a large gain at t, and the rule pays for regression to the mean. It also pays families that learn fast, which need dose least. Gain here only withholds: conservative excess sets weight, and a stall removes it. `replay-sim` keeps the formula for comparison, on assumed worlds.

## Commands

From the repository root:

```
python3 tools/lab/lab.py demo --out DIR
python3 tools/lab/lab.py dry-run --ledger L --config C --arm A --leg N --pool P
python3 tools/lab/lab.py audit-decisions --ledger L --config C
python3 tools/lab/lab.py verdict --ledger L --rule R --experiment E
python3 tools/lab/lab.py replay-sim --n 100
python3 tools/lab/lab.py replay-history
python3 tools/lab/lab.py propose
python3 tools/lab/lab.py pilots
python3 -m pytest tools/lab -q
```

`integrate.py` reruns the wrapped frozen code on real files, counts only, through `scripts/cpu_budget.py`. `LAB_SCREENS_DIR` names the root's `AI_ML/chat_screens_claude`; the frozen files are hash-pinned in `adapters.py` and the core runs without them.

## Limits

`sim.py` tests the code on assumed curves and is no evidence about any model. `history.py` is observational, one seed per model, one weak base, and an intervention there is conditional on that base, not failed. Lab arm ids avoid PROTOCOL's C0, M1, E1 and E2, which mean other things there. Wording from instruments and probes stays out of this folder.
