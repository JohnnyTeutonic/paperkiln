# The signal read: result

Claude Opus 5.5, 7 Nov 2026. Read once, as `PLAN.md` froze it, by `AI_ML/chat_screens_claude/signal_read.py` (root `fba74a26`) on the instrument committed at `35be0efe`, with the stage B result at `e1a8f40b`. CPU only: T_M by the numpy forward on Windows, T_E replies by a CPU ember server in WSL, each checked for no CUDA library mapped and no GPU device open. Full report: `signal_read_v1.json` beside this note.

**It passes the frozen bar.** Spearman 0.788 over 36 cells (bar 0.5, at least 10 cells). 12 families vary by at least 0.30 nats across the three models (at least 5 needed), so the read is informative.

| model | loss on T_M by family, nats | task points on T_E, range over 12 families |
|---|---:|---|
| `pk_sft_v1` | 2.25 to 3.05 | 0 to 7.7 |
| S55 | 1.83 to 2.68 | 0 to 23.7 |
| S54 | 0.15 to 2.83 | 0 to 49.1 |

Claude Opus 5.5's reading, 7 Nov 2026, not part of the verdict: the centring is within family, so the model effect stays in. Much of the agreement is S54 being lower on loss and higher on points across the board. Within S54, the families it was dosed on (correction, passage, recall_abstain, whose_name, roleplay) sit at 0.15 to 0.60 nats and 23 to 49 points, and the stage A families sit at 1.5 to 2.8 nats and 0 to 40 points. So the pass shows that the dev signal tracks the primary across these models and families. The bar does not test whether it ranks one model's families as the controller would use it; within S54 the order is close but not exact. `like_recall`, `affirm_in_role` and `nameless_role` stay at 0 to 6 points on all three models.

**Reported, not judged, asked for in GPT-6's review of 7 Nov 2026** (`signal_rankings.py` in the lane, `signal_rankings_v1.json` beside this note). The pass above is pooled and is not evidence that the signal ranks one model's families. Within each model, raw ranks over the 12 families, which also carry each family's difficulty:

| Model | Spearman | Without families at 0 points | At 0 points | Ties | Signal's top three in T_E's top three |
|---|---:|---:|---|---|---:|
| S54 | 0.578 | 0.451 | `like_recall` | two at 2 | 0 |
| S55 | 0.324 | 0.262 | `affirm_in_role`, `like_recall`, `nameless_role`, `roleplay` | four at 0 | 2 |
| `pk_sft_v1` | −0.246 | −0.289 | `correction`, `how_are_you`, `nameless_role`, `roleplay` | four at 0, two at 2, two at 6 | 0 |

No family is at 0 points on all three. These lines change no rule; the next step tests the signal's use directly (PROTOCOL section 33).
