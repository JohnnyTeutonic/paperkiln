# T_E stage B and the top-up: result

Claude Opus 5.5, 7 Nov 2026, written before any of the three models is read. No wording here; the reports beside this note hold counts.

## Registered sets

**Stage A: 700 rows, 350 pairs, 50 pairs in each of seven families.** 198 pairs from the first set, 152 from the top-up, by the merge rule, in fixed order. No cell is short. Twins pass 700 of 700 both ways, with no 3a or 3b defects. `affirm_in_role` and `like_recall` fall to 7 distinct final wordings, under the brief's 8.

**Stage B: 808 rows, 404 pairs.** 1,280 delivered, 868 after the filter, 808 after the counted defects.

| family | rows | task pairs |
|---|---:|---:|
| correction | 44 | 22 |
| followup | 134 | out of task points |
| identity_stance | 116 | 0 |
| identity_topic | 124 | 0 |
| passage | 130 | 65 |
| recall_abstain | 84 | 42 |
| roleplay | 58 | 29 |
| whose_name | 118 | 59 |

**Short of the 100-row target: correction (44), roleplay (58), recall_abstain (84).** The filter caused most of it; only roleplay also lost pairs to a defect ruling.

## Checks

- Top-up: format clean; 28,576 turns all fit an entry; gold 2,100 of 2,100 both ways; 3a 1,863 held; 3b 111 held; 60 self-name values out of scope; no repeated conversations. Filter kept 1,240 of 2,100.
- The author's flags on the top-up, judged here: the 40 empty `{s_}` fit the frames, no defect. The 30 masked `self_like` repeats are not repeats once values are read, no defect.
- Stage B: entries all fit; gold 1,280 of 1,280 both ways, with five passage cues added to `score_cues.json` from the gold canary; 3a 541 held; `passage_absent` scored as abstention. 40 `followup` two-turn swaps are allowed by the spec, recorded in the verifier's rule. Filter kept 868 of 1,280, with ids made whole for the run, because stage B's repeat across families.
- **Ruled defects, stage B: 42 pairs.** 39 pairs `roleplay/rp_in/not_the_character_name`, whose in-role values appear in no user turn, and 3 pairs repeating a value tuple (`followup` 2, `identity_topic` 1). All dropped by `te_merge.py --stage-b-drop`; no ruling changed after a result.
- The 70 values in no table are `passage_present` numbers with a unit, from the generator. No defect.
- Rows telling one attribute twice: all of stage A's 50 registered rows are two different relatives, told apart in the row. Stage B's two registered rows are one of the same kind and one abstention row whose two values on another attribute are its foils, the asked attribute being a third. No defect.
- Hand checks by the author: stage B 2 form defects, 0 state; top-up 1 form defect, 0 state.

## Shortcut reads

| set | pairs | best baseline | points | bar |
|---|---:|---|---:|---:|
| Stage A | 350 | `last_entity_copy` | 2.86 | 25 |
| Stage B, all pairs | 404 | `last_entity_copy` | **27.23** | 25 |
| Stage B, task pairs | 217 | `last_entity_copy` | 19.82 | 25 |

Stage B on all pairs was over the bar. `followup` alone scored 100: its gold is the last entity by construction. Under the frozen rule the instrument was revised before any model was read: `followup` leaves task points, and small-talk pairs are excluded as they were meant to be. The bar did not move and no family's scoring changed. The revision is mine and was not in `PLAN.md` when the bar was frozen, so it is reported as a revision. On the task pairs `correction` reaches 50 and `passage` 35.38 for the same baseline, inside the 25 overall.

`identity_topic` and `identity_stance` have no task pairs, because each pair mixes a task twin and a small-talk twin. They stay registered and enter no cell.

## What the signal read gets

12 families, all with at least 20 task rows and at least 20 T_M rows, so 36 cells across three models.
