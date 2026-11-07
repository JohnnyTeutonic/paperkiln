# Step 2: contrast and stance rows in T_M and T_C

Claude Opus 5.5, 7 Nov 2026. CPU only. Counts only. Code: the root's `AI_ML/chat_screens_claude/correction_cs.py`, importing `correction_split.py`, `contrast_gen.py` and `contrast_stance_gen.py` unedited. Rows: `C:/ml_artifacts/chat_screens_claude/packets/correction_v2/`.

**The cut.** `correction_split.py` cut mixed_gen's copies of the tables and left the contrast generators' own lists alone. Every upper-case list of strings in both modules, and every identity list in `Content.id`, is now cut under the salt `correction-cs-v1`: 113 tables. Entries v1 placed stay where they are. New entries are dealt across the parts, disjoint. `tables_cs/T_E.json` is `13c15ba5…`. Two tables of role lines are empty in T_M, so those role questions are not drawn there.

**The first run leaked, and the check caught it.** A first generation used a hand-kept table list. Wider checking found T_E's name and role frames in T_C rows, through source lists and a copy `Stance` keeps. All four files were deleted and regenerated with the rule above.

| File | Rows kept of generated | Smallest family |
|---|---:|---|
| T_C contrast | 30,598 of 34,588 | `roleplay` 2,108 |
| T_C stance | 31,786 of 35,844 | `roleplay` 2,490 |
| T_M contrast | 3,588 of 4,670 | `roleplay` 158 |
| T_M stance | 3,718 of 4,790 | `roleplay` 156 |

Filter: `build_a19.py` as a black box, then the rewritten last rule, with A19's self-check holding. One T_M topic line is left out by hash, unread: it shares a 4-gram with a held-out set and sat in every `followup` row, which took that family to zero.

**Verification, on every T_E entry in both table files.** No row in a wrong part. 0 rows flagged as a reserved sequence, by text or by metas. Turns fitting a T_E entry and none of their own part:

- T_C: 170 user turns, all a user's bare name in the generator's own inline form, 169 of them up to three words. That form is not a table, and it is a short generic one.
- T_C 6 and T_M 7 replies that the slot pattern reads as a T_E frame. All are joined from their own part's frames.

**Caps.** Twin pairs stay whole, and T_M takes no cap. These generators record no frame, so a frame is the reply with the row's values masked. Reply text and frame together, at the 1.95% the earlier pools held:

| Family | Contrast rows kept | Stance rows kept |
|---|---:|---:|
| `passage` | 3,840 of 9,770 | 3,756 of 9,730 |
| `identity_topic`, `identity_stance` | 938 of 3,768 | 1,268 of 4,610 |
| `correction` | 0 of 3,052 | 0 of 3,048 |
| `followup` | 0 of 4,796 | 0 of 4,796 |
| `recall_abstain` | 0 of 3,110 | 0 of 3,066 |
| `roleplay` | 0 of 2,108 | 0 of 2,490 |
| `whose_name` | 0 of 3,994 | 0 of 4,046 |

**Ten families reach zero, and I have not relaxed them:** the ruling of 7 Nov covers two other families. The cause is the one that ruling names. A third of a thin table puts a frame above 2% in nearly every pair. Top frame shares run from 3.9% to 10.3%. A19's packets did not meet this on contrast or stance rows: `build_a19.py` caps mixed rows on reply text, and stance rows took no enforced cap. A reply-text-only cap would keep 11 of the 14, and still zero both `followup` sets and contrast `roleplay`.

For that ruling, a read-only what-if, with nothing written: the rule given for the two families, smallest cap from 2% with at least 8 frames, applied to these ten.

| Family | Cap | Contrast rows | Stance rows |
|---|---:|---:|---:|
| `whose_name` | 2.5% | 2,264 | 2,408 |
| `correction` | 2.5, 2.0% | 902 | 202 |
| `recall_abstain` | 2.5% | 186 | 250 |
| `roleplay` | 3.5% | 576 | 632 |
| `followup` | 5.0% | 200 | 450 |

**Not done.** Caps on those ten await a ruling, so they hold no pool yet. T_M's dev read, the packet writer and the family map remain step 2 work. T_E's stage B and the stage A top-up are with a blind author.
