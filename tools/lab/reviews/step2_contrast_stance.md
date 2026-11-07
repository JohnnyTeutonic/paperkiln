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

**Not done.** The 2% caps have not run on these rows, so they are candidates, not a pool. `nameless_role` and `affirm_in_role` stay at zero in T_C, because recovering them means relaxing that cap, which is a data rule and not mine to move. T_M's dev read, the packet writer and the family map remain step 2 work. T_E's stage B needs a blind author: the spec is written.
