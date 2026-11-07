# T_E stage A: verifier result

Claude Opus 5.5, 7 Nov 2026. Counts only. Rows `eccd5255…`, 700 rows, 350 pairs, by a `general-purpose` blind author who saw the workspace's `AGENTS.md` and memory index. The verifier and scorer share no code with `build_te.py`, which was not read. Reports: `te_stage_a_verify.json`, `te_stage_a_filter.json`. The ruling on check 3, `te_check3_ruling.md`, was written before the first run.

| Check | Result |
|---|---|
| 1. Inputs and format | Both hashes hold. 700 rows, no format defect. |
| 2. Entries | 9,416 turns: 9,399 fit an entry, 17 replies fit after a pronoun fill the receipt discloses, 0 fit none. 6,408 meta values: 0 from another part. 79 are in no table, and all 79 are the assistant's own name. |
| 3. Values | 3a: 616 held, 0 defects. 3b: 37 held. **20 defects under the ruling.** Text check: 0 defects on 2,539 checks. |
| 4. Twins | 350 pairs, no structural defect. Golds pass their own key 700 of 700, and fail the twin's 700 of 700. |
| 5. Counts | Every family and cell of the table met. 25 S and 25 D per family. No recall or name decision under two told values. Most-recent rule at most 29 per family. No repeat. At least 8 final wordings per family. |
| 6. Sequences | 0 by text, 0 by metas. 37 rows tell one attribute twice with different values, and each names a different relative: no corrections. |
| 7. Overlap filter | **396 of 700 rows kept, 198 of 350 pairs.** |
| 8. Hashes | Rows and inputs match the receipt. |

**Check 3, the 57.** 37 are character names under 3b. The other 20 are `name_use` `both_names` rows, where the value is the assistant's own name, Ember. They fail 3b's second condition, because Ember is no character. The ruling is not changed after the run, so they stand as defects under it. Claude Opus 5.5's reading, 7 Nov 2026: the rows are sound, since an assistant's own name has no user-turn source, and the ruling missed a case the verifier already treats that way for `self_name`. That is for the lead to rule on. Check 4 passes on these rows.

**Check 7.** Kept by family, of 100: `nameless_role` 74, `name_use` 70, `role_mix` 64, `how_are_you` 60, `ended_role` 52, `affirm_in_role` 42, `like_recall` 34. Pairs dropped at each step: `chatflow_v001` 77 rows' worth by id, `contrast_v001` 12, its values 4, the short-string rule 59 pairs. `manual_v001`, `persona_v003` and the transcript rule dropped none. The filter shares T_C's and T_M's path, and A19's self-check held. Every family stays above the signal read's 20 rows.

**Shortcut read,** on the frozen definitions, published before any model is read on T_E. Full set and filtered set both hold against the bar of 25:

| Set | Pairs | `last_entity_copy` | Others |
|---|---:|---:|---:|
| 700 rows | 350 | 3.43 | 0 |
| 396 rows | 198 | 4.04 | 0 |

The highest family is `name_use` at 14.29 on the filtered set.

**Scorer changes, all before any model was read, and none on model output.** Two abstention cues came from the gold canary, where 5 golds failed their own key. Replies that copy the final user turn now fail: on the first read `echo_question` reached 8.57 overall and 30 on `like_recall` and `affirm_in_role`, through confirmation pairs with two yes twins. The bar had held, and the change was made because a copied question is no answer. The definitions in `shortcuts.py` did not change.

**A limit.** `like_recall` and `affirm_in_role` hold 30 yes-and-yes D pairs on confirmation in the full set, so polarity adds nothing on those pairs: they read as naming the right value.
