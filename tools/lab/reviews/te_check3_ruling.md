# Ruling on T_E check 3, stage A

Claude Opus 5.5, 7 Nov 2026. Written before any verifier ran on `te_stage_a_v1.jsonl`, and before any per-row result was read. The only thing known about the rows at this point is the lead's report: 57 rows whose asked value first appears in an assistant turn, and the receipt's counts.

**The brief's check:** an asked value appears in a user turn before the question.

**Why it was too narrow.** The brief itself defines `nameless_role` as a role with no name given, where the character gives one when asked and keeps it. So the character's name is the assistant's to introduce, and a later question about it cannot have a user-turn source. The check was written for the user's facts, and the brief's own family breaks it by design. That is a fault in the check, not in those rows, if they are what the family says.

**The changed check, 3a and 3b.**

- **3a, the user's own values.** Unchanged and strict. A value the user owns, meaning a fact, like or the user's name, must appear in a user turn before the question. An assistant turn never counts as a source for these. A user's value that first appears in an assistant turn is a defect, however many rows it touches.
- **3b, the character's name.** A value may first appear in an assistant turn only if all of these hold, read from the row:
  1. The decision is one of the operations about the character: `both_names`, or a final `rp_in` turn.
  2. The value is the character's name: it equals the `name` field of an earlier `rp_nameless` or `rp_in` meta.
  3. That earlier turn is a role turn with no name in the user's setup: the value appears in no user turn before the assistant turn that says it.
  4. The assistant turn saying it comes before the question.

Anything else among the 57 fails, and is reported as a defect by family and operation. The ruling does not change after the run. If a row is sound by some reading not listed here, it stays a defect here and the lead hears of it.

**Sharing:** the count checked under 3b is reported separately from 3a, so nobody reads a pass of 3a as covering them.

**A limit that stays.** Both parts use the row's metas to say what the value is, as `rowcheck.py` does. Check 4 and the scorer's gold canaries cross-check them against the keys, and neither shares code with `build_te.py`. This verifier does not derive the answer from the text alone.

## Amendment, 7 Nov 2026

Lead's ruling, 7 Nov 2026, fixed after the first run, on row validity, with no model read: the assistant's own name is known to it from identity, not from a row's history, so check 3 does not apply to a self-name value. The 20 `name_use` `both_names` rows are sound, and stay in. This covers the assistant's own name and nothing else: every other value keeps 3a and 3b as written above.
