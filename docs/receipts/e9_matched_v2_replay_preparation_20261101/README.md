# Offline replay helper for E9 matched v2

Prepared only. No E9 matched compute, provider request, model evaluation or actual-result replay has occurred here. All tests use synthetic receipts patterned after the existing checker fixtures, and execute the exact frozen v2 checker. The frozen stage and Claude files are unchanged.

Use after the actual campaign diagnostics are home:

```
python3 e9_matched_replay_v2.py --results /PATH/TO/ACTUAL/JOB_OUTPUT --stage /mnt/c/ml_artifacts/kvq/stage-e9-matched-v2-20261101 --receipt /NEW/IMMUTABLE/replay.json
```

The results directory must contain `out.zip`, `gate_summary.json` and `execution.json`. The helper verifies the exact frozen checker and full STAGING manifest hashes, then compiles and executes those verified source bytes directly, ignoring potentially stale or unverified bytecode caches. It validates the complete ZIP inventory before extracting into a temporary directory (4096 members, 128 MiB each, 512 MiB total), rejecting duplicate names, traversal, symlinks, special files, encrypted entries and file/directory collisions. External and archived verdict/execution files must agree byte for byte. The returned full manifest must match v2 exactly if present. Inputs are rehashed after replay to detect mutation.

The unchanged checker recomputes all twelve cells and the original three criteria from archived raw metric receipts. Transported criteria and cells are compared structurally, including scalar types. Wrapper-added errors are retained in the returned original archive but need not equal the checker's raw criterion errors. A passing verdict requires an empty explicit error list; a clean scientific negative requires every local criterion error and only the two known wrapper additions. The output records source/input/member hashes and the full locally recomputed checker summary. It is written exclusively and never overwrites an existing receipt.

`PASS` requires both explicit boolean flags true, execution exit code zero, completion marker true, all twelve valid cells, transported matrix agreement and the locally replayed pass. No production adoption is implied.

`COMPLETE_SCIENTIFIC_NEGATIVE` requires twelve valid matching cells with locally failed criteria, both reported flags false, exit code one, completion marker false and the explicit registered-criteria-failure reason. The VM intentionally exits this way when its checker fails; the wrapper adds errors. Other complete-matrix failures are `OPERATIONAL_OR_VERDICT_NEGATIVE`; missing/invalid raw matrices are `SETUP_OR_INCOMPLETE_NEGATIVE`. Archive/integrity disagreement raises and cannot manufacture a verdict. A valid negative writes its receipt and exits one; a pass exits zero.

Limits: this replays reported metric receipts and recorded model/input hash lines. It does not independently rerun device execution or hash model binaries that are absent from the returned archive. Original strict and A1 negatives remain unchanged. Admission, lifecycle, provider cleanup and spending review are outside this helper.

Validation: `python3 -m pytest test_e9_matched_replay_v2.py -q` gives 34 passing tests. They exercise passing and scientific-negative fixtures, timeouts despite complete matrices, incomplete setup, each required pass predicate, mismatched external/archive/criteria/cells/manifest, boolean substitutions, strict JSON, ZIP hazards/bounds and immutable output through the actual CLI subprocess.
