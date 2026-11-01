# DPO startup observation analysis

The read-only `tools/colab_chat/analyze_dpo_startup.py` interprets the already-published opt-in startup markers. It requires the expected reference-pair count and resume step, preserves the raw stderr hash, rejects malformed fields, duplicate JSON keys, nonfinite/decreasing clocks, inconsistent counts and phase order, and emits only completed phase durations and last observations. Inputs are bounded and reports are exclusively created. It does not make provider/API calls, run a model, alter a checkpoint guard, modify source/input bindings or dispatch a new campaign.

Independent review and 39 quick tests pass, including replay of the actual compiled tiny CPU CLI marker log, every valid partial prefix, malformed/truncated records, detectable mixed streams, resume-step binding and original/output preservation. The attached tiny result is not full-size or CUDA timing. Markers have no PID/run ID: callers must establish that stderr belongs to one process. A coherent cross-process splice cannot be ruled out by marker content alone.

Example (an actual new timing campaign must first be separately admitted):

```sh
python3 tools/colab_chat/analyze_dpo_startup.py SINGLE_PROCESS_STDERR --expected-pairs 1445 --start-step 0 --output NEW_OBSERVATION.json
```

The diagnostic clock starts just before reference-model loading, after initial training-model setup. Complete first-iteration markers do not prove a complete durable checkpoint. A last marker identifies the last observed event, not the exact location/cause of a later halt. No phase markers means unobserved, not a fast or failed setup phase. Best-effort telemetry loss can leave a partial or invalid stream and must not be promoted to a training failure.

Next-run decision contract: after the existing S-full/E9 serial queue, construct a fresh source/input-bound diagnostic admission under the original cumulative DPO cap, retaining the dataset, numerical recipe, 10-step checkpoint interval and 180-second guard. Enable only the reviewed phase switch. Preserve all prior setup/guard negatives and all new stderr/manager evidence. If setup does not finish, measure its observed progress without extrapolating a completion time from it; identify a concrete optimisation or separately justified prospective operational amendment before a training retry. If setup finishes but the first training iteration/checkpoint does not, diagnose that separately. A complete durable checkpoint and sustained cadence are required before interpreting a successful continuation; diagnostic markers never reset the guard. This receipt is preparation, not dispatch, financial admission or adoption.
