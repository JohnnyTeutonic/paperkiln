# The paperkiln IR conformance suite

The cases every engine that runs paperkiln IR documents should pass. The normative description is
[`spec/IR_SPEC.md`](../IR_SPEC.md) section 11: the fixture formats are in 11.2, the adapter contract in
11.3 and the pass rules in 11.4. This page covers only how to run the suite and how to add an engine.

## Contents

| Directory | Cases | What they check |
|---|---|---|
| `ops/` | one file per op, several cases each | one node on given inputs: outputs as f32 bit patterns, class (EXACT or ROUNDING) and tolerance, and recurrent state after the last row |
| `models/<name>/` | one model case each | a small document with F32 weights (`<name>.gguf`), prompts, the reference's greedy continuation, log-probabilities, argmax and (under `depth_route`) depth traces |
| `state/` | traces | sessions of prefill, incremental decode, truncate, snapshot and restore: logits, kv rows held (eager and lazy eviction), depths and recurrent tensors |
| `refuse/` | one fault each | documents and GGUFs every consumer must refuse, each with the rule of IR_SPEC section 10.1 it breaks |

`manifest.json` lists every case with its kind and profile. Models whose names start with `pk_` were
trained by paperkiln's `mtstudio` (each directory also holds `<name>.safetensors` and `mtstudio.json`,
so mtstudio can score it); `tp_` models are third-party graphs (`tools/gguf_ir.py`'s builders) with
random weights. The expected values come from the numpy reference, `tools/ir_numpy.py`.

## Running it

    python tools/ir_conformance.py --adapter ir_numpy                    # the reference
    MTSTUDIO_BIN=build/mtstudio python tools/ir_conformance.py --adapter mtstudio
    python tools/ir_conformance.py --adapter cmd:"my_engine --conformance"
    python tools/ir_conformance.py --adapter ir_numpy --only trace/ --json results.json

Each case prints PASS, FAIL or SKIP; a case an engine cannot run is a SKIP, never a PASS. The exit
status is 1 on any FAIL.

## Adding an engine

Write a program that reads one JSON request per line on stdin and writes one JSON reply per line on
stdout (IR_SPEC section 11.3). Start with `hello`, `check`, `open`, `forward` (with the `logits`
report) and `close`: that covers the model cases and the refusals. The state traces then need the
`kv_rows`, `kv_positions`, `depth` and `recurrent` reports and `truncate`, `snapshot` and `restore`;
the op cases need `run_node`. Reply `{"id": n, "skip": "..."}` to anything not built yet.

## Regenerating

    MTSTUDIO_BIN=... python tools/ir_conformance.py --build-models OUT   # train the pk_ models into OUT
    python tools/ir_conformance.py --regenerate                         # everything else, from numpy

Copy `OUT/<name>/` into `models/` before regenerating. Regeneration is deterministic: the same models
give the same files. Change expected values only by regenerating, in the commit that changes the
reference or the specification.
