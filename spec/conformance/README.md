# The keramos IR conformance suite

The cases every engine that runs keramos IR documents should pass. The normative description is
[`IR_SPEC.md`](../IR_SPEC.md) section 11: the fixture formats are in 11.2, the adapter contract in
11.3 and the pass rules in 11.4. This page covers only how to run the suite and how to add an engine.

## Contents

| Directory | Cases | What they check |
|---|---|---|
| `ops/` | one file per op, several cases each | one node on given inputs: outputs as f32 bit patterns, class (EXACT or ROUNDING) and tolerance, and recurrent state after the last row |
| `models/<name>/` | one model case each | a small document with F32 weights (`<name>.gguf`), prompts, the reference's greedy continuation, log-probabilities, argmax and (under `depth_route`) depth traces |
| `state/` | traces | sessions of prefill, incremental decode, truncate, snapshot and restore: logits, kv rows held (eager and lazy eviction), depths and recurrent tensors |
| `refuse/` | one fault each | documents and GGUFs every consumer must refuse, each with the rule of IR_SPEC section 10.1 it breaks |
| `accept/` | `documents.json` | the documents every consumer must accept: every suite model, every rewritten document, every worked document of `../examples/` |
| `digest/` | `vectors.json` | the IR digest (IR_SPEC 3.10): RFC 8785 serialisations and sha256 of numbers, strings, member names and every suite model's document, and texts that have no digest |
| `quant/` | `blocks.json` | stored blocks of every GGUF storage type and their f32 values (IR_SPEC 4.7), bit for bit on every backend |
| `grad/` | `recurrence.json`, `attention.json` | gradients of recurrence nodes (IR_SPEC 6.10.5) and of attention with `relative_kv` (5.15) against central differences in float64 |
| `chat/` | `render.json` | chat template records and conversations: the rendered text or the refusal (IR_SPEC 8.1.3) |
| `recurrence/` | rewritten documents | suite models with their hand-written recurrent ops written as recurrences (IR_SPEC 6.10.6); the traces `state/<model>_recurrence.json` run them in place of each file's own document and expect the original document's values |

`manifest.json` lists every case with its kind and profile. Models whose names start with `pk_` were
trained by paperkiln's `mtstudio` (each directory also holds `<name>.safetensors` and `mtstudio.json`,
so mtstudio can score it); `tp_` models are third-party graphs (`tools/gguf_ir.py`'s builders) with
random weights. The expected values come from the numpy reference, `tools/ir_numpy.py`.

Comments inside the fixtures (`_comment` members and the models' documents) name files by their paths
in paperkiln, which vendors this repository at `spec/` (`spec/IR_SPEC.md` is `IR_SPEC.md` here).

## Running it

    python tools/ir_conformance.py --adapter ir_numpy                    # the reference
    python tools/ir_conformance.py --adapter cmd:"my_engine --conformance"
    python tools/ir_conformance.py --adapter ir_numpy --only trace/ --json results.json
    MTSTUDIO_BIN=build/mtstudio python tools/ir_conformance_mtstudio.py  # paperkiln's adapter, in paperkiln

Each case prints PASS, FAIL or SKIP; a case an engine cannot run is a SKIP, never a PASS. The exit
status is 1 on any FAIL.

## Adding an engine

Write a program that reads one JSON request per line on stdin and writes one JSON reply per line on
stdout (IR_SPEC section 11.3). Start with `hello`, `check`, `open`, `forward` (with the `logits`
report) and `close`: that covers the model cases, the refusals and the acceptance cases (`check` with `gguf`
null checks a document alone). The state traces then need the
`kv_rows`, `kv_positions`, `depth` and `recurrent` reports and `truncate`, `snapshot` and `restore`;
the op cases need `run_node`. The digest vectors need `digest`, the stored blocks `dequantize`, and
the gradient cases (for engines that train) `run_node_grad`, and the chat cases `render_chat`. Reply `{"id": n, "skip": "..."}` to
anything not built yet.

## Regenerating

    python tools/ir_conformance.py --regenerate     # every expected value, from numpy

The `pk_` models come from paperkiln (`MTSTUDIO_BIN=... python tools/ir_conformance_mtstudio.py
--build-models OUT` there, the recipes beside it); copy `OUT/<name>/` into `models/` before
regenerating. Regeneration is deterministic: the same models
give the same files. Change expected values only by regenerating, in the commit that changes the
reference or the specification, and only in the environment that generated them
(`../requirements-reference.txt`: CPython 3.13.14, numpy 2.3.5, Windows 11 x86-64; IR_SPEC section 4.6).
Another numpy build passes the suite but can move ROUNDING values by an ulp
(`receipts/reference_environment.json`).
