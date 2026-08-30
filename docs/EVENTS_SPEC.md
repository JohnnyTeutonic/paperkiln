# events.jsonl — contract specification v1

*The trainer is not the contract. This file is.*

Every paperkiln training run emits `events.jsonl`: one JSON object per
line, append-only, written as training proceeds. Everything downstream —
the studio dashboard (`mtstudio serve`), Atlas row extraction
(`tools/atlas_extract.py`), sweep aggregation (`mtsweep`), the seed
lottery (`tools/seed_lottery.py`), and every pre-registered analysis in
`experiments/` — consumes this file and nothing else. A run whose
events.jsonl is complete is fully analyzable even if the trainer that
produced it no longer exists.

That inversion is deliberate: **any trainer on any hardware can join the
registry by emitting this contract.** `tools/coalfire_events.py` proves
the sidecar pattern — a ~100-line adapter that translates a foreign
trainer's human log into events.jsonl buys that trainer the entire
toolchain without patching it. A PyTorch loop needs about ten lines of
`json.dumps` to comply natively.

Validation: `python tools/validate_events.py <events.jsonl>` — exit 0
iff the file satisfies this spec. Registry submissions are checked with
it; a findings.jsonl row's `receipts` must point at files that pass.

## General rules

- One JSON object per line. UTF-8. No trailing commas, no comments.
- Every object has an `"event"` key naming its type.
- **Consumers MUST ignore unknown keys and unknown event types.**
  Producers MAY add fields freely. This is what makes v1 stable:
  additions never break a reader; renaming or removing a REQUIRED field
  bumps the major version.
- Numeric fields are JSON numbers, never strings.

## Event types

### `start` — REQUIRED, first line
```json
{"event":"start","name":"run_000_c00_s1","steps":3600}
```
`name` (string) run identifier; `steps` (int) the intended budget.

### `data` — RECOMMENDED
```json
{"event":"data","tokens":400001,"val_tokens":20000,"vocab":4096}
```
Corpus provenance: training tokens, held-out tokens, vocabulary size.

### `model` — REQUIRED, before the first `step`
```json
{"event":"model","family":"gpt2","d":256,"layers":2,"heads":8,
 "seed":1,"attention":"exact","lr":0.001,"batch":4,"T":256,
 "params":3742720,"norm":"layernorm","activation":"gelu",
 "position":"learned","d_ff":1024,"vocab":4096,"accum":1}
```
The scope anchor. REQUIRED keys: `family`, `d`, `layers`, `seed`.
Every field a claim scopes on (attention flavor, window, sinks, lr,
batch) MUST appear here — **analyses read lanes from `model` events,
never from directory names** (the refuse-to-run guard in every
pre-registration since sparse_s1_boundary).

### `step` — REQUIRED, one per optimizer step
```json
{"event":"step","step":1,"loss":8.3618,"grad_norm":5.3166}
```
REQUIRED: `step` (1-based, strictly increasing), `loss`.
RECOMMENDED: `grad_norm`.

### `eval` — REQUIRED for any claim about validation behavior
```json
{"event":"eval","step":100,"val_loss":4.5456}
```

### `export` — OPTIONAL
Artifact export record (e.g. GGUF path and hash).

### `done` — REQUIRED for a completed run, last line
```json
{"event":"done","final_step":3600,"best_val":3.4008,
 "early_stopped":false,"wall_seconds":19149.46}
```
A file without `done` is an in-flight or killed run: consumers may
tail it live but no completed-run claim may cite it.

## Resume segments

A killed-and-resumed run APPENDS a fresh `start` + `model` and
continues emitting into the same file (mtsweep's resumability works
this way; 2 of the 10 banked sparse_s1_boundary receipts are
two-segment files). Rules:

- Each segment begins with `start` then `model` before its first
  `step`; `step` numbers are strictly increasing *within* a segment
  and may restart across a segment boundary.
- The `model` scope keys (`family`, `d`, `layers`, `seed`, and every
  lane field) MUST be identical across segments — a mismatch means the
  file mixes two different runs and is INVALID.
- Consumers MUST resolve duplicate step/eval numbers across segments
  by **last occurrence wins** (the resumed segment supersedes the tail
  of the killed one).
- A completed run has exactly one `done`, in the final segment.

## What the contract buys

| Consumer | Reads | Produces |
|---|---|---|
| `mtstudio serve` | tail of step/eval | live dashboard |
| `mtsweep` | done + model | resumability, aggregation |
| `tools/atlas_extract.py` | model + eval + done | Atlas findings row |
| `tools/seed_lottery.py` | model.seed + eval trajectory | seed-dependence exhibits |
| pre-registered `analyze.py` | everything | RESULTS.md with receipts |

## Provenance — which code produced this receipt

The events stream records what a run DID. Until 31 Aug 2026 nothing
recorded what BUILT it, and that hole cost a night: a full Colab sweep
ran a CPU-only `mtstudio` because the CUDA wiring lived in an
uncommitted working copy, and no receipt could have revealed it. A
registry whose receipts cannot name their own binary is one silent
rebuild away from unreproducible.

`mtsweep` therefore writes **`provenance.json` beside `result.json` in
every run directory** (and once at the sweep root):

```json
{
  "repo_commit": "89760cddcb97...",
  "repo_dirty": false,
  "repo_dirty_files": [],
  "binary_path": "/content/mtstudio",
  "binary_sha256": "00c496e9...",
  "binary_bytes": 1506008,
  "sweep": "/abs/path/sweep.json",
  "platform": "...", "python": "3.11.x"
}
```

`mtsweep --require-clean` REFUSES to run from a dirty tree. That is the
setting for any pre-registered experiment: a receipt whose
`repo_dirty` is true is honest about its results and useless as a
reproduction target.

## Submission standard (Atlas)

A registry submission is: the `findings.jsonl` row (mandatory `scope`,
`seeds`, `check`) **plus the events.jsonl files it cites as receipts,
all passing `validate_events.py`**. Scalar claims without per-seed,
per-budget trajectories are not auditable and are not accepted — the
budget-reversal and seed-lottery results are the demonstration of why.

*v1, 2026-08-30. Additive changes only within v1; consumers ignore
what they don't know.*
