# Contributing to the Atlas — findings, runs, and truth rows

The Atlas is a cumulative science of neural architectures. Its unit of
contribution is not a feature — it is a **receipted claim**. Three ways
in, easiest first.

## 1. Grow the extraction benchmark (no compute needed)

`papers/flavor_bench.py` scores the contribution-vs-mention extractor
against papers with hand-verified ground truth. Adding one paper:

1. Pick an arXiv paper whose architecture flavors (norm / activation /
   position) you can verify from the paper's own text.
2. Add a row to `TRUTH` in `papers/flavor_bench.py` — only the fields
   you are CERTAIN of; omission beats guessing. If the paper's true
   value is outside the candidate lattice (Primer's squared ReLU,
   DeBERTa's relative positions), the truth is `None`: you have added a
   designed negative, which is worth more than a positive.
3. Run `python papers/flavor_bench.py`. If your paper exposes a scorer
   failure, open an issue with the verbatim sentence — every cue in the
   scorer was earned from a trace like yours.

## 2. Submit runs (any engine that speaks events.jsonl)

The event stream is the contract, not the trainer. Anything that emits
`start/step/eval/done` JSONL joins the toolchain — the studio dashboard,
`tools/atlas_extract.py` (behavioural features), and `tools/mtsweep.py`
aggregation. `tools/coalfire_events.py` is the worked example: a sidecar
that translates another trainer's human log without touching its code.

Runs become Atlas rows; rows from hardware and engines we don't have
extend every finding's scope column. Include: the spec or config that
produced the run, seed, and engine name in the `start` event.

## 3. Register a finding (the full contribution)

A finding is a row in `atlas/findings.jsonl`:

```json
{"id": "...", "date": "YYYY-MM-DD", "claim": "one falsifiable sentence",
 "metric": "...", "effect": 0.0, "se": 0.0, "t": 0.0,
 "design": "pb12 | pb12f | grid-2^k | ...", "runs": 0, "seeds": 3,
 "scope": "family, param range, corpus, budget — the claim's borders",
 "status": "supported", "manifest": "path/to/sweep.json",
 "receipts": ["paths/that/exist/in/this/repo"],
 "check": {"metric": "...", "factor": "...", "direction": -1, "min_abs_t": 2.0}}
```

House rules, enforced by `python tools/atlas_findings.py validate`:

- **Receipts must exist on disk.** A claim without its raw rows is a
  tweet, not a finding.
- **≥3 seeds.** Single-seed results are uninformative — we learned this
  by retracting one (`SRD-recall`).
- **Scope is part of the claim.** "Muon is better" is not a finding;
  "Muon improves best_val at 0.95–2.7M params on TinyStories at 400
  steps, t=−6.2" is.
- **The `check` field makes it reproducible**: expected direction (or
  `"expect": "null"`) and a t threshold, so
  `python tools/reproduce.py <id> --run` can issue a machine verdict.
- **Nulls are findings.** A designed contrast that shows nothing
  (`S3-ctx-null`) earns `supported` like any positive.
- **Nothing is ever deleted.** Wrong later? Status becomes `superseded`
  (with `superseded_by`) or `retracted` (with a `note` that explains
  the post-mortem). The registry's corrections are its credibility:
  see `S2-ctx` (aliased with data budget), `S2-spike-metric` (the
  metric measured init transients), `SRD-gate-conc` (novelty, not
  retrieval). If your replication of an existing finding FAILS, that
  is a contribution — append it, don't argue in an issue thread.

Pre-registration (`experiments/SRD_PREREG_R2.md` is the worked example) is the
gold standard for anything surprising: design, predictions and decision
rules committed before the runs, results appended after, whatever they
say.
