# Documentation map

**Start here.** Everything written in this repo has a place below. If
you are an assistant picking this up cold, read
[`../ROADMAP.md`](../ROADMAP.md) first — it is the one list of what is
left to do — then come back for context.

## The four kinds of document

| kind | where | what it is |
|---|---|---|
| **What needs doing** | [`../ROADMAP.md`](../ROADMAP.md) | the single canonical list: open work, decisions waiting on Jonathan, parked lines |
| **Why things are this way** | [`decisions/`](decisions/) | judgement calls with their reasoning, so they are not re-litigated or silently reversed |
| **How it works** | `docs/*.md` (this directory) | design, specs, phase plans — the engineering record |
| **What we found** | [`../atlas/`](../atlas/) and [`../experiments/`](../experiments/) | the research: registry, pre-registrations, results, receipts |

## Entry points by question

**"What should I work on?"** and **"What is Jonathan blocking on?"**
→ [`../ROADMAP.md`](../ROADMAP.md)

**"Why was it built this way?"**
→ [`decisions/`](decisions/)

**"What has this project actually established?"**
→ [`../atlas/FINDINGS.md`](../atlas/FINDINGS.md) — the registry, one row
per claim, retractions included. Then
[`../atlas/THEOREM_CROSSING.md`](../atlas/THEOREM_CROSSING.md) and
[`../atlas/SEED_LOTTERY.md`](../atlas/SEED_LOTTERY.md) for the two
results the methodology rests on.

**"What happened recently?"**
→ [`../CHANGELOG.md`](../CHANGELOG.md), then `git log`.

## This directory

| file | subject |
|---|---|
| [`DESIGN.md`](DESIGN.md) | the tape, ops, module system — core architecture |
| [`EVENTS_SPEC.md`](EVENTS_SPEC.md) | the `events.jsonl` contract. **The most portable thing here** — any trainer that emits it joins the toolchain |
| [`CUDA_PHASE_B.md`](CUDA_PHASE_B.md) | Phase B1: device residency (complete) |
| [`CUDA_PHASE_B2.md`](CUDA_PHASE_B2.md) | Phase B2: training-step residency (complete, T4-validated, adoption gate passed), with the CUDA defect records |
| [`ROPE_HEADS.md`](ROPE_HEADS.md) | the llama RoPE head-coverage defect (fixed 6 Oct 2026) and the re-run showing the nine affected Stage 2–3 findings hold |
| [`SPARSE_ATTENTION.md`](SPARSE_ATTENTION.md) | the sparse-attention research ledger, including its negatives |
| [`STUDIO_PLAN.md`](STUDIO_PLAN.md) | the studio / spec-driven driver |
| [`ECOSYSTEM.md`](ECOSYSTEM.md) | how paperkiln, coalfire.cpp and ember.cpp fit together |
| [`TECH_TRANSFER.md`](TECH_TRANSFER.md) | mechanisms imported from papers |
| [`receipts/`](receipts/) | raw validation logs from hardware runs — the evidence behind "T4-validated" |
| [`history/`](history/) | completed phase docs, kept for provenance |

## The research side

- [`../atlas/EXPERIMENT_DESIGNER.md`](../atlas/EXPERIMENT_DESIGNER.md) —
  the implemented conditional effect model and uncertainty-directed experiment
  proposer: matched-seed contrasts, costed batches, legal sweep export, and a
  synthetic end-to-end demonstration. No completed-study reanalysis.

- [`../atlas/ARCHITECTURE_ATLAS.md`](../atlas/ARCHITECTURE_ATLAS.md) —
  the lab charter: how a claim earns a row.
- [`../atlas/FINDINGS.md`](../atlas/FINDINGS.md) +
  `findings.jsonl` — the registry. Machine-checkable `check` field,
  receipts on disk, retractions as rows rather than deletions.
- [`../atlas/PAPER_PLAN.md`](../atlas/PAPER_PLAN.md) — what becomes
  which paper, and the honest gaps in each.
- [`../experiments/atlas_selection/README.md`](../experiments/atlas_selection/README.md)
  — explicitly **post-hoc archival decision audit** of transfer_s2: source
  shortlists, target calibration, held-out seeds, random-policy controls and
  exclusion/calibration loss. No preregistered claim or new training; the
  parent study's frozen analyses remain unchanged. Results and replay commands
  are linked there.
- [`../experiments/`](../experiments/) — one directory per experiment.
  The invariant: **`PREREGISTRATION.md` and `analyze.py` are committed
  together, before any run exists.** `RESULTS.md` and `receipts/` arrive
  after. If you find an experiment where the analysis postdates the
  data, that is a defect worth flagging.

## Conventions worth knowing before you change anything

1. **Pre-registration is a licence, not a formality.** The commit that
   introduces `PREREGISTRATION.md` + `analyze.py` is the anchor a
   directional claim cites. Amending a licensed rule is allowed only
   pre-data, and must be recorded as a dated **Amendment** in that file
   with its reasoning — never edited in silently. See
   `../experiments/transfer_s1/PREREGISTRATION.md` for both an amendment
   and a set of lesser clarifications.
2. **Test the analysis before the data exists.** Every `analyze.py`
   should have a `smoke_analyze.py` beside it that fabricates the run
   layout and drives each decision rule through all its branches. This
   has caught three errors that would otherwise have produced wrong
   published verdicts.
3. **Receipts must name their own binary.** `mtsweep` writes
   `provenance.json` (commit, dirty flag, binary sha256) beside every
   `result.json`; `--require-clean` refuses to run from a dirty tree.
4. **Negatives get published.** Retractions are rows in the registry.
   A benchmark reporting zero errors is reporting that its test set is
   too easy.
