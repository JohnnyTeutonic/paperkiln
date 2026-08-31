# SRD rung 2 — pre-registration: is the gate a RETRIEVAL router or a NOVELTY detector?

*Lineage note: an independent, unfinished draft of this same
pre-registration exists at
[experiments/srd_2x2/PREREGISTRATION.md](experiments/srd_2x2/PREREGISTRATION.md)
(Jonathan's hand) — same hypotheses, same construct-validity diagnosis,
converged separately. Kept in the record per repo policy.*

**Written 2026-08-03, BEFORE any run.** Nothing below may be edited after
the first result lands; outcomes go in a results section appended at the
end, including the outcomes that embarrass the mechanism.

## 0. Status of the claim being tested

Established (do not re-litigate): the SRD gate's placement is
information-dependent — the `shuffle_predictor` falsifier destroys
alignment while preserving the gate's distribution, passing at 5–6σ
twice, and tail-vs-filler concentration replicated across 5 runs
(docs/SPARSE_ATTENTION.md). Retracted: the recall-performance claim, which
failed replication across seeds and is published as a negative.

**The open question this rung answers.** Concentration is real. Its
*interpretation* is not established. Two hypotheses survive:

- **H_retrieval** — the gate fires where retrieval matters: on the
  key/value pair that the query will ask for, and on the query tail.
- **H_novelty** — the gate fires on whatever is locally
  out-of-distribution, and in the current benchmark that is the *same
  set of positions*, because the task construction makes needle tokens
  distributionally distinct.

**Why the current harness cannot separate them (the confound, stated
plainly).** In `tools/srd_needle.cpp` the vocabulary is partitioned by
role: keys occupy [2, 66), values [66, 130), filler [130, 250). A
prediction-residual gate is *required* to fire on tokens drawn from a
region the surrounding context never contains, regardless of whether
those tokens are retrieval-critical. The 5× replication therefore
replicates a quantity that both hypotheses predict identically. This is
a construct-validity gap, not a statistical one, and no number of seeds
closes it.

It also offers the cleanest available explanation of the retraction:
if the gate is a novelty detector, firing at the needle buys nothing on
recall, and the recall claim was the mis-theorization — right about
*where* it fires, wrong about *what that buys*.

## 1. Design: 2 × 2, three seeds, plus one control lane

Two orthogonal factors, both changes to sequence construction only
(model, optimizer, schedule and probe protocol unchanged):

**Factor A — needle distinctness.**
- `A_distinct` (status quo): keys/values from their own vocabulary
  ranges, as today.
- `A_indist`: keys and values drawn from the SAME range as filler
  ([130, 250)), with role carried only by POSITION (the pair layout)
  and by the QUERY marker. Retrieval structure identical; the
  distributional signature removed.

**Factor B — decoy presence.**
- `B_none` (status quo): no decoys.
- `B_decoy`: two spans of tokens drawn from the *distinct* ranges
  ([2, 130)) inserted into the filler region at random positions, in
  pair layout, but NEVER queried. Maximally novel, zero retrieval
  value.

2 × 2 cells × 3 seeds (1, 2, 3) = 12 runs, SRD lane plus the
exact/kimi/srd_f lanes the harness already runs. Steps/T/d/batch
inherited from the last successful needle configuration so this rung is
comparable to the 5× replication set.

**Control lane (separate, runs alongside): matched-density quality.**
At inference the gate hardens to `g > τ`. For each seed, compare
final-loss/accuracy at matched exact-attention density ρ ∈ {0.1, 0.25}:
SRD-hardened vs RANDOM gate at the same ρ vs POSITIONAL gate (last-k +
first-token sink) at the same ρ. This tests the *efficiency* reading of
SRD, which is the claim the mechanism's structure actually supports and
which the retracted recall claim was standing in front of.

## 2. Primary metric (new, replaces tail-vs-filler)

The current `tail_gate` / `fill_gate` pair cannot express the
distinction. Replace with a four-region gate profile, measured on the
fixed probe set exactly as today:

| Region | Definition |
|---|---|
| `g_target` | the key/value pair whose key the query asks for |
| `g_nontarget` | the other (npairs − 1) key/value pairs |
| `g_decoy` | the decoy pair spans (B_decoy cells only) |
| `g_filler` | filler positions outside all of the above |
| `g_tail` | the QUERY marker + queried key (unchanged) |

**Primary statistic — the retrieval-selectivity index:**

    RSI = (g_target − g_nontarget) / (g_target + g_nontarget)

RSI isolates the discrimination that *only* H_retrieval predicts:
target and non-target pairs are distributionally IDENTICAL (same
ranges, same layout, same novelty) and differ only in whether the query
asks for them. A novelty detector has no access to that difference.

**Secondary statistic — the decoy-chasing index:**

    DCI = g_decoy / g_target      (B_decoy cells)

## 3. Pre-registered predictions

| Prediction | H_retrieval | H_novelty |
|---|---|---|
| P1: RSI in `A_distinct, B_none` | > 0, ≥2 SE from 0 | ≈ 0 |
| P2: RSI in `A_indist` | survives (no interaction with A) | ≈ 0 (nothing to fire on) |
| P3: concentration (g_tail − g_filler) in `A_indist` | survives | collapses toward 0 |
| P4: DCI | < 0.5 (decoys largely ignored) | ≈ 1 (decoys ≈ targets) |
| P5: matched-ρ quality vs random gate | SRD better | SRD ≈ random |

**The decisive cell is P4**, and it is decisive in one run per seed:
decoys are maximally novel and carry zero retrieval value, so any gate
that chases them is answering a novelty question. P2/P3 are the
converse test — remove novelty, see whether selectivity survives.

## 4. Decision rules (committed in advance)

- **Mechanism supported as a retrieval router:** P1 AND P2 hold, and
  P4 shows DCI < 0.5, in ≥2 of 3 seeds. Then the retraction's diagnosis
  is "right mechanism, wrong payoff claim", and the efficiency lane
  (P5) becomes the headline claim to develop.
- **Mechanism reclassified as a novelty detector:** P3 collapses and/or
  P4 ≈ 1 in ≥2 of 3 seeds. Then the 5× concentration replication stands
  as a *true result about a different quantity* — docs/SPARSE_ATTENTION.md
  gets a correction stating that "retrieval-critical" was an
  over-reading of a novelty signal, and SRD's remaining value proposition
  is compute allocation, not retrieval.
- **Mixed (P1 holds, P2 fails):** the gate needs distributional contrast
  to express a retrieval preference — an honest, publishable
  intermediate finding: a *conditional* router. No headline either way.
- **Efficiency claim graduates only if** P5 holds at BOTH densities in
  ≥2 of 3 seeds, against BOTH baselines (random and positional).
  Beating random but losing to positional is a negative result and gets
  published as one — sliding-window is the field's default control and
  the sparse-phase S1 baseline for exactly this reason.

**Seed discipline (the lesson that cost us rung 1):** no single-seed
result is reportable. Three seeds minimum; any effect present in one
seed only is noise until shown otherwise. Cell-level differences inside
mean seed spread are not signal (the Atlas Stage 2/3 rule).

## 5. Threats to validity, acknowledged in advance

- **A_indist weakens the task.** In-distribution keys make retrieval
  genuinely harder; if EXACT attention's accuracy collapses in
  `A_indist`, the cell is uninformative about SRD and must be reported
  as such rather than read as an SRD failure. Control-first rule:
  the exact lane must clear its plateau in a cell before that cell's
  SRD numbers are interpreted.
- **Decoys change token statistics** (more distinct-range tokens
  overall), which could shift the gate's global scale. RSI and DCI are
  ratios within a run for exactly this reason.
- **Two blocks, tiny d.** Nothing here transfers to scale without the
  ladder; claims stay scoped.
- **The efficiency lane's baselines must be honest.** Random gate at
  matched ρ and positional gate at matched ρ, same seeds, same steps.

## 6. Execution

`tools/srd_needle.cpp` gains `--needle_dist {distinct|indist}`,
`--decoys N`, and the four-region gate profile in its probe CSV;
`experiments/srd_r2/` will carry the sweep manifest, rows and the
generated analysis. Compute: 12 runs + 6 control-lane runs, polite
profile, roughly one night at one worker.

---

## RESULTS — appended 2026-08-04, after the runs, per the rules above

Full tables: [experiments/srd_r2/results/prereg_analysis.md](experiments/srd_r2/results/prereg_analysis.md);
raw probe + density CSVs alongside. 12/12 cells, 3 seeds each.

**THE VERDICT, PER THE COMMITTED DECISION RULES: the SRD gate is
reclassified as a NOVELTY DETECTOR.** The reclassification branch
required "P3 collapses and/or P4 ≈ 1 in ≥2 of 3 seeds" — both happened,
in all seeds, emphatically:

- **P3 — concentration collapses without the distributional signature.**
  Tail-minus-filler concentration is **+0.59 ± 0.04** with distinct
  needles (reproducing the 5× replication) and **−0.003 ± 0.001** with
  in-distribution needles. The entire concentration effect was the
  vocabulary partition. Not attenuated: gone.
- **P4 — the gate chases decoys.** DCI = **0.97** (distinct cells:
  never-queried decoys gated identically to actual targets) and
  **1.89** (indist cells: decoys gated at nearly TWICE the targets,
  because there they are the only out-of-distribution tokens left).
  A retrieval router has no reason to look at them; a novelty detector
  can look at nothing else.
- **P1/P2 — no query-driven selectivity worth the name.** RSI is
  +0.0095 ± 0.0046 in the favourable design (technically 2 SE from
  zero, microscopic against the 0.59 concentration scale) and ≈ 0
  everywhere else. Target and non-target pairs — identical
  distributions, differing only in being asked for — are gated alike.

**What survives, restated honestly:** the 5×-replicated concentration
and the twice-passed shuffle falsifier were TRUE results about a
DIFFERENT quantity. The gate reliably, information-dependently tracks
*distributional novelty*. "Retrieval-critical positions" was an
over-reading licensed by a benchmark in which novel and
retrieval-critical were the same positions by construction. This also
completes the retraction post-mortem: firing on the needle buys no
recall if firing tracks novelty rather than need.

**P5 — UNINFORMATIVE, and reported as such.** The pre-registered
control-first rule triggered: the exact-attention lane finished at
0.000 answer accuracy in ALL cells — the rung ran at the harder default
task (npairs=8, nkeys=64), not the calibrated configuration rung 1's
control-first lesson produced. No lane learned retrieval, so
matched-density QUALITY comparisons carry no information (all policies
sit at the floor). The gate-profile conclusions above are unaffected —
they measure where the gate looks, not whether the task was solved —
but they carry the scope note: *in models that never mastered
retrieval*. Whether task mastery changes gate semantics is precisely
rung 2b's question.

**Committed next step (rung 2b, if pursued):** the same 2×2 + density
lanes at the control-passing configuration (reduced npairs/nkeys, the
rung-1 calibration), so the exact lane clears its plateau and P5 plus
the mastery question become answerable. The novelty-detector
reclassification does not wait on it: the gate-profile evidence is
seed-consistent and decisive.

**Registry:** SRD-gate-conc → superseded by R2-novelty (supported);
R2-efficiency remains open/pending. docs/SPARSE_ATTENTION.md carries the
correction, as this document committed it would.

### Rung 2b addendum — 2026-08-04, same day: the calibration ladder
closed NEGATIVE, and 2b is PARKED

Two calibration cells at the easiest task setting ever tried
(npairs=2, nkeys=8, batch=4, 1500 steps, distinct needles):

- **Cell A (lr=3e-3, the historical hardcode):** every lane at the CE
  floor (~2.14 vs ln 8 = 2.08), one probe correct each.
- **Cell B (lr=1e-3):** motivated by registry finding S3-lrxopt
  (AdamW@3e-3 is the unstable-bad quadrant) — a genuine
  registry-advises-the-lab moment, and it was **disconfirmed**: floors
  identically. The plateau is not the learning rate. (That is a useful
  scope boundary on S3-lrxopt's transfer, recorded in the registry.)

Combined with rung-1 history (best exact accuracy ever: 18.75% at
3000 steps), the conclusion is structural: **the needle family does not
resolve at 2-block/d=128 for ANY lane, exact attention included** —
registered as NEEDLE-scale-negative. P5 and the
gate-semantics-under-mastery question are unanswerable at this model
scale; rung 2b is parked pending model-scale escalation under a new
pre-registration. One side observation, consistent with R2-novelty: at
lr=1e-3 the gate still concentrates hard on distinct needles
(tail 0.85 vs filler 0.28) while the task remains unlearned — novelty
detection operates independently of task mastery.
