# PAPER PLAN — the methodology paper

*Started 2026-08-06. Argument spine only: what the paper claims, what
receipt backs each claim, and what is MISSING. No drafting until the
gaps close (strength over shipping). Companion paper — the
paper-to-model extraction pipeline — is deliberately SEPARATE; see
§7.*

Working title: **"Replication Is Not Enough: Pre-Registered Ablation
for Architecture Claims, and a Registry That Enforces It"**

Venue: JMLR (unoccupied in the current spread; MLJ holds the ablation
metric paper, JAIR holds the multi-agent governance paper).

---

## 1. The thesis, in one paragraph

Architecture claims in machine learning are evaluated against a bar
that is insufficient for the claims being made. The normal evidentiary
package — a mechanism, an ablation table, a replication, sometimes a
falsifier — can be fully satisfied by a mechanism that is real, whose
placement is causally driven by the signal it is said to track, and
which is still WRONG about what it buys. We demonstrate this with a
worked case in which every conventional check passed and the
interpretation was nonetheless false, show that the error was
detectable only by a pre-registered de-confounding design, and provide
an implemented discipline (a findings registry with machine-checkable
reproduction and cite-or-refuse advising) under which such errors are
recorded, downgraded, and superseded rather than silently propagated.

## 2. The case study (the paper's spine — this is the contribution)

Surprise-Routed Density attention (SRD): a learned gate routes tokens
between exact and linear attention paths.

| Evidence stage | What passed | Receipt |
|---|---|---|
| Mechanism | gate concentrates on needle positions | `SRD-gate-conc` (superseded) |
| Replication | concentration replicated **5×** | same row |
| Falsifier | shuffled-predictor lane worse at **5–6σ**, twice | `docs/SPARSE_ATTENTION.md` graduation + T=256 runs |
| Downstream claim | recall improves | `SRD-recall` — **RETRACTED** |
| De-confounding | needle vocab partitioned BY ROLE (keys/vals/filler), decoys crossed | `experiments/srd_r2/PREREGISTRATION_R2.md` |
| Verdict | gate is a **novelty detector**, not a retrieval router | `R2-novelty` |

The pivot: needles in the standard benchmark are out-of-distribution
BY CONSTRUCTION, so "concentrates on retrieval sites" and "concentrates
on distributionally novel sites" are **observationally identical** under
the conventional protocol. Both are true of the same gate. Only one is
the claim people would cite. The in-distribution cell collapses the
effect (+0.59 → −0.003) while the decoy contrast (DCI 0.97 vs 1.89)
separates the hypotheses.

**Why this generalizes and is not an anecdote:** the confound is
structural, not incidental. Any benchmark that marks the positions it
wants a mechanism to find will mark them with *something*, and that
something is a rival explanation for any mechanism that fires on them.
This is a general hazard for interpretability-flavored architecture
claims, and the paper should say so in exactly those terms.

## 3. The proposed discipline (implemented, not proposed-in-principle)

1. **Pre-registration with committed direction.** Design, predictions,
   decision rules in version control before runs. Receipt: three
   pre-registrations in-repo with git timestamps preceding their own
   result files (`experiments/srd_r2/PREREGISTRATION_R2.md`, `experiments/sparse_s1_window/`).
2. **A findings registry.** Each claim = one row: scope, effect, SE, t,
   seeds, manifest, receipts on disk, and a machine `check`. Status is
   mutable (`supported` / `replicated` / `superseded` / `retracted` /
   `pending`); rows are never deleted. Receipt: `atlas/findings.jsonl`,
   16 rows, validator enforces link and status rules.
3. **Machine-checkable reproduction.** `reproduce.py` re-derives the
   registered effect from the receipts and returns REPLICATED /
   DID-NOT-REPLICATE / UNDERPOWERED. Receipt: 6/6 on committed rows.
4. **Cite-or-refuse advising.** The registry answers configuration
   questions with citations or refuses; it surfaces corrections
   automatically, so a superseded claim cannot be cited as live.

## 4. The reflexive demonstration (the part reviewers will test)

The discipline is shown catching **our own** errors, in public, with
commit hashes:

- `SRD-recall` retracted after failed replication.
- `SRD-gate-conc` superseded by `R2-novelty` — our own mechanism story
  reclassified against us.
- `S2-ctx`, `S2-spike-metric` retracted (the latter a metric-definition
  error we shipped and then killed).
- `NEEDLE-scale-negative`: a registry-derived hypothesis (`S3-lrxopt`
  → lr=1e-3) was DISCONFIRMED, recorded as a scope boundary rather
  than dropped.
- `S1-swa-beats-exact`: published, then **downgraded to pending** on
  referee review (paired t=−3.39 at df=2, below the two-tailed critical
  value; direction not pre-specified so no one-tailed rescue), then
  re-earned at n=10 (t=−10.35, 10/10 seeds). The full sequence is in
  the commit history.

**The strongest single sentence available to this paper:** the registry
chose the learning rate for the experiment that produced our only
positive result — the failures funded the finding.

## 4b. The result that may outrank the case study (added 2026-08-07)

`S1c-budget-reversal` is the strongest thing in the registry, and it is
a negative about our own positive.

Paired Δ (sliding-window minus full attention) moves from **−0.047 at
400 steps to +0.012 at 1200** — same direction on every seed, change
t = 5.09. The sparse advantage we published four days earlier is a
short-budget artifact that reverses with training.

**Why this may belong in the abstract rather than §4.** The SRD case
study demonstrates the failure mode on OUR mechanism. This one
generalizes to a practice the field runs constantly: comparing a sparse
attention variant against dense attention at a small scale and a short
budget. We show the SIGN of that comparison flipping between two
budgets that are both short. The claim we can defend is narrow and
about reporting discipline — state the budget, vary it, report both —
and the paper must NOT overreach into "published results are wrong."
But a referee will recognise the shape of the risk immediately,
because it is their own protocol.

**It also contains the cleanest demonstration of what pre-registration
buys, and it is a within-week A/B.** S1's manifest expected
match-or-cost, so its surprising direction forbade a one-tailed test
and the finding had to be downgraded at n=3. S1-c committed the
reversal direction to git before running, so the one-tailed test IS
licensed — and it needs to be, because the reversal clears the
one-tailed critical value (2.132) and misses the two-tailed (2.776) by
0.04. Same lab, same effect size regime, same week, opposite
statistical entitlements, decided entirely by what was written down
first. That is a better argument for the discipline than any amount of
prose about it, and it should be shown as a table.

## 5. GAPS — nothing is drafted until these close

- **G1 — SCALE.** Everything is 2-block, d=128, T=256. First referee
  question is whether the discipline survives where the claims people
  care about live. Answer = the Stage 4 scale ladder (queued; likely
  Colab). **Blocking.**
- **G2 — n=1 ON THE REFLEXIVE CLAIM.** Every row came from us. "This
  discipline catches errors" is far stronger when a claim from an
  outside run enters and gets constrained. `CONTRIBUTING-FINDINGS.md`
  is the door; nobody has walked through it. **Blocking for the
  strongest version; the paper survives without it by scoping the
  claim to a single lab.**
- **G3 — the case study is one mechanism.** A second, independent
  instance of "conventional bar passed, interpretation wrong" would
  move this from case study to phenomenon. The V2/CoD line may supply
  one; do not manufacture one.
  **PARTIALLY CLOSED 2026-08-06 — and this one is stronger than SRD in
  one specific respect.** `S1b-window-monotone` is a second instance of
  the same failure shape (a real, replicated effect carrying a
  mechanism story that turned out to be unsupported), but unlike SRD it
  was caught **PROSPECTIVELY**: the two hypotheses were declared
  mutually exclusive at a named rung, the decision rule bound us to
  amend our own supported row if the wrong one won, the commit
  predates the runs, and we executed the amendment. SRD demonstrates
  the discipline recovering from an error already made; S1-b
  demonstrates it catching one at the moment of formation. The paper
  wants BOTH, in that order — retrospective repair, then prospective
  interception. That pairing is a better argument than two SRDs would
  be.
- **G4 — prior art sweep not done.** Pre-registration in ML (Bayesian
  workflow, the ML Reproducibility Challenge, registered reports in
  psychology and their ML analogues), experiment-tracking systems
  (W&B, MLflow) and their explicit NON-claims about epistemics.
  Position against these precisely — the novelty is the *enforced
  status lifecycle plus machine-checked reproduction*, not "tracking
  experiments," and the paper dies if that line is fuzzy.

### G4a — POSITIONING (settled 2026-08-06, do not drift from it)

**Do NOT frame the contribution as "the integration."** Reviewers read
"system X combines A, B and C" as an absence of scientific
contribution; it survives in a software track and struggles in a
science venue. The integration is real and is the distinctive asset,
but it must be argued through the CAPABILITY it enables.

**The capability: a CLOSED LOOP, not a record.** W&B and MLflow are
passive stores that make no epistemic claims — a deliberate design
choice on their part, not a gap. Our registry participates in the
design of the next experiment and can be falsified by it. Two
receipts, the second stronger than the first:
- `S3-lrxopt` selected the learning rate for the S1 cell.
- The registry advised a needle-task hypothesis, was **WRONG**, and the
  wrongness became `NEEDLE-scale-negative`, a scope boundary on the
  advising row. A logging tool cannot be wrong; being wrong and
  capturing it as structure is the whole point.

**The comparison class is the argument.** Against MLOps tooling this
looks like a feature. The correct reference class is the **clinical
trials registry** (ClinicalTrials.gov, and the Cochrane apparatus) —
institutions a field built once it concluded its evidentiary base
needed structural enforcement rather than good intentions.
Pre-registration is itself an import from that lineage. Framed this
way the paper is "we ported an evidentiary institution into ML, and
here is what had to change."

**What had to change IS the novel contribution:** a trial registry
cannot re-run its trials. ML experiments are cheap and rerunnable, so
the registry can VERIFY ITS OWN ROWS — `reproduce.py` re-derives the
registered effect from the receipts and returns
REPLICATED / DID-NOT-REPLICATE / UNDERPOWERED. That capability does not
exist in the institution being ported from, and does not exist in the
ML tooling being distinguished from. State it in those terms.

## 6. What this paper does NOT claim

- Not that sliding windows are a contribution (they are prior art; our
  S1 row is a *baseline*, and its role in the paper is to show the
  discipline applied to an uninteresting-but-real result).
- Not that SRD is a good mechanism. It is the specimen.
- Not that the tiny-scale findings transfer. Scope is stated in every
  row and must be stated in every claim in the paper.

## 7. The companion paper (separate, do not merge)

Paper-to-model extraction: arXiv → architecture with base+delta
inheritance resolution, measured against a **40-paper truth set**
(grouped AUROC 0.905 [0.789–1.000] vs naive 0.841; post-veto pooled
0.825; 66/92 verdicts, 3 wrong, abstention-first). The CI-band target
is met as of 31 Aug 2026. Merging the two papers would give one paper
doing both jobs at 70%.

### 7.1 The error taxonomy — no longer a gap, now a result

This section used to read "inheritance accounting needs an error
taxonomy". Growing the truth set 26 → 40 produced one, empirically,
by breaking the extractor three times. **Every failure is a case where
the paper's own sentence is locally misleading**, which is what makes
them worth publishing rather than merely fixing:

| # | failure | paper | what the text says | why it fools a scorer |
|---|---|---|---|---|
| 1 | **attributed adoption** | Megatron-LM | "both GPT-2 and BERT use GeLU … whereas the original transformer uses ReLU" | the model's own flavor is stated ONLY as a property of its ancestors; the contrasted alternative sits in a bare declarative clause and outscores it |
| 2 | **future-work mention** | Cerebras-GPT | "features worth exploring in future work include … RoPE and ALiBi" | the strongest possible NON-adoption signal is scored as evidence FOR adoption; compounded by "a GPT-3-like architecture" not registering as an inheritance cue |
| 3 | **compound name shadowing** | LaMDA | "gated-GELU activation" | a compound flavor name contains a simpler one as a substring; bare-GELU matches inside gated-GELU (which IS GeGLU) |

Two families, and the split is the contribution:

- **Evidential (1, 2).** What a paper *uses* is frequently stated
  indirectly — attributed to an ancestor, inherited, or present only
  under negation — while what it *does not* use often appears in a
  crisp declarative sentence. Any scorer that prefers direct-looking
  mentions inherits this bias. **The positive control matters here:**
  Qwen2 says "we follow Qwen with the usage of SwiGLU … RMSNorm",
  which is also attribution, and scores correctly. So the failure is
  not attribution per se but attribution with no first-person adoption
  verb — a well-defined syntactic case, not a vague weighting problem.
- **Lexical (3).** Independent of evidence quality: the flavor lattice
  needs longest-match-wins and a `gated-X → XGLU` normalisation. Note
  the benchmark had been counting "family-level assertions (GLU naming
  soup)" for weeks — circling this bug without catching it, because no
  truth-set paper used the `gated-X` spelling until LaMDA.

**Methodological point for the paper, and the reason this section is a
result rather than an apology:** all three were invisible at 26 papers
and none required a new metric — only more ground truth. A benchmark
that reports zero errors is not measuring the hard cases; it is
reporting that its truth set is too easy. The failures are registered
in `flavor_bench.py::KNOWN_WRONG` with diagnoses and named fixes, the
gate still fails on any NEW wrong assertion, and a KNOWN_WRONG entry
that stops reproducing is reported as FIXED so the registry cannot
decay into an excuse list.

**Honest limitation to state in the paper:** the truth set is
annotator-single (one reader, evidence quoted in-file). Fields a paper
does not state are omitted rather than inferred from sibling models —
e.g. InternLM2's norm/activation are left blank though it is in fact
RMSNorm/SwiGLU, because the paper states them only in a related-work
sentence about LLaMA. That rule costs coverage and buys the ability to
say the truth set contains no guesses.
