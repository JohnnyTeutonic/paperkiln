# Decisions only Jonathan can make

*Things an assistant should not decide unilaterally, with enough context
that he can answer each in a minute. Remove an item once it is settled
and record the outcome in [`../decisions/`](../decisions/).*

## Open

### 1. PyPI names
`paperkiln-fetch` and `paperkiln` are unclaimed. The wheel builds and
the package is smoke-tested. Publishing needs his credentials. Two
minutes, and the risk of waiting is someone else taking the name.

### 2. H-SCALAR's `|rho|` rule — amend to signed rho, or leave it?
The licensed rule adopts "scalars don't transfer" iff `|rho| < 0.5`, so
a strongly NEGATIVE correlation counts as scalars *transferring* — even
though a reversed ranking is the worst possible outcome for anyone
screening at small scale. It was left licensed and unamended; a warning
now fires whenever `rho <= -0.5` so the threshold cannot speak for that
case unchallenged. Amending it to signed rho is still clean **pre-data**
and stops being clean the moment the M arm lands.

### 3. Is re-running arm S on CUDA worth ~6 T4-hours?
The banked 15-seed CPU cohort could have served as the S arm directly.
It was made the numerics-BRIDGE reference instead, and S is re-run on
CUDA, so the primary S-vs-M comparison is single-venue. Costs ~6
T4-hours; buys a comparison that is not confounded by backend. Reversible
by reverting to the banked cohort if he would rather have the hours.

### 4. Venue for the Price paper
`philosophy_of_mind/structural_limits_consciousness.tex` is
reviewer-proofed and compiles clean. Shortlist was NoC / C&C /
Erkenntnis, pending his pick. Nothing else blocks submission.

## Settled recently (kept briefly, then delete)

- **Commit `tools/mtstudio.cpp` + `tools/parity_model.hpp`** — done
  31 Aug (`e64d203`) on his instruction. Unblocked four things at once:
  CUDA in mtstudio, deep-SWA to flex, highway residual, and
  `window`/`sinks` in the model event, without which transfer_s1 could
  not satisfy its own refuse-to-run guard.
- **The open `[OPEN]` design choices in the transfer pre-registration** —
  delegated to the assistant on 31 Aug and closed in the licence commit
  `3fa55ae`. Reasoning is in that file; several were settled by the S1e
  data rather than by preference.
