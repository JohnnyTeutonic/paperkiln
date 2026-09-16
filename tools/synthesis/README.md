# Synthesis search

A generator and scoper for architectural research ideas of the kind
"take a mechanism from elsewhere in ML, put it at a component of the
transformer where it negates a deficit the component has by
construction, keep one invariant, and state the falsifier". Built 16
September 2026 after the "diffusion inside the KV cache" idea, which had
that shape, turned out to be an ICLR 2026 paper (Bottlenecked
Transformers, arXiv 2505.16950) found only after the excitement. The
point of the tool is to run the literature check before the excitement,
on hundreds of candidates at once, so that what reaches a human is
already open.

## The move, made mechanical

The idea that started this was not two words paired at random. It was:

1. a **component** and a **deficit** it has by construction (the KV
   cache is append-only);
2. a **mechanism** from elsewhere whose native operation negates the
   deficit (masked denoising rewrites entries from bidirectional context);
3. an **invariant** that keeps the system valid after the graft (outputs
   stay causal because edits only affect future reads);
4. a **falsifier** at the scale we can run.

`vocab.py` encodes 1 and 2 as data: 25 components with 55 deficits, each
deficit tagged with the operations that would negate it; 50 mechanisms
from ML, statistics, neuroscience, signal processing and systems, each
tagged with the operations it performs natively. A candidate is any
triple whose tags intersect: 573 of them at the time of writing. The
language model does 3 and 4 for each candidate and may reject a triple
as incoherent or trivial. The scoper then does what we failed to do by
hand: it searches for the idea under vocabulary the idea does not use.

## Stages

```
python tools/synthesis/pipeline.py enumerate            # candidates.jsonl
python tools/synthesis/pipeline.py generate [--limit N] # ideas.jsonl (Sonnet 5; ~8 per call)
python tools/synthesis/pipeline.py scope    [--limit N] # scoped.jsonl (Haiku queries, arXiv API, Sonnet judge)
python tools/synthesis/pipeline.py score                # ranked.md
python tools/synthesis/pipeline.py all --limit 16 --control   # smoke run with controls
```

Every stage caches by candidate id in `results/`; rerunning does only new
work. The key comes from `ANTHROPIC_API_KEY` or `AI_ML/agora/.env`.

**Generate.** Each triple becomes a JSON record: name, claim in the form
"mechanism applied to component so that new capability", placement (what
is computed, where, how often, what is trained), invariant, falsifier
(hypothesis, metric, baseline, nano-scale protocol, GPU-hours), self-rated
feasibility and splash (1 to 5), and the generator's own guess at the
nearest known work. About a third of triples are rejected by the
generator.

**Scope.** Three kinds of query, all fielded arXiv API syntax:

- mechanism level, from the query writer;
- operation level, what is done to which component in plain words with
  no mechanism name, from the query writer AND deterministic backstops
  built from the component's aliases and the ops' synonyms in
  `vocab.py` (this is what finds a paper that calls your denoising step
  "consolidation");
- purpose level, the capability in the words a title would use.

Up to nine queries, ten results each, three seconds apart. The judge
reads the proposal and up to forty abstracts and returns TAKEN, ADJACENT
or OPEN, the closest papers with a quoted phrase from each abstract as
evidence, what twist would survive, and a confidence.

**Score.** Rank = novelty (OPEN 30, ADJACENT 10, TAKEN 0) + 2 × splash +
feasibility. `results/ranked.md` lists the controls, the full table, and
the top OPEN ideas in full with their falsifiers and closest papers.

## Controls

`--control` injects candidates whose verdict is already known. A run is
not to be trusted on the rest unless the controls pass:

- `ctl-kvdiff`: masked denoising at the KV cache. Expected TAKEN, citing
  2505.16950. The first version of the scoper returned OPEN on this
  control because the query writer searched by mechanism name; the
  backstop queries were added in response.
- `ctl-choquet`: Choquet-capacity expert selection at the MoE router.
  Expected ADJACENT: the scoper found arXiv 2607.26164, an infrared
  spectroscopy paper with a Choquet-integral MoE decoder, which we had
  not known of. It aggregates expert outputs; the proposal selects the
  expert set. The judge's same-purpose rule is what separates the two.

## What it is not

It is not an idea generator to be believed. It is a filter. An OPEN
verdict means the arXiv API, queried nine ways, returned nothing the
judge could match. Before any OPEN idea is worked on: read the closest
papers it names, run two more searches by hand with your own vocabulary,
and check Semantic Scholar and OpenAlex, which the tool does not query
(rate-limited and noisy respectively at the time of writing). Feasibility
and splash are the generator's self-ratings, calibrated to the testbed
statement in `vocab.py`; treat them as a first sort, not a judgement.

## Cost

Generation about 75 calls for the full set; scoping about 575 ideas ×
(one Haiku call, nine arXiv queries, one Sonnet judge call with ~40
abstracts). Order of ten dollars and five hours for a full pass; the
arXiv spacing dominates the time.
