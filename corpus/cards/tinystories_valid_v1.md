# Corpus card: `tinystories_valid_v1`

*TinyStories validation file, first 20 MiB (quickstart corpus). Rendered by `corpus/make_cards.py` from `corpus/registry.jsonl` and `tinystories_valid_v1.stats.json`; edit those, not this file.*

**Status: a placeholder, not yet a card** — the measured-effect section is PENDING until the C1 study reports.

## Provenance

- Source: https://huggingface.co/datasets/roneneldan/TinyStories @ `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Licence: cdla-sharing-1.0 (dataset card at the pinned revision)
- Fetch: `python corpus/fetchers/tinystories.py valid_v1` (pinned, idempotent, sha256-verified)
- sha256: `94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4`
- Selection ledger: `corpus/ledger/0000-legacy-unknown.md`

## Composition

| measure | value |
|---|---|
| bytes | 19447282 |
| documents | 21990 |
| document length, bytes (p50 / p90 / p99) | 774 / 1258 / 2372 |
| document length, tokens (p50 / p90 / p99) | 177 / 307 / 590 |
| tokens in file (mtstudio tokenizer boundaries) | 4455561 |
| non-ASCII bytes | 21460 |
| CRLF line endings | 0 |
| end-of-story marker present | True |

## Coverage under the training vocabulary

PENDING: the tokenizer (`own frequency vocabulary built by tools/get_tinystories_data.py (not yet pinned: depends on the gguf package's writer)`) is not pinned yet.

## Quality signals

- Exact duplicate rate (documents): 0.0000
- Near-duplicate rate (MinHash): PENDING (Phase H)
- n-gram perplexity, half/half split: PENDING (Phase H)
- Language distribution: PENDING (Phase H)

## Contamination

PENDING: n-gram overlap against every set in `corpus/eval/` is measured when those sets are frozen (Phase E). Not asserted.

## Measured effect

PENDING — not a card yet. Filled by the C1 corpus study (charter §8.3): bits per byte on the fixed evaluation panel, matched tokens, 5 seeds, with the seed spread.
