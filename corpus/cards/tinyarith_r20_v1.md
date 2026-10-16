# Corpus card: `tinyarith_r20_v1`

*tinyarith level R=20: arithmetic word problems, operands 0..20. Rendered by `corpus/make_cards.py` from `corpus/registry.jsonl` and `tinyarith_r20_v1.stats.json`; edit those, not this file.*

**Status: a placeholder, not yet a card** — the measured-effect section is PENDING until the C1 study reports.

## Provenance

- Source: corpus/fetchers/tinyarith.py @ `generator in this repository; seed 20261016, version v1`
- Licence: same as this repository (generated)
- Fetch: `python corpus/fetchers/tinyarith.py` (pinned, idempotent, sha256-verified)
- sha256: `f5d0600c96d41277732ebb3921d3b75a794b45fca7805c4bde8729a4be5945c2`
- Selection ledger: `corpus/ledger/0002-v1-candidates.md`

## Composition

| measure | value |
|---|---|
| bytes | 234753 |
| documents | 2152 |
| document length, bytes (p50 / p90 / p99) | 94 / 100 / 104 |
| document length, tokens (p50 / p90 / p99) | 24 / 24 / 25 |
| tokens in file (mtstudio tokenizer boundaries) | 50749 |
| non-ASCII bytes | 0 |
| CRLF line endings | 0 |
| end-of-story marker present | True |

## Coverage under the training vocabulary

Tokenizer: `explicit vocabulary tinyarith_r20_v1/vocab.gguf (120 tokens, numbers 0..60)` (cap 4096), trainer read cap `data.max_tokens = 0`.

| measure | value |
|---|---|
| tokens the trainer reads | 52901 |
| bytes the trainer reads | 234753 of 234753 |
| tokens per byte | 0.2253 |
| `<unk>` rate (all / held-out tail) | 0.0000 / 0.0000 |
| `<unk>` from non-ASCII bytes | 0 tokens |
| distinct unknown types | 0 |

Most frequent unknown types: .

## Quality signals

- Exact duplicate rate (documents): 0.0019
- Near-duplicate rate (MinHash): PENDING (Phase H)
- n-gram perplexity, half/half split: PENDING (Phase H)
- Language distribution: PENDING (Phase H)

## Contamination

PENDING: n-gram overlap against every set in `corpus/eval/` is measured when those sets are frozen (Phase E). Not asserted.

## Measured effect

PENDING — not a card yet. Filled by the C1 corpus study (charter §8.3): bits per byte on the fixed evaluation panel, matched tokens, 5 seeds, with the seed spread.
