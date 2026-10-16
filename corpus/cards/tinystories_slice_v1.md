# Corpus card: `tinystories_slice_v1`

*TinyStories research slice (first 5000 train stories). Rendered by `corpus/make_cards.py` from `corpus/registry.jsonl` and `tinystories_slice_v1.stats.json`; edit those, not this file.*

**Status: a placeholder, not yet a card** — the measured-effect section is PENDING until the C1 study reports.

## Provenance

- Source: https://huggingface.co/datasets/roneneldan/TinyStories @ `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Licence: cdla-sharing-1.0 (dataset card at the pinned revision)
- Fetch: `python corpus/fetchers/tinystories.py slice_v1` (pinned, idempotent, sha256-verified)
- sha256: `672622b80c9052da17adaefe93a9f34bb51bdfaffdd70f4fc485fd593c16036b`
- Selection ledger: `corpus/ledger/0000-legacy-unknown.md`

## Composition

| measure | value |
|---|---|
| bytes | 4172294 |
| documents | 5000 |
| document length, bytes (p50 / p90 / p99) | 774 / 1103 / 2107 |
| document length, tokens (p50 / p90 / p99) | 177 / 264 / 520 |
| tokens in file (mtstudio tokenizer boundaries) | 972728 |
| non-ASCII bytes | 17743 |
| CRLF line endings | 5000 |
| end-of-story marker present | False |

## Coverage under the training vocabulary

Tokenizer: `vocab_chat7b_cap4096_v1` (cap 4096), trainer read cap `data.max_tokens = 400000`.

| measure | value |
|---|---|
| tokens the trainer reads | 400001 |
| bytes the trainer reads | 1717152 of 4172294 |
| tokens per byte | 0.2329 |
| `<unk>` rate (all / held-out tail) | 0.1909 / 0.2130 |
| `<unk>` from non-ASCII bytes | 6037 tokens |
| distinct unknown types | 3505 |

Most frequent unknown types: `.` (31242), `,` (15482), `"` (7502), `!` (3188), `\xe2` (1175), `\xa2` (1147), `?` (946), `\x82` (920), `\xc3` (920), `\xac` (910).

## Quality signals

- Exact duplicate rate (documents): 0.0000
- Near-duplicate rate (MinHash): PENDING (Phase H)
- n-gram perplexity, half/half split: PENDING (Phase H)
- Language distribution: PENDING (Phase H)

## Contamination

PENDING: n-gram overlap against every set in `corpus/eval/` is measured when those sets are frozen (Phase E). Not asserted.

## Measured effect

PENDING — not a card yet. Filled by the C1 corpus study (charter §8.3): bits per byte on the fixed evaluation panel, matched tokens, 5 seeds, with the seed spread.
