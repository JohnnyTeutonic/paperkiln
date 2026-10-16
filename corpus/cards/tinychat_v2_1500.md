# Corpus card: `tinychat_v2_1500`

*TinyChat v2, 1500 dialogues, seed 11 (synthetic). Rendered by `corpus/make_cards.py` from `corpus/registry.jsonl` and `tinychat_v2_1500.stats.json`; edit those, not this file.*

**Status: a placeholder, not yet a card** — the measured-effect section is PENDING until the C1 study reports.

## Provenance

- Source: tools/get_tinychat_data.py + tools/tinychat_v2.py @ `generator in this repository; seed and count fixed per id`
- Licence: same as this repository (generated)
- Fetch: `python corpus/fetchers/tinychat.py v2_1500` (pinned, idempotent, sha256-verified)
- sha256: `1d68d1ee554279b9274fd0c4da6255a3e6ce648712d0bf5259dbd585582cc8b4`
- Selection ledger: `corpus/ledger/0001-tinychat.md`

## Composition

| measure | value |
|---|---|
| bytes | 393560 |
| documents | 1500 |
| document length, bytes (p50 / p90 / p99) | 243 / 380 / 461 |
| document length, tokens (p50 / p90 / p99) | 60 / 93 / 114 |
| tokens in file (mtstudio tokenizer boundaries) | 91502 |
| non-ASCII bytes | 0 |
| CRLF line endings | 0 |
| end-of-story marker present | True |

## Coverage under the training vocabulary

Tokenizer: `own frequency vocabulary (tinychat_v2_1500/tinychat_vocab.gguf, pinned in corpus/hashes/tinychat_v2_1500.sha256)` (cap 4096), trainer read cap `data.max_tokens = 400000`.

| measure | value |
|---|---|
| tokens the trainer reads | 93002 |
| bytes the trainer reads | 393560 of 393560 |
| tokens per byte | 0.2363 |
| `<unk>` rate (all / held-out tail) | 0.0000 / 0.0000 |
| `<unk>` from non-ASCII bytes | 0 tokens |
| distinct unknown types | 0 |

Most frequent unknown types: .

## Quality signals

- Exact duplicate rate (documents): 0.0000
- Near-duplicate rate (MinHash): PENDING (Phase H)
- n-gram perplexity, half/half split: PENDING (Phase H)
- Language distribution: PENDING (Phase H)

## Contamination

PENDING: n-gram overlap against every set in `corpus/eval/` is measured when those sets are frozen (Phase E). Not asserted.

## Measured effect

PENDING — not a card yet. Filled by the C1 corpus study (charter §8.3): bits per byte on the fixed evaluation panel, matched tokens, 5 seeds, with the seed spread.
