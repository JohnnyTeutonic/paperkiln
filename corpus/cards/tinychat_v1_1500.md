# Corpus card: `tinychat_v1_1500`

*TinyChat v1, 1500 dialogues, seed 7 (synthetic). Rendered by `corpus/make_cards.py` from `corpus/registry.jsonl` and `tinychat_v1_1500.stats.json`; edit those, not this file.*

**Status: a placeholder, not yet a card** — the measured-effect section is PENDING until the C1 study reports.

## Provenance

- Source: tools/get_tinychat_data.py @ `generator in this repository; seed and count fixed per id`
- Licence: same as this repository (generated)
- Fetch: `python corpus/fetchers/tinychat.py v1_1500` (pinned, idempotent, sha256-verified)
- sha256: `c18129efc35cb7db1b69818ce0edcd5e9a2b8baf2377c52f830c717140a56f35`
- Selection ledger: `corpus/ledger/0001-tinychat.md`

## Composition

| measure | value |
|---|---|
| bytes | 369202 |
| documents | 1500 |
| document length, bytes (p50 / p90 / p99) | 233 / 332 / 370 |
| document length, tokens (p50 / p90 / p99) | 57 / 81 / 90 |
| tokens in file (mtstudio tokenizer boundaries) | 85494 |
| non-ASCII bytes | 0 |
| CRLF line endings | 0 |
| end-of-story marker present | True |

## Coverage under the training vocabulary

Tokenizer: `own frequency vocabulary (tinychat_v1_1500/tinychat_vocab.gguf, pinned in corpus/hashes/tinychat_v1_1500.sha256)` (cap 4096), trainer read cap `data.max_tokens = 400000`.

| measure | value |
|---|---|
| tokens the trainer reads | 86994 |
| bytes the trainer reads | 369202 of 369202 |
| tokens per byte | 0.2356 |
| `<unk>` rate (all / held-out tail) | 0.0000 / 0.0000 |
| `<unk>` from non-ASCII bytes | 0 tokens |
| distinct unknown types | 0 |

Most frequent unknown types: .

## Quality signals

- Exact duplicate rate (documents): 0.0053
- Near-duplicate rate (MinHash): PENDING (Phase H)
- n-gram perplexity, half/half split: PENDING (Phase H)
- Language distribution: PENDING (Phase H)

## Contamination

PENDING: n-gram overlap against every set in `corpus/eval/` is measured when those sets are frozen (Phase E). Not asserted.

## Measured effect

PENDING — not a card yet. Filled by the C1 corpus study (charter §8.3): bits per byte on the fixed evaluation panel, matched tokens, 5 seeds, with the seed spread.
