# Corpus card: `tinychat_v2_4000`

*TinyChat v2, 4000 dialogues, seed 11 (synthetic). Rendered by `corpus/make_cards.py` from `corpus/registry.jsonl` and `tinychat_v2_4000.stats.json`; edit those, not this file.*

**Status: a placeholder, not yet a card** — the measured-effect section is PENDING until the C1 study reports.

## Provenance

- Source: tools/get_tinychat_data.py + tools/tinychat_v2.py @ `generator in this repository; seed and count fixed per id`
- Licence: same as this repository (generated)
- Fetch: `python corpus/fetchers/tinychat.py v2_4000` (pinned, idempotent, sha256-verified)
- sha256: `d195c7ee0e44e995ad04751c4e06f9691b0a517d818533f0d490d74bbc88310c`
- Selection ledger: `corpus/ledger/0001-tinychat.md`

## Composition

| measure | value |
|---|---|
| bytes | 1049149 |
| documents | 4000 |
| document length, bytes (p50 / p90 / p99) | 245 / 377 / 464 |
| document length, tokens (p50 / p90 / p99) | 61 / 93 / 114 |
| tokens in file (mtstudio tokenizer boundaries) | 243871 |
| non-ASCII bytes | 0 |
| CRLF line endings | 0 |
| end-of-story marker present | True |

## Coverage under the training vocabulary

Tokenizer: `own frequency vocabulary (tinychat_v2_4000/tinychat_vocab.gguf, pinned in corpus/hashes/tinychat_v2_4000.sha256)` (cap 4096), trainer read cap `data.max_tokens = 2000000`.

| measure | value |
|---|---|
| tokens the trainer reads | 247871 |
| bytes the trainer reads | 1049149 of 1049149 |
| tokens per byte | 0.2363 |
| `<unk>` rate (all / held-out tail) | 0.0000 / 0.0000 |
| `<unk>` from non-ASCII bytes | 0 tokens |
| distinct unknown types | 0 |

Most frequent unknown types: .

## Quality signals

- Exact duplicate rate (documents): 0.0022
- Near-duplicate rate (MinHash): PENDING (Phase H)
- n-gram perplexity, half/half split: PENDING (Phase H)
- Language distribution: PENDING (Phase H)

## Contamination

PENDING: n-gram overlap against every set in `corpus/eval/` is measured when those sets are frozen (Phase E). Not asserted.

## Measured effect

PENDING — not a card yet. Filled by the C1 corpus study (charter §8.3): bits per byte on the fixed evaluation panel, matched tokens, 5 seeds, with the seed spread.
