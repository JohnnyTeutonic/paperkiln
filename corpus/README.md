# corpus/ — every training corpus, traceable

*Every training corpus in paperkiln must be traceable to a documented
selection decision, with the alternatives that were rejected and the reason
for each rejection recorded in the same place*
([charter 0001](../docs/decisions/0001-posttrain-charter.md) §8).

```bash
make corpus        # rebuild every corpus and vocabulary, byte-exact, sha256-verified
make corpus-test   # the measurement and fetcher tests
```

| Path | What it holds |
|---|---|
| `registry.jsonl` | one record per corpus or vocabulary ever used, with status `active`, `deprecated`, `retired`, `candidate` or `retired-at-scale` |
| `ledger/` | one selection decision per file. It answers the seven questions: task, target, alternatives, reasons for rejection, contamination, licence, and the baseline to beat. |
| `fetchers/` | one script per corpus: pinned revision, deterministic path, idempotent, fails loudly on a hash mismatch (`_common.py` holds the discipline) |
| `hashes/` | `sha256sum`-format pins, written once per corpus id and never re-pinned |
| `cards/` | corpus cards, rendered by `make_cards.py` from the registry and the measured `*.stats.json`. A card is a placeholder until its measured effect exists. |
| `vocab/` | committed token lists for vocabularies that otherwise live only in unpublished model files |
| `eval/` | frozen, versioned held-out sets (Phase E; none frozen yet) |
| `data/` | the fetched bytes (gitignored) |

Measurements come from `tools/corpus_quality.py`. Its tokenizer is a
byte-exact mirror of the trainer's (`include/microtorch/word_tokenizer.hpp`)
and is cross-checked against it (`experiments/corpus_c0/`).

**A new corpus** needs a ledger entry before it is trained on. The entry
names at least three rejected alternatives and states, as a number, the
baseline the corpus must beat. The corpus also needs a fetcher that pins its
source, a `--pin` run whose hashes are committed, a stats file, and a
rendered card.
