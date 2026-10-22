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

**Dialogue pretraining text** (lever 2, [ledger 0004](ledger/0004-dialogue-pretraining.md)):
`fetchers/dialogue_pt.py` builds SODA, OpenAssistant, UltraChat-200k and
everyday-conversations as separate corpora, each checked against the eval sets by
`decontam.py`. `mix_pretrain.py` then mixes them with FineWeb-Edu into a main part and an
annealing tail. Per-source counts by reason are in each corpus's `report.json`. As built on
21 Oct 2026 (Colab data job; ledger `dialogue_pt_v1.ledger.json`):

| Source | Rows in | Kept | Dropped (reason: count) | Text | Answer flags |
|---|---:|---:|---|---:|---:|
| SODA | 1,191,582 | 1,139,202 | non-alternating 39,280; empty turn 12,988; duplicate 58; decontaminated 54 | 834 MB | 2,537 |
| UltraChat-200k | 207,865 | 203,783 | decontaminated 4,051; empty turn 31 | 1,185 MB | 5,575 |
| OpenAssistant (oasst2 + oasst1) | 13,854 trees | 5,388 | not English or not ready 8,453; decontaminated 8; too short 5 | 10.6 MB | 126 |
| everyday-conversations | 2,260 | 2,256 | decontaminated 4 | 1.9 MB | 49 |

No source has a system turn, so none was stripped or dropped for one. UltraChat's
decontamination share (2%) is the highest because `chat_prompts_v1` takes half its prompts from
UltraChat's own test split. Answer flags mark a simple-QA fact or science answer near its
question's words; they are reported, not dropped.
SODA is CC-BY-4.0: Kim et al., 2023, "SODA: Million-scale Dialogue Distillation with Social
Commonsense Contextualization" (allenai/soda).

**Human-written dialogue** (no distillation, [ledger 0005](ledger/0005-human-dialogue.md),
survey [HUMAN_DIALOGUE_SURVEY.md](HUMAN_DIALOGUE_SURVEY.md)): `fetchers/human_dialogue.py` builds
one corpus per source (`human_<source>_v1`) from dialogue written by people only, for the
tiny-chat experiments. The default set has permissive licences; share-alike, non-commercial and
LLM-risk sources are behind `--allow-sa`, `--allow-nc` and `--allow-llm-risk`. The large sources
build on Colab (`tools/colab_chat/human_dialogue_v1.json`).

**A new corpus** needs a ledger entry before it is trained on. The entry
names at least three rejected alternatives and states, as a number, the
baseline the corpus must beat. The corpus also needs a fetcher that pins its
source, a `--pin` run whose hashes are committed, a stats file, and a
rendered card.
