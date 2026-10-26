# ember's draft table from paperkiln: byte identity (27 Oct 2026)

`tools/draft_table.py` (mtstudio's opt-in `export.draft_table`) builds ember's `EMBER_DRAFT=table` n-gram
table (ember.cpp bace837, `experiments/draft_2026-10-27`). Its builder is ember's `dr_table.py`, vendored.
On the corpora ember measured, run in WSL under the CPU budget (2 cores), it reproduces both of ember's
tables byte for byte:

| receipt | sources | GGUF (vocabulary) | tokens (stream sha256) | table sha256 | ember's |
|---|---|---|---|---|---|
| `h_c8_p10.bin.json` | the 13 human dialogue sources' `text.txt` (tiny-chat SFT data) | pick_e1_L16 | 40,123,699 (58c41de4...) | 5a58236a... | `h_c8_p10.bin`, equal |
| `s_c8_p10.bin.json` | SODA's `text.txt` (first 150 MB) + OASST's | pk_base_v1 | 36,161,921 (312b9180...) | ade26f5a... | `s_c8_p10.bin`, equal |
| `h_c8_p10_from_dialogues_jsonl.bin.json` | the same 13 sources' `dialogues.jsonl` (`{"messages"}` rows, rendered by the GGUF's paperkiln/1) | pick_e1_L16 | 40,123,699 (58c41de4...) | 5a58236a... | `h_c8_p10.bin`, equal |

The token streams' sha256 equal ember's `human.u16` and `soda.u16` (ember's `receipts/tables.json`). Settings
are ember's chosen ones, the tool's defaults: orders 1-4, contexts seen at least 8 times, top share at least
0.10, redundant back-off entries pruned (10.3 MB and 8.4 MB; entries 14,151 / 267,966 / 231,134 / 422,618 and
14,689 / 132,267 / 154,980 / 459,193). Wall time 1.5-4 minutes each. The tables themselves are not kept here
(ember's are at `C:/ml_artifacts/ember_draft/tables/`).

Run (WSL, from the repository): `python3 tools/draft_table.py build --gguf <model.gguf> --out <table.bin>
<sources...> [--max-bytes 150000000]`.
