#!/usr/bin/env python3
"""Render corpus/cards/<id>.md from corpus/registry.jsonl and the measured
corpus/cards/<id>.stats.json, so no number on a card is typed by hand.

    python corpus/make_cards.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PENDING = "PENDING"


def fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return "n/a" if x is None else str(x)


def card(r, s):
    co = s["composition"]
    cv = s.get("coverage_under_training_vocab")
    L = [f"# Corpus card: `{r['id']}`", "",
         f"*{r['name']}. Rendered by `corpus/make_cards.py` from "
         f"`corpus/registry.jsonl` and `{r['id']}.stats.json`; edit those, not this file.*", "",
         "**Status: a placeholder, not yet a card** — the measured-effect section is "
         f"{PENDING} until the C1 study reports.", "",
         "## Provenance", "",
         f"- Source: {r['source_uri']} @ `{r['revision']}`",
         f"- Licence: {r['license']}",
         f"- Fetch: `python {r['fetcher'].split(' ')[0]} {' '.join(r['fetcher'].split(' ')[1:])}` "
         "(pinned, idempotent, sha256-verified)",
         f"- sha256: `{r['sha256']}`",
         f"- Selection ledger: `{r['selection_ledger']}`", "",
         "## Composition", "",
         "| measure | value |", "|---|---|",
         f"| bytes | {co['bytes']} |",
         f"| documents | {co['documents']} |",
         f"| document length, bytes (p50 / p90 / p99) | {co['doc_bytes']['p50']} / {co['doc_bytes']['p90']} / {co['doc_bytes']['p99']} |",
         f"| document length, tokens (p50 / p90 / p99) | {co['doc_tokens']['p50']} / {co['doc_tokens']['p90']} / {co['doc_tokens']['p99']} |",
         f"| tokens in file (mtstudio tokenizer boundaries) | {co['tokens_total']} |",
         f"| non-ASCII bytes | {co['nonascii_bytes']} |",
         f"| CRLF line endings | {co['crlf_lines']} |",
         f"| end-of-story marker present | {co['has_eos_marker']} |", ""]
    if cv:
        L += ["## Coverage under the training vocabulary", "",
              f"Tokenizer: `{r['tokenizer']}` (cap {cv['cap']}), trainer read cap "
              f"`data.max_tokens = {cv['max_tokens']}`.", "",
              "| measure | value |", "|---|---|",
              f"| tokens the trainer reads | {cv['tokens']} |",
              f"| bytes the trainer reads | {cv['bytes_consumed']} of {co['bytes']} |",
              f"| tokens per byte | {fmt(cv['tokens_per_byte'])} |",
              f"| `<unk>` rate (all / held-out tail) | {fmt(cv['unk_rate'])} / {fmt(cv['val_unk_rate'])} |",
              f"| `<unk>` from non-ASCII bytes | {cv['unk_nonascii_byte_tokens']} tokens |",
              f"| distinct unknown types | {cv['unk_distinct_types']} |", "",
              "Most frequent unknown types: " +
              ", ".join(f"`{w}` ({c})" for w, c in cv["top_unk"][:10]) + ".", ""]
    else:
        L += ["## Coverage under the training vocabulary", "",
              f"{PENDING}: the tokenizer (`{r['tokenizer']}`) is not pinned yet.", ""]
    L += ["## Quality signals", "",
          f"- Exact duplicate rate (documents): {fmt(co['exact_duplicate_rate'])}",
          f"- Near-duplicate rate (MinHash): {PENDING} (Phase H)",
          f"- n-gram perplexity, half/half split: {PENDING} (Phase H)",
          f"- Language distribution: {PENDING} (Phase H)", "",
          "## Contamination", "",
          f"{PENDING}: n-gram overlap against every set in `corpus/eval/` is measured "
          "when those sets are frozen (Phase E). Not asserted.", "",
          "## Measured effect", "",
          f"{PENDING} — not a card yet. Filled by the C1 corpus study (charter §8.3): "
          "bits per byte on the fixed evaluation panel, matched tokens, 5 seeds, with "
          "the seed spread.", ""]
    return "\n".join(L)


def main():
    with open(os.path.join(HERE, "registry.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    for r in rows:
        sp = os.path.join(HERE, "cards", f"{r['id']}.stats.json")
        if r.get("corpus_card") and os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                s = json.load(f)
            if s["sha256"] != r["sha256"]:
                raise SystemExit(f"{r['id']}: stats were measured on different bytes")
            with open(os.path.join(HERE, "cards", f"{r['id']}.md"), "w", encoding="utf-8",
                      newline="\n") as f:
                f.write(card(r, s))
            print("card:", r["id"])


if __name__ == "__main__":
    main()
