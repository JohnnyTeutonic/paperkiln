#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paperfetch — drop an arXiv id, get the paper's architecture back.

    paperfetch 1706.03762                     # evidence-carrying summary
    paperfetch 2302.13971 --emit-hf cfg.json  # HF config.json, provenance inside
    paperfetch 2302.13971 --json arch.json    # normalized extraction
    paperfetch --tex local.tex 0000.00000     # parse a local LaTeX file

Every extracted value carries the evidence snippet it was read from;
fields the sweep cannot resolve are declared, never silently guessed.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import _fetch
from .hf_emit import build_hf_config


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="paperfetch", description=__doc__.splitlines()[0])
    ap.add_argument("arxiv_id")
    ap.add_argument("--tex", help="parse a local .tex file instead of fetching")
    ap.add_argument("--json", help="write the normalized extraction here")
    ap.add_argument("--emit-hf", metavar="PATH",
                    help="write a Hugging Face config.json here (llama- or "
                         "gpt2-family, chosen from the extracted flavors; "
                         "evidence + declared defaults ride inside under "
                         "the _paperkiln key)")
    ap.add_argument("--emit-cpp", help="write microtorch C++ here")
    ap.add_argument("--emit-html", help="write the diff-to-paper view here")
    args = ap.parse_args()

    tex = (open(args.tex, encoding="utf-8", errors="replace").read()
           if args.tex else _fetch.fetch_source(args.arxiv_id))
    arch = _fetch.extract(args.arxiv_id, tex)

    print(f"arXiv:{arch.arxiv_id}  {arch.title or ''}".strip())
    if arch.inherits:
        print(f"  [inherits {arch.inherits['ancestor']} — "
              f"{arch.inherits['evidence'][:60]}]")
    for k, f in arch.fields.items():
        tag = f" ({f.verdict} {f.score})" if f.verdict else ""
        print(f"  {k:15} = {f.value!s:8}{tag}  [{f.evidence[:60]}]")
        if f.runner_up:
            print(f"  {'':15}   vs {f.runner_up['value']} "
                  f"({f.runner_up['score']}) — contested, verify by hand")
    if arch.unresolved:
        print(f"  unresolved: {', '.join(arch.unresolved)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_fetch.to_json(arch), fh, indent=2)
        print(f"wrote {args.json}")
    if args.emit_hf:
        cfg, family, defaults = build_hf_config(arch)
        with open(args.emit_hf, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"wrote {args.emit_hf}  ({family}-family)")
        if defaults:
            noted = ", ".join(f"{k}={v}" for k, v in defaults.items())
            print(f"  defaults applied (declared in _paperkiln): {noted}")
        print("  load with: transformers.AutoConfig.from_pretrained("
              f"'{args.emit_hf}')")
    if args.emit_cpp:
        with open(args.emit_cpp, "w", encoding="utf-8") as fh:
            fh.write(_fetch.emit_cpp(arch))
        print(f"wrote {args.emit_cpp}")
    if args.emit_html:
        with open(args.emit_html, "w", encoding="utf-8") as fh:
            fh.write(_fetch.emit_html(arch))
        print(f"wrote {args.emit_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
