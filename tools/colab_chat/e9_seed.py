#!/usr/bin/env python3
"""Seed the store with E9's generator and receipts, which every D corpus build pulls (e9_data_<ARM>.json):

    python3 tools/colab_chat/e9_seed.py --diversity DIR --store-repo OWNER/NAME --store-token-file FILE --record OUT.json

Two sets beside base 2's seed sets: seed/e9 (the level generator's modules and e9_build.py, pulled to
chat/upload/e9/diversity, two folders below the folder holding scenario_engine/, where levels.py finds the engine) and
seed/e9_receipts (results/stream_receipt_<LEVEL>.json, pulled to chat/upload/e9/diversity/results). base2_seed.py is
imported and run with these sets in place of its own; nothing of it is edited, and a set already in the store with other
bytes is refused, not overwritten.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base2_seed as S  # noqa: E402  (base 2's seeding tool, unchanged)
import hf_store         # noqa: E402

GENERATOR = ("levels.py", "l2_statements.py", "l3_domains.py", "l4_operations.py", "generate_level.py",
             "validate_level.py", "e9_build.py")
LEVELS = ("L1", "L2", "L3", "L4", "L4-ops")


def e9_sets(diversity):
    sets = [("seed/e9", [os.path.join(diversity, f) for f in GENERATOR]),
            ("seed/e9_receipts", [os.path.join(diversity, "results", f"stream_receipt_{lv}.json") for lv in LEVELS])]
    out = []
    for prefix, paths in sets:
        files = {}
        for p in paths:
            if not os.path.isfile(p):
                raise hf_store.StoreError(f"{prefix}: {p} is missing")
            sha, n = hf_store.sha256_file(p)
            files[os.path.basename(p)] = {"bytes": n, "sha256": sha, "source": os.path.abspath(p)}
        out.append((prefix, files))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--diversity" not in argv:
        raise SystemExit("--diversity DIR: E9's folder in the research tree")
    i = argv.index("--diversity")
    diversity = argv[i + 1]
    del argv[i:i + 2]
    S.plan = lambda a: e9_sets(diversity)      # in this process only
    return S.main(argv)


if __name__ == "__main__":
    sys.exit(main())
