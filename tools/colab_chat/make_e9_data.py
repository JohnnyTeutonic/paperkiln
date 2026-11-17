#!/usr/bin/env python3
"""E9's D-arm corpus build, derived from base 2's (base2_data.sh) by a fixed list of substitutions.

    python tools/colab_chat/make_e9_data.py            write e9_data.sh beside this file
    python tools/colab_chat/make_e9_data.py --check    exit 1 unless e9_data.sh is what this writes

e9_data.sh is base2_data.sh's stage 3 for one D arm: the same held-out panel, docfilter, recipe from N's measured
sizes, shards, mix, ids, share checks (14% to 16% a part) and token report against N, with:
  * the level's generator and validator (E9's generate_level.py and validate_level.py, uploaded at E9_DIR, which must
    sit two folders below the upload folder that holds scenario_engine/) in place of the engine's;
  * a check that those modules and the engine's are the bytes the level's stream receipt was made with, and, with
    E9_SHARD0_SHA256 set (the receipt's text_sha256), that shard 0 is the receipt's bytes before any other shard;
  * the arm's names (E9_ARM) in place of S's: corpus/<ARM>, recipe_<ARM>, <ARM>.{main,tail}.ids, ...;
  * its own receipt branch (e9-build:build/<ARM>/stage3), and the per-arm dose manifest (e9_build.py manifest), which
    fails the stage outside [14.5%, 15.5%];
  * only stage 3: stages 1 and 2 are base 2's, read from base2-build; the held-out scenario panel is regenerated to
    the same bytes and not pushed again.
tests/test_e9_data.py holds the script to this list and rehearses it on base 2's mini fixture with a local store.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "base2_data.sh")
OUT = os.path.join(HERE, "e9_data.sh")

HEAD = """# E9 (scenario diversity): a D arm's corpus, base 2's stage 3 with a level's stream. GENERATED from base2_data.sh by
# make_e9_data.py; do not edit. Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
#   E9_ARM=D2 E9_LEVEL=L2 [E9_DIR=...] [E9_SHARD0_SHA256=...] bash tools/colab_chat/e9_data.sh OUT VOCAB_GGUF stage3
# The rest of this header is base 2's, for the stages this script inherits.
"""

GLOBAL = [
    ('MODE=${3:?stage1, stage2, stage3, all, rehearsal or loadprobe}',
     'MODE=${3:?stage3}\n'
     'ARM=${E9_ARM:?E9_ARM: the D arm, D2, D3, D4, D4ops or a second-seed arm}\n'
     'E9_LEVEL=${E9_LEVEL:?E9_LEVEL: L2, L3, L4 or L4-ops}\n'
     'E9_DIR=${E9_DIR:-/content/chat/upload/e9/diversity}\n'
     'EB=e9-build', 1),
    ('WORK="$OUT/work"', 'WORK="$OUT/work_$ARM"', 1),
    ('python3 "$ENGINE/generate.py" --seed "$SEED" --docs "$SHARD_DOCS"',
     'python3 "$E9_DIR/generate_level.py" --level "$E9_LEVEL" --seed "$SEED" --docs "$SHARD_DOCS"', 1),
    ('--out "$d/raw_$k.txt" --meta "$d/raw_$k.meta.jsonl" > /dev/null',
     '--out "$d/raw_$k.txt" --meta "$d/raw_$k.meta.jsonl" > /dev/null\n'
     '  if [ "$k" = 0 ] && [ -n "${E9_SHARD0_SHA256:-}" ]; then\n'
     '    [ "$(sha256sum "$d/raw_0.txt" | cut -d " " -f 1)" = "$E9_SHARD0_SHA256" ] \\\n'
     '      || { echo "e9_data: shard 0 is not the bytes of the level\'s stream receipt"; exit 1; }\n'
     '  fi', 1),
    ('python3 "$ENGINE/validate.py" "$d/raw_$k.txt"',
     'python3 "$E9_DIR/validate_level.py" --level "$E9_LEVEL" "$d/raw_$k.txt"', 1),
    ('  local k=$1 d="$OUT/scn" led bind', '  local k=$1 d="$OUT/scn_$ARM" led bind', 1),
    ('  bind="$(ENGINE_INPUTS) --input $d/panel_scenario.txt --input $SRC/human_val.txt',
     '  bind="$(ENGINE_INPUTS) $(for f in "$E9_DIR"/*.py; do echo "--input $f"; done) --key level=$E9_LEVEL '
     '--input $d/panel_scenario.txt --input $SRC/human_val.txt', 1),
    ('  PINS "$ENGINE" engine   # the generator\'s files are the pinned ones, or nothing is generated',
     '  PINS "$ENGINE" engine   # the generator\'s files are the pinned ones, or nothing is generated\n'
     '  python3 "$E9_DIR/e9_build.py" check-sources --receipt "$E9_DIR/results/stream_receipt_$E9_LEVEL.json" \\\n'
     '    --engine "$ENGINE" --e9 "$E9_DIR" || { echo "e9_data: the generator is not the receipt\'s"; exit 1; }', 1),
    ('    3) echo "main:corpus/S main:corpus/panel_scenario" ;;', '    3) echo "main:corpus/$ARM" ;;', 1),
    ('  if in_store "$BB" "build/stage$n"; then', '  if in_store "$EB" "build/$ARM/stage$n"; then', 1),
    ('    [ -s "$OUT/base2_stage$n.json" ] || pull_set "$BB" "build/stage$n" "$OUT"',
     '    [ -s "$OUT/e9_${ARM}_stage$n.json" ] || pull_set "$EB" "build/$ARM/stage$n" "$OUT"', 1),
    ('  rm -f "$OUT/base2_stage$n.json"   # a receipt', '  rm -f "$OUT/e9_${ARM}_stage$n.json"   # a receipt', 1),
    ('  stage1) run_stage 1 ;;\n  stage2) run_stage 2 ;;\n', '', 1),
    ('  all) run_stage 1; run_stage 2; run_stage 3 ;;\n  rehearsal) rehearsal ;;\n  loadprobe) loadprobe "${4:-}" ;;\n',
     '', 1),
    ('*) echo "mode is stage1, stage2, stage3, all, rehearsal or loadprobe"', '*) echo "mode is stage3"', 1),
]

# Inside build3() and publish3() only: S's names become the arm's.
STAGE3 = [
    ('  local d="$OUT/scn" k shards part bind', '  local d="$OUT/scn_$ARM" k shards part bind', 1),
    ('recipe_S', 'recipe_$ARM', None),
    ('$OUT/S.', '$OUT/$ARM.', None),
    ('ids_S', 'ids_$ARM', None),
    ('share_S.', 'share_$ARM.', None),
    ('S_in_N', '${ARM}_in_N', None),
    ('mix_and_ids S ', 'mix_and_ids "$ARM" ', 1),
    ('drop_mix_text S ', 'drop_mix_text "$ARM" ', 1),
    ('"S.main.ids=', '"$ARM.main.ids=', 1),
    ('"S.tail.ids=', '"$ARM.tail.ids=', 1),
    ('  PINS "$OUT" build-S\n',
     '  # base 2 recorded S\'s pins here; a D arm is pinned (arm-$ARM) from this stage ledger\'s file hashes\n'
     '  python3 "$E9_DIR/e9_build.py" manifest --arm "$ARM" --level "$E9_LEVEL" --main "$OUT/share_$ARM.main.json" \\\n'
     '    --tail "$OUT/share_$ARM.tail.json" --scenario "$OUT/scenario.ledger.json" --out "$OUT/e9_manifest_$ARM.json" \\\n'
     '    > /dev/null || { echo "e9_data: the arm\'s dose is outside its band (e9_manifest_$ARM.json)"; exit 1; }\n', 1),
    ('base2_stage3', 'e9_${ARM}_stage3', None),
    ('--json "pins=$OUT/base2_pins_observed.json"',
     '--json "pins=$OUT/base2_pins_observed.json" --json "manifest=$OUT/e9_manifest_$ARM.json"', 1),
    ('  push_set main corpus/panel_scenario "$OUT/panel_scenario.ids" "$OUT/panel_scenario.json"\n', '', 1),
    ('push_set main corpus/S ', 'push_set main "corpus/$ARM" ', 1),
    ('"$OUT/share_$ARM.tail.json" "$OUT/token_counts.json"',
     '"$OUT/share_$ARM.tail.json" "$OUT/token_counts.json" "$OUT/e9_manifest_$ARM.json"', 1),
    ('--json "published_S=$OUT/published_corpus_S.json" --json "published_panel_scenario=$OUT/published_corpus_panel_scenario.json"',
     '--json "published_$ARM=$OUT/published_corpus_$ARM.json"', 1),
    ('push_set "$BB" build/stage3 ', 'push_set "$EB" "build/$ARM/stage3" ', 1),
    ('  rm -rf "$OUT/scn" "$WORK"', '  rm -rf "$OUT/scn_$ARM" "$WORK"', 1),
]


def apply(text, subs, where):
    for old, new, n in subs:
        k = text.count(old)
        if (n is not None and k != n) or k == 0:
            raise SystemExit(f"{where}: {old!r} occurs {k} times, expected {n or 'at least once'}")
        text = text.replace(old, new)
    return text


def build():
    src = open(SRC, encoding="utf-8").read()
    a = src.index("build3() {")
    b = src.index("rehearsal() {", a)
    mid = apply(src[a:b], STAGE3, "stage 3")
    for leftover in ("corpus/S", "S.ledger", "_S.", "build-S", "base2_stage3"):
        if leftover in mid:
            raise SystemExit(f"stage 3 still names S: {leftover!r}")
    text = apply(src[:a] + mid + src[b:], GLOBAL, "script")
    first, rest = text.split("\n", 1)
    return first + "\n" + HEAD + rest


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = build()
    if "--check" in argv:
        same = os.path.exists(OUT) and open(OUT, encoding="utf-8").read() == text
        print("equal" if same else "different")
        return 0 if same else 1
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.chmod(OUT, 0o755)
    print("written e9_data.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
