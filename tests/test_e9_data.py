"""E9's D-arm corpus build (e9_data.sh): the generated script, and a rehearsal on base 2's mini fixture.

The rehearsal runs base 2's stages 1 and 2 (base2_data.sh) on the mini fixture's local store, then e9_data.sh's
stage 3 for D2 (level L2) and D4 (level L4) on another VM folder: the level's stream is generated and filtered, the
mix is built and tokenised, the shares are checked, the dose manifest is written and in band, the corpus and the
arm's receipt are published and read back, and a rerun finds the stage finished. Base 2's S is never built.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(ROOT / "tests"))
import make_e9_data as MD  # noqa: E402

E9_SRC = Path(os.environ.get("E9_SRC", "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/"
                                       "AI_ML/chat_screens_claude/cue_selective/diversity"))
E9_FILES = MD_FILES = ("levels.py", "l2_statements.py", "l3_domains.py", "l4_operations.py", "generate_level.py",
                       "validate_level.py", "e9_build.py")


def test_the_script_is_generated_from_base2s_by_the_listed_substitutions():
    assert MD.main(["--check"]) == 0
    text = MD.build()
    assert "generate_level.py" in text and "validate_level.py" in text and "e9_build.py" in text
    a = text.index("build3() {")
    b = text.index("rehearsal() {", a)
    for s in ("corpus/S", "S.ledger", "build-S", "base2_stage3", '"$ENGINE/generate.py" --seed "$SEED" --docs "$SHARD'):
        assert s not in text[a:b], s
    # the held-out panel is still the engine's own, to the same bytes as S's
    assert '"$ENGINE/generate.py" --seed "$SEED" --docs "$HELD_DOCS" --heldout' in text


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    import test_base2_data as T
    if not T.VOCAB.exists() or not (T.ENGINE / "generate.py").exists() or not E9_SRC.exists():
        pytest.skip("needs the 16k vocabulary, the scenario engine and E9's folder")
    m = T.Mini(tmp_path_factory.mktemp("e9_mini"))
    # Since base 2's build was adopted (13 Nov 2026) the pins hold the real build's sources, N, S and scenario panel;
    # the toy corpus has its own, so those groups are set back to "not yet measured" (null), as before the adoption.
    # (Base 2's own mini fixture predates the adoption and fails on these pins; it is not changed here.)
    pins = json.load(open(m.pins))
    for g in ("sources", "N", "S", "rehearsal"):
        pins[g] = {k: None for k in pins[g]}
    pins["panels"]["panel_scenario.ids"] = None
    m.pins.write_text(json.dumps(pins))
    m.run(m.tmp / "vm_a", "stage1")
    m.run(m.tmp / "vm_b", "stage2")
    e9 = m.tmp / "upload" / "e9" / "diversity"          # two folders below the upload folder holding scenario_engine
    (e9 / "results").mkdir(parents=True)
    for f in E9_FILES:
        shutil.copyfile(E9_SRC / f, e9 / f)
    for lv in ("L2", "L4"):
        shutil.copyfile(E9_SRC / "results" / f"stream_receipt_{lv}.json", e9 / "results" / f"stream_receipt_{lv}.json")
    m.e9 = e9

    def run(out, arm, level, expect=0, **extra):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "BASE2_STORE_CONFIG": str(m.vm_config(out)),
               "BASE2_PINS": str(m.pins), "BASE2_ENGINE_DIR": str(m.engine), "BASE2_FIXTURE": str(m.fx),
               "BASE2_THREADS": "2", "BASE2_JOBS": "2", "BASE2_SCN_SHARD_DOCS": "400", "BASE2_SCN_HELDOUT_DOCS": "200",
               "BASE2_MIN_FREE_GB": "0", "BASE2_TIMING_BYTES": "150000", "BASE2_KEEP_MIX_TEXT": "1",
               "BASE2_STORE_RETRY_S": "0", "BASE2_SCN_VALIDATE": "0", "PYTHONDONTWRITEBYTECODE": "1",
               "E9_ARM": arm, "E9_LEVEL": level, "E9_DIR": str(e9), "E9_REHEARSAL": "1", **extra}
        r = subprocess.run(["bash", "tools/colab_chat/e9_data.sh", str(out), str(m.vocab), "stage3"], cwd=ROOT,
                           env=env, capture_output=True, text=True)
        assert r.returncode == expect, (r.stdout[-3000:], r.stderr[-3000:])
        return r.stdout + r.stderr
    m.e9run = run
    m.d2 = m.tmp / "vm_d2"
    m.out_d2 = run(m.d2, "D2", "L2")
    m.out_d2_again = run(m.d2, "D2", "L2")
    m.d4 = m.tmp / "vm_d4"
    m.out_d4 = run(m.d4, "D4", "L4")
    return m


def test_rehearsal_builds_publishes_and_finds_the_stage_finished(rehearsal):
    m = rehearsal
    assert "stage 3 done" in m.out_d2
    assert "stage 3: finished already" in m.out_d2_again
    for arm in ("D2", "D4"):
        corpus = m.store / "main" / "corpus" / arm
        assert (corpus / f"{arm}.main.ids").exists() and (corpus / f"{arm}.tail.ids").exists()
        assert (corpus / f"e9_manifest_{arm}.json").exists()
        assert (m.store / "e9-build" / "build" / arm / "stage3").exists()
    assert not (m.store / "main" / "corpus" / "S").exists()          # base 2's S is never built here


def test_rehearsal_dose_manifest_is_in_band_and_the_streams_differ(rehearsal):
    m = rehearsal
    man = {a: json.load(open(d / f"e9_manifest_{a}.json")) for a, d in (("D2", m.d2), ("D4", m.d4))}
    for a, x in man.items():
        assert x["in_band"] and 0.145 <= x["dose_share"] <= 0.155, (a, x["dose_share"])
        for part in ("main", "tail"):
            assert json.load(open((m.d2 if a == "D2" else m.d4) / f"share_{a}.{part}.json"))["ok"] is True
    assert man["D2"]["stream_sha256"] != man["D4"]["stream_sha256"]
    led = json.load(open(m.d2 / "e9_D2_stage3.json"))
    assert led["built"]["manifest"]["arm"] == "D2"
    # B6: the manifest says at what seed and size it was built, and that it was a rehearsal
    assert man["D2"]["seed"] == 20261111 and man["D2"]["shard_docs"] == 400 and man["D2"]["rehearsal"] is True
    for name in ("scenario_D2.ledger.json", "token_counts_D2.json", "scenario_sample_D2.json"):
        assert (m.d2 / name).exists(), name                     # arm-specific names


def test_shard0_is_held_to_the_receipt_at_the_receipts_size(rehearsal):
    """With the receipt at the rehearsal's size (seed 20261111, 400 documents), shard 0 must be its bytes: the right
    hash builds, a wrong one stops the build before any other shard. (The engine's validator cannot pass at this
    size, so the rehearsal runs with it off; at the campaign's size it runs on every shard.)"""
    m = rehearsal
    p = m.e9 / "results" / "stream_receipt_L2.json"
    keep = p.read_bytes()
    ref = m.tmp / "ref_shard0.txt"
    subprocess.run([sys.executable, str(m.e9 / "generate_level.py"), "--level", "L2", "--seed", "20261111", "--docs",
                    "400", "--start", "0", "--out", str(ref), "--quiet"], check=True, capture_output=True)
    good = __import__("hashlib").sha256(ref.read_bytes()).hexdigest()
    try:
        r = json.loads(keep)
        r.update(docs=400, seed=20261111, text_sha256=good)
        p.write_text(json.dumps(r))
        out = m.e9run(m.tmp / "vm_d2_right", "D2r", "L2")       # a fresh arm: D2 is finished in the store
        assert "stage 3 done" in out
        r["text_sha256"] = "0" * 64
        p.write_text(json.dumps(r))
        out = m.e9run(m.tmp / "vm_d2_wrong", "D2x", "L2", expect=1)
        assert "shard 0 is not the bytes of the level's stream receipt" in out
    finally:
        p.write_bytes(keep)


def test_pins_adopt_from_a_real_stage_receipt(rehearsal, tmp_path):
    """B4: e9_pins.py takes the arm's id hashes from the stage receipt's published-set manifest."""
    import e9_pins as EP
    m = rehearsal
    ep = tmp_path / "e9_pins.json"
    ep.write_text((CC / "e9_pins.json").read_text())
    assert EP.main(["adopt-ledger", str(m.d2 / "e9_D2_stage3.json"), "--e9-pins", str(ep)]) == 0
    got = json.load(open(ep))["D2"]
    for part in ("main", "tail"):
        assert got[f"D2.{part}.ids"] == EP.P.sha256(str(m.d2 / f"D2.{part}.ids"))


def test_another_size_without_e9_rehearsal_is_refused(rehearsal):
    """B6: a VM left at a rehearsal's size (or a stray seed) builds nothing unless E9_REHEARSAL=1 says it is one."""
    out = rehearsal.e9run(rehearsal.tmp / "vm_d2_norehearsal", "D2n", "L2", expect=1, E9_REHEARSAL="0",
                          BASE2_SCN_VALIDATE="1", BASE2_SCN_HELDOUT_DOCS="20000")
    assert "E9_REHEARSAL is not 1" in out and "shard 0 is not the bytes" in out, out[-2000:]


def test_the_validator_cannot_be_switched_off_outside_a_rehearsal(rehearsal):
    """N2: BASE2_SCN_VALIDATE=0 is refused unless E9_REHEARSAL=1."""
    out = rehearsal.e9run(rehearsal.tmp / "vm_d2_noval", "D2w", "L2", expect=1, E9_REHEARSAL="0",
                          BASE2_SCN_VALIDATE="0")
    assert "the level validator must run on every shard" in out


def test_the_level_validator_runs_through_the_build_scripts_command_line(rehearsal):
    """validate_level.py, as e9_data.sh calls it. At 400 documents the engine's rates (cue and genre shares, endings)
    cannot be met, so the build must stop on the validator, and its report must be the level validator's."""
    m = rehearsal
    # the held-out panel is validated first, by the engine's own validator; at its campaign size (20,000) it passes
    out = m.e9run(m.tmp / "vm_d2_validate", "D2v", "L2", expect=1, BASE2_SCN_VALIDATE="1",
                  BASE2_SCN_HELDOUT_DOCS="20000")
    assert "scenario shard 0 fails the engine's validator" in out, out[-2000:]
    scn = m.tmp / "vm_d2_validate" / "scn_D2v"
    log = (scn / "validate_0.log").read_text() if (scn / "validate_0.log").exists() else "(no log)"
    assert (scn / "validate_0.json").exists(), log[-3000:]
    rep = json.load(open(scn / "validate_0.json"))
    assert rep["level"] == "L2" and rep["verdict"] == "FAIL" and "new_layout_docs" in rep


def test_a_changed_generator_is_refused(rehearsal):
    m = rehearsal
    p = m.e9 / "l2_statements.py"
    keep = p.read_bytes()
    try:
        p.write_bytes(keep + b"\n# changed\n")
        out = m.e9run(m.tmp / "vm_d3", "D3", "L2", expect=1)
        assert "the generator is not the receipt's" in out
    finally:
        p.write_bytes(keep)
