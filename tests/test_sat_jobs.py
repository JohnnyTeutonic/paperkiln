"""SAT's jobs file (tools/colab_chat/make_sat.py) against the token ladder's N1B: equal but for the registered changes;
base 2's two source-pin tests restated for the changed trainer; the VM's sat_check wrapper on the CPU trainer.
Claude Opus 5.5, 26 Nov 2026."""
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
sys.path.insert(0, str(ROOT / "tests"))
import make_base2 as M  # noqa: E402
import make_ladder as L  # noqa: E402
import make_sat as S  # noqa: E402


def load(name):
    return json.loads((CC / name).read_text(encoding="utf-8"))


def job(d, name):
    return next(j for j in d["jobs"] if j["name"] == name)


def test_file_is_the_builder_output():
    assert S.main(["--check"]) == 0
    assert load(S.OUT_FILE) == json.loads(M.text(S.arm()))


def test_n1b_on_disk_is_the_ladders():
    assert load("ladder_N1B.json") == json.loads(M.text(L.arm("N1B")))


def test_equal_to_n1b_but_for_the_registered_changes():
    s, n = copy.deepcopy(load(S.OUT_FILE)), load("ladder_N1B.json")
    # 1. symbol_targets, on the training job's spec only
    train = job(s, "pt_base2_SAT")
    assert train["spec"]["train"].pop("symbol_targets") == S.symbol_targets()
    # 2. the pin and check jobs, and training waits for both
    assert [j["name"] for j in s["jobs"]][:4] == ["pins", "sat_pin", "sat_check", "pt_base2_SAT"]
    s["jobs"] = [j for j in s["jobs"] if j["name"] not in ("sat_pin", "sat_check")]
    assert train["after"] == ["pins", "sat_pin", "sat_check"]
    train["after"] = ["pins"]
    # 3. the four SAT files among the store's inputs, last
    assert s["store_inputs"][-3:] == S.SAT_SETS
    del s["store_inputs"][-3:]
    # 4. the release turn (SYMW's)
    assert s.pop("store_release_offset_min") == 20 and "store_release_offset_min" not in n
    # 5. comment and metadata
    for k in ("_comment", "_base2"):
        s.pop(k), n.pop(k)
    # 6. names
    assert json.loads(json.dumps(s).replace("SAT", "N1B")) == n


def test_symbol_targets_are_the_registered_ones():
    st = job(load(S.OUT_FILE), "pt_base2_SAT")["spec"]["train"]["symbol_targets"]
    assert (st["w0"], st["half_life"], st["cut_step"], st["role_classes"], st["pointer_dim"]) == (1.0, 2000, 13288, 8, 64)
    assert st["twins"]["every"] == 10
    for k in ("labels", "eval_labels"):
        assert st[k].startswith(M.DATA + "/")
    tr = job(load(S.OUT_FILE), "pt_base2_SAT")["spec"]
    tail_from = int(tr["data"]["anneal"]["from_fraction"] * tr["train"]["steps"])
    assert st["cut_step"] <= tail_from == 51000 and not tr["train"].get("fused_head")
    # The registered cut: the first step at which w falls below 1% of w0.
    import math
    assert st["cut_step"] == math.ceil(st["half_life"] * math.log2(100))


def test_pin_job_checks_all_four_files():
    pin = job(load(S.OUT_FILE), "sat_pin")
    assert pin["light"] and pin["halt_on_failure"]
    cmd = pin["cmd"][2]
    for f in S.FILES:
        assert f"grep -q '^{S.SAT_SHA256[f]}  {M.DATA}/{f}$' {{out}}/sat_pin.txt" in cmd
        assert re.fullmatch(r"[0-9a-f]{64}", S.SAT_SHA256[f])
    files = {f for st in S.SAT_SETS for f in st["files"]}
    assert files == set(S.FILES) and all(st["prefix"].startswith("corpus/sat_") for st in S.SAT_SETS)


def test_placeholder_hashes_are_marked_and_fail_closed():
    """Until the files are built the hashes are the all-zero placeholder (no file hashes to it, so sat_pin fails and
    the session halts); the metadata says so. When they are replaced, none may be left."""
    b = load(S.OUT_FILE)["_base2"]["symbol_targets"]
    placeholders = [f for f, h in S.SAT_SHA256.items() if h == S.PLACEHOLDER]
    assert b["hashes_are_placeholders"] == bool(placeholders)
    assert placeholders in ([], list(S.FILES)), "replace all four hashes together"


def test_check_job_runs_the_training_spec():
    d = load(S.OUT_FILE)
    chk, train = job(d, "sat_check"), job(d, "pt_base2_SAT")
    assert chk["light"] and chk["halt_on_failure"] and chk["after"] == ["pins", "sat_pin"]
    argv = chk["cmd"]
    spec = json.loads(argv[argv.index("--spec-json") + 1])
    assert argv[argv.index("--windows") + 1] == "8"
    assert spec.pop("name") == "base2_SAT_check"
    t = copy.deepcopy(train["spec"])
    t.pop("name")
    assert spec == t


def test_schedule_marks_and_seed_are_n1bs():
    s = load(S.OUT_FILE)
    train = job(s, "pt_base2_SAT")
    tr = train["spec"]["train"]
    assert tr["steps"] == 60000 and tr["seed"] == 1 and tr["lr_schedule"]["warmup_steps"] == 4000
    assert train["snapshot_steps"] == [10000, 58500, 60000] and train["env"] == {"MICROTORCH_CHECKPOINT_TXN": "1"}
    assert s["_base2"]["positions"] == 60000 * 16 * 1024


def test_no_n1b_name_left():
    assert "N1B" not in json.dumps(load(S.OUT_FILE)["jobs"])


TRAINER_TEST_SKIPS = ("test_trainer_sources_are_the_measured_commit", "test_one_window_a_row_from_one_generator_in_the_source")


def test_trainer_differs_from_base2s_measured_commit_in_mtstudio_only():
    """Base 2's test_trainer_sources_are_the_measured_commit fails here by design (SAT changes tools/mtstudio.cpp): this
    is its replacement. Against base 2's measured trainer commit, only tools/mtstudio.cpp differs among the sources."""
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    import test_base2_jobs as B
    paths = ["src", "include", "third_party", "tools/mtstudio.cpp", "CMakeLists.txt"]
    d = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", B.TRAINER, "--", *paths], capture_output=True, text=True)
    assert d.returncode == 0 and d.stdout.split() == ["tools/mtstudio.cpp"], d.stdout


def test_one_draw_a_row_with_the_twins():
    """Base 2's test_one_window_a_row_from_one_generator_in_the_source, restated for the changed lines: inside the
    micro-step loop each row takes one draw (one for SFT rows, one for windows) and hands it to take_window, whose
    offset is that draw modulo the main corpus's limit, or the twin corpus's for a twin row (the draw is taken either
    way); a resume replays start_step x accum x batch draws; the check mode draws from a copy of the generator."""
    src = (ROOT / "tools" / "mtstudio.cpp").read_text()
    loop = src[src.index("for (int k = 0; !grpo && !po && k < s.accum; ++k) {"):]
    loop = loop[:loop.index('mark("data");')]
    assert loop.count("rng()") == 2 and "for (int b = 0; b < s.batch; ++b) {" in loop
    assert "const auto draw = rng();" in loop and "take_window(step, k, b, draw, src, lim," in loop
    tw = src[src.index("auto take_window = [&]"):]
    tw = tw[:tw.index("return twin;")]
    assert "rng" not in tw
    assert "const size_t at = twin ? draw % twin_lim : draw % lim;" in tw
    assert "g % static_cast<uint64_t>(s.sat.twin_every) == 0" in tw and "step < s.sat.cut_step" in tw
    assert "for (int i = 0; i < start_step * s.accum * s.batch; ++i) rng();" in src
    chk = src[src.index("if (sat_check_n > 0) {"):]
    chk = chk[:chk.index("// Train.")]
    assert "std::mt19937 g = rng;" in chk and "rng()" not in chk


@pytest.mark.skipif(__import__("test_mtstudio_chat").BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
def test_sat_check_vm_on_the_cpu_trainer(tmp_path):
    """The VM wrapper end to end on the CPU build: pass on SAT's synthetic spec; fail (non-zero) without labels."""
    import test_sat_targets as T
    from test_mtstudio_chat import BIN
    import sat_check_vm as V
    T.setup(tmp_path)
    spec = json.loads(T.spec(tmp_path, tmp_path / "unused", T.sat_cfg(tmp_path)).read_text())
    out = tmp_path / "chk"
    assert V.main(["--out", str(out), "--windows", "8", "--spec-json", json.dumps(spec), "--bin", BIN]) == 0
    rec = json.loads((out / "sat_check.json").read_text())
    assert rec["ok"] and rec["result"][0]["equal"] and len(rec["checks"]) == 16
    spec["train"]["symbol_targets"]["labels"] = str(tmp_path / "none.satl")
    spec["train"]["symbol_targets"].pop("twins")          # no targets anywhere: a check that cannot fail is refused
    assert V.main(["--out", str(tmp_path / "chk2"), "--windows", "8", "--spec-json", json.dumps(spec), "--bin", BIN]) == 1
    spec["train"]["symbol_targets"]["labels"] = str(tmp_path / "bad.satl")
    assert V.main(["--out", str(tmp_path / "chk3"), "--windows", "8", "--spec-json", json.dumps(spec), "--bin", BIN]) == 1
