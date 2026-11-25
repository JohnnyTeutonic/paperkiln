"""SYMW's jobs file (tools/colab_chat/make_symw.py) against the token ladder's N1B: equal but for the registered changes.
Claude Opus 5.5, 26 Nov 2026."""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
import make_base2 as M  # noqa: E402
import make_ladder as L  # noqa: E402
import make_symw as S  # noqa: E402


def load(name):
    return json.loads((CC / name).read_text(encoding="utf-8"))


def test_file_is_the_builder_output():
    assert S.main(["--check"]) == 0
    assert load(S.OUT_FILE) == json.loads(M.text(S.arm()))


def test_n1b_on_disk_is_the_ladders():
    assert load("ladder_N1B.json") == json.loads(M.text(L.arm("N1B")))


def test_pin_is_a_real_hash():
    assert re.fullmatch(r"[0-9a-f]{64}", S.SYMW_IDS_SHA256)
    pin = next(j for j in load(S.OUT_FILE)["jobs"] if j["name"] == "symw_pin")
    assert f"'^{S.SYMW_IDS_SHA256} '" in pin["cmd"][2] and pin["halt_on_failure"] and pin["light"]


def test_equal_to_n1b_but_for_the_registered_changes():
    s, n = copy.deepcopy(load(S.OUT_FILE)), load("ladder_N1B.json")
    # 1. the warm-up, on the training job's spec only
    train = next(j for j in s["jobs"] if j["name"] == "pt_base2_SYMW")
    assert train["spec"]["data"].pop("warmup") == {"corpus": f"{M.DATA}/symw.ids", "steps": 2000}
    # 2. the pin job, and training waits for it
    assert [j["name"] for j in s["jobs"]][:2] == ["pins", "symw_pin"]
    s["jobs"] = [j for j in s["jobs"] if j["name"] != "symw_pin"]
    assert train["after"] == ["pins", "symw_pin"]
    train["after"] = ["pins"]
    # 3. the symbolic corpus among the store's inputs, last
    assert s["store_inputs"].pop() == S.SYMW_SET
    # 4. the release turn
    assert s.pop("store_release_offset_min") == 20 and "store_release_offset_min" not in n
    # 5. comment and metadata
    for k in ("_comment", "_base2"):
        s.pop(k), n.pop(k)
    # 6. names
    t = json.dumps(s).replace("SYMW", "N1B")
    assert json.loads(t) == n


def test_schedule_marks_and_seed_are_n1bs():
    s = load(S.OUT_FILE)
    train = next(j for j in s["jobs"] if j["name"] == "pt_base2_SYMW")
    tr = train["spec"]["train"]
    assert tr["steps"] == 60000 and tr["seed"] == 1 and tr["lr_schedule"]["warmup_steps"] == 4000
    assert train["snapshot_steps"] == [10000, 58500, 60000]
    tail_from = int(train["spec"]["data"]["anneal"]["from_fraction"] * tr["steps"])
    assert train["spec"]["data"]["warmup"]["steps"] < tail_from
    b = s["_base2"]
    assert b["warmup"]["positions"] == 2000 * 16 * 1024 == 32768000
    assert b["positions"] == 60000 * 16 * 1024


def test_no_n1b_name_left():
    assert "N1B" not in json.dumps(load(S.OUT_FILE)["jobs"])


TRAINER_TEST_SKIPS = ("test_trainer_sources_are_the_measured_commit", "test_one_window_a_row_from_one_generator_in_the_source")


def test_trainer_differs_from_base2s_measured_commit_in_mtstudio_only():
    """Base 2's test_trainer_sources_are_the_measured_commit fails here by design (SYMW changes tools/mtstudio.cpp): this
    is its replacement. Against base 2's measured trainer commit, only tools/mtstudio.cpp differs among the sources."""
    import subprocess
    import pytest
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    import test_base2_jobs as B
    paths = ["src", "include", "third_party", "tools/mtstudio.cpp", "CMakeLists.txt"]
    d = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", B.TRAINER, "--", *paths], capture_output=True, text=True)
    assert d.returncode == 0 and d.stdout.split() == ["tools/mtstudio.cpp"], d.stdout


def test_one_draw_a_row_with_the_warmup():
    """Base 2's test_one_window_a_row_from_one_generator_in_the_source, restated for the changed line: inside the
    micro-step loop each row takes one draw (one for SFT rows, one for windows), the window offset is that draw modulo the
    chosen corpus's limit whatever the corpus, and a resume replays start_step x accum x batch draws."""
    src = (ROOT / "tools" / "mtstudio.cpp").read_text()
    loop = src[src.index("for (int k = 0; !grpo && !po && k < s.accum; ++k) {"):]
    loop = loop[:loop.index('mark("data");')]
    assert loop.count("rng()") == 2 and "for (int b = 0; b < s.batch; ++b) {" in loop
    assert "const auto draw = rng();" in loop and "const size_t at = draw % lim;" in loop
    assert "for (int i = 0; i < start_step * s.accum * s.batch; ++i) rng();" in src
    sel = src[src.index("const bool warming = "):src.index("const bool warming = ") + 700]
    assert "src = warming ? warmup_ids : annealing ? anneal_ids : ids;" in sel
