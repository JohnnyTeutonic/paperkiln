"""SATW's jobs file (tools/colab_chat/make_satw.py) against SAT's (make_sat.py): equal but for the recency weight,
the names, the release turn and the metadata. Claude Opus 5.5, 26 Nov 2026."""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
import make_base2 as M  # noqa: E402
import make_sat as S  # noqa: E402
import make_satw as SW  # noqa: E402


def load(name):
    return json.loads((CC / name).read_text(encoding="utf-8"))


def job(d, name):
    return next(j for j in d["jobs"] if j["name"] == name)


def check_spec(d):
    argv = job(d, "sat_check")["cmd"]
    return json.loads(argv[argv.index("--spec-json") + 1])


def test_file_is_the_builder_output():
    assert SW.main(["--check"]) == 0
    assert load(SW.OUT_FILE) == json.loads(M.text(SW.arm()))


def test_sat_on_disk_is_the_builders():
    assert load(S.OUT_FILE) == json.loads(M.text(S.arm()))


def test_equal_to_sat_but_for_the_weight_names_and_turn():
    w, s = copy.deepcopy(load(SW.OUT_FILE)), load(S.OUT_FILE)
    # 1. the weight, on the training spec and on the check's copy of it, and nowhere else
    want = {"nearest_right": SW.RECENCY_WEIGHT}
    assert job(w, "pt_base2_SATW")["spec"]["train"]["symbol_targets"].pop("recency_weight") == want
    chk = check_spec(w)
    assert chk["train"]["symbol_targets"].pop("recency_weight") == want
    argv = job(w, "sat_check")["cmd"]
    argv[argv.index("--spec-json") + 1] = json.dumps(chk, sort_keys=True)
    assert "recency_weight" not in json.dumps(w["jobs"]).replace('\\"', '"')
    # 2. the release turn: its own, neither SAT's nor W169's (0)
    assert w.pop("store_release_offset_min") == SW.OFFSET_MIN == 40 and s.pop("store_release_offset_min") == S.OFFSET_MIN == 20
    assert SW.OFFSET_MIN not in (S.OFFSET_MIN, 0) and w["store_release_every_min"] == s["store_release_every_min"] == 60
    # 3. comment and metadata
    bw, bs = w.pop("_base2"), s.pop("_base2")
    w.pop("_comment"), s.pop("_comment")
    assert bw.pop("symbol_targets").pop("recency_weight") == want
    bs.pop("symbol_targets")
    assert bw.pop("arm") == "SATW" and bs.pop("arm") == "SAT" and bw == bs
    # 4. names
    assert json.loads(re.sub(r"SATW(?![A-Za-z])", "SAT", json.dumps(w))) == s


def test_the_check_runs_satws_own_spec():
    d = load(SW.OUT_FILE)
    chk, train = check_spec(d), copy.deepcopy(job(d, "pt_base2_SATW")["spec"])
    assert chk.pop("name") == "base2_SATW_check" and train.pop("name") == "base2_SATW"
    assert chk == train and chk["train"]["symbol_targets"]["recency_weight"] == {"nearest_right": SW.RECENCY_WEIGHT}
    assert job(d, "pt_base2_SATW")["after"] == ["pins", "sat_pin", "sat_check"]


def test_names_and_weight():
    d = load(SW.OUT_FILE)
    t = job(d, "pt_base2_SATW")
    assert t["store_branch"] == "ckpt-SATW" and t["model_name"] == "base2_SATW"
    assert t["spec"]["out_dir"] == "/content/chat/pt_base2_SATW"
    assert sorted(j["name"] for j in d["jobs"] if j["name"].startswith("rung_")) == \
           ["rung_SATW_s10000", "rung_SATW_s58500", "rung_SATW_s60000"]
    assert not re.search(r"SAT(?![A-Za-z])", json.dumps(d["jobs"]))
    assert 0 < SW.RECENCY_WEIGHT <= 1
    # The four SAT files and their pins are SAT's.
    assert d["store_inputs"] == load(S.OUT_FILE)["store_inputs"] and job(d, "sat_pin") == job(load(S.OUT_FILE), "sat_pin")
