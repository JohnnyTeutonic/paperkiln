"""E9's launch path (e9_launch.py), pins (e9_pins.py), seed sets (e9_seed.py) and corpus-build jobs files.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CC = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(CC))
import e9_launch as EL   # noqa: E402
import e9_pins as EP     # noqa: E402
import make_e9 as E9     # noqa: E402

LADDER_COMMIT = "672501bb"
BASELINES = {"N1B": "valid", "S1B": "valid", "who": "lead", "when": "2026-12-01"}


def git_show(rel):
    return subprocess.run(["git", "show", f"{LADDER_COMMIT}:{rel}"], cwd=ROOT, check=True, capture_output=True).stdout


def test_ladder_and_base2_files_used_by_the_ladder_are_unchanged():
    for rel in ("tools/colab_chat/ladder_launch.py", "tools/colab_chat/base2_pins.py", "tools/colab_chat/base2_pins.json",
                "tools/colab_chat/make_ladder.py", "tools/colab_chat/make_base2.py", "tools/colab_chat/base2_seed.py",
                "tools/colab_chat/base2_data.sh", "tools/colab_chat/runner_multi.py"):
        assert (ROOT / rel).read_bytes() == git_show(rel), rel


def test_arms_order_jobs_and_hours():
    names = [EL.ARMS[n]["arm"] for n in EL.ARMS]
    assert set(names) == {"D2", "N1Bs50", "D3", "S1Bs50", "D4", "D2s50", "D3s50", "D4s50", "D4ops"}
    assert EL.ARMS["pk-e9-N1Bs50"]["before"] == "pk-e9-D2" and EL.ARMS["pk-e9-S1Bs50"]["before"] == "pk-e9-D3"
    for r in EL.REPLICATES:
        assert EL.ARMS[f"pk-e9-{r}"]["before"] == "pk-e9-D4"
    for n, a in EL.ARMS.items():
        assert (ROOT / a["jobs"]).exists(), a["jobs"]
        assert a["rung"] == "1B"
    EL.install()
    argv = EL.runner_argv("pk-e9-D3", "a" * 40, "2026-12-05T00:00:00Z")
    assert "tools/colab_chat/e9_D3.json" in argv and "--max-hours" in argv and argv[argv.index("--max-hours") + 1] == "26"
    assert argv[argv.index("--deadline-utc") + 1] == "2026-12-05T00:00:00Z"
    assert "tools/colab_chat/ladder_N1Bs50.json" in EL.runner_argv("pk-e9-N1Bs50", "a" * 40, "2026-12-05T00:00:00Z")


def _arms_in(tmp):
    arms = {}
    for n, a in EL.ARMS.items():
        d = tmp / n
        d.mkdir()
        arms[n] = dict(a, state_dir=str(d))
    return arms


def test_the_ledger_fits_six_arms_and_refuses_a_seventh(tmp_path, monkeypatch):
    arms = _arms_in(tmp_path)
    monkeypatch.setattr(EL, "ladder_ended", lambda cmdlines=None, arms=None: None)
    monkeypatch.setattr(EL, "_ORIG_DECIDE", lambda *a, **k: ("wait", "delegated"))
    order = ["pk-e9-D2", "pk-e9-D3", "pk-e9-D4", "pk-e9-N1Bs50", "pk-e9-S1Bs50", "pk-e9-D3s50"]
    (tmp_path / "launch").mkdir()
    (tmp_path / "launch" / "BASELINES.json").write_text(json.dumps(BASELINES))
    (tmp_path / "launch" / "REPLICATE.json").write_text(json.dumps({"arm": "D3s50", "who": "lead"}))
    for n in order:
        assert EL.decide_arm(n, "a" * 40, 0, None, None, None, launch_dir=str(tmp_path / "launch"), arms=arms,
                             cmdlines=[]) == ("wait", "delegated"), n
        (Path(arms[n]["state_dir"]) / EL.OWN_COPY).write_text("{}")          # it started
    assert abs(EL.committed_units(arms) - (EL.BUILD_RESERVE_UNITS + 6 * EL.ARM_WORST_UNITS)) < 1e-6
    assert EL.committed_units(arms) <= EL.CAP_UNITS
    home = json.dumps({"campaign_home": True, "store_release_final_done": True})
    for n in order[:3] + ["pk-e9-D3s50"]:                 # N1Bs50 and S1Bs50 still out: D4ops waits, across slots
        (Path(arms[n]["state_dir"]) / "state.json").write_text(home)
    assert EL.decide_arm("pk-e9-D4ops", "a" * 40, 0, None, None, None, launch_dir=str(tmp_path / "launch"),
                         arms=arms, cmdlines=[])[0] == "wait"
    for n in ("pk-e9-N1Bs50", "pk-e9-S1Bs50"):
        (Path(arms[n]["state_dir"]) / "state.json").write_text(home)
    kind, what = EL.decide_arm("pk-e9-D4ops", "a" * 40, 0, None, None, None, launch_dir=str(tmp_path / "launch"),
                               arms=arms, cmdlines=[])
    assert kind == "halt" and what[0] == "E9_CAP"


def test_the_replicate_and_d4ops_wait_for_the_leads_choice(tmp_path, monkeypatch):
    arms = _arms_in(tmp_path)
    monkeypatch.setattr(EL, "ladder_ended", lambda cmdlines=None, arms=None: None)
    monkeypatch.setattr(EL, "_ORIG_DECIDE", lambda *a, **k: ("wait", "delegated"))
    ld = tmp_path / "launch"
    ld.mkdir()
    assert EL.decide_arm("pk-e9-D2", "a" * 40, 0, None, None, None, launch_dir=str(ld), arms=arms,
                         cmdlines=[])[0] == "wait"          # no BASELINES.json: nothing of E9 starts
    (ld / "BASELINES.json").write_text(json.dumps(BASELINES))
    k = lambda n: EL.decide_arm(n, "a" * 40, 0, None, None, None, launch_dir=str(ld), arms=arms, cmdlines=[])  # noqa
    assert k("pk-e9-D3s50")[0] == "wait" and k("pk-e9-D4ops")[0] == "wait"
    (ld / "REPLICATE.json").write_text(json.dumps({"arm": "D3s50", "who": "lead"}))
    assert k("pk-e9-D2s50")[0] == "halt" and k("pk-e9-D3s50") == ("wait", "delegated")
    assert k("pk-e9-D4ops")[0] == "wait"                     # the replicate is not home yet


def test_the_replicate_choice_freezes_once_one_has_started(tmp_path):
    arms = _arms_in(tmp_path)
    ld = tmp_path / "launch"
    ld.mkdir()
    (Path(arms["pk-e9-D2s50"]["state_dir"]) / EL.OWN_COPY).write_text("{}")
    (ld / "REPLICATE.json").write_text(json.dumps({"arm": "D3s50", "who": "lead"}))
    assert EL.replicate_choice(str(ld), arms) == "D2s50"


def test_the_balance_check_counts_the_plan_not_every_entry(tmp_path):
    arms = _arms_in(tmp_path)
    need = EL.units_needed(None, arms, 0.0, None)
    assert abs(need - (6 * 26 * 1.94 + EL.LL.UNITS_MARGIN)) < 1e-6, need     # six planned arms, nothing started: 322.6
    assert need <= EL.CAP_UNITS


def test_nothing_starts_before_the_ladder_has_ended(tmp_path):
    ladder = {}
    for n in ("pk-tl-N1B", "pk-tl-S1B"):
        d = tmp_path / n
        d.mkdir()
        ladder[n] = {"state_dir": str(d)}
    (tmp_path / "pk-tl-N1B" / "state.json").write_text(json.dumps({"launches": 1}))       # launched, not stopped
    assert EL.ladder_ended(cmdlines=[], arms=ladder) is not None
    (tmp_path / "pk-tl-N1B" / "state.json").write_text(json.dumps({"launches": 1, "campaign_home": True,
                                                                   "store_release_final_done": True}))
    assert EL.ladder_ended(cmdlines=[], arms=ladder) is None
    assert EL.ladder_ended(cmdlines=["python3 runner_multi.py --name pk-tl-S1B"], arms=ladder) is not None


def test_e9_pins_refuse_until_adopted_and_check_like_base2(tmp_path):
    d, up = tmp_path / "data", tmp_path / "upload"
    d.mkdir()
    up.mkdir()
    for f in ("D2.main.ids", "D2.tail.ids"):
        (d / f).write_bytes(f.encode())
    base = json.load(open(CC / "base2_pins.json"))
    for name in base["panels"]:
        (d / name).write_bytes(name.encode())
        base["panels"][name] = EP.P.sha256(str(d / name))
    for name in base["vocab"]:
        (up / name).write_bytes(b"v")
        base["vocab"][name] = EP.P.sha256(str(up / name))
    bp = tmp_path / "base2_pins.json"
    bp.write_text(json.dumps(base))
    ep = tmp_path / "e9_pins.json"
    ep.write_text((CC / "e9_pins.json").read_text())
    with pytest.raises(SystemExit):
        EP.main([str(d), "arm-D2", "--upload", str(up), "--pins", str(bp), "--e9-pins", str(ep)])
    led = tmp_path / "e9_D2_stage3.json"
    led.write_text(json.dumps({"built": {"files": {f: {"sha256": EP.P.sha256(str(d / f))} for f in ("D2.main.ids", "D2.tail.ids")}}}))
    assert EP.main(["adopt-ledger", str(led), "--e9-pins", str(ep)]) == 0
    assert EP.main([str(d), "arm-D2", "--upload", str(up), "--pins", str(bp), "--e9-pins", str(ep)]) == 0
    (d / "D2.tail.ids").write_bytes(b"other")
    with pytest.raises(SystemExit):
        EP.main([str(d), "arm-D2", "--upload", str(up), "--pins", str(bp), "--e9-pins", str(ep)])
    assert "arm-D2" not in EP.P.MODES                         # base 2's modes untouched after the call


def test_corpus_build_jobs_pull_e9_beside_the_engine():
    for base, level in E9.LEVEL.items():
        j = E9.data_jobs(base)
        to = {s["prefix"]: s["to"] for s in j["store_inputs"]}
        assert to["seed/scenario_engine"] == "chat/upload/scenario_engine"
        assert to["seed/e9"] == "chat/upload/e9/diversity" and to["seed/e9_receipts"] == "chat/upload/e9/diversity/results"
        cmd = j["jobs"][0]["cmd"]
        assert cmd[:3] == ["env", f"E9_ARM={base}", f"E9_LEVEL={level}"] and "E9_REHEARSAL=1" not in cmd
        assert "tools/colab_chat/e9_data.sh" in cmd and cmd[-1] == "stage3"
