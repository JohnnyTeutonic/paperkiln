"""E9's launch path (e9_launch.py), pins (e9_pins.py) and corpus-build jobs files.

Claude Opus 5.5 (a fork of the coordinating session), 18 Nov 2026.
"""
import json
import os
import subprocess
import sys
import time
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
HOME = json.dumps({"campaign_home": True, "store_release_final_done": True})


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
    for a in list(EL.ARMS.values()) + list(EL.BUILDS.values()):
        assert (ROOT / a["jobs"]).exists(), a["jobs"]
    assert not {a["state_dir"] for a in EL.ARMS.values()} & {b["state_dir"] for b in EL.BUILDS.values()}
    EL.install()
    argv = EL.runner_argv("pk-e9-D3", "a" * 40, "2026-12-05T00:00:00Z")
    assert "tools/colab_chat/e9_D3.json" in argv and argv[argv.index("--max-hours") + 1] == "26"
    assert argv[argv.index("--deadline-utc") + 1] == "2026-12-05T00:00:00Z"
    assert "tools/colab_chat/ladder_N1Bs50.json" in EL.runner_argv("pk-e9-N1Bs50", "a" * 40, "2026-12-05T00:00:00Z")
    assert EL.GUARD_STOP_AT == 283.4


def _arms_in(tmp):
    arms = {}
    for n, a in EL.ARMS.items():
        d = tmp / n
        d.mkdir()
        arms[n] = dict(a, state_dir=str(d))
    return arms


def _builds_in(tmp):
    out = {}
    for n, b in EL.BUILDS.items():
        d = tmp / n
        d.mkdir()
        out[n] = dict(b, state_dir=str(d))
    return out


def test_the_ledger_fits_six_arms_and_refuses_a_seventh(tmp_path, monkeypatch):
    arms, builds = _arms_in(tmp_path), _builds_in(tmp_path)
    monkeypatch.setattr(EL, "ladder_ended", lambda cmdlines=None, arms=None, launch_dir=None, now=None: None)
    monkeypatch.setattr(EL, "_ORIG_DECIDE", lambda *a, **k: ("wait", "delegated"))
    ld = tmp_path / "launch"
    ld.mkdir()
    (ld / "BASELINES.json").write_text(json.dumps(BASELINES))
    (ld / "REPLICATE.json").write_text(json.dumps({"arm": "D3s50", "who": "lead"}))
    order = ["pk-e9-D2", "pk-e9-D3", "pk-e9-D4", "pk-e9-N1Bs50", "pk-e9-S1Bs50", "pk-e9-D3s50"]
    k = lambda n: EL.decide_arm(n, "a" * 40, 0, None, None, None, launch_dir=str(ld), arms=arms, cmdlines=[], builds=builds)  # noqa
    for n in order:
        assert k(n) == ("wait", "delegated"), n
        (Path(arms[n]["state_dir"]) / EL.OWN_COPY).write_text("{}")
    assert abs(EL.committed_units(arms, builds) - (3 * EL.BUILD_UNITS + 6 * EL.ARM_WORST_UNITS)) < 1e-6
    for n in order[:3] + ["pk-e9-D3s50"]:
        (Path(arms[n]["state_dir"]) / "state.json").write_text(HOME)
    assert k("pk-e9-D4ops")[0] == "wait"                    # N1Bs50 and S1Bs50 still out: D4ops waits, across slots
    for n in ("pk-e9-N1Bs50", "pk-e9-S1Bs50"):
        (Path(arms[n]["state_dir"]) / "state.json").write_text(HOME)
    kind, what = k("pk-e9-D4ops")
    assert kind == "halt" and what[0] == "E9_CAP"
    # a fifth build start (D4ops's) counted too: it would also have to fit
    (Path(builds["pk-e9-data-D4ops"]["state_dir"]) / EL.BUILD_COPY).write_text("{}")
    assert abs(EL.committed_units(arms, builds) - (4 * EL.BUILD_UNITS + 6 * EL.ARM_WORST_UNITS)) < 1e-6


def test_the_replicate_and_d4ops_wait_for_the_leads_choice(tmp_path, monkeypatch):
    arms, builds = _arms_in(tmp_path), _builds_in(tmp_path)
    monkeypatch.setattr(EL, "ladder_ended", lambda cmdlines=None, arms=None, launch_dir=None, now=None: None)
    monkeypatch.setattr(EL, "_ORIG_DECIDE", lambda *a, **k: ("wait", "delegated"))
    ld = tmp_path / "launch"
    ld.mkdir()
    k = lambda n: EL.decide_arm(n, "a" * 40, 0, None, None, None, launch_dir=str(ld), arms=arms, cmdlines=[], builds=builds)  # noqa
    assert k("pk-e9-D2")[0] == "wait"                       # no BASELINES.json: nothing of E9 starts
    (ld / "BASELINES.json").write_text(json.dumps(BASELINES))
    assert k("pk-e9-D3s50")[0] == "wait" and k("pk-e9-D4ops")[0] == "wait"
    (ld / "REPLICATE.json").write_text(json.dumps({"arm": "D3s50", "who": "lead"}))
    assert k("pk-e9-D2s50")[0] == "halt" and k("pk-e9-D3s50") == ("wait", "delegated")


def test_the_replicate_choice_freezes_once_one_has_started(tmp_path):
    arms = _arms_in(tmp_path)
    ld = tmp_path / "launch"
    ld.mkdir()
    (Path(arms["pk-e9-D2s50"]["state_dir"]) / EL.OWN_COPY).write_text("{}")
    (ld / "REPLICATE.json").write_text(json.dumps({"arm": "D3s50", "who": "lead"}))
    assert EL.replicate_choice(str(ld), arms) == "D2s50"


def test_the_balance_check_counts_the_plan_not_every_entry(tmp_path):
    need = EL.units_needed(None, _arms_in(tmp_path), 0.0, None)
    assert abs(need - (6 * 26 * 1.94 + EL.LL.UNITS_MARGIN)) < 1e-6, need


def _ladder(tmp, deadline_in_s=3600):
    ladder, ld = {}, tmp / "ladder_launch"
    ld.mkdir()
    (ld / "launch.json").write_text(json.dumps({"deadline_utc": EL.LL.iso(time.time() + deadline_in_s)}))
    for n in ("pk-tl-N1B", "pk-tl-S1B"):
        d = tmp / n
        d.mkdir()
        ladder[n] = {"state_dir": str(d)}
    return ladder, str(ld)


def test_the_ladder_has_ended_only_when_admitted_stopped_and_not_running(tmp_path):
    ladder, ld = _ladder(tmp_path)
    assert EL.ladder_ended(cmdlines=[], arms=ladder, launch_dir=str(tmp_path / "nowhere")) is not None   # never admitted
    (Path(ladder["pk-tl-N1B"]["state_dir"]) / "state.json").write_text(HOME)
    # S1B never launched, holds no HALTED, and the ladder's week is not over: it could still start
    assert "could still start" in EL.ladder_ended(cmdlines=[], arms=ladder, launch_dir=ld)
    (Path(ladder["pk-tl-S1B"]["state_dir"]) / "HALTED").write_text("{}")
    assert EL.ladder_ended(cmdlines=[], arms=ladder, launch_dir=ld) is None
    for line in (["python3", "runner_multi.py", "--name", "pk-tl-S1B"], ["python3", "x.py", "--name=pk-tl-S1B"],
                 "python3 runner_multi.py --name pk-tl-S1B"):
        assert EL.ladder_ended(cmdlines=[line], arms=ladder, launch_dir=ld) is not None, line


def test_a_never_launched_ladder_arm_counts_as_ended_once_the_week_is_over(tmp_path):
    ladder, ld = _ladder(tmp_path, deadline_in_s=-60)
    (Path(ladder["pk-tl-N1B"]["state_dir"]) / "state.json").write_text(HOME)
    assert EL.ladder_ended(cmdlines=[], arms=ladder, launch_dir=ld) is None


def test_a_build_waits_writes_its_deadline_once_and_is_halted_past_it(tmp_path, monkeypatch):
    arms, builds = _arms_in(tmp_path), _builds_in(tmp_path)
    ended = {"why": "not yet"}
    monkeypatch.setattr(EL, "ladder_ended", lambda cmdlines=None, arms=None, launch_dir=None, now=None: ended["why"])
    off = {"why": "seed 2 live"}
    kb = lambda n, now: EL.decide_build(n, "a" * 40, now, arms=arms, builds=builds, cmdlines=[],  # noqa
                                         seed2=lambda now=None: off["why"], launch_dir=str(tmp_path))
    t0 = 1_800_000_000.0
    assert kb("pk-e9-data-D2", t0)[0] == "wait"                         # the ladder has not ended
    ended["why"] = None
    assert kb("pk-e9-data-D2", t0)[0] == "wait"                         # seed 2 is not off
    off["why"] = None
    kind, argv = kb("pk-e9-data-D2", t0)
    assert kind == "exec" and argv[argv.index("--max-hours") + 1] == "6"
    dl = argv[argv.index("--deadline-utc") + 1]
    assert dl == EL.LL.iso(t0 + 6 * 3600) and "--store-campaign" not in argv
    assert "tools/colab_chat/e9_data_D2.json" in argv and builds["pk-e9-data-D2"]["state_dir"] in argv
    kind, argv2 = kb("pk-e9-data-D2", t0 + 3 * 3600)                    # a restart keeps the first deadline
    assert argv2[argv2.index("--deadline-utc") + 1] == dl
    assert kb("pk-e9-data-D2", t0 + 6 * 3600 + 1)[0] == "halt"         # past it: halted for a person
    assert abs(EL.committed_units(arms, builds) - 3 * EL.BUILD_UNITS) < 1e-6   # started D2 is one of the reserved three


def test_d4ops_build_waits_until_a_planned_arm_is_known_not_to_start(tmp_path, monkeypatch):
    arms, builds = _arms_in(tmp_path), _builds_in(tmp_path)
    monkeypatch.setattr(EL, "ladder_ended", lambda cmdlines=None, arms=None, launch_dir=None, now=None: None)
    kb = lambda: EL.decide_build("pk-e9-data-D4ops", "a" * 40, 1_800_000_000.0, arms=arms, builds=builds,  # noqa
                                 cmdlines=[], seed2=lambda now=None: None, launch_dir=str(tmp_path))
    assert kb()[0] == "wait"
    (Path(arms["pk-e9-S1Bs50"]["state_dir"]) / "HALTED").write_text("{}")      # a planned arm that will not start
    assert kb()[0] == "exec"


def test_decide_arm_under_install_reaches_the_ladders_gate_with_e9s_folders(tmp_path, monkeypatch):
    """Integration: install(), real temporary folders, the ladder's own decide_arm below E9's conditions."""
    ladder, ld = _ladder(tmp_path)
    for x in ladder.values():
        (Path(x["state_dir"]) / "state.json").write_text(HOME)
    monkeypatch.setattr(EL, "LADDER_ARMS", ladder)
    monkeypatch.setattr(EL, "LADDER_LAUNCH_DIR", ld)
    arms, builds = _arms_in(tmp_path), _builds_in(tmp_path)
    e9 = tmp_path / "e9_launch"
    e9.mkdir()
    (e9 / "BASELINES.json").write_text(json.dumps(BASELINES))
    EL.install()
    kind, what = EL.LL.decide_arm("pk-e9-D2", "a" * 40, time.time(), None, lambda: [], lambda: 1000.0,
                                  launch_dir=str(e9), arms=arms, cmdlines=[], builds=builds)
    # E9's conditions hold; the ladder's gate decides, reading seed 2 from the LADDER's folder and GO from E9's
    assert kind in ("refuse", "wait"), (kind, what)
    assert "GO" in str(what) or "seed 2" in str(what), what
    assert not (e9 / "launch.json").exists()                            # nothing admitted without a GO record


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
    led.write_text(json.dumps({"published_D2": {"files": {f: {"sha256": EP.P.sha256(str(d / f))}
                                                          for f in ("D2.main.ids", "D2.tail.ids")}}}))
    assert EP.main(["adopt-ledger", str(led), "--e9-pins", str(ep)]) == 0
    assert EP.main([str(d), "arm-D2", "--upload", str(up), "--pins", str(bp), "--e9-pins", str(ep)]) == 0
    (d / "D2.tail.ids").write_bytes(b"other")
    with pytest.raises(SystemExit):
        EP.main([str(d), "arm-D2", "--upload", str(up), "--pins", str(bp), "--e9-pins", str(ep)])
    assert "arm-D2" not in EP.P.MODES


def test_corpus_build_jobs_pull_e9_beside_the_engine():
    for base, level in E9.LEVEL.items():
        j = E9.data_jobs(base)
        to = {s["prefix"]: s["to"] for s in j["store_inputs"]}
        assert to["seed/scenario_engine"] == "chat/upload/scenario_engine"
        assert to["seed/e9"] == "chat/upload/e9/diversity" and to["seed/e9_receipts"] == "chat/upload/e9/diversity/results"
        cmd = j["jobs"][0]["cmd"]
        assert cmd[:3] == ["env", f"E9_ARM={base}", f"E9_LEVEL={level}"] and "E9_REHEARSAL=1" not in " ".join(cmd)
        assert "tools/colab_chat/e9_data.sh" in cmd and cmd[-1] == "stage3"
