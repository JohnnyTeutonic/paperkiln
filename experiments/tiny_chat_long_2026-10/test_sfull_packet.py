"""Prospective S-full recipe, exact-input and fail-closed completion regressions; no model/API calls."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools/colab_chat"))
import prepare_sfull as prep  # noqa: E402
import finalize_sfull as final  # noqa: E402
import sfull_verify_inputs as verify  # noqa: E402


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_l8_recipe_and_only_registered_mask_delta(seed):
    path, doc, source = prep.build(seed)
    original = json.loads(prep.frozen(source))
    jobs = {j["name"]: j for j in doc["jobs"]}
    assert len(jobs) == 6 and doc["max_parallel"] == 1
    pt = jobs[f"pt_c12_Sfull_r{seed}"]
    old_pt = next(j for j in original["jobs"] if j["name"].startswith("pt_"))
    expected = copy.deepcopy(old_pt["spec"])
    expected["arch"]["state_tokens"]["mask"] = "causal"
    assert pt["spec"] == expected
    assert pt["spec"]["arch"]["custom"]["layers"] == 8
    assert "sfull_inputs" in pt["after"]
    assert jobs["gate"]["cmd"][jobs["gate"]["cmd"].index("--jobs") + 1] == path
    assert jobs["data"] == next(j for j in original["jobs"] if j["name"] == "data")
    post = jobs[f"post_c12_Sfull_r{seed}"]
    assert post["cmd"][post["cmd"].index("--seed") + 1] == str(seed)
    assert post["cmd"][post["cmd"].index("--cell") + 1] == "Sfull"
    assert post["cmd"][post["cmd"].index("--spec") + 1].endswith("cell_Sfull.json")
    old_post = next(j for j in original["jobs"] if j["name"].startswith("post_"))
    assert post["cmd"] == [v.replace("Sgist", "Sfull") for v in old_post["cmd"]]
    assert "--sft-batching" in post["cmd"]


@pytest.mark.parametrize("seed", [True, False, 0, 4, 5, "1"])
def test_no_unregistered_initial_seed(seed):
    with pytest.raises(ValueError):
        prep.build(seed)


def test_trigger_is_exact_audited_five_seed_bytes():
    assert prep.trigger(prep.TRIGGER.read_bytes())["verdict"] == "SUPPORTED"
    with pytest.raises(ValueError):
        prep.trigger(prep.TRIGGER.read_bytes() + b" ")


@pytest.fixture
def data_case(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    (repo / "corpus/eval").mkdir(parents=True)
    vocab = data / "vocab.gguf"
    vocab.write_bytes(b"vocabulary")
    (data / "mix.ids").write_bytes(b"1234")
    (data / "human/source").mkdir(parents=True)
    (data / "human/source/dialogues.jsonl").write_bytes(b"raw human source")
    for name in ["carry5_v1.jsonl", "carry12_v1.jsonl"]:
        (repo / "corpus/eval" / name).write_bytes(name.encode())
    report = {"rows_kept": 5}
    dump(data / "human_decontam.json", report)
    ref = {
        "args": {"out": "/old", "vocab": str(vocab)},
        "vocab_sha256": verify.sha(vocab),
        "steps": {"decontam": report},
        "human": {
            "files": {
                "source/dialogues.jsonl": verify.sha(
                    data / "human/source/dialogues.jsonl"
                )
            },
            "carry5_v1_sha256": verify.sha(repo / "corpus/eval/carry5_v1.jsonl"),
            "carry12_v1_sha256": verify.sha(repo / "corpus/eval/carry12_v1.jsonl"),
        },
        "outputs": {"mix.ids": {"bytes": 4, "sha256": verify.sha(data / "mix.ids")}},
    }
    reference = tmp_path / "reference.json"
    dump(reference, ref)
    monkeypatch.setattr(verify, "REFERENCE_SHA", verify.sha(reference))
    actual = copy.deepcopy(ref)
    actual["args"]["out"] = "/new"
    dump(data / "data.ledger.json", actual)
    return data, reference, repo


def test_exact_data_and_operational_path_only(data_case):
    assert verify.verify(*data_case)["verified"] is True


@pytest.mark.parametrize(
    "target",
    [
        "output",
        "source",
        "decontamination",
        "evaluation",
        "vocab",
        "ledger",
        "reference",
    ],
)
def test_changed_scientific_input_refuses(data_case, target):
    data, reference, repo = data_case
    paths = {
        "output": data / "mix.ids",
        "source": data / "human/source/dialogues.jsonl",
        "decontamination": data / "human_decontam.json",
        "evaluation": repo / "corpus/eval/carry12_v1.jsonl",
        "vocab": data / "vocab.gguf",
        "reference": reference,
    }
    if target == "ledger":
        value = json.loads((data / "data.ledger.json").read_text())
        value["args"]["new_semantic_setting"] = True
        dump(data / "data.ledger.json", value)
    else:
        paths[target].write_bytes(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        verify.verify(data, reference, repo)


@pytest.fixture
def case(tmp_path, monkeypatch):
    art = tmp_path / "art"
    home = tmp_path / "home"
    states = tmp_path / "campaigns"
    out = tmp_path / "out"
    proof = tmp_path / "proof"
    proof.mkdir()
    marker = tmp_path / "source.py"
    marker.write_text("frozen")
    monkeypatch.setattr(final.common, "CPU_BUDGET", marker)
    monkeypatch.setattr(final, "sources", lambda: [marker])
    monkeypatch.setattr(final, "LAUNCH", proof / "launch.json")
    monkeypatch.setattr(final, "MANIFEST", proof / "manifest.json")
    monkeypatch.setattr(final, "REFERENCE", final.PROOF / "data_reference.json")
    pin = "a" * 40
    ref = final.common.load_json(final.REFERENCE)
    corpus = [
        json.loads(x)
        for x in (final.REPO / "corpus/eval/carry12_v1.jsonl").read_text().splitlines()
    ]
    row = {
        "mode": "post",
        "arch": {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240},
        "Y12": -0.1,
        "provenance": {"paperkiln_commit": pin},
        "carry12": {
            "context": 1024,
            "meta": {
                "items": 1452,
                "max_new": 48,
                "self_scope": "target",
                "near_self": False,
                "scorer": "mtstudio",
                "spot": 20,
                "sha256": "de24f2c726a1006e8e61542dd3b2dfbf8677b9e6fe689277cca035305bd32797",
            },
            "gen_engine": {"engine": "ember", "spot": {"pass": True}},
            "gate": {"replies": 17424, "argmax_identical": 17424},
            "pools": {
                "acc": {
                    "primary": {"self": {"mean": 0.4}, "twin": {"mean": 0.5}},
                    "context": {"twin": {"mean": 0.5}, "noctx": {"mean": 0.25}},
                }
            },
        },
    }
    hashes = {}
    for cell in final.CELLS:
        for seed in final.SEEDS:
            base = art / f"c12_{cell}_r{seed}"
            value = copy.deepcopy(row)
            value.update(cell=cell, seed=seed)
            for key, steps in [("pt_result", 10000), ("sft_result", 1500)]:
                value[key] = {
                    "T": 1024,
                    "batch": 16,
                    "accum": 1,
                    "seed": seed,
                    "final_step": steps,
                    "steps_requested": steps,
                    "early_stopped": False,
                }
            dump(base / f"post_c12_{cell}_r{seed}/summary.json", value)
            dump(base / "data/data.ledger.json", ref)
            dump(base / "gate/gate.json", {"ok": True, "problems": []})
            if cell != "Sfull":
                for suffix in [
                    f"post_c12_{cell}_r{seed}/summary.json",
                    "data/data.ledger.json",
                    "gate/gate.json",
                ]:
                    path = base / suffix
                    hashes[str(path.relative_to(art))] = final.common.sha(path)
                continue
            dump(base / "cuda_gate/cuda_gate.json", {"ok": True, "failed": []})
            dump(
                base / "sfull_inputs/verified_inputs.json",
                {
                    "verified": True,
                    "reference_sha256": prep.REFERENCE_SHA,
                    "outputs_sha256": {
                        k: v["sha256"] for k, v in ref["outputs"].items()
                    },
                    "no_data_mutation": True,
                    "vocab_sha256": ref["vocab_sha256"],
                    "raw_human_sources_verified": len(ref["human"]["files"]),
                    "evaluation_sources_verified": 2,
                },
            )
            for path in [
                base / f"pt_c12_Sfull_r{seed}/pt_c12_Sfull_r{seed}_spec.json",
                base / f"post_c12_Sfull_r{seed}/sft_spec.json",
            ]:
                dump(
                    path,
                    {
                        "arch": {
                            "state_tokens": {"k": 4, "mask": "causal"},
                            "custom": row["arch"],
                        }
                    },
                )
            target = base / f"post_c12_Sfull_r{seed}/carry12/m"
            target.mkdir(parents=True)
            (target / "items.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "id": r["id"],
                            "gate": [True] * r["target_turn"],
                            "server_errors": 0,
                        }
                    )
                    + "\n"
                    for r in corpus
                )
            )
            (target / "gen.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "id": r["id"],
                            "self_scope": "target",
                            "self": [{}] * r["target_turn"],
                        }
                    )
                    + "\n"
                    for r in corpus
                )
            )
            dump(
                states / f"pk_c12_sfull_r{seed}_runner/state.json",
                {
                    "campaign_home": True,
                    "jobs": {"post": {"home": True, "status": "FINISHED"}},
                    "runpod": {"pod": None, "volume": None},
                },
            )
    dump(final.MANIFEST, {"artifacts_sha256": {}, "baseline_artifact_sha256": hashes})
    dump(
        final.LAUNCH,
        {
            "parent": prep.PIN,
            "manifest_sha256": final.common.sha(final.MANIFEST),
            "commit": pin,
        },
    )
    f = final.Finalizer(art, home, out, campaign_state_root=states)

    def run(command, **kwargs):
        assert "--cores" in command and command[command.index("--cores") + 1] == "1"
        final.calculate(art, Path(command[-1]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(final.subprocess, "run", run)
    return f, art, marker, states


def test_complete_atomic_report_and_repeat_handoff(case):
    f, art, marker, states = case
    assert f.tick() == "COMPLETE"
    report = final.common.load_json(f.output / "reports/analysis_hs0.json")
    assert report["HS0_Sfull_minus_B"]["verdict"] == "EQUIVALENT"
    assert report["historical_HS1_nonidentical"] == 6
    assert report["interpretation_requires_parity_review"] is True
    assert report["no_automatic_followup_launch"] is True
    assert f.tick() == "COMPLETE"


def test_missing_input_waits_without_analysis(case):
    f, art, marker, states = case
    (art / "c12_Sfull_r3/post_c12_Sfull_r3/summary.json").unlink()
    assert f.tick() == "WAITING"
    assert not (f.output / "reports").exists()


def test_nonhome_waits_then_accepts_final_state(case):
    f, art, marker, states = case
    path = states / "pk_c12_sfull_r3_runner/state.json"
    value = final.common.load_json(path)
    value["campaign_home"] = False
    dump(path, value)
    assert f.tick() == "WAITING"
    value["campaign_home"] = True
    dump(path, value)
    assert f.tick() == "COMPLETE"


@pytest.mark.parametrize(
    "problem",
    [
        "source",
        "baseline",
        "pin",
        "shape",
        "mask",
        "data",
        "strict_summary",
        "strict_raw",
        "duplicate",
        "extra_seed",
        "cuda",
    ],
)
def test_mismatch_halts_and_preserves_flag(case, problem):
    f, art, marker, states = case
    f.state()
    base = art / "c12_Sfull_r1"
    path = base / "post_c12_Sfull_r1/summary.json"
    value = final.common.load_json(path)
    if problem == "source":
        marker.write_text("changed")
    elif problem == "baseline":
        (art / "c12_B_r1/post_c12_B_r1/summary.json").write_text("{}")
    elif problem in ["pin", "shape", "strict_summary"]:
        if problem == "pin":
            value["provenance"]["paperkiln_commit"] = "wrong"
        elif problem == "shape":
            value["arch"]["layers"] = 16
        else:
            value["carry12"]["gate"]["argmax_identical"] -= 1
        dump(path, value)
    elif problem == "mask":
        dump(
            base / "post_c12_Sfull_r1/sft_spec.json",
            {"arch": {"state_tokens": {"k": 4, "mask": "turn_state"}}},
        )
    elif problem == "data":
        dump(base / "data/data.ledger.json", {"changed": True})
    elif problem == "strict_raw":
        raw = base / "post_c12_Sfull_r1/carry12/m/items.jsonl"
        rows = [json.loads(x) for x in raw.read_text().splitlines()]
        rows[0]["gate"][0] = False
        raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif problem == "duplicate":
        dump(art / "duplicate/post_c12_Sfull_r1/summary.json", value)
    elif problem == "extra_seed":
        value["seed"] = 4
        dump(art / "extra/post_c12_Sfull_r4/summary.json", value)
    else:
        dump(base / "cuda_gate/cuda_gate.json", {"ok": False, "failed": ["x"]})
    assert f.tick() == "HALTED"
    assert (f.home / "HALTED").exists()
    assert f.tick() == "HALTED"


def test_partial_analysis_output_refuses(tmp_path):
    dump(
        tmp_path / final.REPORT,
        {"scope": final.SCOPE, "partial": True, "seeds": [1, 2]},
    )
    with pytest.raises(ValueError):
        final.Finalizer.verify_outputs(tmp_path)


def test_prospective_source_freeze_required_bytes(case, tmp_path):
    f, art, marker, states = case
    freeze = tmp_path / "prospective.json"
    dump(freeze, {"source_hashes": {str(marker.resolve()): final.common.sha(marker)}})
    f.source_freeze = freeze
    assert f.tick() == "COMPLETE"
    value = final.common.load_json(freeze)
    value["source_hashes"][str(marker.resolve())] = "f" * 64
    dump(freeze, value)
    with pytest.raises(ValueError, match="prospective finalizer source freeze"):
        f.tick()


def test_same_total_history_flags_wrong_item_allocation_refused(case):
    f, art, marker, states = case
    path = art / "c12_Sfull_r1/post_c12_Sfull_r1/carry12/m/items.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["gate"].pop()
    rows[1]["gate"].append(True)
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    assert f.tick() == "HALTED"
