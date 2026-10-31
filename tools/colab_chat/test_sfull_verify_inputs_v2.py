"""Portable V2 report-shape correction and unchanged full-input guard checks."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("v2", HERE / "sfull_verify_inputs_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def put(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = HERE.parents[1]
    spec = importlib.util.spec_from_file_location(
        "producer", root / "corpus/decontam.py"
    )
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    data = tmp_path / "data"
    data.mkdir()
    inp = data / "human_all.jsonl"
    inp.write_text(
        "".join(
            json.dumps({"messages": [{"role": "user", "content": x}]}) + "\n"
            for x in ["drop", "keep"]
        )
    )

    class Patterns:
        def check(self, users, text):
            if text == "drop":
                return [{"rule": "prompt", "eval": "fixture:1", "match": "drop"}], []
            return [], [{"rule": "answer", "eval": "fixture:2", "match": "keep"}]

    summary = dc.run(
        str(inp), str(data / "clean"), str(data / "human_decontam.json"), dc=Patterns()
    )
    human = tmp_path / "raw"
    files = {}
    for i in range(13):
        p = human / f"source{i}/dialogues.jsonl"
        p.parent.mkdir(parents=True)
        p.write_bytes(str(i).encode())
        files[str(p.relative_to(human))] = v2.sha(p)
    vocab = tmp_path / "vocab"
    vocab.write_bytes(b"vocab")
    repo = tmp_path / "repo"
    ev = {}
    for n in ("carry5_v1", "carry12_v1"):
        p = repo / "corpus/eval" / f"{n}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(n.encode())
        ev[n + "_sha256"] = v2.sha(p)
    outputs = {}
    for i in range(5):
        p = data / f"product{i}"
        p.write_bytes(b"item" + str(i).encode())
        outputs[p.name] = {"bytes": p.stat().st_size, "sha256": v2.sha(p), "tokens": i}
    ledger = {
        "args": {"vocab": "/original/cloud/vocab", "out": "/original/cloud/data"},
        "steps": {"decontam": summary},
        "human": {"files": files, **ev},
        "vocab_sha256": v2.sha(vocab),
        "outputs": outputs,
    }
    ref = tmp_path / "reference.json"
    put(ref, ledger)
    put(data / "data.ledger.json", ledger)
    monkeypatch.setattr(v2, "REFERENCE_SHA", v2.sha(ref))
    return {
        "data": data,
        "ref": ref,
        "repo": repo,
        "vocab": vocab,
        "human": human,
        "ledger": ledger,
    }


def verify(f):
    return v2.verify(
        f["data"], f["ref"], f["repo"], vocab_path=f["vocab"], human_dir=f["human"]
    )


def test_real_producer_v2_passes_while_original_v1_guard_rejects(fixture):
    f = fixture
    report = json.loads((f["data"] / "human_decontam.json").read_text())
    summary = f["ledger"]["steps"]["decontam"]
    assert report["summary"] == summary
    # Exact V1 final comparison, after all prior checks: detailed != flat summary.
    with pytest.raises(ValueError, match="raw decontamination report differs"):
        if v2.normal(report) != v2.normal(summary):
            raise ValueError("raw decontamination report differs")
    result = verify(f)
    assert result["verified"] is True and result["validation_version"] == 2
    assert (
        result["raw_human_sources_verified"] == 13
        and result["evaluation_sources_verified"] == 2
    )
    assert result["path_only_overrides"]["human_dir"] == str(f["human"])
    assert result["raw_detailed_historical_byte_identity_claimed"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "ledger",
        "product",
        "vocab",
        "raw",
        "eval",
        "reference",
        "summary_missing",
        "summary_list",
        "summary_count",
        "summary_rules",
        "detail_missing",
        "detail_extra",
        "dropped_type",
        "flagged_type",
        "eval_hits_type",
        "eval_hits_bool",
    ],
)
def test_original_and_report_shape_guards_refuse(fixture, kind):
    f = fixture
    data = f["data"]
    reportpath = data / "human_decontam.json"
    if kind == "ledger":
        x = json.loads((data / "data.ledger.json").read_text())
        x["unexpected"] = True
        put(data / "data.ledger.json", x)
    elif kind in ["product", "vocab", "raw", "eval", "reference"]:
        path = {
            "product": data / "product0",
            "vocab": f["vocab"],
            "raw": f["human"] / "source0/dialogues.jsonl",
            "eval": f["repo"] / "corpus/eval/carry5_v1.jsonl",
            "reference": f["ref"],
        }[kind]
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        x = json.loads(reportpath.read_text())
        if kind == "summary_missing":
            del x["summary"]
        if kind == "summary_list":
            x["summary"] = []
        if kind == "summary_count":
            x["summary"]["rows_in"] += 1
        if kind == "summary_rules":
            x["summary"]["rules"]["ngram"] += 1
        if kind == "detail_missing":
            del x["dropped"]
        if kind == "detail_extra":
            x["extra"] = True
        if kind == "dropped_type":
            x["dropped"] = {}
        if kind == "flagged_type":
            x["flagged"] = [1]
        if kind == "eval_hits_type":
            x["eval_hits"] = []
        if kind == "eval_hits_bool":
            x["eval_hits"] = {"fixture:1": True}
        put(reportpath, x)
    with pytest.raises((ValueError, OSError)):
        verify(f)


def relay_fixture(f, monkeypatch, tmp_path):
    result = verify(f)
    ref = f["ledger"]
    monkeypatch.setattr(v2, "VM_RAW_REPORT_SHA", result["detailed_decontam_sha256"])
    witness = {
        "kind": "SUPPLEMENTAL_READ_ONLY_SUMMARY_SHAPE_DIAGNOSTIC_V2",
        "source_commit": v2.FROZEN_REMOTE,
        "original_verifier_sha256": v2.V1_SOURCE_SHA,
        "diagnostic_source_sha256": v2.VM_PATCHED_SOURCE_SHA,
        "only_source_change": {"before": v2.BEFORE, "after": v2.AFTER},
        "original_gate_pass": False,
        "files_or_job_states_modified": False,
        "raw_detailed_historical_byte_identity_claimed": False,
        "raw_report_keys": sorted(["summary", "dropped", "flagged", "eval_hits"]),
        "raw_decontam_sha256": result["detailed_decontam_sha256"],
        "summary": ref["steps"]["decontam"],
        "corrected_contract_result": {
            k: result[k]
            for k in [
                "verified",
                "reference_sha256",
                "ledger_sha256",
                "outputs_sha256",
                "vocab_sha256",
                "raw_human_sources_verified",
                "evaluation_sources_verified",
                "no_data_mutation",
            ]
        },
        "five_products": {
            n: {k: v[k] for k in ["bytes", "sha256"]} for n, v in ref["outputs"].items()
        },
        "raw_human_witness": {
            n: {"expected_sha256": h, "actual_sha256": h}
            for n, h in ref["human"]["files"].items()
        },
        "detailed_lengths": {"dropped": 1, "flagged": 1, "eval_hits": 1},
    }
    wp = tmp_path / "witness.json"
    put(wp, witness)
    monkeypatch.setattr(v2, "VM_WITNESS_SHA", v2.sha(wp))
    source = tmp_path / "diagnostic.py.txt"
    source.write_text("captured diagnostic source")
    monkeypatch.setattr(v2, "VM_SCRIPT_SHA", v2.sha(source))
    protocol = tmp_path / "protocol.json"
    put(
        protocol,
        {
            "remote_commit": v2.FROZEN_REMOTE,
            "controller_root": str(tmp_path),
            "source_sha256": {source.name: v2.sha(source)},
        },
    )
    monkeypatch.setattr(v2, "PROTOCOL_SHA", v2.sha(protocol))
    state = tmp_path / "state.json"
    put(
        state,
        {
            "jobs": {
                "data": {"status": "FINISHED", "home": True},
                "sfull_inputs": {"status": "FAILED"},
            }
        },
    )
    return {
        "witness_path": wp,
        "diagnostic_source_path": source,
        "protocol_manifest_path": protocol,
        "v1_state_path": state,
        "repo": f["repo"],
        "vocab_path": f["vocab"],
        "human_dir": f["human"],
    }


def test_relay_pass_is_explicit_supplement_not_v1_completion(
    fixture, monkeypatch, tmp_path
):
    args = relay_fixture(fixture, monkeypatch, tmp_path)
    before = args["v1_state_path"].read_bytes()
    r = v2.verify_relay(fixture["data"], fixture["ref"], **args)
    assert r["status"] == "SUPPLEMENTAL_V2_INPUTS_VERIFIED_V1_GATE_FAILED"
    assert r["v1_gate_pass"] is False and r["scientific_outcome"] is False
    assert args["v1_state_path"].read_bytes() == before


@pytest.mark.parametrize(
    "kind",
    [
        "witness_hash",
        "script_hash",
        "protocol_hash",
        "witness_raw",
        "witness_eval",
        "witness_source",
        "v1_pass",
        "raw_report",
        "inventory",
    ],
)
def test_relay_witness_and_failure_preservation_guards(
    fixture, monkeypatch, tmp_path, kind
):
    args = relay_fixture(fixture, monkeypatch, tmp_path)
    if kind in ["witness_hash", "script_hash", "protocol_hash", "inventory"]:
        name = {
            "witness_hash": "witness_path",
            "script_hash": "diagnostic_source_path",
            "protocol_hash": "protocol_manifest_path",
            "inventory": "diagnostic_source_path",
        }[kind]
        p = args[name]
        p.write_bytes(p.read_bytes() + b"drift")
        if kind == "inventory":
            monkeypatch.setattr(v2, "VM_SCRIPT_SHA", v2.sha(p))
    elif kind == "v1_pass":
        put(
            args["v1_state_path"],
            {
                "jobs": {
                    "data": {"status": "FINISHED", "home": True},
                    "sfull_inputs": {"status": "FINISHED"},
                }
            },
        )
    elif kind == "raw_report":
        p = fixture["data"] / "human_decontam.json"
        p.write_text(p.read_text() + "\n")
    else:
        p = args["witness_path"]
        w = json.loads(p.read_text())
        if kind == "witness_raw":
            w["raw_human_witness"].pop(next(iter(w["raw_human_witness"])))
        if kind == "witness_eval":
            w["corrected_contract_result"]["evaluation_sources_verified"] = 1
        if kind == "witness_source":
            w["source_commit"] = "wrong"
        put(p, w)
        monkeypatch.setattr(v2, "VM_WITNESS_SHA", v2.sha(p))
    with pytest.raises((ValueError, OSError)):
        v2.verify_relay(fixture["data"], fixture["ref"], **args)


def test_welltyped_but_inconsistent_detail_lengths_refused(fixture):
    p = fixture["data"] / "human_decontam.json"
    value = json.loads(p.read_text())
    value["dropped"] = []
    put(p, value)
    with pytest.raises(ValueError, match="detail lengths"):
        verify(fixture)


def test_frozen_digest_literals_match_captured_diagnostic_contract():
    import re

    for name in (
        "REFERENCE_SHA",
        "V1_SOURCE_SHA",
        "VM_WITNESS_SHA",
        "VM_SCRIPT_SHA",
        "VM_PATCHED_SOURCE_SHA",
        "PROTOCOL_SHA",
        "VM_RAW_REPORT_SHA",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", getattr(v2, name)), name
    # Independently captured patched-source checksum, not derived from V2's literal.
    assert (
        v2.VM_PATCHED_SOURCE_SHA
        == "853896010b1a1e970790d1fff4871fe390071a38bfbf6cd65c7e0ef369282f39"
    )


def cli_fixture(tmp_path, monkeypatch):
    import sys

    out = tmp_path / "new_supplement.json"
    result = {"status": "TEST_PUBLICATION_ONLY", "scientific_outcome": False}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sfull_verify_inputs_v2.py",
            "--data",
            "unused_data",
            "--reference",
            "unused_reference",
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(v2, "verify", lambda *args, **kwargs: result)
    return out, result


def test_cli_publishes_new_complete_json_atomically(tmp_path, monkeypatch, capsys):
    out, result = cli_fixture(tmp_path, monkeypatch)
    v2.main()
    assert json.loads(out.read_text()) == result
    assert json.loads(capsys.readouterr().out) == result
    assert not list(tmp_path.glob("*.tmp.*"))


def test_cli_existing_output_refuses_before_verification_and_preserves_bytes(
    tmp_path, monkeypatch
):
    out, _ = cli_fixture(tmp_path, monkeypatch)
    historical = b"preserved original report bytes\n"
    out.write_bytes(historical)

    def forbidden(*args, **kwargs):
        raise AssertionError("verification must not run when output exists")

    monkeypatch.setattr(v2, "verify", forbidden)
    with pytest.raises(SystemExit) as exc:
        v2.main()
    assert exc.value.code == 2
    assert out.read_bytes() == historical
    assert not list(tmp_path.glob("*.tmp.*"))


def test_cli_publication_race_never_overwrites_concurrent_result(tmp_path, monkeypatch):
    out, _ = cli_fixture(tmp_path, monkeypatch)
    link = v2.os.link

    def racing_link(source, target):
        Path(target).write_bytes(b"concurrent original")
        return link(source, target)

    monkeypatch.setattr(v2.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        v2.main()
    assert out.read_bytes() == b"concurrent original"


def test_cli_failed_fsync_does_not_publish_result(tmp_path, monkeypatch):
    out, _ = cli_fixture(tmp_path, monkeypatch)

    def failure(fd):
        raise OSError("synthetic storage failure")

    monkeypatch.setattr(v2.os, "fsync", failure)
    with pytest.raises(OSError):
        v2.main()
    assert not out.exists()
