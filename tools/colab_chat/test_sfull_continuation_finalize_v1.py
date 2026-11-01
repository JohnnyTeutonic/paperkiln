import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
import sfull_continuation_finalize_v1 as m
from test_sfull_checkpoint_v2 import flat


def put(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))
    return str(p)


def state(seed):
    names = ["cuda_gate", "data", "gate", "sfull_inputs", f"pt_c12_Sfull_r{seed}", f"post_c12_Sfull_r{seed}"]
    return {"campaign_home": True, "endpoint": None, "jobs": {n: {"home": True, "status": "FINISHED"} for n in names}, "runpod": {"pod": None, "volume": None, "retired_volumes": [], "ledger": {"pods": [{"id": "old", "start": 1, "stop": 2, "usd_per_hr": .59}], "volumes": []}}}


@pytest.fixture
def contract(tmp_path):
    roots, baseline, campaigns = {}, {}, []
    anchor = tmp_path / "anchor.py"; anchor.write_text("# immutable source\n")
    pins = {str(anchor): m.sha(anchor)}
    checkout = tmp_path / "operational"
    names = ["runner_multi.py", "sfull_operational_v2.py", "sfull_checkpoint_v2.py", "sfull_checkpoint_transport_v2.py", "boot_sfull_v2_20261101.sh", "sfull_verify_inputs_v2.py", "sfull_sft_template_v2.json", *[f"carry12_sfull_v2_r{s}.json" for s in m.SEEDS]]
    source_pins = {}
    for name in names:
        p = checkout / "tools/colab_chat" / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text("# frozen fixture\n");source_pins[str(p)] = m.sha(p)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "gc.auto=0", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    pins.update(source_pins)
    for c in m.CELLS:
        for seed in m.SEEDS:
            name = f"c12_{c}_r{seed}"; root = tmp_path / name
            p = Path(put(root / f"post_c12_{c}_r{seed}" / "summary.json", {"cell": c, "seed": seed}))
            roots[name] = str(root)
            if c != "Sfull":
                baseline[f"{name}/post_c12_{c}_r{seed}/summary.json"] = m.sha(p)
            else:
                for rel in [f"post_c12_Sfull_r{seed}/carry12/m/items.jsonl", f"post_c12_Sfull_r{seed}/carry12/m/gen.jsonl", f"pt_c12_Sfull_r{seed}/pt_c12_Sfull_r{seed}_spec.json", f"post_c12_Sfull_r{seed}/sft_spec.json", "data/data.ledger.json", "gate/gate.json", "cuda_gate/cuda_gate.json", "sfull_inputs/verified_inputs.json"]:
                    put(root / rel, {})
                for rel, step in [(f"pt_c12_Sfull_r{seed}",10000), (f"post_c12_Sfull_r{seed}/sft",1500)]:
                    folder = flat(root / rel, step)
                    m.checkpoint.publish(folder, step, seed, 16, 1, m.NUMERICAL)
    for seed in m.SEEDS:
        st = tmp_path / f"runner{seed}" / "state.json"
        carry = tmp_path / f"carry{seed}.json"
        put(st, state(seed)); put(carry, state(seed)["runpod"]["ledger"])
        pins[str(st)] = m.sha(st); pins[str(carry)] = m.sha(carry)
        campaigns.append({"seed": seed, "state": str(st), "carried_ledger": str(carry), "protected_original_volumes": [], "operational_source_pins": {**source_pins, str(anchor): m.sha(anchor)}, "operational_checkout": str(checkout), "operational_commit": commit, "original_admission_pins": {str(anchor): m.sha(anchor)}})
    freeze = Path(put(tmp_path / "freeze.json", {"source_hashes": {str(anchor): m.sha(anchor)}}))
    manifest = Path(put(tmp_path / "manifest.json", {"baseline_artifact_sha256": baseline}))
    put(freeze, {"source_hashes": {str(anchor): m.sha(anchor), str(manifest): m.sha(manifest)}})
    for p in (freeze, manifest): pins[str(p)] = m.sha(p)
    return {"numerical_commit": m.NUMERICAL, "seed_cap_usd": 7, "aggregate_cap_usd": 21, "pins": pins, "original_source_freeze": str(freeze), "original_manifest": str(manifest), "frozen_finalizer": str(anchor), "cpu_budget": str(anchor), "artifact_roots": roots, "campaigns": campaigns}


def test_complete_mapping_and_view_references_originals(contract, tmp_path):
    before = {p: m.sha(p) for p in contract["pins"]}
    proof = m.validate(contract, 10)
    assert proof["original_scientific_predicates_unchanged"]
    art, states = m.create_views(contract, tmp_path / "view")
    assert len(list(art.iterdir())) == 9
    assert len(list(states.iterdir())) == 3
    assert (art / "c12_Sfull_r1").resolve() == Path(contract["artifact_roots"]["c12_Sfull_r1"])
    assert {p: m.sha(p) for p in before} == before
    with pytest.raises(ValueError, match="new absolute"):
        m.create_views(contract, tmp_path / "view")


@pytest.mark.parametrize("change", ["missing_seed", "duplicate_seed", "missing_summary", "duplicate_summary", "summary_identity", "alias_root", "baseline_changed", "unsafe_baseline", "wrong_numerical", "wrong_budget", "source_changed", "empty_old_pins", "unbound_state", "wrong_control", "incomplete_control", "missing_raw", "corrupt_optimizer", "wrong_final_step"])
def test_mapping_rejects_ambiguous_incomplete_or_drifting_inputs(contract, tmp_path, change):
    c = copy.deepcopy(contract)
    if change == "missing_seed": c["campaigns"].pop()
    elif change == "duplicate_seed": c["campaigns"][2]["seed"] = 1
    elif change == "missing_summary":
        next(Path(c["artifact_roots"]["c12_Sfull_r3"]).glob("*/summary.json")).unlink()
    elif change == "duplicate_summary": put(Path(c["artifact_roots"]["c12_Sfull_r1"]) / "post_c12_extra/summary.json", {})
    elif change == "summary_identity": put(Path(c["artifact_roots"]["c12_Sfull_r1"]) / "post_c12_Sfull_r1/summary.json", {"cell": "Sfull", "seed": 2})
    elif change == "alias_root": c["artifact_roots"]["c12_Sfull_r2"] = c["artifact_roots"]["c12_Sfull_r1"]
    elif change == "baseline_changed": next(Path(c["artifact_roots"]["c12_B_r1"]).glob("*/summary.json")).write_text('{"cell":"B","seed":1,"altered":true}')
    elif change == "unsafe_baseline":
        manifest = m.read(c["original_manifest"]); manifest["baseline_artifact_sha256"]["../outside"] = "x";put(Path(c["original_manifest"]), manifest);c["pins"][c["original_manifest"]] = m.sha(c["original_manifest"])
        freeze=m.read(c["original_source_freeze"]);freeze["source_hashes"][c["original_manifest"]]=m.sha(c["original_manifest"]);put(Path(c["original_source_freeze"]),freeze);c["pins"][c["original_source_freeze"]]=m.sha(c["original_source_freeze"])
    elif change == "wrong_numerical": c["numerical_commit"] = "b" * 40
    elif change == "wrong_budget": c["seed_cap_usd"] = 8
    elif change == "source_changed": Path(c["frozen_finalizer"]).write_text("changed")
    elif change == "empty_old_pins": c["campaigns"][0]["original_admission_pins"] = {}
    elif change == "unbound_state": c["pins"].pop(c["campaigns"][0]["state"])
    elif change == "missing_raw": (Path(c["artifact_roots"]["c12_Sfull_r1"])/"post_c12_Sfull_r1/carry12/m/items.jsonl").unlink()
    elif change == "corrupt_optimizer": (Path(c["artifact_roots"]["c12_Sfull_r1"])/"pt_c12_Sfull_r1/.checkpoint_v2/CURRENT/optim.safetensors").write_bytes(b"broken")
    elif change == "wrong_final_step":
        folder = flat(Path(c["artifact_roots"]["c12_Sfull_r1"])/"pt_c12_Sfull_r1", 10025)
        m.checkpoint.publish(folder,10025,1,16,1,m.NUMERICAL)
    elif change == "wrong_control": c["campaigns"][0]["operational_commit"] = "f" * 40
    elif change == "incomplete_control": c["campaigns"][0]["operational_source_pins"].pop(str(Path(c["campaigns"][0]["operational_checkout"])/"tools/colab_chat/runner_multi.py"))
    with pytest.raises((ValueError, FileNotFoundError)): m.validate(c, 10)


@pytest.mark.parametrize("change", ["not_home", "unfinished", "endpoint", "pod", "volume", "retired", "creating", "omitted", "modified", "duplicate", "unstopped", "negative", "nan", "over_cap", "unprotected_volume"])
def test_ledger_preserves_costs_and_refuses_owned_resources(change):
    s = state(1); carry = copy.deepcopy(s["runpod"]["ledger"])
    if change == "not_home": s["campaign_home"] = False
    elif change == "unfinished": s["jobs"]["data"]["status"] = "RUNNING"
    elif change == "endpoint": s["endpoint"] = "endpoint"
    elif change in ("pod", "volume", "creating"): s["runpod"][change] = {"id": "owned"}
    elif change == "retired": s["runpod"]["retired_volumes"] = [{"id": "owned"}]
    elif change == "omitted": s["runpod"]["ledger"]["pods"] = []
    elif change == "modified": s["runpod"]["ledger"]["pods"][0]["usd_per_hr"] = 0
    elif change == "duplicate": s["runpod"]["ledger"]["pods"].append(copy.deepcopy(carry["pods"][0]))
    elif change == "unstopped": s["runpod"]["ledger"]["pods"].append({"id": "live", "start": 1, "usd_per_hr": .59})
    elif change == "negative": s["runpod"]["ledger"]["pods"].append({"id": "neg", "start": 2, "stop": 1, "usd_per_hr": .59})
    elif change == "nan": s["runpod"]["ledger"]["pods"].append({"id": "nan", "start": 1, "stop": 2, "usd_per_hr": float("nan")})
    elif change == "over_cap": s["runpod"]["ledger"]["pods"].append({"id": "large", "start": 1, "stop": 36001, "usd_per_hr": 1})
    elif change == "unprotected_volume": s["runpod"]["ledger"]["volumes"].append({"id": "v", "created": 1, "gb": 80})
    with pytest.raises(ValueError): m.validate_ledger(s, carry, [], 10)


def test_preserved_volume_costs_continue_counting_without_ownership():
    s = state(1);s["runpod"]["ledger"]["volumes"] = [{"id": "protected", "created": 1, "gb": 80}]
    carry = copy.deepcopy(s["runpod"]["ledger"])
    assert m.validate_ledger(s, carry, ["protected"], 100) > m.validate_ledger(s, carry, ["protected"], 10)


def test_check_mode_does_not_create_views_or_invoke_finalizer(contract, tmp_path):
    config = Path(put(tmp_path / "contract.json", contract));view = tmp_path / "view"
    cmd = [sys.executable, m.__file__, "--contract", str(config), "--contract-sha256", m.sha(config), "--view", str(view), "--out", str(tmp_path / "out"), "--state-dir", str(tmp_path / "finalizer")]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["status"] == "READY_FOR_MAPPED_FINALIZER"
    assert not view.exists()


def test_real_subprocess_invokes_frozen_finalizer_once_with_mapped_paths(contract, tmp_path):
    frozen = Path(contract["frozen_finalizer"])
    capture = tmp_path / "invocation.json"
    frozen.write_text("import json,sys\nfrom pathlib import Path\nPath(" + repr(str(capture)) + ").write_text(json.dumps(sys.argv[1:]))\n")
    digest = m.sha(frozen);contract["pins"][str(frozen)] = digest
    for c in contract["campaigns"]:
        c["operational_source_pins"][str(frozen)] = digest;c["original_admission_pins"][str(frozen)] = digest
    freeze = Path(contract["original_source_freeze"]);put(freeze, {"source_hashes": {str(frozen): digest, contract["original_manifest"]: m.sha(contract["original_manifest"])}});contract["pins"][str(freeze)] = m.sha(freeze)
    config = Path(put(tmp_path / "contract.json", contract))
    cmd = [sys.executable, m.__file__, "--contract", str(config), "--contract-sha256", m.sha(config), "--view", str(tmp_path / "view"), "--out", str(tmp_path / "out"), "--state-dir", str(tmp_path / "finalizer"), "--execute"]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    argv = m.read(capture)
    assert argv.count("--once") == 1
    assert argv[argv.index("--art")+1] == str(tmp_path / "view/art")
    assert argv[argv.index("--campaign-state-root")+1] == str(tmp_path / "view/states")
    assert argv[argv.index("--source-freeze")+1] == contract["original_source_freeze"]
