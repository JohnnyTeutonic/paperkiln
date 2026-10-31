import json
from pathlib import Path
import pytest
import dpo_checkpoint_v1 as dpo
import dpo_checkpoint_core_v1 as core
from test_dpo_checkpoint_core_v1 import flat


@pytest.fixture
def contract(tmp_path):
    out = tmp_path / "run"
    inputs = {}
    for name in ["vocab", "pairs", "init"]:
        path = tmp_path / name
        path.write_text(name)
        inputs[str(path)] = core.sha(path)
    spec = {
        "name": "pk_dpo_v1",
        "out_dir": str(out),
        "data": {
            "vocab": str(tmp_path / "vocab"),
            "pref_pairs": str(tmp_path / "pairs"),
        },
        "train": {
            "objective": "provenance_po",
            "optimizer": "adamw",
            "seed": 1,
            "batch": 8,
            "accum": 1,
            "steps": 20,
            "checkpoint_every": 10,
            "init_from": str(tmp_path / "init"),
        },
    }
    record = {
        "kind": dpo.KIND,
        "source_commit": "a" * 40,
        "runtime_spec": spec,
        "inputs_sha256": inputs,
        "dataset_manifest_sha256": "c" * 64,
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(record))
    return path, record, out


def test_fresh_sft_initialization_allowed(contract):
    path, record, out = contract
    checked, digest = dpo.load_contract(path)
    assert dpo.prepare(checked, digest, out)["fresh_init_from_allowed"]


@pytest.mark.parametrize(
    "name", ["model.safetensors", "optim.safetensors", "state.txt"]
)
def test_partial_flat_never_cold_resumes(contract, name):
    path, record, out = contract
    out.mkdir()
    (out / name).write_text("partial")
    with pytest.raises(ValueError, match="cold resume"):
        dpo.prepare(record, core.sha(path), out)


def test_full_generation_restored_and_contract_drift_refused(contract):
    path, record, out = contract
    flat(out, 10)
    digest = core.sha(path)
    core.publish(out, 10, 1, 8, 1, "a" * 40, digest)
    (out / "optim.safetensors").unlink()
    dpo.prepare(record, digest, out)
    assert (out / "optim.safetensors").is_file()
    with pytest.raises(ValueError, match="contract differs"):
        dpo.prepare(record, "d" * 64, out)


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch", 2),
        ("accum", 2),
        ("checkpoint_every", 20),
        ("objective", "lm"),
        ("optimizer", "muon"),
        ("seed", 2),
    ],
)
def test_recipe_contract_refuses_unsupported(contract, field, value):
    path, record, out = contract
    record["runtime_spec"]["train"][field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        dpo.load_contract(path)


def test_changed_input_refused(contract):
    path, record, out = contract
    Path(record["runtime_spec"]["train"]["init_from"]).write_text("changed")
    with pytest.raises(ValueError, match="input differs"):
        dpo.prepare(record, core.sha(path), out)


def test_direct_cli_resume_checks_generation_recipe_not_just_contract_hash(contract):
    path, record, out = contract
    flat(out, 10)
    digest = core.sha(path)
    core.publish(out, 10, 2, 8, 1, "a" * 40, digest)
    with pytest.raises(ValueError, match="generation recipe"):
        dpo.prepare(record, digest, out)


def test_boolean_seed_not_integer_recipe(contract):
    path, record, out = contract
    record["runtime_spec"]["train"]["seed"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="integer"):
        dpo.load_contract(path)
