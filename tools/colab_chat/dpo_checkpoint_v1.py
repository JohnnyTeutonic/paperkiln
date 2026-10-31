"""Named, opt-in DPO checkpoint contract. Fresh init_from is not a resume."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import dpo_checkpoint_core_v1 as core

KIND = "DPO_FULL_STATE_CHECKPOINT_V1"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def load_contract(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular DPO contract required")
    contract = json.loads(path.read_text())
    required = {
        "kind",
        "source_commit",
        "runtime_spec",
        "inputs_sha256",
        "dataset_manifest_sha256",
    }
    if set(contract) != required or contract["kind"] != KIND:
        raise ValueError("DPO contract fields differ")
    spec = contract["runtime_spec"]
    train = spec["train"]
    if any(
        type(train[k]) is not int
        for k in ["seed", "batch", "accum", "steps", "checkpoint_every"]
    ):
        raise ValueError("integer DPO runtime recipe required")
    if (
        train["objective"] != "provenance_po"
        or train["optimizer"] != "adamw"
        or train["batch"] != 8
        or train["accum"] != 1
        or train["seed"] != 1
        or train["checkpoint_every"] != 10
        or type(train["steps"]) is not int
        or train["steps"] < 1
    ):
        raise ValueError("unsupported DPO checkpoint recipe")
    if spec.get("name") != "pk_dpo_v1":
        raise ValueError("DPO job name differs")
    for value, size in [
        (contract["source_commit"], 40),
        (contract["dataset_manifest_sha256"], 64),
    ]:
        if (
            not isinstance(value, str)
            or len(value) != size
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("DPO source/dataset hash differs")
    paths = {spec["data"]["vocab"], spec["data"]["pref_pairs"], train["init_from"]}
    if set(contract["inputs_sha256"]) != paths:
        raise ValueError("DPO input inventory differs")
    return contract, core.sha(path)


def check_arguments(contract, out, step, seed, batch, accum, source):
    spec = contract["runtime_spec"]
    t = spec["train"]
    if (
        str(Path(out).resolve()) != str(Path(spec["out_dir"]).resolve())
        or source != contract["source_commit"]
        or (seed, batch, accum) != (t["seed"], t["batch"], t["accum"])
        or not 0 <= step <= t["steps"]
    ):
        raise ValueError("DPO checkpoint invocation differs from contract")


def prepare(contract, digest, out):
    for path, expected in contract["inputs_sha256"].items():
        if core.sha(path) != expected:
            raise ValueError("DPO immutable input differs: " + path)
    root = Path(out) / ".dpo_checkpoint_v1"
    if (root / "CURRENT").is_symlink():
        with core.locked(root, False) as locked:
            _, manifest = core.current_unlocked(locked)
            if manifest["operational_contract_sha256"] != digest:
                raise ValueError("DPO resumed contract differs")
            t = contract["runtime_spec"]["train"]
            if (
                any(
                    manifest["rng"][k] != t[v]
                    for k, v in [
                        ("recipe_seed", "seed"),
                        ("batch", "batch"),
                        ("accum", "accum"),
                    ]
                )
                or manifest["step"] > t["steps"]
            ):
                raise ValueError("DPO resumed generation recipe/budget differs")
        core.restore_flat(root, out, contract["source_commit"])
    elif any(
        (Path(out) / name).exists()
        for name in ["model.safetensors", "optim.safetensors", "state.txt"]
    ):
        raise ValueError(
            "DPO partial flat output has no committed full state; cold resume refused"
        )
    return {"prepared": True, "fresh_init_from_allowed": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "publish"])
    for name in ["out", "source", "spec"]:
        p.add_argument("--" + name, required=True)
    for name in ["step", "seed", "batch", "accum"]:
        p.add_argument("--" + name, required=True, type=int)
    args = p.parse_args()
    contract, digest = load_contract(os.environ["DPO_CHECKPOINT_CONTRACT"])
    check_arguments(
        contract, args.out, args.step, args.seed, args.batch, args.accum, args.source
    )
    if args.action == "prepare":
        if json.loads(Path(args.spec).read_text()) != contract["runtime_spec"]:
            raise ValueError("actual DPO runtime spec differs from admitted recipe")
        result = prepare(contract, digest, args.out)
    else:
        result = core.publish(
            args.out, args.step, args.seed, args.batch, args.accum, args.source, digest
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
