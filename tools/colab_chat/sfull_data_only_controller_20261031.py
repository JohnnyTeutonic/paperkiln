"""Source-bound, data-only S-full controller. Check mode never contacts a provider."""

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BOOT = "tools/colab_chat/boot_sfull_data_only_20261031.sh"
JOBS = "tools/colab_chat/sfull_data_only_jobs_20261031.json"
REMOTE = "f616a0f85447f95f67d0f9bdf6dfef513dc632a3"
BASE = "7edb0ca1b"
REFERENCE = "experiments/tiny_chat_long_2026-10/receipts/sfull_prospective_20261031/data_reference.json"
REFERENCE_SHA = "acd00d74a93299aa0614f98233f837ef3dc30d38ce9120330e9f2f5c2b982cf4"
OPERATIONS = {
    BOOT,
    JOBS,
    "tools/colab_chat/sfull_data_only_finish_20261031.py",
    "tools/colab_chat/test_sfull_data_only_finish_20261031.py",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def blob(root, commit, path):
    return subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{path}"])


def timestamp(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware admission time required")
    return result.timestamp()


def validate_inventory(root, files):
    for name, expected in files.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or digest(root / rel) != expected:
            raise ValueError("controller source inventory differs")


def verify_manifest(path, wanted, root):
    if digest(path) != wanted:
        raise ValueError("manifest hash differs")
    m = json.loads(Path(path).read_text())
    if (
        m["kind"] != "SFULL_DATA_ONLY_REVIEWED_CONTROLLER_V1"
        or m["remote_commit"] != REMOTE
    ):
        raise ValueError("wrong gate or remote source")
    if Path(m["controller_root"]).resolve() != root.resolve():
        raise ValueError("controller location differs")
    validate_inventory(root, m["source_sha256"])
    changed = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-status", BASE, REMOTE], text=True
    ).splitlines()
    if set(changed) != {"A\t" + name for name in OPERATIONS}:
        raise ValueError("original computational tree changed")
    for name in OPERATIONS:
        if (root / name).read_bytes() != blob(root, REMOTE, name):
            raise ValueError("local and remote operational files differ")
    jobs = json.loads((root / JOBS).read_text())["jobs"]
    registered = json.loads(
        blob(root, REMOTE, "tools/colab_chat/carry12_sfull_r1.json")
    )["jobs"]
    old_data = next(j for j in registered if j["name"] == "data")
    if [j["name"] for j in jobs] != ["data", "sfull_inputs"] or jobs[0][
        "cmd"
    ] != old_data["cmd"]:
        raise ValueError("data command differs from frozen job")
    if any("spec" in j or j.get("light") is not True for j in jobs):
        raise ValueError("not command-only jobs")
    if jobs[1]["after"] != "data" or jobs[1]["cmd"] != [
        "python3",
        "tools/colab_chat/sfull_verify_inputs.py",
        "--data",
        "/content/chat/data",
        "--reference",
        REFERENCE,
        "--out",
        "{out}",
    ]:
        raise ValueError("original verifier command differs")
    refraw = blob(root, REMOTE, REFERENCE)
    if hashlib.sha256(refraw).hexdigest() != REFERENCE_SHA:
        raise ValueError("reference differs")
    ref = json.loads(refraw)
    if set(jobs[0]["outputs"]) != set(ref["outputs"]) | {
        "data.ledger.json",
        "human_decontam.json",
    } or jobs[1]["outputs"] != ["verified_inputs.json"]:
        raise ValueError("relay inventory differs")
    if digest(m["vocab_path"]) != ref["vocab_sha256"]:
        raise ValueError("vocabulary differs")
    args = m["runner_argv"]
    if len(args) % 2 or len(set(args[::2])) != len(args) // 2:
        raise ValueError("invalid or repeated runner options")
    opts = dict(zip(args[::2], args[1::2]))
    expected = {
        "--commit": REMOTE,
        "--jobs": JOBS,
        "--boot-script": BOOT,
        "--backend": "runpod",
        "--gpu": "L4",
        "--cloud": "SECURE",
        "--volume-gb": "50",
        "--max-hours": "1.25",
        "--max-usd": "1.25",
        "--name": "pk-sfull-data-1031",
        "--art": m["art_root"],
        "--state-dir": m["state_dir"],
        "--upload": "vocab_llama3_bpe.gguf=" + m["vocab_path"],
    }
    if opts != expected:
        raise ValueError("runner arguments differ from bounded proposal")
    return m


def predecessor_clean(path):
    path = Path(path)
    s = json.loads(path.read_text())
    r = s.get("runpod", {})
    jobs = s.get("jobs")
    if (
        s.get("campaign_home") is not True
        or not jobs
        or any(v.get("home") is not True for v in jobs.values())
    ):
        raise ValueError("serial predecessor is not home")
    if (
        s.get("endpoint")
        or r.get("pod")
        or r.get("creating")
        or r.get("volume")
        or r.get("retired_volumes")
    ):
        raise ValueError("serial predecessor resources remain")
    if any(not e.get("stop") for e in r.get("ledger", {}).get("pods", [])) or any(
        not e.get("deleted") for e in r.get("ledger", {}).get("volumes", [])
    ):
        raise ValueError("serial predecessor cleanup not recorded")
    pid = path.parent / "runner.pid"
    if pid.exists():
        value = int(pid.read_text())
        try:
            os.kill(value, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("serial predecessor driver is alive")


def validate_admission(m, manifest_sha, admission, now, already_admitted=False):
    a = json.loads(Path(admission).read_text())
    if (
        a.get("authorisation") != "ROOT_APPROVED_SFULL_DATA_ONLY"
        or a.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("root admission required")
    if (
        a.get("manual_balance_accessor_checked") is not True
        or a.get("provider_cleanup_checked") is not True
        or a.get("published_remote_commit") != REMOTE
    ):
        raise ValueError("fresh manual provider admission required")
    created, checked, deadline = (
        timestamp(a[k])
        for k in ("admitted_at_utc", "manual_balance_checked_utc", "deadline_utc")
    )
    if (
        not created <= deadline <= created + 4500
        or not now < deadline
        or created > now + 60
    ):
        raise ValueError("75-minute absolute window invalid or expired")
    if not already_admitted and not 0 <= now - checked <= 300:
        raise ValueError("manual balance check is not fresh")
    proofs = a.get("serial_predecessors", [])
    if {x["path"] for x in proofs} != set(m["required_predecessor_states"]):
        raise ValueError("serial predecessor proof set differs")
    for proof in proofs:
        if digest(proof["path"]) != proof["sha256"]:
            raise ValueError("serial predecessor proof changed")
        predecessor_clean(proof["path"])
    return a


def validate_relay(art, ref, normal):
    art = Path(art)
    data = art / "data"
    report = json.loads((art / "sfull_inputs/verified_inputs.json").read_text())
    if (
        report.get("verified") is not True
        or report.get("reference_sha256") != REFERENCE_SHA
        or report.get("raw_human_sources_verified") != 13
        or report.get("evaluation_sources_verified") != 2
        or report.get("no_data_mutation") is not True
    ):
        raise ValueError("original verifier did not pass")
    ledger = json.loads((data / "data.ledger.json").read_text())
    if normal(ledger) != normal(ref) or report["ledger_sha256"] != digest(
        data / "data.ledger.json"
    ):
        raise ValueError("full original ledger differs after relay")
    if normal(json.loads((data / "human_decontam.json").read_text())) != normal(
        ref["steps"]["decontam"]
    ):
        raise ValueError("decontamination report differs")
    if report["vocab_sha256"] != ref["vocab_sha256"] or report["outputs_sha256"] != {
        n: v["sha256"] for n, v in ref["outputs"].items()
    }:
        raise ValueError("verifier output identities differ")
    outputs = {}
    for name, want in ref["outputs"].items():
        p = data / name
        if (
            Path(name).name != name
            or p.stat().st_size != want["bytes"]
            or digest(p) != want["sha256"]
        ):
            raise ValueError("relayed product differs")
        outputs[name] = {"sha256": digest(p), "bytes": p.stat().st_size}
    if len(outputs) != 5:
        raise ValueError("expected five products")
    return outputs


def apply_local_fence(api):
    previous = api.MAX_PODS
    api.MAX_PODS = 2
    return previous


def save_new(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.link(temp, path)
    temp.unlink()


def run(m, manifest_sha, admission, root):
    state = Path(m["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (state / "HALTED").exists():
            raise ValueError("preserved halt requires review")
        if (state / "GATE_COMPLETE.json").exists():
            prior = json.loads((state / "GATE_COMPLETE.json").read_text())
            if (
                prior.get("manifest_sha256") != manifest_sha
                or prior.get("status") != "DATA_INPUT_GATE_VERIFIED_NO_TRAINING"
            ):
                raise ValueError("prior gate identity differs")
            return 0
        api = None
        original = None
        phase = "admission"
        try:
            anchor = state / "admission_anchor.json"
            expected = {
                "manifest_sha256": manifest_sha,
                "admission_sha256": digest(admission),
            }
            if anchor.exists() and json.loads(anchor.read_text()) != expected:
                raise ValueError("admission anchor changed")
            if not anchor.exists() and (
                (state / "state.json").exists()
                or (Path(m["art_root"]).exists() and any(Path(m["art_root"]).iterdir()))
            ):
                raise ValueError("unbound existing campaign artifacts")
            a = validate_admission(
                m, manifest_sha, admission, time.time(), anchor.exists()
            )
            if not anchor.exists():
                save_new(anchor, expected)
            sys.path.insert(0, str(root / "tools/colab_chat"))
            import runpod_api
            import runpod_transport

            api = runpod_api
            if runpod_transport.api is not api:
                raise ValueError("provider fence module mismatch")
            original = apply_local_fence(api)
            phase = "runner"
            spec = importlib.util.spec_from_file_location(
                "sfull_modern_runner", root / "tools/colab_chat/runner_multi.py"
            )
            driver = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(driver)
            sys.argv = [
                str(root / "tools/colab_chat/runner_multi.py"),
                *m["runner_argv"],
                "--deadline-utc",
                a["deadline_utc"],
            ]
            driver.main()
            phase = "relay_verification"
            state_json = json.loads((state / "state.json").read_text())
            if set(state_json.get("jobs", {})) != {"data", "sfull_inputs"} or any(
                v.get("status") != "FINISHED" or v.get("home") is not True
                for v in state_json["jobs"].values()
            ):
                raise ValueError("data-only jobs did not both finish and relay")
            namespace = {
                "__name__": "frozen_sfull_contract",
                "__file__": str(root / "tools/colab_chat/sfull_verify_inputs.py"),
            }
            exec(
                compile(
                    blob(root, REMOTE, "tools/colab_chat/sfull_verify_inputs.py"),
                    "frozen_sfull_verify_inputs.py",
                    "exec",
                ),
                namespace,
            )
            outputs = validate_relay(
                m["art_root"],
                json.loads(blob(root, REMOTE, REFERENCE)),
                namespace["normal"],
            )
            phase = "publish"
            save_new(
                state / "GATE_COMPLETE.json",
                {
                    "status": "DATA_INPUT_GATE_VERIFIED_NO_TRAINING",
                    "manifest_sha256": manifest_sha,
                    "remote_commit": REMOTE,
                    "products": outputs,
                    "scientific_outcome": False,
                },
            )
            return 0
        except BaseException as error:
            if not (state / "HALTED").exists():
                number = getattr(error, "errno", None)
                save_new(
                    state / "HALTED",
                    {
                        "status": "HALTED_REVIEW_REQUIRED",
                        "error_type": type(error).__name__,
                        "phase": phase,
                        "errno": number if type(number) is int else None,
                        "manifest_sha256": manifest_sha,
                        "no_data_policy_waiver": True,
                    },
                )
            raise
        finally:
            if api is not None and original is not None:
                api.MAX_PODS = original


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--admission", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    m = verify_manifest(args.manifest, args.manifest_sha256, root)
    if args.check:
        print(
            json.dumps(
                {
                    "status": "SOURCE_AND_UPLOAD_PREFLIGHT_PASS",
                    "launch_authorised": False,
                    "requires_fresh_root_serial_admission": True,
                    "process_local_pod_fence": 2,
                    "global_caps_unchanged": True,
                    "remote_commit": REMOTE,
                }
            )
        )
        return
    if args.admission is None:
        raise ValueError("root serial admission sidecar required")
    raise SystemExit(run(m, args.manifest_sha256, args.admission, root))


if __name__ == "__main__":
    main()
