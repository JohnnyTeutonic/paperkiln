"""Bounded historical-source reconstruction; never rewrites historical research results."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
import urllib.request

DEADLINE = datetime.datetime(2026, 10, 31, 6, 5, tzinfo=datetime.timezone.utc)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def instrumented_equivalent(original, diagnostic):
    """The tap may add a raw-logit file only; every JSON numerical result must agree."""
    if original != diagnostic:
        raise ValueError("diagnostic tap changed original scorer results")


def run(
    out,
    upload=Path("/content/chat/upload"),
    root=Path("/content/hs1_source"),
    build=Path("/content/hs1_build"),
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("prior reconstruction evidence is immutable")
    report = {
        "status": "ERROR",
        "scope": "New source-bound reconstruction; not recovered historical logits",
        "commands": [],
        "models": {},
        "errors": [],
    }
    env = dict(os.environ)
    original = None
    source = root / "tools/mtstudio.cpp"

    def command(name, argv, timeout=300, local_env=None, cwd=None):
        remaining = (
            DEADLINE - datetime.datetime.now(datetime.timezone.utc)
        ).total_seconds()
        if remaining <= 0:
            raise ValueError("absolute reconstruction deadline expired")
        p = subprocess.Popen(
            list(map(str, argv)),
            cwd=cwd or root,
            env=local_env or env,
            stdout=(out / (name + ".log")).open("xb"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = p.wait(timeout=min(timeout, remaining))
        except (subprocess.TimeoutExpired, InterruptedError):
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise ValueError(name + " timed out") from None
        report["commands"].append(
            {"name": name, "argv": list(map(str, argv)), "returncode": rc}
        )
        if rc:
            raise ValueError(name + " failed")

    try:
        m = json.loads((upload / "hs1_inputs.json").read_text())
        report["input_manifest_sha256"] = sha(upload / "hs1_inputs.json")
        (out / "input_manifest.json").write_text(json.dumps(m, indent=2) + "\n")
        (out / "diagnostic_source.cpp").write_bytes(
            (upload / "mtstudio_diagnostic.cpp").read_bytes()
        )
        for name, h in m["files"].items():
            if sha(upload / name) != h:
                raise ValueError("input hash differs:" + name)
        if sha(upload / "vocab.gguf") != m["vocab_sha256"]:
            raise ValueError("vocabulary hash differs")
        for model in m["models"].values():
            if sha(upload / model["weights_upload"]) != model["weights_sha256"]:
                raise ValueError("checkpoint hash differs")
        env.update(m["runtime_env"])
        root.mkdir(exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("source destination not empty")
        with tarfile.open(upload / "paperkiln_3dc818b2.tar") as archive:
            for member in archive.getmembers():
                if (
                    member.name.startswith("/")
                    or ".." in Path(member.name).parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe source archive")
            archive.extractall(root)
        original = source.read_bytes()
        if sha(source) != m["source_original_mtstudio_sha256"]:
            raise ValueError("original scorer source differs")
        command(
            "hardware",
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            30,
        )
        hardware = (out / "hardware.log").read_text()
        if len(hardware.strip().splitlines()) != 1 or not hardware.startswith(
            "NVIDIA L4,"
        ):
            raise ValueError("requires one actual L4")
        command(
            "configure",
            [
                "cmake",
                "-S",
                root,
                "-B",
                build,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DMICROTORCH_CUDA=ON",
                "-DCUDA_ENABLED=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=89",
            ],
            180,
        )
        command(
            "build_original",
            ["cmake", "--build", build, "--target", "mtstudio", "--parallel", "4"],
            600,
        )
        report["original_binary_sha256"] = sha(build / "mtstudio")
        for run_name, model in m["models"].items():
            result = {"chunks": {}, "export": "NOT_RUN"}
            report["models"][run_name] = result
            for chunk in model["chunks"]:
                name = run_name + "_chunk" + str(chunk["index"])
                output = out / (name + "_original.jsonl")
                command(
                    name + "_original",
                    [
                        build / "mtstudio",
                        "score",
                        upload / model["spec"],
                        "--in",
                        upload / chunk["file"],
                        "--out",
                        output,
                        "--batch-tokens",
                        "16384",
                    ],
                    180,
                )
                result["chunks"][str(chunk["index"])] = {
                    "original": output.name,
                    "target_keys": chunk["target_keys"],
                }
            command(
                run_name + "_export",
                [
                    build / "mtstudio",
                    "export",
                    upload / model["spec"],
                    "--formats",
                    "gguf",
                ],
                120,
            )
            gguf = upload / (model["model_name"] + ".gguf")
            result["export"] = (
                "EXACT_HASH"
                if sha(gguf) == model["original_gguf_sha256"]
                else "HASH_DIFFERENT_NO_EXACT_ENGINE_CLAIM"
            )
            result["gguf_sha256"] = sha(gguf)
        source.write_bytes((upload / "mtstudio_diagnostic.cpp").read_bytes())
        if sha(source) != m["source_diagnostic_mtstudio_sha256"]:
            raise ValueError("diagnostic source differs")
        command(
            "build_diagnostic",
            ["cmake", "--build", build, "--target", "mtstudio", "--parallel", "4"],
            300,
        )
        report["diagnostic_binary_sha256"] = sha(build / "mtstudio")
        for run_name, model in m["models"].items():
            for chunk in model["chunks"]:
                name = run_name + "_chunk" + str(chunk["index"])
                output, logits = (
                    out / (name + "_diagnostic.jsonl"),
                    out / (name + ".f32"),
                )
                local = {
                    **env,
                    "HS1_DIAG_KEYS": "\n" + "\n".join(chunk["target_keys"]) + "\n",
                }
                command(
                    name + "_diagnostic",
                    [
                        build / "mtstudio",
                        "score",
                        upload / model["spec"],
                        "--in",
                        upload / chunk["file"],
                        "--out",
                        output,
                        "--batch-tokens",
                        "16384",
                        "--logits",
                        logits,
                    ],
                    180,
                    local,
                )

                def read(path):
                    return [json.loads(x) for x in path.read_text().splitlines()]

                instrumented_equivalent(
                    read(out / (name + "_original.jsonl")), read(output)
                )
                expected = sum(
                    len(case["request"]["continuation_ids"])
                    for case in model["cases"]
                    if case["request"]["key"] in chunk["target_keys"]
                )
                if logits.stat().st_size != expected * 16384 * 4:
                    raise ValueError("diagnostic logit row inventory differs")
                report["models"][run_name]["chunks"][str(chunk["index"])].update(
                    diagnostic=output.name,
                    logits=logits.name,
                    numerical_json_identical=True,
                    logit_rows=expected,
                )
        # Generation replay is conditional on byte-identical original GGUF exports.
        if all(x["export"] == "EXACT_HASH" for x in report["models"].values()):
            ember = root.parent / "hs1_ember"
            command(
                "ember_clone",
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    "https://github.com/JohnnyTeutonic/ember.cpp.git",
                    ember,
                ],
                180,
            )
            command(
                "ember_checkout",
                ["git", "-C", ember, "checkout", m["generation_source"]],
                30,
            )
            command(
                "ember_configure",
                [
                    "cmake",
                    "-S",
                    ember,
                    "-B",
                    ember / "build",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CUDA_ARCHITECTURES=89",
                ],
                180,
            )
            command(
                "ember_build",
                [
                    "cmake",
                    "--build",
                    ember / "build",
                    "--target",
                    "tinyllama_server",
                    "--parallel",
                    "4",
                ],
                600,
            )
            report["generation_binary_sha256"] = sha(ember / "build/tinyllama_server")
            for run_name, model in m["models"].items():
                directory = out / (run_name + "_ember")
                directory.mkdir()
                (directory / "debugging.log").symlink_to("/dev/null")
                www = directory / "www"
                www.mkdir()
                argv = [
                    str(ember / "build/tinyllama_server"),
                    str(upload / (model["model_name"] + ".gguf")),
                    "8094",
                    "127.0.0.1",
                    str(www),
                    "-1",
                ]
                child = subprocess.Popen(
                    argv,
                    cwd=directory,
                    env=env,
                    stdout=(directory / "stdout.log").open("xb"),
                    stderr=(directory / "stderr.log").open("xb"),
                    start_new_session=True,
                )
                try:
                    health = None
                    for _ in range(120):
                        if datetime.datetime.now(datetime.timezone.utc) >= DEADLINE:
                            raise ValueError("generation deadline expired")
                        try:
                            health = json.loads(
                                urllib.request.urlopen(
                                    "http://127.0.0.1:8094/health", timeout=2
                                ).read()
                            )
                            break
                        except OSError:
                            if child.poll() is not None:
                                raise ValueError("generation server exited")
                            time.sleep(0.5)
                    if not health or os.path.realpath(health.get("model", "")) != str(
                        upload / (model["model_name"] + ".gguf")
                    ):
                        raise ValueError("generation health identity differs")
                    responses = []
                    for case in model["cases"]:
                        for repeat in range(2):
                            remaining = (
                                DEADLINE - datetime.datetime.now(datetime.timezone.utc)
                            ).total_seconds()
                            if remaining <= 0:
                                raise ValueError("generation request deadline expired")
                            body = {
                                "user_input": case["user"],
                                "history": [
                                    {"user": u, "assistant": a}
                                    for u, a in case["history"]
                                ],
                                "temperature": 1.0,
                                "top_k": 1,
                                "top_p": 1.0,
                                "seed": 7,
                                "max_new_tokens": 48,
                                "return_ids": True,
                            }
                            request = urllib.request.Request(
                                "http://127.0.0.1:8094/chat",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"},
                            )
                            response = json.loads(
                                urllib.request.urlopen(
                                    request, timeout=min(30, remaining)
                                ).read()
                            )
                            responses.append(
                                {
                                    "key": case["request"]["key"],
                                    "repeat": repeat,
                                    "request": body,
                                    "response": response,
                                    "prompt_ids_identical": response.get("prompt_ids")
                                    == case["request"]["prompt_ids"],
                                    "ids_identical": response.get("ids")
                                    == case["expected"]["ids"],
                                    "stop_identical": response.get("stop_reason")
                                    == case["expected"]["stop"],
                                }
                            )
                    (out / (run_name + "_generation.json")).write_text(
                        json.dumps(responses, indent=2) + "\n"
                    )
                    report["models"][run_name]["generation_reconstruction"] = responses
                finally:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        else:
            report["generation_skipped"] = (
                "At least one re-export differs from original GGUF hash; no exact-generation-engine claim"
            )
        report["status"] = "SCORER_RECONSTRUCTION_COMPLETE"
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        report["errors"].append(str(exc))
    finally:
        if original is not None:
            source.write_bytes(original)
            report["original_source_restored"] = (
                sha(source) == hashlib.sha256(original).hexdigest()
            )
        report["files_sha256"] = {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        with zipfile.ZipFile(
            out / "out.zip",
            "x",
            compression=zipfile.ZIP_DEFLATED,
            strict_timestamps=False,
        ) as archive:
            for p in out.rglob("*"):
                if p.is_file() and not p.is_symlink() and p.name != "out.zip":
                    archive.write(p, str(p.relative_to(out)))
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0


if __name__ == "__main__":
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(InterruptedError("termination")),
    )
    raise SystemExit(run(sys.argv[1]))
