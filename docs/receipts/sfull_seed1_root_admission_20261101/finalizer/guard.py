"""Exclusive owner for the frozen local finalizer, never a provider controller."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


def atom(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def validate(record):
    args = record["argv"]
    state = Path(record["state_dir"])
    if args[args.index("--state-dir") + 1] != str(state):
        raise ValueError("finalizer state differs")
    if "--calculate" in args or "--once" in args:
        raise ValueError("watchdog requires full frozen finalizer loop")
    for path, digest in record["source_pins"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen finalizer source differs: " + path)
    return args, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch", required=True)
    p.add_argument("--launch-sha256", required=True)
    p.add_argument("--state-dir", required=True)
    a = p.parse_args()
    data = Path(a.launch).read_bytes()
    if hashlib.sha256(data).hexdigest() != a.launch_sha256:
        raise ValueError("finalizer launch binding differs")
    record = json.loads(data)
    args, state = validate(record)
    if str(state) != a.state_dir:
        raise ValueError("finalizer match state differs")
    state.mkdir(parents=True, exist_ok=True)
    stop = state / "STOPPED"
    with (state / "watchdog_owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        if stop.exists():
            return 0
        try:
            result = subprocess.run(args, check=False)
            current = json.loads((state / "state.json").read_text())
            status = current.get("status")
            if (result.returncode == 0 and status == "COMPLETE") or (
                result.returncode == 1 and status == "HALTED"
            ):
                atom(
                    stop,
                    {
                        "status": status,
                        "finalizer_returncode": result.returncode,
                        "finalizer_only": True,
                    },
                )
                return result.returncode
            raise RuntimeError(
                "unexpected finalizer exit/state: " + str((result.returncode, status))
            )
        except Exception as error:
            atom(
                stop,
                {
                    "status": "FINALIZER_OWNER_HALTED",
                    "reason": str(error),
                    "finalizer_only": True,
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
