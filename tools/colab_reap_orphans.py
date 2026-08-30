#!/usr/bin/env python3
"""Reap UNNAMED Colab sessions — the zombie class supervisor.py cannot.

    python3 tools/colab_reap_orphans.py            # list only
    python3 tools/colab_reap_orphans.py --kill     # stop every orphan
    python3 tools/colab_reap_orphans.py --kill --keep fixv,scl2

`colab sessions` prints `[name] endpoint ...` for sessions whose
name→token mapping lives in ~/.config/colab-cli/sessions.json, and
`[?] endpoint ...` for ones it cannot name — sessions created by a
process whose config is gone (a killed workflow agent, a reclaimed
supervisor, a crashed shell). Those burn units with nothing watching
them, and `colab stop -s` cannot address them: it takes a NAME.

supervisor.py's session registry solves this only for sessions IT
created (it snapshots each config at creation). Orphans from any other
process were unkillable from the CLI.

They are not. `stop` authenticates on the control plane (OAuth), not
with the per-session token, so a config carrying just the endpoint is
enough — verified 2026-08-30 against a leaked T4 that had been idling
after its creating agent died:

    colab --config /tmp/orphan.json stop -s orphan1   -> terminated

This tool automates that. It NEVER touches the real config file, and it
refuses to kill anything it can name (a named session belongs to a live
owner) — pass --keep for extra belt and braces.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

COL = os.path.expanduser("~/.local/bin/colab")
ROW = re.compile(r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)")


def sessions() -> list:
    p = subprocess.run([COL, "sessions"], capture_output=True, text=True,
                       timeout=120)
    out = []
    for line in (p.stdout or "").splitlines():
        m = ROW.match(line.strip())
        if m:
            out.append((m.group("name"), m.group("endpoint")))
    return out


def stop_orphan(endpoint: str) -> bool:
    """Stop a session addressed only by endpoint, via a scratch config."""
    cfg = {
        "orphan": {
            "name": "orphan",
            "token": "",
            "url": f"https://8080-{endpoint}-b.us-west1-1.prod.colab.dev",
            "endpoint": endpoint,
            "variant": "GPU",
        }
    }
    fd, path = tempfile.mkstemp(suffix=".json", prefix="reap_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f)
        p = subprocess.run([COL, "--config", path, "stop", "-s", "orphan"],
                           capture_output=True, text=True, timeout=120)
        return "terminated" in (p.stdout + p.stderr).lower()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill", action="store_true",
                    help="actually stop the orphans (default: list only)")
    ap.add_argument("--keep", default="",
                    help="comma-separated endpoints/names never to touch")
    args = ap.parse_args()
    keep = {s.strip() for s in args.keep.split(",") if s.strip()}

    rows = sessions()
    if not rows:
        print("no active sessions")
        return 0
    orphans = [ep for name, ep in rows
               if name == "?" and ep not in keep]
    for name, ep in rows:
        tag = "ORPHAN" if (name == "?" and ep not in keep) else "owned "
        print(f"  {tag}  [{name}] {ep}")
    if not orphans:
        print("no orphans")
        return 0
    if not args.kill:
        print(f"{len(orphans)} orphan(s); rerun with --kill to stop them")
        return 0
    for ep in orphans:
        ok = stop_orphan(ep)
        print(f"  {'reaped' if ok else 'FAILED'}: {ep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
