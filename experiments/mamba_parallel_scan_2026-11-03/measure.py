#!/usr/bin/env python3
"""Run a command and record its elapsed time and peak memory (vm_run.sh's prefill steps):

    python3 measure.py OUT.json -- COMMAND [ARGS...]

The command inherits stdin, stdout and stderr, so the caller's redirections apply to it. When it exits, OUT.json
is written atomically: the command line, exit status, elapsed wall seconds, user and system CPU seconds, and the
peak resident set of its largest waited-for descendant (getrusage RUSAGE_CHILDREN ru_maxrss, KiB on Linux). The
exit status is the command's own (128 + N for a command ended by signal N), also when OUT.json cannot be written:
that failure goes to stderr, and the missing receipt fails the step in vm_run.sh. A stop that kills this process too
leaves no OUT.json; the step's own status records that. Standard library only (the VM image's python3)."""
import json
import os
import resource
import subprocess
import sys
import time


def main(argv):
    if len(argv) < 3 or argv[1] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    out, cmd = argv[0], argv[2:]
    t0 = time.monotonic()
    try:
        rc = subprocess.run(cmd).returncode
    except OSError as e:
        print(f"measure.py: {e}", file=sys.stderr)
        rc = 127
    elapsed = time.monotonic() - t0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    code = rc if rc >= 0 else 128 - rc
    rec = {"cmd": cmd, "exit": code, "elapsed_s": round(elapsed, 3), "user_s": round(ru.ru_utime, 3),
           "sys_s": round(ru.ru_stime, 3), "max_rss_kib": ru.ru_maxrss}
    # the command's status stands even if the receipt cannot be written: the failure is reported here, and the
    # missing receipt fails the step in vm_run.sh
    tmp = out + ".tmp"
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(rec) + "\n")
        os.replace(tmp, out)
    except OSError as e:
        print(f"measure.py: could not write {out}: {e}", file=sys.stderr)
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
