"""The vendored keramos tree (spec/): the IR's specification, schema, reference tools and conformance
suite, copied from the keramos repository at a pinned tag. spec/KERAMOS_PIN.json records the tag, its
full commit and the commit's manifest (every file's git blob sha1), so a local edit under spec/ is
visible: tests/test_keramos_pin.py fails on one, and tools/sync_keramos.sh refuses to overwrite one.

    python tools/keramos_pin.py check                  # spec/ against the pin; exit 1 on any difference
    python tools/keramos_pin.py manifest SRC COMMIT    # a keramos commit's manifest, as the pin stores it
    python tools/keramos_pin.py write SRC TAG          # record TAG of the keramos checkout SRC as the pin

Standard library only. Text files are hashed as git stores them (CRLF read as LF, as a Windows checkout
with core.autocrlf writes them; keramos stores text with LF); a file with a NUL byte in its first 8000
bytes is binary and hashed as is.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "spec"
PIN = SPEC / "KERAMOS_PIN.json"
REPOSITORY = "https://github.com/JohnnyTeutonic/keramos"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def blob_sha1(path):
    """The git blob id of a file's content as git would store it."""
    data = Path(path).read_bytes()
    if b"\0" not in data[:8000]:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tree(root=SPEC):
    """{relative posix path: blob sha1} of every vendored file under root (the pin itself excluded)."""
    root = Path(root)
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.resolve() == PIN.resolve() or p.suffix == ".pyc":
                continue
            out[p.relative_to(root).as_posix()] = blob_sha1(p)
    return out


def load_pin(path=PIN):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def differences(pin=None, root=SPEC):
    """Messages for every file that is missing, extra or different against the pin's manifest."""
    pin = pin or load_pin()
    want, have = pin["files"], tree(root)
    msgs = [f"missing: {p}" for p in sorted(set(want) - set(have))]
    msgs += [f"not in keramos {pin['tag']}: {p}" for p in sorted(set(have) - set(want))]
    msgs += [f"edited: {p}" for p in sorted(set(want) & set(have)) if want[p] != have[p]]
    return msgs


def _git(src, *args):
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(src), *args], capture_output=True,
                          check=True).stdout


def manifest(src, commit):
    """{path: blob sha1} of a keramos commit (regular files; links and submodules are not vendored)."""
    out = {}
    for rec in _git(src, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, sha = meta.split()
        if kind == b"blob" and mode in (b"100644", b"100755"):
            out[path.decode("utf-8")] = sha.decode()
    return dict(sorted(out.items()))


def tag_commit(src, tag):
    """The full commit a tag of the keramos checkout names (a tag, not a branch or a bare commit)."""
    _git(src, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}")
    return _git(src, "rev-parse", f"refs/tags/{tag}^{{commit}}").decode().strip()


def write(src, tag, path=PIN):
    commit = tag_commit(src, tag)
    pin = {"_comment": "spec/ is keramos (the IR specification) at this tag, vendored by tools/sync_keramos.sh; "
                       "tests/test_keramos_pin.py checks spec/ against files (git blob sha1 per path). Edit "
                       "keramos, tag it, and sync; never edit spec/ in place.",
           "repository": REPOSITORY, "tag": tag, "commit": commit, "files": manifest(src, commit)}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(pin, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return pin


def main(argv):
    if argv[:1] == ["check"]:
        pin = load_pin()
        msgs = differences(pin)
        for m in msgs:
            print(m)
        print(f"spec/ {'differs from' if msgs else 'is'} keramos {pin['tag']} ({pin['commit'][:12]}): "
              f"{len(pin['files'])} files")
        return 1 if msgs else 0
    if argv[:1] == ["manifest"] and len(argv) == 3:
        print(json.dumps(manifest(argv[1], argv[2]), indent=1))
        return 0
    if argv[:1] == ["write"] and len(argv) == 3:
        pin = write(argv[1], argv[2])
        print(f"pinned keramos {pin['tag']} = {pin['commit']} ({len(pin['files'])} files)")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
