"""Shared fetcher discipline (docs/decisions/0001-posttrain-charter.md, section 9):
pinned source, deterministic output path, idempotent, sha256 verified, loud
failure on mismatch. Every corpus fetcher imports this; nothing else writes
under corpus/data/.
"""
from __future__ import annotations

import hashlib
import os
import sys

CORPUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CORPUS_DIR, "data")      # gitignored: bytes, not records
HASH_DIR = os.path.join(CORPUS_DIR, "hashes")    # committed: the record


class HashMismatch(RuntimeError):
    pass


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def out_path(corpus_id, filename):
    d = os.path.join(DATA_DIR, corpus_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def expected_hashes(corpus_id):
    """{filename: sha256} from corpus/hashes/<id>.sha256 (sha256sum format),
    or {} when the corpus has not been pinned yet."""
    p = os.path.join(HASH_DIR, f"{corpus_id}.sha256")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                h, name = line.split(None, 1)
                out[name.lstrip("*")] = h
    return out


def pin_hashes(corpus_id, files, note):
    """Write corpus/hashes/<id>.sha256 the first time a corpus is fetched.
    Refuses to overwrite: a pinned hash changes only by a new corpus id."""
    p = os.path.join(HASH_DIR, f"{corpus_id}.sha256")
    if os.path.exists(p):
        raise RuntimeError(f"{p} exists; a pinned corpus is never re-pinned")
    os.makedirs(HASH_DIR, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {note}\n")
        for name, h in sorted(files.items()):
            f.write(f"{h}  {name}\n")


def finish(corpus_id, paths, note, pin=False):
    """Verify every produced file against its pinned hash. With pin=True and
    no pin on record, record the hashes instead (first fetch of a new id)."""
    got = {os.path.basename(p): sha256_file(p) for p in paths}
    want = expected_hashes(corpus_id)
    if not want:
        if not pin:
            raise HashMismatch(f"{corpus_id}: no pinned hashes; run with --pin once, "
                               "then commit corpus/hashes/")
        pin_hashes(corpus_id, got, note)
        print(f"{corpus_id}: PINNED {got}")
        return got
    for name, h in got.items():
        if want.get(name) != h:
            raise HashMismatch(f"{corpus_id}/{name}: sha256 {h} != pinned {want.get(name)}")
    print(f"{corpus_id}: OK (sha256 verified: {', '.join(sorted(got))})")
    return got


def already_done(corpus_id, filenames):
    """Idempotence: every file exists and matches its pin."""
    want = expected_hashes(corpus_id)
    if not want:
        return False
    for name in filenames:
        p = os.path.join(DATA_DIR, corpus_id, name)
        if not os.path.exists(p) or sha256_file(p) != want.get(name):
            return False
    return True


def die(msg):
    print(f"FETCH FAILED: {msg}", file=sys.stderr)
    sys.exit(1)
