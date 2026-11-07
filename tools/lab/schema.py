"""Canonical JSON, hashes and field checks shared by the lab. Claude Opus 5.5, 7 Nov 2026. Standard library only."""
from __future__ import annotations

import hashlib
import json
import math
import re

SHA = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
GENESIS = "0" * 64


class Malformed(ValueError):
    """A record, receipt or configuration that fails its schema. Nothing malformed is ever stored."""


def canon(obj) -> str:
    """One text for one value: sorted keys, no spaces, NaN and infinity refused."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(obj) -> str:
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(d, key, kind, where):
    if not isinstance(d, dict):
        raise Malformed(f"{where}: not an object")
    if key not in d:
        raise Malformed(f"{where}: missing {key}")
    v = d[key]
    if kind is float:
        return number(v, f"{where}.{key}")
    if kind is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise Malformed(f"{where}.{key}: not an integer")
        return v
    if not isinstance(v, kind):
        raise Malformed(f"{where}.{key}: not {getattr(kind, '__name__', kind)}")
    if kind is str and not v.strip():
        raise Malformed(f"{where}.{key}: empty")
    return v


def number(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Malformed(f"{where}: not a finite number")
    return float(v)


def nonneg(v, where):
    x = number(v, where)
    if x < 0:
        raise Malformed(f"{where}: negative")
    return x


def sha_field(d, key, where):
    v = need(d, key, str, where)
    if not SHA.match(v):
        raise Malformed(f"{where}.{key}: not a sha256")
    return v


def one_of(v, allowed, where):
    if v not in allowed:
        raise Malformed(f"{where}: {v!r} is not one of {sorted(allowed)}")
    return v


def rounded(x, digits=6):
    """Floats enter a decision record at fixed precision, so its hash does not depend on the last bit of a platform's libm."""
    return round(float(x), digits)
