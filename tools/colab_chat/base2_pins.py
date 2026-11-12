#!/usr/bin/env python3
"""Base 2's pins: sources, scenario engine, mixes' ids, panels and inputs (BASE2_PLAN.md section 5, item 9, in the author's
research tree). Claude Fable 5.2, 11 Nov 2026.

    python3 tools/colab_chat/base2_pins.py DIR MODE [--upload DIR] [--pins FILE] [--out COPY.json]
    python3 tools/colab_chat/base2_pins.py pin-engine ENGINE_DIR [--commit HASH] [--repin] [--pins FILE]   # at home
    python3 tools/colab_chat/base2_pins.py adopt OBSERVED.json [...] [--pins FILE]                         # at home
    python3 tools/colab_chat/base2_pins.py adopt-receipts BUILD_ART [--pins FILE] [--repo ROOT]          # at home

tools/colab_chat/base2_pins.json (or --pins, or BASE2_PINS) holds sha256 by group. A null pin is one not yet measured.

    MODE       reads                                                           a null pin
    vocab      vocab: the vocabulary in --upload                               refuses
    sources    files: S-full's outputs and the filtered dialogue sources,      refuses
               the hashes base A's data job met (tc_b1_pins.json "files")
    engine     engine: the scenario engine's files in DIR                      refuses
    stage1     sources: the joined FineWeb-Edu text, Simple English Wikipedia  records
               and the dialogue sources as the mixes read them
    build-N    N: the natural mix's ids                                        records
    build-S    S: the scenario mix's ids; panels                               records
    arm-N      N, panels; inputs: the vocabulary in --upload                   refuses
    arm-S      S, panels; the vocabulary                                       refuses
    arm-C      S, panels; the vocabulary and base A's final weights            refuses
    rehearsal  the vocabulary; rehearsal: the rehearsal's ids                  records the ids, refuses the vocabulary

Every mode writes what it hashed to DIR/base2_pins_observed.json under its own name and exits non-zero on any difference,
so an arm never starts on ids, panels, a vocabulary or weights that differ from the pinned ones. `adopt` copies observed
hashes into the null pins, refuses to change a pin that is set, and refuses a record that carries differences.
`adopt-receipts` is the adoption after the build, in one step (adopt_receipts below); `pin-engine` records the engine's files as they stand
(the engine was still being revised on 11 Nov 2026, so its pins are null until its author is done) and refuses to change
them without --repin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_FILES = ("banks.py", "engine.py", "frames.py", "generate.py", "rules.py", "validate.py")
# mode: ([(group, base)], record). base "dir" is DIR, "upload" is --upload.
MODES = {
    "vocab": ([("vocab", "upload")], False),
    "sources": ([("files", "dir")], False),
    "engine": ([("engine", "dir")], False),
    "stage1": ([("sources", "dir")], True),
    "build-N": ([("N", "dir")], True),
    "build-S": ([("S", "dir"), ("panels", "dir")], True),
    "arm-N": ([("N", "dir"), ("panels", "dir"), ("vocab", "upload")], False),
    "arm-S": ([("S", "dir"), ("panels", "dir"), ("vocab", "upload")], False),
    "arm-C": ([("S", "dir"), ("panels", "dir"), ("vocab", "upload"), ("a_weights", "upload")], False),
    "rehearsal": ([("vocab", "upload"), ("rehearsal", "dir")], True),
}
NEVER_RECORDED = ("vocab", "a_weights", "files", "engine")  # a null pin in these refuses in every mode


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pins_path(arg=None):
    return arg or os.environ.get("BASE2_PINS") or os.path.join(HERE, "base2_pins.json")


def group(pins, name):
    g = pins[name]
    return g["files"] if name == "engine" else g


def check(pins, d, mode, upload=None):
    """(differences, observed) for one mode."""
    groups, record = MODES[mode]
    bad, seen = [], {}
    for name, base in groups:
        root = d if base == "dir" else upload
        for rel, want in sorted(group(pins, name).items()):
            if rel.startswith("_"):
                continue
            p = os.path.join(root, rel)
            got = sha256(p) if os.path.isfile(p) else None
            seen[f"{name}/{rel}"] = got
            if got is None:
                bad.append(f"{name}/{rel}: missing")
            elif want is None:
                if not record or name in NEVER_RECORDED:
                    bad.append(f"{name}/{rel}: no pin; {mode} refuses")
            elif got != want:
                bad.append(f"{name}/{rel}: {got} against the pinned {want}")
        if name == "engine":  # a module beside the pinned ones could be imported in their place
            for f in engine_files(root):
                if f not in group(pins, name):
                    bad.append(f"engine/{f}: in the engine's folder and not pinned")
    return bad, seen


def write_pins(path, pins):
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(pins, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)


def engine_files(d):
    """The engine's Python files, tests left out: what base2_seed.py sends to the store."""
    return sorted(f for f in os.listdir(d) if f.endswith(".py") and not f.startswith("test_"))


def pin_engine(pins, d, commit=None, repin=False):
    names = engine_files(d)
    missing = [f for f in ENGINE_FILES if f not in names]
    if missing:
        raise SystemExit(f"not the scenario engine's folder: no {missing}")
    got = {f: sha256(os.path.join(d, f)) for f in names}
    old = pins["engine"]["files"]
    changed = [f for f in set(old) | set(got) if old.get(f) is not None and old.get(f) != got.get(f)]
    if changed and not repin:
        raise SystemExit(f"the engine is pinned and these files differ: {changed}; --repin to replace the pins")
    pins["engine"] = {"commit": commit, "files": got}
    return got


def adopt(pins, observed):
    """Observed hashes into null pins. Returns the pins set; raises on a set pin that differs. A record that carries
    differences (a missing file, a hash against a set pin) comes from a run that was refused: nothing of it is adopted."""
    carrying = {m: r["differences"] for m, r in observed.items() if r.get("differences")}
    if carrying:
        raise SystemExit("the observed record carries differences; nothing adopted:\n" +
                         "\n".join(f"{m}: {d}" for m, ds in sorted(carrying.items()) for d in ds))
    done = []
    for mode, rec in observed.items():
        for k, got in rec["observed"].items():
            name, rel = k.split("/", 1)
            g = group(pins, name)
            if got is None or rel not in g:
                continue
            if g[rel] is None:
                if name in NEVER_RECORDED:
                    continue
                g[rel] = got
                done.append(k)
            elif g[rel] != got:
                raise SystemExit(f"{k}: observed {got}, pinned {g[rel]}; a set pin is never replaced")
    return done


ARM_GROUPS = ("N", "S", "panels", "vocab", "a_weights")   # what the arms' modes read: none may be null at the arms
FW_HASHES = [f"fineweb_edu_shard{n}_v1.sha256" for n in (2, 3, 4, 5)]   # pinned on the build's first fetch


def adopt_receipts(pins, build, repo):
    """The adoption after the build, from its home output folder `build` (the runner's ART: build/data/ and
    build/load_probe/). Every check is made before anything is written:
    - each stage receipt's "built.pins" (the hashes the build took of what it published), adopted as `adopt` does;
    - the load probe's own hashes of S's ids and the three panels, as it pulled them back from the store
      (load_probe/loadprobe_pins.json), equal to stage 3's, key by key, with no difference recorded;
    - no pin an arm reads (ARM_GROUPS) left null;
    - the FineWeb-Edu shard 2 to 5 hash files the build's first fetch wrote (plan, work item 6: committed in
      corpus/hashes/, beside shards 0 and 1): written there, or equal to what is there.
    Returns (adopted pins, files written, relative to repo); raises SystemExit naming every refusal."""
    data, bad = os.path.join(build, "data"), []
    stages = {}
    for n in (1, 2, 3):
        p = os.path.join(data, f"base2_stage{n}.json")
        try:
            got = (json.load(open(p, encoding="utf-8")).get("built") or {}).get("pins")
        except (OSError, ValueError) as e:
            got, bad = None, bad + [f"stage {n}: receipt not read ({type(e).__name__}: {p})"]
        if got is not None and not isinstance(got, dict):
            got = None
        if got is None and not any(b.startswith(f"stage {n}:") for b in bad):
            bad.append(f"stage {n}: its receipt carries no built.pins")
        stages[n] = got or {}
    s3 = (stages[3].get("build-S") or {}).get("observed") or {}
    try:
        probe = json.load(open(os.path.join(build, "load_probe", "loadprobe_pins.json"), encoding="utf-8"))["build-S"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        probe, bad = {}, bad + [f"load probe: loadprobe_pins.json not read ({type(e).__name__})"]
    if probe.get("differences"):
        bad += [f"load probe: {d}" for d in probe["differences"]]
    want = {k: v for k, v in s3.items() if k.split("/", 1)[0] in ("S", "panels")}
    seen = {k: v for k, v in (probe.get("observed") or {}).items() if k.split("/", 1)[0] in ("S", "panels")}
    if not want:
        bad.append("stage 3: no build-S hashes of S's ids and the panels")
    for k in sorted(set(want) | set(seen)):
        if want.get(k) is None or want.get(k) != seen.get(k):
            bad.append(f"{k}: stage 3 hashed {want.get(k)}, the load probe pulled {seen.get(k)}")
    adopted = []
    for n in (1, 2, 3):
        try:
            adopted += adopt(pins, stages[n])
        except SystemExit as e:
            bad.append(f"stage {n}: {e}")
    null = [f"{g}/{k}" for g in ARM_GROUPS for k, v in group(pins, g).items() if v is None and not k.startswith("_")]
    bad += [f"{k}: still no pin after the adoption, and an arm reads it" for k in null]
    files = []
    for f in FW_HASHES:
        src, dst = os.path.join(data, f), os.path.join(repo, "corpus", "hashes", f)
        if not os.path.isfile(src):
            bad.append(f"{f}: not in the build's outputs")
            continue
        new = open(src, "rb").read()
        if os.path.exists(dst):
            if open(dst, "rb").read() != new:
                bad.append(f"corpus/hashes/{f}: committed, and the build's first fetch wrote another")
        else:
            files.append((dst, new))
    if bad:
        raise SystemExit("nothing adopted, nothing written:\n" + "\n".join(bad))
    for dst, new in files:
        with open(dst + ".tmp", "wb") as fh:
            fh.write(new)
        os.replace(dst + ".tmp", dst)
    return sorted(set(adopted)), [os.path.relpath(d, repo) for d, _ in files]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", help="the data dir, or pin-engine / adopt / adopt-receipts")
    ap.add_argument("mode", help="a mode, the engine dir, the first observed file, or the build's home output folder")
    ap.add_argument("more", nargs="*")
    ap.add_argument("--upload", default="/content/chat/upload")
    ap.add_argument("--pins")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(HERE)), help="adopt-receipts: the repo root")
    ap.add_argument("--out", help="a second copy of this mode's record (a job's own output folder)")
    ap.add_argument("--commit")
    ap.add_argument("--repin", action="store_true")
    a = ap.parse_args(argv)
    path = pins_path(a.pins)
    pins = json.load(open(path, encoding="utf-8"))
    if a.dir == "pin-engine":
        got = pin_engine(pins, a.mode, a.commit, a.repin)
        write_pins(path, pins)
        print(json.dumps({"engine": got, "commit": a.commit}, indent=1))
        return 0
    if a.dir == "adopt-receipts":
        done, written = adopt_receipts(pins, a.mode, a.repo)
        write_pins(path, pins)
        commit = sorted([os.path.relpath(os.path.abspath(path), a.repo), *written])
        print(json.dumps({"adopted": done, "written": written, "commit_files": commit}, indent=1))
        return 0
    if a.dir == "adopt":
        done = []
        for f in [a.mode, *a.more]:
            done += adopt(pins, json.load(open(f, encoding="utf-8")))
        write_pins(path, pins)
        print(json.dumps({"adopted": sorted(set(done))}, indent=1))
        return 0
    if a.mode not in MODES:
        raise SystemExit(f"mode is one of {', '.join(MODES)}")
    bad, seen = check(pins, a.dir, a.mode, a.upload)
    obs = os.path.join(a.dir, "base2_pins_observed.json")
    all_seen = {}
    if os.path.exists(obs):
        try:
            all_seen = json.load(open(obs))
        except ValueError:
            all_seen = {}
    all_seen[a.mode] = {"observed": seen, "differences": bad}
    with open(obs + ".tmp", "w") as f:
        json.dump(all_seen, f, indent=1)
    os.replace(obs + ".tmp", obs)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({a.mode: all_seen[a.mode]}, f, indent=1)
    if bad:
        raise SystemExit("not the pinned bytes:\n" + "\n".join(bad))
    print(json.dumps({"pins": "equal", "mode": a.mode, "files": len(seen)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
