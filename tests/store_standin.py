"""Stand-ins for the checkpoint store's tests (test_hf_store.py, test_ckpt_courier.py, test_runner_store.py).

make_gen() writes one transactional generation as include/microtorch/checkpoint_txn.hpp does (three files and
MANIFEST.json under ckpt_gen/g<step>, the `live` symlink, the flat names as links through it, the generation before
kept and older ones removed). TRAINER is a stand-in for `mtstudio run SPEC` built on the same layout: its weights
and optimiser bytes at each step are a hash chain over the previous step's weights AND optimiser bytes, so a run that
resumes from a whole generation ends on the same bytes as an uninterrupted one, and a resume with a cold optimiser
or from other weights does not. It resumes as the trainer does: from the live generation (falling back to the one
before it when the live one fails its manifest), else from flat files that jobs_vm.restore() placed.
"""
import hashlib
import json
import os
import shutil

GEN_FORMAT = "paperkiln.ckpt_gen/1"
FILES = ("model.safetensors", "optim.safetensors", "state.txt")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(tag, size=4096):
    h = hashlib.sha256(tag).digest()
    return (h * (size // len(h) + 1))[:size]


def next_state(model, optim, step):
    """One training step of the stand-in: both files depend on both files of the step before."""
    o = blob(b"o" + hashlib.sha256(optim + model + str(step).encode()).digest(), 2048)
    m = blob(b"m" + hashlib.sha256(model + o).digest(), 4096)
    return m, o


def state_at(step, seed=0):
    m, o = blob(b"init-m%d" % seed), blob(b"init-o%d" % seed, 2048)
    for s in range(1, step + 1):
        m, o = next_state(m, o, s)
    return m, o


def commit(out, step, model, optim, state=None, flat_links=True):
    """Publish one generation: tmp dir, rename, the live pointer by one rename, flat links, prune to two."""
    gen = os.path.join(out, "ckpt_gen")
    os.makedirs(gen, exist_ok=True)
    name = f"g{step}"
    tmp = os.path.join(gen, name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    state = state or f"{step}\n" + json.dumps({"adamw_t": [step], "best_val": 1.0}) + "\n"
    files = {"model.safetensors": model, "optim.safetensors": optim, "state.txt": state.encode()}
    for f, b in files.items():
        with open(os.path.join(tmp, f), "wb") as fh:
            fh.write(b)
    with open(os.path.join(tmp, "MANIFEST.json"), "w") as fh:
        json.dump({"format": GEN_FORMAT, "step": step,
                   "files": {f: {"bytes": len(b), "sha256": sha(b)} for f, b in files.items()}}, fh)
    shutil.rmtree(os.path.join(gen, name), ignore_errors=True)
    os.rename(tmp, os.path.join(gen, name))
    try:
        prev = os.readlink(os.path.join(gen, "live"))
    except OSError:
        prev = None
    os.symlink(name, os.path.join(gen, "live.tmp"))
    os.replace(os.path.join(gen, "live.tmp"), os.path.join(gen, "live"))
    for f in FILES if flat_links else ():
        p = os.path.join(out, f)
        if not os.path.islink(p):
            if os.path.exists(p):
                os.remove(p)
            os.symlink(os.path.join("ckpt_gen", "live", f), p)
    for d in os.listdir(gen):
        if d.startswith("g") and d[1:].isdigit() and d not in (name, prev):
            shutil.rmtree(os.path.join(gen, d))


def make_gen(out, step, seed=0):
    """The generation of `step` in out, with the stand-in's bytes for that step."""
    m, o = state_at(step, seed)
    commit(out, step, m, o)
    return m, o


TRAINER = '''#!{py}
# Stand-in for `mtstudio run SPEC` with MICROTORCH_CHECKPOINT_TXN=1 (tests/store_standin.py).
import json, os, sys, time
sys.path.insert(0, {tests!r})
import store_standin as S
spec = json.load(open(sys.argv[2]))
out, tr = spec["out_dir"], spec["train"]
os.makedirs(out, exist_ok=True)
ev = open(os.path.join(out, "events.jsonl"), "a")
def emit(e):
    ev.write(json.dumps(e) + "\\n")
    ev.flush()
def whole(d, step):
    try:
        m = json.load(open(os.path.join(d, "MANIFEST.json")))
        return m["step"] == step and all(S.sha(open(os.path.join(d, f), "rb").read()) == m["files"][f]["sha256"] for f in S.FILES)
    except (OSError, ValueError, KeyError):
        return False
step, model, optim = 0, S.blob(b"init-m%d" % tr.get("seed", 0)), S.blob(b"init-o%d" % tr.get("seed", 0), 2048)
gen = os.path.join(out, "ckpt_gen")
if os.path.islink(os.path.join(gen, "live")):
    live = int(os.readlink(os.path.join(gen, "live"))[1:])
    for s in sorted((int(d[1:]) for d in os.listdir(gen) if d.startswith("g") and d[1:].isdigit()), reverse=True):
        if s <= live and whole(os.path.join(gen, "g%d" % s), s):
            step = s
            model = open(os.path.join(gen, "g%d" % s, "model.safetensors"), "rb").read()
            optim = open(os.path.join(gen, "g%d" % s, "optim.safetensors"), "rb").read()
            emit({{"event": "resume", "step": step, "optimizer": "restored", "from": "generation"}})
            break
    else:
        sys.exit("no whole generation; refusing to resume")
elif os.path.exists(os.path.join(out, "state.txt")):
    lines = open(os.path.join(out, "state.txt")).read().split("\\n")
    step = int(lines[0])
    model = open(os.path.join(out, "model.safetensors"), "rb").read()
    op = os.path.join(out, "optim.safetensors")
    rec = json.loads(lines[1]) if len(lines) > 1 and lines[1].strip() else {{}}
    warm = os.path.exists(op) and "adamw_t" in rec
    optim = open(op, "rb").read() if warm else S.blob(b"cold", 2048)
    emit({{"event": "resume", "step": step, "optimizer": "restored" if warm else "cold", "from": "flat"}})
every = tr.get("checkpoint_every", 1)
for s in range(step + 1, tr["steps"] + 1):
    time.sleep(float(os.environ.get("STANDIN_STEP_SLEEP", "0")))
    model, optim = S.next_state(model, optim, s)
    emit({{"event": "step", "step": s, "loss": int.from_bytes(model[:4], "big") / 2 ** 32}})
    if s % every == 0 or s == tr["steps"]:
        S.commit(out, s, model, optim)
        emit({{"event": "checkpoint", "step": s, "bytes": len(model) + len(optim)}})
open(os.path.join(out, spec["name"] + ".safetensors"), "wb").write(model)
json.dump({{"final_step": tr["steps"]}}, open(os.path.join(out, "result.json"), "w"))
'''


def write_trainer(path, py):
    with open(path, "w") as f:
        f.write(TRAINER.format(py=py, tests=os.path.dirname(os.path.abspath(__file__))))
    os.chmod(path, 0o755)
    return path
