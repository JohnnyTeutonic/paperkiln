#!/usr/bin/env python3
"""VM side of the checkpoint store (tools/colab_chat/hf_store.py): the courier and the restore of a new VM.

    python3 ckpt_courier.py restore --config STORE.json --jobs JOBS.json --chat /content/chat
    python3 ckpt_courier.py run     --config STORE.json --jobs JOBS.json --chat /content/chat

The jobs it serves are the training jobs named in the config's "branches" ({job: ckpt branch}); they must train with
MICROTORCH_CHECKPOINT_TXN=1, because only a committed generation (CHAT/<job>/ckpt_gen/g<step>, named by `live`) is
ever pushed, and a restore installs one.

run (a daemon the boot starts beside jobs_vm.py, in a process group of its own). Each tick, per job:
- marks first, and locally before any network call: a mark snapshot (CHAT/<job>/rung_<step>/, or a generation at
  one of the job's "snapshot_steps") is hard-linked into CHAT/<job>/courier_stage/, also while a push is in
  flight, so the rung job that later drops the weights cannot lose it; staged marks are pushed to main before the
  generation, once, and never removed;
- when `live` names a generation newer than the store's newest, its files are hard-linked into the staging folder
  and checked against the trainer's own MANIFEST.json (a save or a prune during staging shows here and the tick is
  skipped). The branch head is then read again: a generation written under a higher epoch (the runner's launch
  count), or by another boot at a step above what this courier knew, means a second session owns the branch, and
  this courier stops pushing for good, says so in its log and status, and writes CHAT/COURIER_FENCED. Otherwise
  the generation goes up as one commit against that head, with its campaign, job, spec and inputs hashes, epoch
  and boot id. A step already in the store is never written again.
- prune: only behind two generations this VM has verified, by its own push or by the download of its restore, and
  never at or above a mark that is staged and not yet pushed. Storage release is off unless the config says so.
A failed push is retried with a growing wait (at most ten minutes) and always with the newest generation; the
trainer never waits on any of this. Every store call runs in a child with a time limit.
Status: CHAT/courier_status.json. Log: CHAT/courier.log. One courier per VM (CHAT/courier.pid, created exclusively).

restore (the boot, before jobs_vm.py). Per unfinished job: the newest generation of its branch that verifies by
sha256 is downloaded to a temporary folder and installed as a trainer generation: CHAT/<job>/ckpt_gen/g<step>
renamed into place whole, then the `live` pointer by one rename, which is the commit point. A boot killed before it
leaves the earlier state and the next boot installs again; the trainer validates the live generation by its own
manifest before it resumes, optimiser included. This VM's own checkpoint counts only as a live generation that
passes that manifest. The restore fails, and the boot with it, when the store cannot be read, when no generation
verifies, when the branch lists nothing at or above the config's "min_step" for the job (the step home has seen
pushed), when the generation is of another campaign, job, spec or inputs, or when the branch is empty and the job is
not in "first_start". Mark snapshots that an unfinished rung job needs come from main into CHAT/<job>/rung_<step>/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_store  # noqa: E402
from hf_store import GEN_FILES, MARK_FILES, StoreError, VerifyError, bounded, file_step, gen_step  # noqa: E402,F401

BACKOFF_MAX = 600
FLAT = ("model.safetensors", "optim.safetensors", "state.txt")


def status(chat, name):
    p = os.path.join(chat, f"STATUS_{name}")
    return open(p).read().strip() if os.path.exists(p) else "NONE"


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def copy_events(src, dst, step):
    """events.jsonl in whole lines, cut after the `checkpoint` event of `step` when there is one, so a resume's
    record does not carry steps the checkpoint never held."""
    try:
        data = open(src, "rb").read()
    except OSError:
        return False
    end = data.rfind(b"\n") + 1
    cut, pos = end, end
    for _ in range(50000):  # the event is near the end; never scan a whole run's record
        if pos <= 0:
            break
        start = data.rfind(b"\n", 0, pos - 1) + 1
        line = data[start:pos]
        if b'"checkpoint"' in line:
            try:
                e = json.loads(line)
            except ValueError:
                e = {}
            if e.get("event") == "checkpoint" and isinstance(e.get("step"), int):
                if e["step"] == step:
                    cut = pos
                if e["step"] <= step:
                    break
        pos = start
    with open(dst, "wb") as f:
        f.write(data[:cut])
    return True


def identity(cfg, chat, job):
    """What every generation of this job carries and every restore checks."""
    try:
        inputs = json.load(open(os.path.join(chat, "store_inputs.json"))).get("inputs_sha256")
    except (OSError, ValueError):
        inputs = None
    return {"campaign": cfg.get("campaign"), "job": job["name"], "spec_sha256": hf_store.spec_hash(job),
            "inputs_sha256": inputs}


def boot_id(chat):
    """This boot's id: written once by the boot, made here when a test or a hand run has none."""
    p = os.path.join(chat, "boot_id")
    if not os.path.exists(p):
        os.makedirs(chat, exist_ok=True)
        with open(p, "w") as f:
            f.write(uuid.uuid4().hex)
    return open(p).read().strip()


class Courier:
    def __init__(self, store, chat, jobs, branches, log=print, min_interval=0, push_timeout=None, keep=2,
                 clock=time.time, cfg=None, boot=None):
        """jobs: {name: job dict} of the jobs file; branches: {training job: ckpt branch}; cfg: the store config."""
        self.store, self.chat, self.jobs, self.branches = store, chat, jobs, dict(branches)
        self._log, self.min_interval, self.push_timeout, self.keep, self.clock = log, min_interval, push_timeout, keep, clock
        self.cfg = cfg or {}
        self.epoch, self.boot = int(self.cfg.get("epoch") or 0), boot or boot_id(chat)
        self.s = {n: {"branch": b, "head": None, "known": False, "marks": None, "failures": 0, "next_try": 0,
                      "last_error": None, "pushed_at": None, "live_step": None, "refused": None, "pushes": 0,
                      "fenced": None, "verified": set(self.restored(n))}
                  for n, b in self.branches.items()}

    def log(self, msg):
        self._log(hf_store.redact(msg, self.store.b.token))

    def call(self, fn):
        return bounded(fn, self.push_timeout, self.store.b.token, while_waiting=self.stage_all_marks)

    def out(self, name):
        return os.path.join(self.chat, name)

    def stage(self, name):
        return os.path.join(self.chat, name, "courier_stage")

    def restored(self, name):
        """Steps this VM has verified before this courier started: the restore's download, and an earlier courier's
        own pushes on this same boot."""
        steps = []
        try:
            rec = json.load(open(os.path.join(self.chat, "store_restore", f"{name}.json")))
            if rec.get("verified_step") is not None and rec.get("boot_id") == self.boot:
                steps.append(rec["verified_step"])
        except (OSError, ValueError):
            pass
        try:
            st = json.load(open(os.path.join(self.chat, "courier_status.json")))
            if st.get("boot_id") == self.boot:
                steps += st["jobs"][name].get("verified", [])
        except (OSError, ValueError, KeyError):
            pass
        return steps

    # -- staging (local only: no call here may touch the network)
    def live(self, name):
        link = os.path.join(self.out(name), "ckpt_gen", "live")
        try:
            target = os.readlink(link)
        except OSError:
            return None
        step = gen_step(os.path.basename(target))
        return None if step is None else (step, os.path.join(self.out(name), "ckpt_gen", os.path.basename(target)))

    def stage_generation(self, name, step, gen_dir):
        """The generation's files under courier_stage/g<step>, whole by the trainer's manifest, or VerifyError."""
        dst = os.path.join(self.stage(name), f"g{step}")
        if os.path.isdir(dst):
            try:
                hf_store.check_generation(dst, step)
                return dst  # staged for a push that failed: the same bytes, pushed again
            except VerifyError:
                shutil.rmtree(dst, ignore_errors=True)
        tmp = dst + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        try:
            for f in GEN_FILES:
                link_or_copy(os.path.join(gen_dir, f), os.path.join(tmp, f))
            hf_store.check_generation(tmp, step)
        except (OSError, VerifyError) as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise VerifyError(f"generation g{step} changed while it was staged ({e})") from None
        copy_events(os.path.join(self.out(name), "events.jsonl"), os.path.join(tmp, "events.jsonl"), step)
        os.replace(tmp, dst)
        for d in os.listdir(self.stage(name)):  # an older staged generation is never pushed after a newer one
            if gen_step(d.split(".")[0]) is not None and d != f"g{step}":
                shutil.rmtree(os.path.join(self.stage(name), d), ignore_errors=True)
        return dst

    def stage_mark(self, name, step, src):
        """Hard links of a mark's two files under courier_stage/rung_<step>; False when src is not at that step."""
        dst = os.path.join(self.stage(name), f"rung_{step}")
        if os.path.isdir(dst):
            return True
        tmp = f"{dst}.tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        try:
            for f in MARK_FILES:
                link_or_copy(os.path.join(src, f), os.path.join(tmp, f))
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        if file_step(os.path.join(tmp, "state.txt")) != step:
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        os.replace(tmp, dst)
        return True

    def stage_marks(self, name):
        """Every mark on disk that is not known to be in the store, linked into the staging folder now."""
        out, done = self.out(name), self.s[name]["marks"] or set()
        if not os.path.isdir(out):
            return
        os.makedirs(self.stage(name), exist_ok=True)
        for d in sorted(os.listdir(out)):
            m = re.fullmatch(r"rung_(\d+)", d)
            if m and int(m.group(1)) not in done:
                self.stage_mark(name, int(m.group(1)), os.path.join(out, d))
        live = self.live(name)  # a generation at a mark step holds the same weights: no wait for jobs_vm's copy
        if live and live[0] in self.jobs[name].get("snapshot_steps", []) and live[0] not in done:
            self.stage_mark(name, live[0], live[1])

    def stage_all_marks(self):
        for name in self.branches:
            try:
                self.stage_marks(name)
            except OSError:
                pass

    def staged_marks(self, name):
        st = self.stage(name)
        return sorted(int(m.group(1)) for m in (re.fullmatch(r"rung_(\d+)", d) for d in os.listdir(st)) if m) \
            if os.path.isdir(st) else []

    # -- one pass
    def fail(self, s, what, e, marks=False):
        """A failure and its growing wait. Marks wait on their own clock: main refusing a mark never holds a
        generation back, and the other way round."""
        n, t = ("mark_failures", "mark_next") if marks else ("failures", "next_try")
        s[n] = s.get(n, 0) + 1
        wait = min(BACKOFF_MAX, 15 * 2 ** min(s[n], 10))
        s[t] = self.clock() + wait
        if marks:
            self.log(hf_store.redact(f"{what}: {type(e).__name__}: {e}", self.store.b.token)[:400]
                     + f"; kept staged, retry in {wait} s (failure {s[n]})")
            return
        s["last_error"] = hf_store.redact(f"{what}: {type(e).__name__}: {e}", self.store.b.token)[:400]
        self.log(f"{s['last_error']}; training goes on, retry in {wait} s (failure {s['failures']})")

    def fence(self, name, head, why):
        s = self.s[name]
        s["fenced"] = {"why": why, "head_step": head.get("step"), "head_epoch": head.get("epoch"),
                       "head_boot": head.get("boot_id"), "at": time.time()}
        with open(os.path.join(self.chat, "COURIER_FENCED"), "a") as f:
            f.write(json.dumps({"job": name, **s["fenced"]}) + "\n")
        self.log(f"{name}: FENCED: {why}. Another session owns {s['branch']} (its newest generation g{head.get('step')}, "
                 f"epoch {head.get('epoch')}, boot {str(head.get('boot_id'))[:8]}; this courier is epoch {self.epoch}, boot "
                 f"{self.boot[:8]}). Nothing more is pushed from this VM.")

    def foreign(self, name, head):
        """Why the branch belongs to another session now, or None."""
        s = self.s[name]
        if head.get("step") is None:
            return None
        if head.get("epoch") is not None and head["epoch"] > self.epoch:
            return f"a generation of epoch {head['epoch']} is in the store"
        if head.get("boot_id") not in (None, self.boot) and s["known"] and head["step"] > (s["head"] or -1):
            return f"another boot has pushed g{head['step']} since this courier last looked"
        return None

    def contact(self, name):
        """What the store already holds for this job, read once; False while the store cannot be read."""
        s = self.s[name]
        if s["known"]:
            return True
        if self.clock() < s["next_try"]:
            return False
        try:
            self.call(lambda: self.store.b.ensure_branch(s["branch"]))
            head = self.call(lambda: self.store.head(s["branch"]))
            s["marks"] = set(self.call(lambda: self.store.mark_steps(name)))
        except Exception as e:  # noqa: BLE001
            self.fail(s, f"{name}: store not read", e)
            return False
        why = self.foreign(name, head)
        s["head"], s["known"], s["failures"] = head["step"], True, 0
        self.log(f"{name}: store branch {s['branch']} holds step {s['head']}, marks {sorted(s['marks'])}")
        if why:
            self.fence(name, head, why)
        return True

    def push_marks(self, name):
        s = self.s[name]
        if not self.contact(name) or s["fenced"] or self.clock() < s.get("mark_next", 0):
            return
        ident = identity(self.cfg, self.chat, self.jobs[name])
        for step in self.staged_marks(name):
            src = os.path.join(self.stage(name), f"rung_{step}")
            try:
                if step not in s["marks"]:
                    new = self.call(lambda step=step, src=src: self.store.push_mark(name, step, src, ident))
                    self.log(f"{name}: mark at step {step} " + ("pushed" if new else "already in the store"))
                s["marks"].add(step)
                s["mark_failures"] = 0
                shutil.rmtree(src, ignore_errors=True)
            except Exception as e:  # noqa: BLE001
                self.fail(s, f"{name}: mark {step} not pushed", e, marks=True)
                return

    def generation(self, name):
        s = self.s[name]
        live = self.live(name)
        if live is None:
            return
        step, gen_dir = live
        s["live_step"] = step
        if not self.contact(name) or s["fenced"]:
            return
        head = s["head"]
        if head is not None and step < head:
            if s["refused"] != step:
                s["refused"] = step
                self.log(f"{name}: live generation g{step} is behind the store's g{head}; refused, nothing pushed")
            return
        s["refused"] = None
        if step == head or self.clock() < s["next_try"]:
            return
        ended = status(self.chat, name) in ("FINISHED", "FAILED")
        if self.min_interval and s["pushed_at"] and self.clock() - s["pushed_at"] < self.min_interval and not ended:
            return
        try:
            staged = self.stage_generation(name, step, gen_dir)
        except VerifyError as e:
            self.log(f"{name}: {e}; next tick")
            return
        if step in self.jobs[name].get("snapshot_steps", []) and step not in s["marks"]:
            self.stage_mark(name, step, staged)  # the mark goes first; until it is in the store the prune keeps g<step>
            self.push_marks(name)
        t0 = self.clock()
        try:
            now = self.call(lambda: self.store.head(s["branch"]))  # the branch as it is now, not as it was at start
            why = self.foreign(name, now)
            if why:
                self.fence(name, now, why)
                return
            if step in now["steps"]:  # written once: a step in the store is left as it is
                s["head"] = max(step, now["step"])
                shutil.rmtree(staged, ignore_errors=True)
                self.log(f"{name}: g{step} is already in the store; not written again")
                return
            ident = {**identity(self.cfg, self.chat, self.jobs[name]), "epoch": self.epoch, "boot_id": self.boot}
            man = self.call(lambda: self.store.push_generation(s["branch"], name, step, staged, ident, now["commit"]))
        except Exception as e:  # noqa: BLE001 a ConflictError too: the next pass reads the head again
            self.fail(s, f"{name}: generation g{step} not pushed", e)
            return
        mb = sum(f["bytes"] for f in man["files"].values()) / 1e6
        s.update(head=step, failures=0, last_error=None, pushed_at=self.clock(), pushes=s["pushes"] + 1)
        s["verified"].add(step)  # these bytes were hashed here and the commit holding them was accepted
        shutil.rmtree(staged, ignore_errors=True)
        self.log(f"{name}: generation g{step} pushed ({mb:.1f} MB in {self.clock() - t0:.0f} s)")
        try:
            protect = [m for m in self.staged_marks(name) if m not in s["marks"]]
            gone = self.call(lambda: self.store.prune(s["branch"], self.keep, sorted(s["verified"]), protect))
            if gone:
                self.log(f"{name}: generations {gone} removed from the store")
                if self.store.reclaim_on:
                    n = self.call(lambda: self.store.release(self.boot))
                    self.log(f"{name}: storage release: " + ("another writer holds the lock" if n is None else f"{n} objects deleted"))
        except Exception as e:  # noqa: BLE001 the verified generations are in the store; the next push prunes again
            self.log(f"{name}: prune or release failed ({type(e).__name__}: {e}); will try after the next push")

    def tick(self):
        self.stage_all_marks()  # before any network call
        for name in self.branches:
            for step in (self.push_marks, self.generation, self.push_marks):
                try:
                    step(name)
                except Exception as e:  # noqa: BLE001 nothing here may end the daemon
                    self.log(f"{name}: courier pass failed ({type(e).__name__}: {e})")
        self.write_status()

    def idle(self):
        """True when every job has ended and all it left is in the store (or this courier is fenced off it)."""
        for name, s in self.s.items():
            if status(self.chat, name) not in ("FINISHED", "FAILED"):
                return False
            if s["fenced"]:
                continue
            live = self.live(name)
            if live and (not s["known"] or s["head"] is None or live[0] > s["head"]):
                return False
            if self.staged_marks(name):
                return False
        return True

    def write_status(self):
        rec = {"updated": time.time(), "boot_id": self.boot, "epoch": self.epoch, "huggingface_hub": hf_store.hub_version(),
               "jobs": {n: {"branch": s["branch"], "pushed_step": s["head"], "pushed_at": s["pushed_at"],
                            "live_step": s["live_step"], "pushes": s["pushes"], "marks": sorted(s["marks"] or []),
                            "staged_marks": self.staged_marks(n), "verified": sorted(s["verified"]),
                            "failures": s["failures"], "last_error": s["last_error"], "refused_step": s["refused"],
                            "fenced": s["fenced"]} for n, s in self.s.items()}}
        p = os.path.join(self.chat, "courier_status.json")
        with open(p + ".tmp", "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(p + ".tmp", p)


# ---------------------------------------------------------------- restore
def local_step(chat, name):
    """The step this VM holds for the job: its live generation, and only when that generation is whole by the
    trainer's manifest. A state.txt number alone is never trusted."""
    gen = os.path.join(chat, name, "ckpt_gen")
    try:
        step = gen_step(os.path.basename(os.readlink(os.path.join(gen, "live"))))
        hf_store.check_generation(os.path.join(gen, f"g{step}"), step)
        return step
    except (OSError, VerifyError, TypeError):
        return 0


def install(chat, name, src, step):
    """A verified generation folder becomes the job's live trainer generation. The folder is renamed into ckpt_gen
    whole; the `live` pointer then moves by one rename, the commit point; the flat names follow as links through it
    (the trainer makes them itself when they are missing)."""
    out = os.path.join(chat, name)
    gen = os.path.join(out, "ckpt_gen")
    os.makedirs(gen, exist_ok=True)
    ev = os.path.join(src, "events.jsonl")
    if os.path.exists(ev):
        os.replace(ev, os.path.join(out, "events.jsonl"))
    if os.path.exists(os.path.join(src, hf_store.MANIFEST)):
        os.remove(os.path.join(src, hf_store.MANIFEST))
    dst = os.path.join(gen, f"g{step}")
    shutil.rmtree(dst, ignore_errors=True)  # never live here: local_step() said this VM is behind `step`
    os.rename(src, dst)
    hf_store.check_generation(dst, step)
    fd = os.open(gen, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp = os.path.join(gen, "live.install")
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(f"g{step}", tmp)
    os.replace(tmp, os.path.join(gen, "live"))  # the commit point
    for f in FLAT:
        p = os.path.join(out, f)
        if os.path.lexists(p):
            os.remove(p)
        os.symlink(os.path.join("ckpt_gen", "live", f), p)


def restore(store, chat, jobs, branches, log=print, cfg=None):
    """See the module docstring. Returns {job: record}; raises when a job's branch cannot be read, verified or
    trusted. Transport failures are retried inside the store (its `tries` and `delay`), then raised."""
    cfg = cfg or {}
    out = {}
    boot = boot_id(chat)
    os.makedirs(os.path.join(chat, "store_restore"), exist_ok=True)
    for name, branch in branches.items():
        if status(chat, name) in ("FINISHED", "FAILED"):
            continue
        ident = identity(cfg, chat, jobs[name])
        floor = int((cfg.get("min_step") or {}).get(name) or 0)
        have = local_step(chat, name)
        stage = os.path.join(chat, "store_restore", name)
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        t0 = time.time()
        step, rejected = store.pull_generation(branch, stage, expect=ident, min_step=floor)
        rec = {"job": name, "branch": branch, "store_step": step, "local_step": have, "rejected": rejected,
               "placed": False, "verified_step": step, "boot_id": boot, "min_step": floor, "at": time.time()}
        if step is None:
            if name not in (cfg.get("first_start") or []):
                raise hf_store.IdentityError(f"{name}: {branch} holds no generation and a first start is not admitted "
                                             "for this job; refusing to start it from step 0")
            log(f"{name}: the store holds no generation on {branch}; first start admitted, the job starts from its spec")
        elif step <= have:
            log(f"{name}: this VM holds a whole generation at step {have}, the store step {step}; nothing installed")
        else:
            install(chat, name, os.path.join(stage, f"g{step}"), step)
            rec["placed"] = True
            log(f"{name}: generation g{step} of {branch} verified and installed for the resume "
                f"({time.time() - t0:.0f} s" + (f"; refused first: {rejected}" if rejected else "") + ")")
        shutil.rmtree(stage, ignore_errors=True)
        with open(os.path.join(chat, "store_restore", f"{name}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        out[name] = rec
    # Mark snapshots for rung jobs that have not run: jobs_vm.py fails a rung whose snapshot is missing ("rung lost").
    listed = {}
    for j in jobs.values():
        ro = j.get("rung_of")
        if not ro or ro["job"] not in branches or status(chat, j["name"]) in ("FINISHED", "FAILED"):
            continue
        pt, step = ro["job"], int(ro["step"])
        dst = os.path.join(chat, pt, f"rung_{step}")
        if os.path.exists(os.path.join(dst, "model.safetensors")):
            continue
        if pt not in listed:
            listed[pt] = store.mark_steps(pt)
        if step not in listed[pt]:
            continue
        shutil.rmtree(dst + ".tmp", ignore_errors=True)
        ident = identity(cfg, chat, jobs[pt])
        store.pull_mark(pt, step, dst + ".tmp", expect={"campaign": ident["campaign"], "job": pt})
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(dst + ".tmp", dst)
        log(f"{pt}: mark snapshot rung_{step} verified and placed for {j['name']}")
    return out


def store_jobs(cfg, jobs):
    """{training job: branch} from the config, for jobs this jobs file holds."""
    return {n: b for n, b in (cfg.get("branches") or {}).items() if n in jobs and "cmd" not in jobs[n]}


def take_pid(chat):
    """One courier per VM. The file is created exclusively; one left by a dead process is replaced, once."""
    p = os.path.join(chat, "courier.pid")
    for _ in range(2):
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return
        except FileExistsError:
            try:
                other = int(open(p).read())
            except (OSError, ValueError):
                other = None
            if other and other != os.getpid() and os.path.exists(f"/proc/{other}"):
                sys.exit(f"another courier (pid {other}) is running; refusing")
            try:
                os.remove(p)
            except OSError:
                pass
    sys.exit("courier pid file could not be taken; refusing")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "restore"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--chat", default="/content/chat")
    ap.add_argument("--tick", type=float, default=20)
    ap.add_argument("--min-interval", type=float, default=None,
                    help="seconds between generation pushes, for a store that limits commits (default: the config's "
                         "\"min_push_interval_s\", else 0: every save)")
    ap.add_argument("--push-timeout", type=float, default=1800, help="seconds before a store call is ended; 0: none")
    ap.add_argument("--retry-delay", type=float, default=20)
    ap.add_argument("--limit", type=float, default=0, help="restore: seconds before the whole restore is ended; 0: none")
    a = ap.parse_args(argv)
    cfg = hf_store.load_config(a.config)
    jobs = {j["name"]: j for j in json.load(open(a.jobs))["jobs"]}
    branches = store_jobs(cfg, jobs)
    logf = os.path.join(a.chat, "courier.log")

    def log(msg):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
        with open(logf, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    token = None
    try:
        store = hf_store.open_store(cfg, log=log, delay=a.retry_delay)
        token = store.b.token
        if a.cmd == "restore":
            bounded(lambda: len(restore(store, a.chat, jobs, branches, log=log, cfg=cfg)), a.limit, token)
            return 0
    except Exception as e:  # noqa: BLE001 one redacted line; the boot's ERR trap does the rest
        log(hf_store.redact(f"courier {a.cmd} failed: {type(e).__name__}: {e}", token))
        return 1
    take_pid(a.chat)
    stop = []

    def ended(*_):
        stop.append(1)
        hf_store.end_children()  # a push in flight holds the token in memory: it goes with the daemon

    signal.signal(signal.SIGTERM, ended)
    gap = a.min_interval if a.min_interval is not None else float(cfg.get("min_push_interval_s") or 0)
    c = Courier(store, a.chat, jobs, branches, log=log, min_interval=gap, push_timeout=a.push_timeout or None, cfg=cfg)
    log(f"courier up: {branches}, epoch {c.epoch}, boot {c.boot[:8]}, huggingface_hub {hf_store.hub_version()}"
        + (f", at most one generation push every {gap:.0f} s" if gap else ""))
    quiet = 0
    while not stop:
        c.tick()
        quiet = quiet + 1 if c.idle() else 0
        if quiet >= 2:
            log("every job has ended and its last generation and marks are in the store; courier done")
            break
        time.sleep(a.tick)
    c.write_status()
    try:
        os.remove(os.path.join(a.chat, "courier.pid"))
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
