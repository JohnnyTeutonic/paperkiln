#!/usr/bin/env python3
"""VM side of the checkpoint store (tools/colab_chat/hf_store.py): the courier and the restore of a new VM.

    python3 ckpt_courier.py restore --config STORE.json --jobs JOBS.json --chat /content/chat
    python3 ckpt_courier.py run     --config STORE.json --jobs JOBS.json --chat /content/chat

The jobs it serves are the training jobs named in the config's "branches" ({job: ckpt branch}); they must train with
MICROTORCH_CHECKPOINT_TXN=1, because only a committed generation (CHAT/<job>/ckpt_gen/g<step>, named by `live`) is
ever pushed.

run (a daemon the boot starts beside jobs_vm.py). Each tick, per job:
- a mark snapshot (CHAT/<job>/rung_<step>/, or a generation at one of the job's "snapshot_steps") is hard-linked into
  CHAT/<job>/courier_stage/ at once, so the rung job that later drops the weights cannot lose it, then pushed to
  main, once, and never removed;
- when `live` names a generation newer than the store's, its files are hard-linked into the staging folder, checked
  against the trainer's own MANIFEST.json (a save or a prune during staging shows here and the tick is skipped),
  and pushed with events.jsonl as one commit; then generations older than the newest two are removed from the store.
  A step behind the store's is refused. A failed push is retried with a growing wait (at most ten minutes) and
  always with the newest generation; the trainer never waits on any of this.
Status: CHAT/courier_status.json. Log: CHAT/courier.log. One courier per VM (CHAT/courier.pid).

restore (the boot, before jobs_vm.py). Per unfinished job: the newest generation of its branch that verifies by
sha256 (the one before it when the newest does not) goes to CHAT/upload/<job>/ as jobs_vm.restore() reads it
(model.safetensors.part00, optim.safetensors.part00, state.txt, events.jsonl), then READY; nothing is placed when
this VM's own checkpoint is at or past that step. It fails (and the boot with it) when the store cannot be read or
holds generations of which none verifies: a job never starts from step 0 because a pull failed. Mark snapshots
that an unfinished rung job needs come from main into CHAT/<job>/rung_<step>/.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_store  # noqa: E402
from hf_store import GEN_FILES, MARK_FILES, StoreError, VerifyError, file_step, gen_step  # noqa: E402

BACKOFF_MAX = 600


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


def bounded(fn, timeout, token=None):
    """fn() in a forked child with a time limit, so a hung upload cannot stop the courier. Its JSON-able result,
    or StoreError carrying the child's (redacted) error text."""
    if not timeout:
        return fn()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = {"ok": fn()}
        except BaseException as e:  # noqa: BLE001
            out = {"err": hf_store.redact(f"{type(e).__name__}: {e}", token)[:2000]}
        try:
            os.write(w, json.dumps(out).encode())
        finally:
            os._exit(0)
    os.close(w)
    t0 = time.time()
    try:
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.time() - t0 > timeout:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                raise StoreError(f"store call over {timeout} s: ended")
            time.sleep(0.2)
        buf = b""
        while True:
            b = os.read(r, 1 << 16)
            if not b:
                break
            buf += b
    finally:
        os.close(r)
    try:
        out = json.loads(buf)
    except ValueError:
        raise StoreError("store call ended without a result") from None
    if "err" in out:
        raise StoreError(out["err"])
    return out["ok"]


class Courier:
    def __init__(self, store, chat, jobs, branches, log=print, min_interval=0, push_timeout=None, keep=2,
                 clock=time.time):
        """jobs: {name: job dict} of the jobs file; branches: {training job: ckpt branch}."""
        self.store, self.chat, self.jobs, self.branches = store, chat, jobs, dict(branches)
        self._log, self.min_interval, self.push_timeout, self.keep, self.clock = log, min_interval, push_timeout, keep, clock
        self.s = {n: {"branch": b, "head": None, "known": False, "marks": None, "failures": 0, "next_try": 0,
                      "last_error": None, "pushed_at": None, "live_step": None, "refused": None, "pushes": 0}
                  for n, b in self.branches.items()}

    def log(self, msg):
        self._log(hf_store.redact(msg, self.store.b.token))

    def call(self, fn):
        return bounded(fn, self.push_timeout, self.store.b.token)

    def out(self, name):
        return os.path.join(self.chat, name)

    def stage(self, name):
        return os.path.join(self.chat, name, "courier_stage")

    # -- staging
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
        tmp = dst + ".tmp"
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

    # -- one pass
    def fail(self, s, what, e):
        s["failures"] += 1
        wait = min(BACKOFF_MAX, 15 * 2 ** min(s["failures"], 10))
        s["next_try"] = self.clock() + wait
        s["last_error"] = hf_store.redact(f"{what}: {type(e).__name__}: {e}", self.store.b.token)[:400]
        self.log(f"{s['last_error']}; training goes on, retry in {wait} s (failure {s['failures']})")

    def contact(self, name):
        """What the store already holds for this job, read once; False while the store cannot be read."""
        s = self.s[name]
        if s["known"]:
            return True
        if self.clock() < s["next_try"]:
            return False
        try:
            self.call(lambda: self.store.b.ensure_branch(s["branch"]))
            # The newest generation that is whole in the listing: a damaged newer one is pushed over, not skipped.
            s["head"] = (self.call(lambda: self.store.whole_steps(s["branch"])) or [None])[-1]
            s["marks"] = set(self.call(lambda: self.store.mark_steps(name)))
        except Exception as e:  # noqa: BLE001
            self.fail(s, f"{name}: store not read", e)
            return False
        s["known"], s["failures"] = True, 0
        self.log(f"{name}: store branch {s['branch']} holds step {s['head']}, marks {sorted(s['marks'])}")
        return True

    def marks(self, name):
        s, out = self.s[name], self.out(name)
        os.makedirs(self.stage(name), exist_ok=True)
        done = s["marks"] or set()
        for d in sorted(os.listdir(out)) if os.path.isdir(out) else []:
            m = re.fullmatch(r"rung_(\d+)", d)
            if m and int(m.group(1)) not in done:
                self.stage_mark(name, int(m.group(1)), os.path.join(out, d))
        if not self.contact(name) or self.clock() < s["next_try"]:
            return
        for d in sorted(os.listdir(self.stage(name))):
            m = re.fullmatch(r"rung_(\d+)", d)
            if not m:
                continue
            step, src = int(m.group(1)), os.path.join(self.stage(name), d)
            try:
                if step not in s["marks"]:
                    new = self.call(lambda step=step, src=src: self.store.push_mark(name, step, src))
                    self.log(f"{name}: mark at step {step} " + ("pushed" if new else "already in the store"))
                s["marks"].add(step)
                shutil.rmtree(src, ignore_errors=True)
            except Exception as e:  # noqa: BLE001
                self.fail(s, f"{name}: mark {step} not pushed", e)
                return

    def generation(self, name):
        s = self.s[name]
        live = self.live(name)
        if live is None:
            return
        step, gen_dir = live
        s["live_step"] = step
        if not self.contact(name):
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
            self.stage_mark(name, step, staged)  # the same weights: the mark does not wait for jobs_vm's copy
        t0 = self.clock()
        try:
            man = self.call(lambda: self.store.push_generation(s["branch"], name, step, staged))
        except Exception as e:  # noqa: BLE001
            self.fail(s, f"{name}: generation g{step} not pushed", e)
            return
        mb = sum(f["bytes"] for f in man["files"].values()) / 1e6
        s.update(head=step, failures=0, last_error=None, pushed_at=self.clock(), pushes=s["pushes"] + 1)
        shutil.rmtree(staged, ignore_errors=True)
        self.log(f"{name}: generation g{step} pushed ({mb:.1f} MB in {self.clock() - t0:.0f} s)")
        try:
            gone = self.call(lambda: self.store.prune(s["branch"], self.keep))
            if gone:
                self.log(f"{name}: generations {gone} removed from the store")
        except Exception as e:  # noqa: BLE001 the newest two are in the store; the next push prunes again
            self.log(f"{name}: prune failed ({type(e).__name__}: {e}); will try after the next push")

    def tick(self):
        for name in self.branches:
            for step in (self.generation, self.marks):
                try:
                    step(name)
                except Exception as e:  # noqa: BLE001 nothing here may end the daemon
                    self.log(f"{name}: courier pass failed ({type(e).__name__}: {e})")
        self.write_status()

    def idle(self):
        """True when every job has ended and all it left is in the store."""
        for name, s in self.s.items():
            if status(self.chat, name) not in ("FINISHED", "FAILED"):
                return False
            live = self.live(name)
            if live and (not s["known"] or s["head"] is None or live[0] > s["head"]):
                return False
            st = self.stage(name)
            if os.path.isdir(st) and any(d.startswith("rung_") for d in os.listdir(st)):
                return False
        return True

    def write_status(self):
        rec = {"updated": time.time(), "jobs": {
            n: {"branch": s["branch"], "pushed_step": s["head"], "pushed_at": s["pushed_at"], "live_step": s["live_step"],
                "pushes": s["pushes"], "marks": sorted(s["marks"] or []), "failures": s["failures"],
                "last_error": s["last_error"], "refused_step": s["refused"]} for n, s in self.s.items()}}
        p = os.path.join(self.chat, "courier_status.json")
        with open(p + ".tmp", "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(p + ".tmp", p)


# ---------------------------------------------------------------- restore
def local_step(chat, name):
    """The step this VM already holds for the job: its live generation, its flat checkpoint, or a placed upload."""
    out, up = os.path.join(chat, name), os.path.join(chat, "upload", name)
    steps = [0]
    try:
        steps.append(gen_step(os.path.basename(os.readlink(os.path.join(out, "ckpt_gen", "live")))) or 0)
    except OSError:
        pass
    steps.append(file_step(os.path.join(out, "state.txt")) or 0)
    if os.path.exists(os.path.join(up, "READY")):
        steps.append(file_step(os.path.join(up, "state.txt")) or 0)
    return max(steps)


def restore(store, chat, jobs, branches, log=print, tries=4, delay=20):
    """See the module docstring. Returns {job: record}; raises when a job's branch cannot be read or verified."""
    token = store.b.token
    out = {}
    for name, branch in branches.items():
        if status(chat, name) in ("FINISHED", "FAILED"):
            continue
        have = local_step(chat, name)
        stage = os.path.join(chat, "store_restore", name)
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        t0 = time.time()
        step, rejected = hf_store.retry(lambda branch=branch, stage=stage: store.pull_generation(branch, stage),
                                        tries=tries, delay=delay, log=log, what=f"{name}: pull from {branch}", token=token)
        rec = {"job": name, "branch": branch, "store_step": step, "local_step": have, "rejected": rejected,
               "placed": False, "at": time.time()}
        if step is None:
            log(f"{name}: the store holds no generation on {branch}; the job starts from its spec")
        elif step <= have:
            log(f"{name}: this VM holds step {have}, the store step {step}; nothing placed")
        else:
            up, g = os.path.join(chat, "upload", name), os.path.join(stage, f"g{step}")
            os.makedirs(up, exist_ok=True)
            for f in os.listdir(up):
                if f == "READY" or f.split(".part")[0] in ("model.safetensors", "optim.safetensors", "state.txt", "events.jsonl"):
                    os.remove(os.path.join(up, f))
            for f in ("model.safetensors", "optim.safetensors"):
                os.replace(os.path.join(g, f), os.path.join(up, f + ".part00"))
            os.replace(os.path.join(g, "state.txt"), os.path.join(up, "state.txt"))
            if os.path.exists(os.path.join(g, "events.jsonl")):
                os.replace(os.path.join(g, "events.jsonl"), os.path.join(up, "events.jsonl"))
            open(os.path.join(up, "READY"), "w").close()  # last: it names a whole, verified generation
            rec["placed"] = True
            log(f"{name}: generation g{step} of {branch} verified and placed for the resume "
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
            listed[pt] = hf_store.retry(lambda pt=pt: store.mark_steps(pt), tries=tries, delay=delay, log=log,
                                        what=f"{pt}: marks listing", token=token)
        if step not in listed[pt]:
            continue
        shutil.rmtree(dst + ".tmp", ignore_errors=True)
        hf_store.retry(lambda pt=pt, step=step, dst=dst: store.pull_mark(pt, step, dst + ".tmp"), tries=tries, delay=delay,
                       log=log, what=f"{pt}: mark {step}", token=token)
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(dst + ".tmp", dst)
        log(f"{pt}: mark snapshot rung_{step} verified and placed for {j['name']}")
    return out


def store_jobs(cfg, jobs):
    """{training job: branch} from the config, for jobs this jobs file holds."""
    return {n: b for n, b in (cfg.get("branches") or {}).items() if n in jobs and "cmd" not in jobs[n]}


def take_pid(chat):
    p = os.path.join(chat, "courier.pid")
    try:
        other = int(open(p).read())
        if other != os.getpid() and os.path.exists(f"/proc/{other}"):
            sys.exit(f"another courier (pid {other}) is running; refusing")
    except (OSError, ValueError):
        pass
    open(p, "w").write(str(os.getpid()))


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
        store = hf_store.open_store(cfg, log=log)
        token = store.b.token
        if a.cmd == "restore":
            restore(store, a.chat, jobs, branches, log=log, delay=a.retry_delay)
            return 0
    except Exception as e:  # noqa: BLE001 one redacted line; the boot's ERR trap does the rest
        log(hf_store.redact(f"courier {a.cmd} failed: {type(e).__name__}: {e}", token))
        return 1
    take_pid(a.chat)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    gap = a.min_interval if a.min_interval is not None else float(cfg.get("min_push_interval_s") or 0)
    c = Courier(store, a.chat, jobs, branches, log=log, min_interval=gap, push_timeout=a.push_timeout or None)
    log(f"courier up: {branches}" + (f", at most one generation push every {gap:.0f} s" if gap else ""))
    quiet = 0
    while not stop:
        c.tick()
        quiet = quiet + 1 if c.idle() else 0
        if quiet >= 2:
            log("every job has ended and its last generation and marks are in the store; courier done")
            break
        time.sleep(a.tick)
    c.write_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
