#!/usr/bin/env python3
"""fakecolab.py plus the VM side of the checkpoint store, for runner_multi.py --ckpt-store hf (tests/test_runner_store.py).

Every call goes to fakecolab.main() unchanged. Around it, as boot_base2.sh and the courier daemon would on a VM:
- after the boot exec: a boot id is written and ckpt_courier.restore() installs each job's newest whole generation,
  and the marks unfinished rung jobs need, from the store named in chat/store.json (the fake trainer reads flat
  files, so the installed generation's files are copied to them); a restore that refuses ends as BOOT_FAILED with
  the token removed, as the boot's trap does;
- after each vm_status exec (one simulated training step): the fake trainer's flat checkpoint becomes a
  transactional generation (tests/store_standin.commit) and one courier pass runs (ckpt_courier.Courier.tick).
Scenario keys in fake.json: "boot_fail" (the boot ends as BOOT_FAILED), "boot_hang" (it never reports a status),
"courier_off_from": N (no courier pass from the Nth vm_status call on: a store that stops receiving).
The store is the "dir:" repo the runner was given, so nothing leaves the machine. FAKE_STORE_TOOLS is the folder
holding hf_store.py and ckpt_courier.py (the copy beside the runner under test).
"""
import json
import os
import shutil
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ["FAKE_STORE_TOOLS"])
import ckpt_courier  # noqa: E402
import fakecolab  # noqa: E402
import hf_store  # noqa: E402
import store_standin  # noqa: E402

CHAT = fakecolab.CHAT
CFG = os.path.join(CHAT, "store.json")
FLAT = ("model.safetensors", "optim.safetensors", "state.txt")


def note(msg):
    with open(os.path.join(fakecolab.ROOT, "store_vm.log"), "a") as f:
        f.write(msg + "\n")


def jobs_and_branches():
    jobs = {j["name"]: j for j in json.load(open(os.environ["FAKE_JOBS"]))["jobs"]}
    cfg = json.load(open(CFG))
    return jobs, ckpt_courier.store_jobs(cfg, jobs), cfg


def boot():
    d = fakecolab.load()
    status = os.path.join(CHAT, "STATUS")
    if d.get("boot_fail") or d.get("boot_hang"):
        os.remove(status)
        if d.get("boot_fail"):
            open(status, "w").write("BOOT_FAILED\n")
            hf_store.forget_token(CFG, chat=CHAT)
        return
    with open(os.path.join(CHAT, "boot_id"), "w") as f:
        f.write(uuid.uuid4().hex)
    jobs, branches, cfg = jobs_and_branches()
    try:
        recs = ckpt_courier.restore(hf_store.open_store(cfg, log=note, delay=0), CHAT, jobs, branches, log=note, cfg=cfg)
    except Exception as e:  # noqa: BLE001 the real boot's ERR trap
        note(f"launch {d['launches']}: boot failed: {type(e).__name__}: {e}")
        open(status, "w").write("BOOT_FAILED\n")
        hf_store.forget_token(CFG, chat=CHAT)
        return
    for name, rec in recs.items():
        note(f"launch {d['launches']}: {name} store_step {rec['store_step']} placed {rec['placed']}")
        if rec["placed"]:  # the fake trainer writes flat files in place: give it copies, not links into the generation
            out = os.path.join(CHAT, name)
            for f in FLAT:
                os.remove(os.path.join(out, f))
                shutil.copyfile(os.path.join(out, "ckpt_gen", f"g{rec['store_step']}", f), os.path.join(out, f))


def after_step():
    d = fakecolab.load()
    jobs, branches, cfg = jobs_and_branches()
    for name in branches:
        out = os.path.join(CHAT, name)
        step = fakecolab.ckpt_step(out)
        if not os.path.exists(os.path.join(out, "optim.safetensors")):
            continue
        if step and not os.path.isdir(os.path.join(out, "ckpt_gen", f"g{step}")):
            store_standin.commit(out, step, open(os.path.join(out, "model.safetensors"), "rb").read(),
                                 open(os.path.join(out, "optim.safetensors"), "rb").read(),
                                 state=open(os.path.join(out, "state.txt")).read(), flat_links=False)
    if d.get("courier_off_from") and d.get("status_calls", 0) >= d["courier_off_from"]:
        return
    ckpt_courier.Courier(hf_store.open_store(cfg, log=note, delay=0), CHAT, jobs, branches, log=note, cfg=cfg).tick()


def main():
    argv = sys.argv[1:]
    code = open(argv[argv.index("-f") + 1]).read() if argv and argv[0] == "exec" else ""
    rc = fakecolab.main()
    if code and rc == 0 and fakecolab.load()["alive"] and os.path.exists(CFG):
        if "boot_jobs.sh" in code:
            boot()
        elif "VMJSON" in code:
            after_step()
    return rc


if __name__ == "__main__":
    sys.exit(main())
