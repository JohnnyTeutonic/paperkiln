#!/usr/bin/env python3
"""fakecolab.py plus the VM side of the checkpoint store, for runner_multi.py --ckpt-store hf (tests/test_runner_store.py).

Every call goes to fakecolab.main() unchanged. Around it, as boot_base2.sh and the courier daemon would on a VM:
- after the boot exec: the uploaded token is adopted (hf_store.adopt_token) and ckpt_courier.restore() places each
  job's newest whole generation, and the marks unfinished rung jobs need, from the store named in chat/store.json;
- after each vm_status exec (one simulated training step): the fake trainer's flat checkpoint becomes a
  transactional generation (tests/store_standin.commit) and one courier pass runs (ckpt_courier.Courier.tick).
The store is the "dir:" repo the runner was given, so nothing leaves the machine. FAKE_STORE_TOOLS is the folder
holding hf_store.py and ckpt_courier.py (the copy beside the runner under test).
"""
import json
import os
import sys

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


def note(msg):
    with open(os.path.join(fakecolab.ROOT, "store_vm.log"), "a") as f:
        f.write(msg + "\n")


def jobs_and_branches():
    jobs = {j["name"]: j for j in json.load(open(os.environ["FAKE_JOBS"]))["jobs"]}
    return jobs, ckpt_courier.store_jobs(json.load(open(CFG)), jobs)


def boot():
    d = fakecolab.load()
    hf_store.adopt_token(CFG, os.path.join(CHAT, "upload", "store_token"), os.path.join(fakecolab.VM, "shm"))
    jobs, branches = jobs_and_branches()
    recs = ckpt_courier.restore(hf_store.open_store(CFG, log=note), CHAT, jobs, branches, log=note)
    for name, rec in recs.items():
        note(f"launch {d['launches']}: {name} store_step {rec['store_step']} placed {rec['placed']}")
        if rec["placed"]:  # the fake's own restore removes PENDING with READY
            open(os.path.join(CHAT, "upload", name, "PENDING"), "w").close()


def after_step():
    jobs, branches = jobs_and_branches()
    for name in branches:
        out = os.path.join(CHAT, name)
        step = fakecolab.ckpt_step(out)
        if not os.path.exists(os.path.join(out, "optim.safetensors")):
            continue  # a checkpoint the fake just restored (it places weights and step only): not a new save
        if step and not os.path.isdir(os.path.join(out, "ckpt_gen", f"g{step}")):
            store_standin.commit(out, step, open(os.path.join(out, "model.safetensors"), "rb").read(),
                                 open(os.path.join(out, "optim.safetensors"), "rb").read(),
                                 state=open(os.path.join(out, "state.txt")).read(), flat_links=False)
    ckpt_courier.Courier(hf_store.open_store(CFG, log=note), CHAT, jobs, branches, log=note).tick()


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
