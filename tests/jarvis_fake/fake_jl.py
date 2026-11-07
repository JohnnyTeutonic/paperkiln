#!/usr/bin/env python3
"""A stand-in for the `jl` CLI: one account in FAKE_JL_DIR/account.json, every call appended to
FAKE_JL_DIR/calls.jsonl. No network. JSON shapes are those of jl 0.2.19 (read from its source and its
real `status`, `gpus`, `list` and `filesystem list` replies, 7 Nov 2026).

account.json keys beyond the data: "create" chooses how a create behaves:
  ok               the VM comes up Running
  refused          an {"error"} reply before any POST: no VM
  fail_after_post  the VM exists (Creating) and jl reports an error, as its 600 s poll does
  hang_after_post  the VM exists (Creating) and jl never returns, as when killed mid-poll
and "fail" is a list of command names that fail outright; "lying_destroy" answers success and leaves
the VM; "destroy_fails" fails that many destroys.
"""
import json
import os
import sys
import time

DIR = os.environ["FAKE_JL_DIR"]
ACCOUNT = os.path.join(DIR, "account.json")
PERSON = {"user_id": "person@example.org", "name": "SENTINEL-NAME", "address1": "SENTINEL-ADDRESS",
          "phone_number": "SENTINEL-PHONE", "city": None, "country": "AU", "address2": None,
          "state": None, "zip_code": None, "tax_id": None}


def load():
    with open(ACCOUNT) as f:
        return json.load(f)


def save(d):
    tmp = f"{ACCOUNT}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, ACCOUNT)


def reply(obj, rc=0):
    print(json.dumps(obj, indent=2))
    sys.exit(rc)


def die(msg):
    reply({"error": msg}, 1)


def opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def new_instance(d, argv, status):
    d["next_id"] += 1
    vm = {"machine_id": d["next_id"], "cost": 0.0, "runtime": 0, "gpu_type": opt(argv, "--gpu"), "ram": 124,
          "storage_gb": int(opt(argv, "--storage")), "cores": 28, "template": "vm", "fs_id": int(opt(argv, "--fs-id")),
          "num_gpus": 1, "url": None, "ssh_command": None, "status": status, "name": opt(argv, "--name"),
          "is_spot": d.get("return_spot", False), "public_ip": None, "region": opt(argv, "--region")}
    if status == "Running":
        vm["public_ip"] = "203.0.113.7"
        vm["ssh_command"] = "ssh -o StrictHostKeyChecking=no -p 22 ubuntu@203.0.113.7"
    d["instances"].append(vm)
    save(d)
    return vm


def main():
    argv = sys.argv[1:]
    with open(os.path.join(DIR, "calls.jsonl"), "a") as f:
        f.write(json.dumps(argv) + "\n")
    d = load()
    cmd = argv[0]
    key = " ".join(argv[:2]) if cmd == "filesystem" else cmd
    if key in d.get("fail", []):
        die(f"simulated failure of {key}")
    if d.get("garbage"):
        print("SENTINEL-NAME is not JSON")
        sys.exit(0)
    if "--json" not in argv:
        sys.exit("fake jl: every call must pass --json")
    if cmd == "status":
        reply({"user": PERSON, "balance": {"balance": d["balance"], "grants": 0.0}, "resources": {}, "currency": "USD"})
    if cmd == "gpus":
        reply([{"gpu_type": "A30", "region": "IN2", "num_free_devices": 9, "effective_num_free_devices": 9,
                "price_per_hour": 0.41, "spot_price": 0.29, "workload_type": "container"},
               {"gpu_type": "L4", "region": "IN2", "num_free_devices": 6, "effective_num_free_devices": 6,
                "price_per_hour": 0.44, "spot_price": 0.29, "workload_type": "container"},
               {"gpu_type": "L4", "region": "IN2", "num_free_devices": d["stock"], "effective_num_free_devices": d["stock"],
                "price_per_hour": d["price"], "spot_price": 0.29, "workload_type": "vm"}])
    if cmd == "list":
        reply(d["instances"])
    if cmd == "destroy":
        mid = int(argv[1])
        if d.get("destroy_fails", 0) > 0:
            d["destroy_fails"] -= 1
            save(d)
            die("simulated destroy failure")
        if not any(v["machine_id"] == mid for v in d["instances"]):
            die(f"Instance {mid} not found. Check the ID with: jl list")
        if not d.get("lying_destroy"):
            d["instances"] = [v for v in d["instances"] if v["machine_id"] != mid]
            save(d)
        reply({"success": True, "machine_id": mid})
    if cmd == "create":
        mode = d.get("create", "ok")
        if mode == "refused":
            die("No L4 available in IN2")
        if mode == "fail_after_post":
            new_instance(d, argv, "Creating")
            die("Instance creation failed: unknown error")
        if mode == "hang_after_post":
            new_instance(d, argv, "Creating")
            time.sleep(3600)
        reply(new_instance(d, argv, "Running"))
    if argv[:2] == ["filesystem", "list"]:
        reply(d["filesystems"])
    if argv[:2] == ["filesystem", "create"]:
        d["next_fs"] += 1
        d["filesystems"].append({"fs_id": d["next_fs"], "fs_name": opt(argv, "--name"),
                                 "storage": int(opt(argv, "--storage")), "region": opt(argv, "--region")})
        save(d)
        reply({"success": True, "fs_id": d["next_fs"]})
    if argv[:2] == ["filesystem", "remove"]:
        d["filesystems"] = [f for f in d["filesystems"] if f["fs_id"] != int(argv[2])]
        save(d)
        reply({"success": True})
    die(f"fake jl: unknown command {argv}")


if __name__ == "__main__":
    main()
