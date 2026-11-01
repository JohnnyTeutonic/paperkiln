"""Root-only serial continuation constructor. Does not activate the watchdog."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DPO=Path("/home/jonat/pk_dpo_layout_v3_20261101_runner/state.json")
CONFIG=Path("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/scripts/colab_driver.json")


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def predecessor_ready(s):
    if s.get("campaign_home") is not True or s.get("endpoint"):
        raise ValueError("DPO not completely home")
    r=s.get("runpod",{})
    if r.get("pod") or r.get("volume") or r.get("retired_volumes") or r.get("creating"):
        raise ValueError("DPO still owns resources")
    if any(not p.get("stop") for p in r["ledger"]["pods"]) or any(not v.get("deleted") for v in r["ledger"]["volumes"]):
        raise ValueError("DPO resource ledger not closed")
    jobs=s.get("jobs",{})
    if not jobs or any(j.get("home") is not True for j in jobs.values()):
        raise ValueError("DPO job outputs not home")


def prepare():
    probe=ROOT/"probe_review.json"
    review=json.loads(probe.read_text())
    owner=json.loads((ROOT/"owner_probe.json").read_text())
    if owner.get("pass") is not True or owner["review_sha256"]!=sha(probe):
        raise ValueError("final source-bound owner probe missing")
    validation=json.loads((ROOT/"guard_validation.json").read_text())
    if validation["passed"]<20:
        raise ValueError("guard regressions missing")
    previous=json.loads(DPO.read_text());predecessor_ready(previous)
    cfg=json.loads(CONFIG.read_text())
    if any(d.get("enabled") and "no GPU/provider control" not in d["name"] for d in cfg["drivers"]):
        raise ValueError("another GPU driver enabled")
    source=Path(review["source_checkout"])
    if subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()!=review["source_control_commit"]:
        raise ValueError("operational source checkout differs")
    subprocess.run(["git","-C",str(source),"diff","--exit-code","HEAD","--"],check=True,stdout=subprocess.PIPE)
    sys.path.insert(0,str(source/"tools/colab_chat"));import runpod_api as api
    balance, rate=api.balance();pods=api.list_pods();volumes=api.list_volumes()
    if pods:raise ValueError("another GPU campaign owns provider compute")
    if balance<=0:raise ValueError("provider balance exhausted")
    if not any(v["id"]==review["protected_old_volume"] for v in volumes):
        raise ValueError("protected original volume missing from provider inventory")
    ledger=json.loads((Path(review["state_dir"])/"state.json").read_text())["runpod"]["ledger"]
    spent=api.ledger_usd(ledger)
    if not 0<=spent<7:raise ValueError("original seed budget exhausted or invalid")
    now=dt.datetime.now(dt.timezone.utc)
    stamp=lambda t:t.replace(microsecond=0).isoformat().replace("+00:00","Z")
    review["probe_only"]=False;review["dispatch_authorized"]=True
    review["admitted_at_utc"]=stamp(now);review["deadline_utc"]=stamp(now+dt.timedelta(hours=23.5))
    argv=review["runner_argv"];argv[argv.index("--deadline-utc")+1]=review["deadline_utc"]
    record={"checked_at_utc":stamp(now),"gpu_campaigns_active":0,"previous_campaign_home":True,"previous_owned_resources_deleted":True,"previous_state_sha256":sha(DPO),"seed_spent_usd_at_admission":spent,"seed_remaining_usd_at_admission":7-spent,"original_volume_preserved":True,"new_owned_volume_required":True,"continuation_is_same_seed_reservation":True,"caps_reset":False}
    if (ROOT/"admission.json").exists():raise ValueError("admission already exists; never refresh its deadline")
    serial=ROOT/"serial_dispatch_check.json";serial.write_text(json.dumps(record,indent=2)+"\n")
    review["pins"][str(serial)]=sha(serial)
    for path in [ROOT/"prepare_dispatch.py",ROOT/"activate_watchdog.py",ROOT/"guard_validation.json",ROOT/"owner_probe.json",ROOT/"activation_validation.json"]:
        review["pins"][str(path)]=sha(path)
    path=ROOT/"admission.json";path.write_text(json.dumps(review,indent=2)+"\n")
    guard={"__file__":str(ROOT/"guarded_continuation.py"),"__name__":"dispatch_guard"}
    guard_bytes=(ROOT/"guarded_continuation.py").read_bytes()
    if hashlib.sha256(guard_bytes).hexdigest() != review["pins"][str(ROOT/"guarded_continuation.py")]:
        raise ValueError("guard source changed before execution")
    exec(compile(guard_bytes,str(ROOT/"guarded_continuation.py"),"exec"),guard)
    guard["validate"](path,sha(path),argv[1:]);guard["validate_checkpoint"](review)
    result={"review":str(path),"review_sha256":sha(path),"guard_argv":["python3",str(ROOT/"guarded_continuation.py"),"--review",str(path),"--review-sha256",sha(path),"--"]+argv[1:],"deadline_utc":review["deadline_utc"],"provider_clear":True,"watchdog_enabled":False}
    (ROOT/"dispatch.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"SERIAL_ADMISSION_READY_NOT_ENABLED","deadline_utc":review["deadline_utc"],"review_sha256":sha(path)}))
    return result

if __name__=="__main__":prepare()
