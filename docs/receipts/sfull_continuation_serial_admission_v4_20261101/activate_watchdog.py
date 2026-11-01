"""Enable only the existing watchdog entry after immutable serial admission."""
from pathlib import Path
import json,hashlib,datetime,os,shlex,sys
ROOT=Path(__file__).resolve().parent
CONFIG=Path("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/scripts/colab_driver.json")

def activate():
    record=json.loads((ROOT/"dispatch.json").read_text());review=Path(record["review"])
    if hashlib.sha256(review.read_bytes()).hexdigest()!=record["review_sha256"]:
        raise ValueError("dispatch review changed")
    r=json.loads(review.read_text())
    if review != ROOT/"admission.json":
        raise ValueError("unexpected admission path")
    expected=["python3",str(ROOT/"guarded_continuation.py"),"--review",str(review),"--review-sha256",record["review_sha256"],"--"]+r["runner_argv"][1:]
    if record["guard_argv"] != expected:
        raise ValueError("dispatch command differs from reviewed guard/runner")
    # Execute the pinned guard source, never a cached bytecode file.
    guard={"__file__":str(ROOT/"guarded_continuation.py"),"__name__":"activation_guard"}
    guard_bytes=(ROOT/"guarded_continuation.py").read_bytes()
    if hashlib.sha256(guard_bytes).hexdigest() != r["pins"][str(ROOT/"guarded_continuation.py")]:
        raise ValueError("guard source changed before execution")
    exec(compile(guard_bytes,str(ROOT/"guarded_continuation.py"),"exec"),guard)
    guard["validate"](review,record["review_sha256"],r["runner_argv"][1:])
    guard["validate_checkpoint"](r)
    if not r.get("dispatch_authorized") or r.get("probe_only"):
        raise ValueError("not a production admission")
    checked=datetime.datetime.fromisoformat(json.loads((ROOT/"serial_dispatch_check.json").read_text())["checked_at_utc"].replace("Z","+00:00"))
    if not 0<=(datetime.datetime.now(datetime.timezone.utc)-checked).total_seconds()<=120:
        raise ValueError("serial admission check is stale")
    if (ROOT/"watchdog_activation.json").exists():
        raise ValueError("already activated; do not mutate admission or deadline")
    raw=CONFIG.read_bytes();cfg=json.loads(raw)
    for d in cfg["drivers"]:
        if d.get("enabled") and "no GPU/provider control" not in d["name"]:
            raise ValueError("another GPU campaign is enabled: "+d["name"])
    if any(d.get("state_dir")==r["state_dir"] for d in cfg["drivers"]):
        raise ValueError("duplicate state entry")
    sys.path.insert(0,str(Path(r["source_checkout"])/"tools/colab_chat"));import runpod_api as api
    if api.list_pods():raise ValueError("provider GPU slot is occupied")
    state=json.loads((Path(r["state_dir"])/"state.json").read_text())
    if api.ledger_usd(state["runpod"]["ledger"])>=7:
        raise ValueError("original seed cap exhausted")
    entry={"name":"S-full seed1 continuation V4 oneL4 cumulative7USD checkpoint180s", "enabled":True,"workdir":r["source_checkout"],"state_dir":r["state_dir"],"stop_file":r["state_dir"]+"/HALTED","match":"--state-dir "+r["state_dir"],"command":shlex.join(record["guard_argv"])+" >> "+r["state_dir"]+"/guard.log 2>&1"}
    cfg["drivers"].append(entry)
    if CONFIG.read_bytes()!=raw:raise ValueError("watchdog config changed during admission")
    tmp=CONFIG.with_suffix(".sfull-activation.tmp");tmp.write_text(json.dumps(cfg,indent=4)+"\n");os.replace(tmp,CONFIG)
    out={"entry":entry,"review_sha256":record["review_sha256"],"deadline_utc":r["deadline_utc"],"enabled_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"only_existing_watchdog":True}
    (ROOT/"watchdog_activation.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"enabled":entry["name"],"deadline_utc":r["deadline_utc"]}))

if __name__=="__main__":activate()
