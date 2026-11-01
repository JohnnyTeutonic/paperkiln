"""Fail-closed S-full seed-one continuation. No direct credential access."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def flag(argv, name):
    if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
        raise ValueError("unique flag required: " + name)
    return argv[argv.index(name) + 1]


def validate(path, digest, supplied):
    if sha(path) != digest:
        raise ValueError("admission hash differs")
    r = json.loads(Path(path).read_text())
    if r["kind"] != "SFULL_CONTINUATION_ADMISSION_V4" or r["status"] != "ADMITTED":
        raise ValueError("not admitted")
    if not r.get("probe_only"):
        admitted = dt.datetime.fromisoformat(r["admitted_at_utc"].replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(r["deadline_utc"].replace("Z", "+00:00"))
        if admitted.tzinfo is None or not 0 < (end-admitted).total_seconds() <= 23.5*3600:
            raise ValueError("continuation absolute window exceeds original upper bound")
    argv = r["runner_argv"]
    if supplied != argv[1:]:
        raise ValueError("runner argv differs")
    for name, expected in {"--backend":"runpod", "--gpu":"L4", "--cloud":"SECURE", "--volume-gb":"80", "--max-usd":"7", "--max-hours":"23.5", "--commit":r["source_control_commit"], "--jobs":r["jobs_relative"], "--boot-script":"tools/colab_chat/boot_sfull_v2_20261101.sh", "--state-dir":r["state_dir"], "--art":r["art_dir"], "--deadline-utc":r["deadline_utc"]}.items():
        if flag(argv,name) != expected:
            raise ValueError("source/resource/path flag differs: " + name)
    deadline = dt.datetime.fromisoformat(r["deadline_utc"].replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline <= dt.datetime.now(dt.timezone.utc):
        raise ValueError("fixed continuation deadline expired")
    if (r["numerical_commit"] != "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1" or r["seed"] != 1 or r["aggregate_cap_usd"] != 21 or r["per_seed_cap_usd"] != 7):
        raise ValueError("scientific/budget contract differs")
    if Path(r["jobs_path"]) != Path(r["source_checkout"])/r["jobs_relative"] or r["jobs_relative"] != "tools/colab_chat/carry12_sfull_v2_r1.json":
        raise ValueError("relative VM jobs binding differs")
    if Path(argv[1]).resolve() != Path(r["source_checkout"])/"tools/colab_chat/runner_multi.py":
        raise ValueError("runner source directory differs")
    if r["art_dir"] == r["original_art_dir"] or r["state_dir"] == str(Path(r["original_state"]).parent):
        raise ValueError("historical paths cannot be reused")
    for file, pin in r["pins"].items():
        if not Path(file).is_file() or Path(file).is_symlink() or sha(file) != pin:
            raise ValueError("source/input pin differs: " + file)
    if sha(r["original_state"]) != r["original_state_sha256"]:
        raise ValueError("historical state changed")
    original = json.loads(Path(r["original_state"]).read_text())
    s = json.loads((Path(r["state_dir"])/"state.json").read_text())
    if s.get("campaign_home") or (Path(r["state_dir"])/"HALTED").exists():
        raise ValueError("home or halted continuation cannot relaunch")
    owned = s["runpod"]
    def resource_id(value):
        return value.get("id") if isinstance(value, dict) else value
    volume_ids = [resource_id(owned.get("volume")), *[resource_id(x) for x in owned.get("retired_volumes",[])]]
    if r["protected_old_volume"] in volume_ids:
        raise ValueError("protected original volume cannot be owned")
    prior = json.loads(Path(r["carried_ledger"]).read_text())
    if prior != original["runpod"]["ledger"]:
        raise ValueError("carried ledger does not equal original historical state")
    if any(not entry.get("stop") for entry in prior["pods"]):
        raise ValueError("historical pod is not stopped")
    for kind in ["pods","volumes"]:
        current = owned["ledger"][kind]
        if any(entry not in current for entry in prior[kind]):
            raise ValueError("prior seed spending omitted or changed")
        ids = [entry["id"] for entry in current]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate resource ledger entries")
    reservation = json.loads(Path(r["reservation_path"]).read_text())
    if reservation["aggregate_cap_usd"] != 21 or reservation["per_seed_cap_usd"] != 7:
        raise ValueError("aggregate reservation differs")
    entries=reservation["reservations"]
    if len(entries)!=1 or entries[0]["seed"]!=1 or entries[0]["cap_usd"]!=7 or entries[0]["state_dir"]!=r["state_dir"]:
        raise ValueError("continuation is not same seed-one reservation")
    uploads=[]
    for i, arg in enumerate(argv):
        if arg=="--upload":
            name, src=argv[i+1].split("=",1)
            p=Path(src)
            if name!=p.name or p.is_symlink() or str(p.resolve())!=src or r["pins"].get(src)!=sha(src):
                raise ValueError("unbound or unsafe upload")
            uploads.append(name)
    if uploads != ["vocab_llama3_bpe.gguf"]:
        raise ValueError("upload inventory differs")
    return r


def validate_checkpoint(r):
    source=Path(r["source_checkout"])/"tools/colab_chat"
    sys.path.insert(0,str(source))
    import sfull_checkpoint_v2 as cp
    import sfull_checkpoint_transport_v2 as tr
    jobs=json.loads(Path(r["jobs_path"]).read_text())["jobs"]
    j=next(j for j in jobs if j["name"]=="pt_c12_Sfull_r1")
    root=tr.root_for(r["art_dir"],j)
    with cp.locked(root,False):
        _, m=cp.current_unlocked(root)
        tr.check_manifest(j,m)
        if m["step"]<675:
            raise ValueError("home checkpoint predates admitted continuation")
    return m["step"]


def main():
    p=argparse.ArgumentParser();p.add_argument("--review",required=True);p.add_argument("--review-sha256",required=True);p.add_argument("argv",nargs=argparse.REMAINDER);a=p.parse_args()
    argv=a.argv[1:] if a.argv[:1]==["--"] else a.argv
    r=validate(a.review,a.review_sha256,argv)
    step=validate_checkpoint(r)
    if os.environ.get("SFULL_OWNER_PROBE")=="1":
        print("SFULL_OWNER_PROBE_VALIDATED_NO_PROVIDER step="+str(step),flush=True);time.sleep(30);return
    if r.get("probe_only") or not r.get("dispatch_authorized"):
        raise ValueError("probe contract cannot dispatch compute")
    runner=Path(r["runner_argv"][1]);sys.path.insert(0,str(runner.parent))
    import runpod_api as api
    if Path(api.__file__).resolve()!=runner.parent/"runpod_api.py":
        raise ValueError("provider module differs")
    # Existing watchdog owns lifecycle; refuse a foreign active pod on restart.
    state=json.loads((Path(r["state_dir"])/"state.json").read_text())
    expected=state["runpod"].get("pod")
    pods=api.list_pods()
    if any(p["id"]!=expected for p in pods):
        raise ValueError("another GPU campaign owns compute")
    if api.ledger_usd(state["runpod"]["ledger"])>=7:
        raise ValueError("original cumulative seed budget exhausted")
    api.MAX_PODS=1
    sys.argv=r["runner_argv"][1:]
    runpy.run_path(str(runner),run_name="__main__")

if __name__=="__main__":
    try:
        main()
    except Exception as error:
        # Fixed local continuation path; never use an unverified review path.
        state = Path(__file__).resolve().parent / "runner"
        state.mkdir(exist_ok=True)
        try:
            with (state / "HALTED").open("x") as marker:
                json.dump({"kind":"SFULL_CONTINUATION_ADMISSION_FAILURE", "error":str(error)},marker)
                marker.flush()
                os.fsync(marker.fileno())
        except FileExistsError:
            pass
        print("S-full continuation admission refused: "+str(error),file=sys.stderr,flush=True)
        raise
