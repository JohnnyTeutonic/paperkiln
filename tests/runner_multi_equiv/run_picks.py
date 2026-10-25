#!/usr/bin/env python3
"""A pick whose post jobs are home, through a replacement VM, end to end on the fake colab CLI (fakecolab.py).

Three training jobs, a post job after each (SFT and scoring: fakecolab.fake_post, deterministic) and a pick
over the three (tiny_chat_run.py pick itself, run on the fake VM's files). The pick is held until after the
VM is reclaimed, so the replacement VM must run it from what it is given.

  A    every post is home at the reclaim; post_a and post_b tie on the primary (pick's rule: the lower seed,
       post_a). OLD (the runner before the fix) reruns all three posts on the replacement VM (the bug). NEW
       reruns post_a only, puts post_b's and post_c's home summary.json at the VM path pick reads (FINISHED,
       never run), uploads only pt_a's final weights, and the pick's best.json and model equal those of a run
       with no reclaim. No home file changes after it first arrives.
  B    post_c is held until after the reclaim (not home): NEW reruns post_b only (the best of the home posts)
       besides running post_c; post_c wins, and best.json equals that of a run with no reclaim.

    python3 tests/runner_multi_equiv/run_picks.py OLD_RUNNER NEW_RUNNER WORKDIR   # WORKDIR on ext4
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eq  # noqa: E402

TOOL = "tools/colab_chat/tiny_chat_run.py"


def jobs(primaries, hold_c=None, hold_pick=None):
    js = [{"name": "data", "cmd": ["x"], "outputs": ["out.txt"]}]
    for x in "abc":
        js.append({"name": f"pt_{x}", "model_name": f"m{x}", "after": "data", "relay_every_steps": 300, "spec": {}})
    for i, x in enumerate("abc"):
        j = {"name": f"post_{x}", "after": f"pt_{x}", "init_job": f"pt_{x}",
             "cmd": ["python3", TOOL, "post", "--out", "{out}", "--seed", str(i + 1)], "outputs": ["summary.json"],
             "fake_post": {"seed": i + 1, "primary": primaries[i]}}
        if x == "c" and hold_c:
            j["fake_start_at"] = hold_c
        js.append(j)
    pick = {"name": "pick", "after": ["post_a", "post_b", "post_c"],
            "cmd": ["python3", TOOL, "pick", "--out", "{out}", "--run", "/content/chat/post_a", "--run",
                    "/content/chat/post_b", "--run", "/content/chat/post_c"],
            "outputs": ["best.json", "model.safetensors", "model.gguf", "spec.json"]}
    if hold_pick:
        pick["fake_start_at"] = hold_pick
    js.append(pick)
    return {"jobs": js}


def sft_runs(work):
    p = os.path.join(work, "fake", "sft_runs.log")
    return [ln.split() for ln in open(p).read().splitlines()] if os.path.exists(p) else []


def best(work):
    """The pick's best.json with the fake VM's root mapped back to /content/, and its model's sha256."""
    art = os.path.join(work, "art", "pick")
    b = json.loads(open(os.path.join(art, "best.json")).read().replace(os.path.join(work, "fake", "vm") + "/",
                                                                          "/content/"))
    m = os.path.join(art, "model.safetensors")
    return b, hashlib.sha256(open(m, "rb").read()).hexdigest() if os.path.exists(m) else None


def home_files(work, names):
    return {n: run_eq.tree(os.path.join(work, "art", n)) for n in names}


def main():
    old, new, work = sys.argv[1:4]
    os.makedirs(work, exist_ok=True)
    checks = {}
    # vm_status calls: 1-2 data; 3-6, 7-10, 11-14 the three training jobs; 15-20 the posts (two calls each);
    # each post's summary and log come home over the next ticks. A: reclaimed at the 24th, pick from the 25th.
    ja = jobs([0.5, 0.5, 0.2], hold_pick=25)
    base_a = run_eq.run(new, os.path.join(work, "a_base"), {"reclaim_at": []}, jobs=ja)
    old_a = run_eq.run(old, os.path.join(work, "a_old"), {"reclaim_at": [24]}, jobs=ja)
    new_a = run_eq.run(new, os.path.join(work, "a_new"), {"reclaim_at": [24]}, jobs=ja)
    wb, wo, wn = (os.path.join(work, x) for x in ("a_base", "a_old", "a_new"))
    checks["A: all three runs reach the end"] = [base_a["rc"], old_a["rc"], new_a["rc"]] == [0, 0, 0]
    checks["A: the reclaim came after every post was home"] = all(
        new_a["log"].find(f"post_{x}: outputs downloaded") < new_a["log"].find("(launch 2)") for x in "abc") and (
        new_a["log"].find("(launch 2)") > 0)
    checks["A: OLD reruns all three posts on the replacement VM (the bug)"] = sorted(
        n for launch, n in sft_runs(wo) if launch == "2") == ["post_a", "post_b", "post_c"]
    checks["A: NEW reruns post_a only (the tie goes to the lower seed)"] = [
        n for launch, n in sft_runs(wn) if launch == "2"] == ["post_a"]
    checks["A: NEW logs the spared posts"] = (
        "post jobs their pick can never choose are not rerun (FINISHED, summary.json from home): "
        "['post_b', 'post_c']" in new_a["log"])
    bb, bn, bo = best(wb), best(wn), best(wo)
    checks["A: pick chose post_a with its model, as with no reclaim"] = (
        bn == bb and bb[0]["best"]["run"] == "/content/chat/post_a" and bb[1] is not None
        and bb[0]["weights_match_summary"] is True and bn[0]["weights_sha256"] == bn[1])
    checks["A: OLD's pick equal too (the fix changes the cost, not the pick)"] = bo == bb
    vm = os.path.join(wn, "fake", "vm", "chat")
    checks["A: post_b, post_c FINISHED on the VM with the home summary.json, no SFT run there"] = all(
        open(os.path.join(vm, f"STATUS_post_{x}")).read().strip() == "FINISHED"
        and open(os.path.join(vm, f"post_{x}", "summary.json"), "rb").read()
        == open(os.path.join(wn, "art", f"post_{x}", "summary.json"), "rb").read()
        and not os.path.exists(os.path.join(vm, f"post_{x}", "sft")) for x in "bc")
    up_n = [c for c in new_a["calls"].splitlines() if c.startswith('["upload"') and "__final" in c]
    up_o = [c for c in old_a["calls"].splitlines() if c.startswith('["upload"') and "__final" in c]
    checks["A: NEW uploads only pt_a's final weights (OLD all three)"] = (
        {c.split("upload/")[1].split("__")[0] for c in up_n} == {"pt_a"}
        and {c.split("upload/")[1].split("__")[0] for c in up_o} == {"pt_a", "pt_b", "pt_c"})
    posts = ["post_a", "post_b", "post_c"]
    checks["A: home post files identical to the no-reclaim run (nothing home overwritten)"] = (
        home_files(wn, posts) == home_files(wb, posts))
    checks["A: every job home and FINISHED"] = all(
        e["home"] and e["status"] == "FINISHED" for e in new_a["state"]["jobs"].values())

    # B: post_c held until after the reclaim (not home then): post_a and post_b home; c wins.
    jb = jobs([0.5, 0.6, 0.9], hold_c=24, hold_pick=24)
    base_b = run_eq.run(new, os.path.join(work, "b_base"), {"reclaim_at": []}, jobs=jb)
    new_b = run_eq.run(new, os.path.join(work, "b_new"), {"reclaim_at": [22]}, jobs=jb)
    wbb, wnb = os.path.join(work, "b_base"), os.path.join(work, "b_new")
    checks["B: both runs reach the end"] = [base_b["rc"], new_b["rc"]] == [0, 0]
    lb = new_b["log"]
    checks["B: post_a, post_b home and post_c not at the reclaim"] = (
        lb.find("post_a: outputs downloaded") < lb.find("(launch 2)")
        and lb.find("post_b: outputs downloaded") < lb.find("(launch 2)") < lb.find("post_c: outputs downloaded"))
    checks["B: the replacement VM reruns post_b (best of the home posts) and runs post_c; never post_a"] = sorted(
        n for launch, n in sft_runs(wnb) if launch == "2") == ["post_b", "post_c"]
    b2, b1 = best(wnb), best(wbb)
    checks["B: pick chose post_c with its model, as with no reclaim"] = (
        b2 == b1 and b1[0]["best"]["run"] == "/content/chat/post_c" and b1[1] is not None)
    checks["B: every job home and FINISHED"] = all(
        e["home"] and e["status"] == "FINISHED" for e in new_b["state"]["jobs"].values())

    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    for name, r in (("A_base", base_a), ("A_old", old_a), ("A_new", new_a), ("B_base", base_b), ("B_new", new_b)):
        open(os.path.join(work, f"{name}.log"), "w").write(r["log"])
        if r["stderr"]:
            print(f"{name} stderr:", r["stderr"][-1500:])
    ok = all(checks.values())
    print("PICKS OK" if ok else "PICKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
