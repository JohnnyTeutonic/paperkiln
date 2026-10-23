"""The recursion arm's Stage 0 entries (PREREGISTRATION.md A3.3): shape specs and a jobs file for
tools/colab_chat/runner_multi.py. Written, not launched. A3.3 names make_jobs.py as the writer of the
registered tiny_chat_s0_rec.json; this file is the paperkiln side's proposal of its entries (the
specs mtstudio accepts and the probe and screen commands), for make_jobs.py to adopt.

    python experiments/tiny_chat_2026-10/recursion/make_probe_jobs.py

Cells, all at E1's 20m_L4 shape (d 640, 10 heads, d_ff 1,760; 20.08M non-embedding parameters):
  R2   0, [4] x 2, 0   effective depth 8   (E1 twin 20m_L8)
  R4   0, [4] x 4, 0   16                  (20m_L16)
  R8   0, [4] x 8, 0   32                  (20m_L32; screens only on A3.3's R8 rule, so none here)
  MC7  1, [2] x 7, 1   16                  (Middle-Cycle, MoR's layout; 20m_L16 and R4)
No input injection (A3.3; arch.custom.recursion.inject stays "none"). Batch 32 x 1 at R2, 16 x 2 at
effective depth 16 and above. Each probe runs alone on the L4 (amendment 2), 300 steps on the 35% mix;
screens at {2.5e-4, 5e-4, 1e-3, 2e-3}, 1,500 steps, seed 101, after their cell's probe; the data job is
tiny_chat_s0.json's (the same bytes). The VM's paperkiln must hold arch.custom.recursion.
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = json.loads((HERE.parent / "specs" / "shape_20m_L4.json").read_text(encoding="utf-8"))
S0 = json.loads((ROOT / "tools" / "colab_chat" / "tiny_chat_s0.json").read_text(encoding="utf-8"))
CORPUS = "/content/chat/data/mix_s35.ids"
RATES = [0.00025, 0.0005, 0.001, 0.002]

CELLS = [  # name, recursion, screened here
    ("20m_R2", {"prelude": 0, "block": 4, "repeats": 2, "coda": 0}, True),
    ("20m_R4", {"prelude": 0, "block": 4, "repeats": 4, "coda": 0}, True),
    ("20m_R8", {"prelude": 0, "block": 4, "repeats": 8, "coda": 0}, False),
    ("20m_MC7", {"prelude": 1, "block": 2, "repeats": 7, "coda": 1}, True),
]


def spec(name, rec):
    s = copy.deepcopy(BASE)
    s["name"] = "tc_" + name
    s["arch"]["custom"]["recursion"] = dict(rec, schedule="fixed", inject="none")
    eff = rec["prelude"] + rec["block"] * rec["repeats"] + rec["coda"]
    s["train"]["batch"], s["train"]["accum"] = (16, 2) if eff >= 16 else (32, 1)
    s["_shape"] = dict(s["_shape"], recursion=rec, effective_layers=eff, stored_layers=4)
    return s


def run_cmd(mode, rel, *extra):
    return ["python3", "tools/colab_chat/tiny_chat_run.py", mode, "--out", "{out}", "--spec", rel,
            "--corpus", CORPUS, *extra]


def main():
    (HERE / "specs").mkdir(exist_ok=True)
    data = copy.deepcopy(next(j for j in S0["jobs"] if j["name"] == "data"))
    data.pop("after", None)
    jobs = [data]
    for name, rec, screened in CELLS:
        path = HERE / "specs" / f"shape_{name}.json"
        path.write_text(json.dumps(spec(name, rec), indent=1) + "\n", encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        jobs.append({"name": f"probe_{name}", "after": "data", "cmd": run_cmd("probe", rel, "--steps", "300"),
                     "outputs": ["summary.json"]})
        for k, lr in enumerate(RATES if screened else []):
            jobs.append({"name": f"screen_{name}_lr{k}", "after": f"probe_{name}",
                         "cmd": run_cmd("screen", rel, "--lr", repr(lr), "--steps", "1500", "--seed", "101"),
                         "outputs": ["summary.json"]})
    doc = {"_comment": "tiny-chat recursion arm (PREREGISTRATION.md A3.3), Stage 0 probes and screens: the "
                       "paperkiln side's proposed entries for make_jobs.py's tiny_chat_s0_rec.json, not launched "
                       "(experiments/tiny_chat_2026-10/recursion/make_probe_jobs.py). One GPU job per VM "
                       "(amendment 2). R8's screens wait for A3.3's R8 rule. Needs a paperkiln with "
                       "arch.custom.recursion on the VM.",
           "max_parallel": 1, "jobs": jobs}
    (HERE / "s0_rec_entries.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"{len(CELLS)} specs and {len(jobs)} jobs written")


if __name__ == "__main__":
    main()
