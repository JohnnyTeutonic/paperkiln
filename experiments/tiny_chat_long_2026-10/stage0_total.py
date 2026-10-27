#!/usr/bin/env python3
"""The measured total of Carry-12's core runs (PREREGISTRATION_DRAFT.md section 10; section 11 item 6), from the
pre-freeze campaign's receipts (make_stage0_jobs.py) and E1 at 60M's measured walls at T 512.

    python experiments/tiny_chat_long_2026-10/stage0_total.py [--campaign-usd X] [--out stage0/total.json]

Per run, on one L4 (A5.10's pattern), in L4 VM-hours:
- pretraining: E1 at 60M's measured pretraining wall for the cell's shape at T 512 (median of its three runs;
  stage0/e1_60m_walls_t512.json) times probe(T 512) / probe(cell at T 1,024), so the in-run evaluations,
  checkpoints and rungs scale with the step time (E1 at 60M ran 9-10% under its probe rate);
- SFT: E1 at 60M's measured SFT seconds (median) times the same ratio (1,500 steps of 16,384 tokens at either T);
- the rest of the post job (panels, Carry-5, base-model Carry-5): E1 at 60M's measured post wall minus its SFT
  (median over seeds; unchanged by T: the same prompts);
- Carry-12 in the post job: the cost run on the shape's E1 at 60M pick (stage0/cost_<pick>.l0.json: eval wall
  at context 1,024, --max-new 48, self history to the target, no near_self) plus its ember build (one per VM);
- per machine: boot, CUDA gate and data job (MACHINE_H, measured on pk-tc-s150-L8 on 26 Oct 2026: pod created
  17:27:08, data FINISHED 17:49:02), once for each of the 12 machines of B, S-gist, P16 and X; D follows B on
  B's machines (SFT and post only).
Cells: B and D at 60m_L8, S-gist at its own probe (60m_L8 + state slots), P16 at 60m_L16, X at its own probe
(60m_L16 + llama_attnres); X's post job is costed at P16's measured times (X's generation is not measured here).
USD at $0.59 per L4-hour (RunPod). The approved planning number for L* = 8 is 126 VM-hours ($74); a measured total
above 1.5 times it goes back to Jonathan before launch (section 10).
"""
import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
S0 = os.path.join(HERE, "stage0")
E1_PROBES = os.path.join(HERE, "..", "tiny_chat_2026-10", "stage0", "attempt1", "s0_60m")
TOKENS = 163_840_000
USD = 0.59
MACHINE_H = (49 * 60 + 2 - (27 * 60 + 8)) / 3600  # 0.365 h
APPROVED = {"vm_hours": 126, "usd": 74}
SEEDS = 3


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def probe_rate(s0, name):
    p = os.path.join(s0, f"{name}.summary.json")
    if not os.path.exists(p):
        return None, None
    s = jload(p)
    return (s["tok_per_s_steady"] if s.get("rc") == 0 and s.get("tok_per_s_steady") else None), s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign-usd", type=float, default=None, help="the pre-freeze campaign's RunPod ledger")
    ap.add_argument("--stage0", default=S0, help="the folder holding the receipts (default stage0/)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    s0 = a.stage0
    walls = jload(os.path.join(s0, "e1_60m_walls_t512.json"))["jobs"]

    def med(kind, shape, key):
        xs = [v[key] for k, v in walls.items() if k.startswith(f"{kind}_e1_{shape}_r") and not v.get("partial")
              and v.get(key)]
        return statistics.median(xs) if xs else None

    shape_of = {"60m_L8": "60m_L8", "60m_L16": "60m_L16"}
    base = {}
    for sh in shape_of:
        sft = med("post", sh, "sft_s")
        base[sh] = {"pt_s_512": med("pt", sh, "wall_s"), "sft_s_512": sft,
                    "post_rest_s": med("post", sh, "wall_s") - sft,
                    "probe_512": jload(os.path.join(E1_PROBES, f"probe_{sh}", "summary.json"))["tok_per_s_steady"]}
    rates, probes = {}, {}
    for name in ("probe60_B_T1024", "probe60_B_T1024_acc2", "probe60_Sgist_T1024", "probe60_P16_T1024",
                 "probe60_X_T1024"):
        rates[name], probes[name] = probe_rate(s0, name)
    b_layout = "probe60_B_T1024" if rates["probe60_B_T1024"] else "probe60_B_T1024_acc2"
    cost = {}
    for sh, pick in (("60m_L8", "pick_e1_60m_L8"), ("60m_L16", "pick_e1_60m_L16"), ("60m_L32", "pick_e1_60m_L32")):
        p = os.path.join(s0, f"cost_{pick}.l0.json")
        if os.path.exists(p):
            r = jload(p)
            c = r["cost_run"]
            build = ((r["eval"].get("ember_build") or {}).get("seconds") or 0) if \
                (r["eval"].get("ember_build") or {}).get("built_here") else 0
            cost[sh] = {"pick": pick, "eval_s": c["eval_wall_s"], "ember_build_s": build, "timing": c["timing"],
                        "gen_engine": c.get("gen_engine"), "served_ctx": c["served_ctx"]}
    missing = [n for n in (b_layout, "probe60_Sgist_T1024", "probe60_P16_T1024", "probe60_X_T1024") if not rates[n]]
    missing += [f"cost_{sh}" for sh in ("60m_L8", "60m_L16") if sh not in cost]
    if missing:
        print(json.dumps({"incomplete": missing}))
        return 1
    cells = {"B": ("60m_L8", b_layout), "D": ("60m_L8", b_layout), "S-gist": ("60m_L8", "probe60_Sgist_T1024"),
             "P16": ("60m_L16", "probe60_P16_T1024"), "X": ("60m_L16", "probe60_X_T1024")}
    rows = {}
    for cell, (sh, probe) in cells.items():
        b = base[sh]
        ratio = b["probe_512"] / rates[probe]
        pt = 0.0 if cell == "D" else b["pt_s_512"] * ratio / 3600
        sft = b["sft_s_512"] * ratio / 3600
        c12 = (cost[sh]["eval_s"] + cost[sh]["ember_build_s"]) / 3600
        per = pt + sft + b["post_rest_s"] / 3600 + c12
        machine = 0.0 if cell == "D" else MACHINE_H
        rows[cell] = {"shape": sh, "probe": probe, "tok_per_s_1024": round(rates[probe]), "step_ratio_512_to_1024":
                      round(ratio, 4), "pt_h": round(pt, 2), "sft_h": round(sft, 2),
                      "post_rest_h": round(b["post_rest_s"] / 3600, 2), "carry12_post_h": round(c12, 2),
                      "per_run_h": round(per, 2), "machine_h": round(machine, 2),
                      "runs": SEEDS, "vm_hours": round(SEEDS * (per + machine), 1)}
    core = round(sum(r["vm_hours"] for r in rows.values()), 1)
    out = {"_comment": __doc__.split("\n\n")[1].strip().splitlines()[0] + " See the docstring of "
                       "experiments/tiny_chat_long_2026-10/stage0_total.py for the formula.",
           "tokens": TOKENS, "usd_per_l4_hour": USD, "machine_h": round(MACHINE_H, 3), "base_t512": base,
           "probes_t1024": {k: {"tok_per_s_steady": rates[k], "peak_gpu_mem_mib": (v or {}).get("peak_gpu_mem_mib"),
                                "rc": (v or {}).get("rc")} for k, v in probes.items()},
           "b_layout": b_layout, "carry12_post": cost, "cells": rows, "core_vm_hours": core,
           "core_usd": round(core * USD, 2), "approved": APPROVED,
           "ratio_to_approved": round(core / APPROVED["vm_hours"], 3),
           "back_to_jonathan_rule": "above 1.5 x the approved planning number",
           "above_rule": core > 1.5 * APPROVED["vm_hours"]}
    if a.campaign_usd is not None:
        out["pre_freeze_campaign"] = {"usd": a.campaign_usd, "l4_hours": round(a.campaign_usd / USD, 2)}
    with open(a.out or os.path.join(s0, "total.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("| cell | shape | T 1,024 tok/s | PT h | SFT h | post (rest) h | Carry-12 post h | per run h | VM-hours (3) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for cell, r in rows.items():
        print(f"| {cell} | {r['shape']} | {r['tok_per_s_1024']:,} | {r['pt_h']} | {r['sft_h']} | {r['post_rest_h']} | "
              f"{r['carry12_post_h']} | {r['per_run_h']} | {r['vm_hours']} |")
    print(f"core {core} VM-hours, ${core * USD:.0f}; {out['ratio_to_approved']} x the approved 126")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
