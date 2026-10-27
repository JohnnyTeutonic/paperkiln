#!/usr/bin/env python3
"""Tables for conversation schema v2's measurement (RESULTS.md, "Conversation schema v2"): instruction
placement on IF-tiny (system vs user turn, bare base rate), per family, the turn-distance contrasts, and the
tool-call set.

    python experiments/chat_general/v2_placement.py IFTINY_DIR [--tools TOOLS_DIR] [--md OUT.md]

IFTINY_DIR holds one folder per model (tools/iftiny_eval.py --placement: summary.json, items.jsonl.gz).
"""
import argparse
import gzip
import json
import math
import os


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iftiny")
    ap.add_argument("--tools")
    ap.add_argument("--md")
    a = ap.parse_args()
    L = []
    models = sorted(t for t in os.listdir(a.iftiny) if os.path.exists(os.path.join(a.iftiny, t, "summary.json")))
    S = {t: json.load(open(os.path.join(a.iftiny, t, "summary.json"), encoding="utf-8")) for t in models}
    L += ["Placement (574 IF-tiny items, greedy, 64 new tokens): compliance per condition, micro over items "
          "[Wilson 95%] and macro over 15 families", "",
          "| model | user | system | bare | macro user | macro system | macro bare | system - user [95%] | "
          "system only / user only |", "|---|---|---|---|---|---|---|---|---|"]
    for t in models:
        p = S[t]["placement"]
        mi, ma, su = p["micro"], p["macro"], p["system_minus_user"]

        def c(x):
            return f"{x['rate']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
        L.append(f"| {t} | {c(mi['user'])} | {c(mi['system'])} | {c(mi['bare'])} | {ma['user']:.3f} | "
                 f"{ma['system']:.3f} | {ma['bare']:.3f} | {su['diff']:+.3f} [{su['ci95'][0]:+.3f}, "
                 f"{su['ci95'][1]:+.3f}] | {su['system_only']} / {su['other_only']} |")
    fams = sorted({f for t in models for f in S[t]["placement"]["families"]})
    L += ["", "Per family: user / system / bare", "", "| family | " + " | ".join(models) + " |",
          "|---|" + "---|" * len(models)]
    for f in fams:
        cells = []
        for t in models:
            x = S[t]["placement"]["families"].get(f)
            cells.append(f"{x['user']['rate']:.2f} / {x['system']['rate']:.2f} / {x['bare']['rate']:.2f} "
                         f"(n {x['user']['n']})" if x else "-")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    # turn distance: system minus user per distance, paired over (family, sequence)
    L += ["", "Turn distance (84 conversations per cell): instr_user, instr_system, nohint; system - user paired",
          "", "| model | d | user | system | nohint | system - user [95%] |", "|---|---|---|---|---|---|"]
    for t in models:
        with gzip.open(os.path.join(a.iftiny, t, "items.jsonl.gz"), "rt", encoding="utf-8") as fh:
            recs = [json.loads(x) for x in fh if x.strip()]
        cells = {}
        for r in recs:
            if r["set"] != "ifdist":
                continue
            _, fam, seq, dd, cond = r["id"].split("/")
            cells.setdefault((dd, cond), {})[(fam, seq)] = int(r["comply"])
        for dd in ("d0", "d1", "d2", "d4"):
            if (dd, "instr_user") not in cells:
                continue
            keys = sorted(cells[(dd, "instr_user")])
            u = [cells[(dd, "instr_user")][k] for k in keys]
            s = [cells[(dd, "instr_system")][k] for k in keys]
            nh = [cells[(dd, "nohint")][k] for k in keys]
            m, lo, hi = paired(s, u)
            L.append(f"| {t} | {dd} | {sum(u) / len(u):.3f} | {sum(s) / len(s):.3f} | {sum(nh) / len(nh):.3f} | "
                     f"{m:+.3f} [{lo:+.2f}, {hi:+.2f}] |")
    L += ["", "| model | prompt check (mismatched / checked, with system) | stops eos / turn / length |",
          "|---|---|---|"]
    for t in models:
        pc, st = S[t].get("prompt_check", {}), S[t].get("stops", {})
        L.append(f"| {t} | {pc.get('mismatched')} / {pc.get('checked')} ({pc.get('with_system')}) | "
                 f"{st.get('eos')} / {st.get('turn')} / {st.get('length')} |")
    if a.tools:
        L += ["", "Tool calls (corpus/eval/tool_calls_v1.jsonl, greedy, 96 new tokens)", "",
              "| model | call made | right name | arguments JSON | arguments equal | false calls | stops eos / turn / length |",
              "|---|---|---|---|---|---|---|"]
        for f in sorted(os.listdir(a.tools)):
            if not f.endswith("_tool_summary.json"):
                continue
            x = json.load(open(os.path.join(a.tools, f), encoding="utf-8"))
            n = x["with_call_items"]
            L.append(f"| {f[:-len('_tool_summary.json')]} | {x['call_made']}/{n} | {x['name_correct']}/{n} | "
                     f"{x['arguments_json']}/{n} | {x['arguments_equal']}/{n} | {x['false_calls']}/{x['no_call_items']} | "
                     f"{x['stops']['eos']} / {x['stops']['turn']} / {x['stops']['length']} |")
    text = "\n".join(L) + "\n"
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
