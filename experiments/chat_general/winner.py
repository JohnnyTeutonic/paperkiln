#!/usr/bin/env python3
"""The winner rule of PLAN.md ("Choosing the winner", fixed 25 Oct 2026, and amendment A1), applied
to one chat_eval_vm.sh run.

    python experiments/chat_general/winner.py RESULTS_DIR [--out verdict.json]

RESULTS_DIR holds the unpacked chat_eval_results.tar.gz (chat/, chat2/, qa/). Prints and writes
both verdicts: "registered" (gate: clean stop >= 0.90 on the frozen prompts and on each source of
chat_prompts_v1) and "A1" (gate: protocol failures, a reply ending on a generated role marker or
containing one, at most 0.10 on each). Then the same ranking: simple-QA correct answers out of 60;
the top gated model wins with a lead of at least 4; models within 3 of the top tie, broken by
ultrachat reference bits/byte (lower), then, within 0.01, by ultrachat repeat fraction, then size.
"""
import argparse
import json
import os

MARKERS = ("<|user|>", "<|assistant|>")
LEAD, BPB_TIE = 4, 0.01


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def protocol_failures(samples, prompts, source=None):
    rs = [r for r in samples if source is None or prompts[r["prompt_index"] % len(prompts)].get("source") == source]
    bad = sum(r["stop"] == "turn" or any(m in r["completion"] for m in MARKERS) for r in rs)
    return bad / len(rs) if rs else 0.0, len(rs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out")
    ap.add_argument("--markdown", help="write the result tables (markdown) here")
    ap.add_argument("--prompts2", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                       "corpus", "eval", "chat_prompts_v1.jsonl"))
    a = ap.parse_args(argv)
    R = a.results
    chat = json.load(open(f"{R}/chat/summary.json"))
    chat2 = json.load(open(f"{R}/chat2/summary.json"))
    qa = json.load(open(f"{R}/qa/qa_summary.json"))
    man = json.load(open(f"{R}/manifest.json")) if os.path.exists(f"{R}/manifest.json") else {"models": {}}
    p2 = read_jsonl(a.prompts2)
    sources = sorted({p.get("source", "") for p in p2})
    rows = {}
    for t in qa:
        n_qa = round(qa[t]["all"] * 60)
        reg = {"frozen": chat[t]["clean_stop"], **{s: chat2[t]["by_source"][s]["clean_stop"] for s in sources}}
        s1 = read_jsonl(f"{R}/chat/{t}_samples.jsonl")
        s2 = read_jsonl(f"{R}/chat2/{t}_samples.jsonl")
        a1 = {"frozen": protocol_failures(s1, None)[0],
              **{s: protocol_failures(s2, p2, s)[0] for s in sources}}
        uc = chat2[t]["by_source"]["ultrachat"]
        arch = man["models"].get(t, {}).get("arch") or {}
        rows[t] = {"qa_correct": n_qa, "qa_all": qa[t]["all"], "clean_stop": reg, "protocol_fail": a1,
                   "gate_registered": all(v >= 0.90 for v in reg.values()),
                   "gate_A1": all(v <= 0.10 for v in a1.values()),
                   "ultrachat_ref_bpb": uc["ref_bpb"], "ultrachat_ref_truncated": uc["ref_truncated"],
                   "ultrachat_ref_dropped": uc["ref_dropped"], "ultrachat_repeat_frac": uc["repeat_frac"],
                   "size": json.dumps(arch.get("custom", {}), sort_keys=True), "moe": bool(arch.get("moe"))}

    def verdict(gate):
        gated = [t for t in rows if rows[t][gate]]
        if not gated:
            return {"winner": None, "why": "no model passes the gate", "gated": []}
        top = max(rows[t]["qa_correct"] for t in gated)
        tied = [t for t in gated if top - rows[t]["qa_correct"] <= LEAD - 1]
        if len(gated) == 1:
            return {"winner": gated[0], "why": f"the only model through the gate (simple-QA {top}/60)", "gated": gated}
        if len(tied) == 1:
            return {"winner": tied[0], "why": f"simple-QA {top}/60, a lead of at least {LEAD}", "gated": gated}
        tied.sort(key=lambda t: rows[t]["ultrachat_ref_bpb"])
        best = [t for t in tied if rows[t]["ultrachat_ref_bpb"] - rows[tied[0]]["ultrachat_ref_bpb"] < BPB_TIE]
        if len(best) == 1:
            return {"winner": best[0], "why": f"tie on simple-QA among {tied}; lowest ultrachat ref bits/byte",
                    "gated": gated}
        best.sort(key=lambda t: rows[t]["ultrachat_repeat_frac"])
        return {"winner": best[0], "why": f"tie on QA and bits/byte among {best}; lowest ultrachat repeat fraction "
                "(size next, by hand, if this ties too)", "gated": gated}

    res = {"rows": rows, "registered": verdict("gate_registered"), "A1": verdict("gate_A1")}
    print(json.dumps({k: res[k] for k in ("registered", "A1")}, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    if a.markdown:
        kinds = ("fact", "science", "arith", "turn")
        L = ["Simple-QA (`simple_qa_v1`, 60 questions, greedy):", "",
             "| model | correct | all | " + " | ".join(kinds) + " |", "|---|---:|---:|" + "---:|" * len(kinds)]
        for t in qa:
            L.append(f"| {t} | {rows[t]['qa_correct']}/60 | {qa[t]['all']:.3f} | "
                     + " | ".join(f"{qa[t].get(k, 0):.2f}" for k in kinds) + " |")
        L += ["", "Chat (temperature 0.7, 64 new tokens; per prompt set and source):", "",
              "| model | set | samples | clean stop | length stops | protocol fails | repeats | ref F1 | "
              "prompt overlap | words | ref bits/byte (cut, dropped) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for t in qa:
            sets = [("frozen", chat[t])] + [(s, chat2[t]["by_source"][s]) for s in sources]
            for name, m in sets:
                L.append(f"| {t} | {name} | {m['samples']} | {m['clean_stop']:.3f} | "
                         f"{1 - m['clean_stop'] - rows[t]['protocol_fail'][name]:.3f} | "
                         f"{rows[t]['protocol_fail'][name]:.3f} | {m['repeat_frac']:.3f} | {m['ref_f1']:.3f} | "
                         f"{m['prompt_overlap']:.3f} | {m['words']:.1f} | {m['ref_bpb']:.3f} "
                         f"({m['ref_truncated']}, {m['ref_dropped']}) |")
        open(a.markdown, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
