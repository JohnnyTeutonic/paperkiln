#!/usr/bin/env python3
"""The verifier for T_E rows: checks 1 to 6 and 8 of the author's brief. Claude Opus 5.5, 7 Nov 2026. Counts and row ids only; no text leaves this script.

It shares no code with build_te.py and was written without reading it. It imports rowcheck.py, te_score.py and, for the two forbidden sequences, mixed_gen's own held-out detectors through the pinned import. Check 7, the overlap filter, is te_filter.py.
Check 3 follows reviews/te_check3_ruling.md, written before this first ran: 3a is the brief's check for the user's own values, 3b admits a character's name the assistant introduced, under four conditions.

  python3 tools/lab/te_verify.py --dir T_E_FOLDER --rows te_stage_a_v1.jsonl --receipt te_stage_a_v1_receipt.json --tables TABLES.json --out REPORT.json [--expect-rows 700]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters as AD  # noqa: E402
import rowcheck as RC  # noqa: E402
import shortcuts as SH  # noqa: E402
import te_score as TS  # noqa: E402
from firewall import check_shortcut_read  # noqa: E402
from schema import sha_file  # noqa: E402

INPUTS = {"tables": "055b6a3634671b7305545b2df8b9d493b56769d15208ffc39bdd0cb3fb636d6e", "attr_meta": "29600559518b02eb51757ec4c06d0889a90487ffac6b97fb3d2d3b3c1848e361"}
STAGE_A = {"like_recall": {("like_recall", "none"): 20, ("like_affirm", "none"): 20, ("like_unknown", "none"): 10},
           "role_mix": {("recall", "role"): 20, ("user_name", "role"): 15, ("abstain", "role"): 15},
           "ended_role": {("recall", "after"): 20, ("user_name", "after"): 10, ("abstain", "after"): 10, ("self_name", "after"): 10},
           "name_use": {("user_name", "none"): 25, ("recall", "none"): 10, ("both_names", "none"): 15},
           "nameless_role": {("rp_sincere", "role"): 20, ("both_names", "role"): 15, ("user_name_unknown", "role"): 15},
           "how_are_you": {("self_name", "none"): 25, ("self_like", "none"): 10, ("abstain", "none"): 15},
           "affirm_in_role": {("like_affirm", "role"): 30, ("like_recall", "role"): 20}}
ROW_FIELDS = ("conversation_id", "segments", "retained_assistant_turns", "provenance", "decision")
PROV_FIELDS = ("kind", "generator_version", "seed", "skill", "family", "pair_id", "variant", "changed_factor", "changed_turns", "values", "turns", "trained", "llm_written", "correction_part", "correction_split")
DEC_FIELDS = ("turn", "k", "c", "form", "template_group", "key", "told_in_order")
USER_VALUE_OPS = ("recall", "user_name", "like_recall", "like_affirm")
ASSISTANT_NAMED_OPS = ("both_names", "rp_in")
OWN_DESIGN_OPS = ("self_name", "self_like", "rp_sincere")     # the assistant's own identity or preference: nothing told, so check 3 does not apply, and they are counted
OWN_NAME = "Ember"     # a self-name value, known from identity and not from the row's history: the lead's ruling of 7 Nov 2026, and this one value only
VALUE_TABLES = ("pool:", "role_list", "role_place:", "role_act:", "rel", "fav_thing", "topic_np", "travel_place", "opinion_pair")
SKIP_FRAME = VALUE_TABLES


def read_rows(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def norm(s):
    return re.sub(r"^(a|an|the) ", "", str(s).strip(), flags=re.I).lower()


def frames(tables, attr):
    """Exact strings and slot patterns, from the two inputs. A slot matches any text; values are checked apart."""
    exact, patterns = set(), []
    entries = [e for name, v in tables.items() if not name.startswith(SKIP_FRAME) for e in v if isinstance(e, str)]
    for e in entries:
        if "{" in e:
            patterns.append(re.sub(r"\\\{[A-Za-z_0-9]+\\\}", ".+?", re.escape(e).replace(r"\{s_\}", "s?")))  # {s_} is a verb's s, empty after a plural
        else:
            exact.add(e.lower())
    return exact, re.compile("(?:" + "|".join(sorted(set(patterns), key=len, reverse=True)) + ")", re.I | re.S)


def fits(text, exact, rx):
    t = text.strip()
    if t.lower() in exact or rx.fullmatch(t):
        return True
    parts = [x for x in re.split(r"(?<=[.!?])\s+", t) if x]
    return len(parts) > 1 and all(x.lower() in exact or rx.fullmatch(x) for x in parts)


def check_format(rows, problems):
    seen = set()
    for r in rows:
        cid = r.get("conversation_id", "?")
        for f in ROW_FIELDS:
            if f not in r:
                problems[f"missing:{f}"] += 1
        p, d = r.get("provenance", {}), r.get("decision", {})
        for f in PROV_FIELDS:
            if f not in p:
                problems[f"missing:provenance.{f}"] += 1
        for f in DEC_FIELDS:
            if f not in d:
                problems[f"missing:decision.{f}"] += 1
        if cid in seen:
            problems["duplicate_id"] += 1
        seen.add(cid)
        if p.get("correction_part") != "T_E" or p.get("correction_split") != "correction-v1" or p.get("variant") not in ("a", "b") or p.get("trained") != "none":
            problems["provenance_values"] += 1
        if not re.fullmatch(r"te-[a-z_]+-\d+-[ab]", cid):
            problems["id_pattern"] += 1
        turns = RC.parse_turns(r)
        if len(turns) != len(p.get("turns", [])):
            problems["turns_do_not_align"] += 1
            continue
        if not all(u.strip() and a.strip() for u, a in turns):
            problems["empty_turn"] += 1
        if d.get("turn") != len(turns) - 1:
            problems["decision_not_final"] += 1
        if d.get("k") not in {m.get("k") for m in p["turns"]}:
            problems["decision_k_not_in_metas"] += 1
        if not 3 <= len(turns) <= 12:
            problems["length"] += 1
        if d.get("c") not in ("none", "role", "after") or d.get("form") not in ("S", "D"):
            problems["decision_values"] += 1
        if not re.fullmatch(r"[0-9a-f]{12}", str(d.get("template_group"))):
            problems["template_group"] += 1


def check_entries(rows, tables, attr, others, out):
    exact, rx = frames(tables, attr)
    un = collections.Counter()
    n = 0
    for r in rows:
        for u, a in RC.parse_turns(r):
            for who, t in (("user", u), ("reply", a)):
                n += 1
                if not fits(t, exact, rx):
                    un[who] += 1
    pool = set()
    for name, v in tables.items():
        if name.startswith(VALUE_TABLES):
            pool |= {norm(x) for x in v if isinstance(x, str)}
    elsewhere = {norm(x) for name, v in others.items() if name.startswith(VALUE_TABLES) for x in v if isinstance(x, str)} - pool
    bad, wrong_part, checked = 0, 0, 0
    for r in rows:
        for m in r["provenance"]["turns"]:
            vals = [m[k] for k in ("value", "other", "name", "role") if isinstance(m.get(k), str)] + [x for x in (m.get("likes") or []) if isinstance(x, str)]
            for v in vals:
                checked += 1
                if norm(v) in elsewhere:
                    wrong_part += 1
                elif norm(v) not in pool:
                    bad += 1
    out["2_entries"] = {"turns_checked": n, "turns_fitting_no_entry": dict(un), "meta_values_checked": checked, "meta_values_in_another_part": wrong_part, "meta_values_in_no_table": bad}


def told_sources(row):
    turns = RC.parse_turns(row)
    i = row["decision"]["turn"]
    return " ".join(u for u, _ in turns[:i]), turns, row["provenance"]["turns"]


def check_values(rows, out):
    """Reviews/te_check3_ruling.md."""
    a_ok, a_bad, b_ok, b_bad, own = collections.Counter(), collections.Counter(), 0, collections.Counter(), collections.Counter()
    no_source = collections.Counter()
    self_name = 0
    for r in rows:
        d = r["decision"]
        fam = r["provenance"]["family"]
        if d["k"] in OWN_DESIGN_OPS:
            own[d["k"]] += 1
            continue
        if d["k"] not in USER_VALUE_OPS + ASSISTANT_NAMED_OPS:
            continue
        users, turns, metas = told_sources(r)
        i = d["turn"]
        for v in d["key"]["accept"]:
            if RC.has(users, v):
                a_ok[d["k"]] += 1
                continue
            if v == OWN_NAME:
                self_name += 1
                continue
            if d["k"] not in ASSISTANT_NAMED_OPS:
                a_bad[f"{fam}/{d['k']}"] += 1
                continue
            is_name = [j for j, m in enumerate(metas[:i]) if m.get("k") in ("rp_nameless", "rp_in") and m.get("name") is not None and norm(m["name"]) == norm(v)]
            said = [j for j in range(i) if RC.has(turns[j][1], v)]
            if not is_name:
                b_bad[f"{fam}/{d['k']}/not_the_character_name"] += 1
            elif not said:
                no_source[f"{fam}/{d['k']}"] += 1
            elif any(RC.has(" ".join(u for u, _ in turns[: j + 1]), v) for j in said[:1]):
                b_bad[f"{fam}/{d['k']}/user_said_it_first"] += 1
            else:
                b_ok += 1
    out["3_values"] = {"3a_user_values_held": sum(a_ok.values()), "3a_defects": dict(a_bad), "3b_character_names_held": b_ok, "3b_defects": dict(b_bad), "self_name_values_out_of_scope": self_name, "no_source_in_any_turn": dict(no_source), "assistant_own_decisions_not_checked_here": dict(own),
                       "rowcheck": RC.check_rows(rows) | {"families": None}}


def pairs_of(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        p = r["provenance"]
        by[(p["family"], p["pair_id"])][p["variant"]] = r
    return by


def check_twins(rows, cues, out):
    by = pairs_of(rows)
    bad = collections.Counter()
    gold_ok, swap_ok, n = 0, 0, 0
    fail = collections.Counter()
    for key, v in by.items():
        if set(v) != {"a", "b"}:
            bad["not_two_rows"] += 1
            continue
        a, b = v["a"], v["b"]
        ta, tb = RC.parse_turns(a), RC.parse_turns(b)
        if len(ta) != len(tb):
            bad["turn_count_differs"] += 1
            continue
        diff = [i for i, (x, y) in enumerate(zip(ta, tb)) if x[0] != y[0]]
        if not diff:
            bad["no_user_turn_differs"] += 1
        if len(diff) > 1:
            bad["more_than_one_user_turn_differs"] += 1
        if sorted(a["provenance"]["changed_turns"]) != sorted(b["provenance"]["changed_turns"]):
            bad["changed_turns_disagree"] += 1
        if diff != sorted(a["provenance"]["changed_turns"]):
            bad["changed_turns_not_the_difference"] += 1
        if a["decision"]["key"] == b["decision"]["key"]:
            bad["same_key"] += 1
        ga, gb = ta[-1][1], tb[-1][1]
        n += 2
        for gold, own, other in ((ga, a, b), (gb, b, a)):
            if TS.passes(gold, own, cues):
                gold_ok += 1
            else:
                fail["gold_fails_own_key:" + own["decision"]["k"]] += 1
            if not TS.passes(gold, other, cues):
                swap_ok += 1
            else:
                fail["gold_passes_twin_key:" + other["decision"]["k"]] += 1
    out["4_twins"] = {"pairs": len(by), "structural_defects": dict(bad), "gold_passes_own_key": f"{gold_ok} of {n}", "gold_fails_twin_key": f"{swap_ok} of {n}", "by_kind": dict(sorted(fail.items()))}


def told_from_metas(row):
    out = []
    for m in row["provenance"]["turns"][:row["decision"]["turn"]]:
        if m.get("k") in ("fact", "ack_name") and m.get("value") is not None:
            out.append(norm(m["value"]))
        if m.get("sub") == "like" and m.get("tell"):
            out += [norm(w) for w in (m.get("words") or [])]
    return out


def mask(row, text):
    vals = sorted({norm(x) for x in _flat(row["provenance"]["values"]) if len(str(x)) > 1}, key=len, reverse=True)
    t = text.lower()
    for v in vals:
        t = t.replace(v, "#")
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", t)).strip()


def _flat(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from _flat(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _flat(x)
    elif v is not None:
        yield v


def check_counts(rows, spec, out):
    by = pairs_of(rows)
    got = collections.defaultdict(collections.Counter)
    forms = collections.Counter()
    for (fam, _), v in by.items():
        b = v.get("b") or next(iter(v.values()))
        op = b["provenance"]["values"].get("pair_op") or b["decision"]["k"]
        got[fam][(op, b["decision"]["c"])] += 1
        forms[(fam, b["decision"]["form"])] += 1
    misses = {}
    for fam, want in spec.items():
        for k, n in want.items():
            if got[fam][k] != n:
                misses[f"{fam}/{k[0]}/{k[1]}"] = [got[fam][k], n]
    few, recent, dec = collections.Counter(), collections.Counter(), collections.Counter()
    words = collections.defaultdict(set)
    dup_conv, dup_val = collections.Counter(), collections.Counter()
    seen_conv, seen_val = set(), set()
    for r in rows:
        d, fam = r["decision"], r["provenance"]["family"]
        turns = RC.parse_turns(r)
        told = told_from_metas(r)
        if d["k"] in ("recall", "user_name") and len(set(told)) < 2:
            few[fam] += 1
        if d["k"] in USER_VALUE_OPS and d["key"]["accept"]:
            dec[fam] += 1
            if told and told[-1] in {norm(x) for x in d["key"]["accept"]}:
                recent[fam] += 1
        words[fam].add(d["template_group"])
        cv = (fam, tuple(mask(r, u) for u, _ in turns), tuple(sorted(str(x) for x in _flat(r["provenance"]["values"]))))
        if cv in seen_conv:
            dup_conv[fam] += 1
        seen_conv.add(cv)
    for (fam, _), v in by.items():
        t = (fam, tuple(sorted(str(x) for r in v.values() for x in _flat(r["provenance"]["values"].get("told", [])))), tuple(sorted(str(x) for r in v.values() for x in _flat(r["provenance"]["values"].get("asked", [])))))
        if t in seen_val:
            dup_val[fam] += 1
        seen_val.add(t)
    out["5_counts"] = {"table_misses": misses, "pairs_by_form": {f"{a}/{b}": n for (a, b), n in sorted(forms.items())}, "recall_or_name_decisions_under_two_told": dict(few), "asked_is_most_recent_by_family": {f: [recent[f], dec[f], 100] for f in sorted(dec)},
                       "most_recent_over_half": sorted(f for f in dec if recent[f] > 50), "distinct_final_wordings": {f: len(v) for f, v in sorted(words.items())}, "repeated_conversations": dict(dup_conv), "repeated_value_tuples": dict(dup_val)}


def check_sequences(rows, out):
    mg = AD.load_frozen("mixed_gen.py")
    txt, ev, corr = 0, 0, 0
    for r in rows:
        users = [u for u, _ in RC.parse_turns(r)]
        if mg.heldout_text(users):
            txt += 1
        if mg.heldout_events(r["provenance"]["turns"], users):
            ev += 1
        attrs = collections.defaultdict(set)
        for m in r["provenance"]["turns"]:
            if m.get("k") == "fact" and m.get("attr") is not None:
                attrs[str(m["attr"])].add(norm(m.get("value")))
        corr += any(len(v) > 1 for v in attrs.values())
    out["6_sequences"] = {"rows_flagged_by_text": txt, "rows_flagged_by_metas": ev, "rows_with_one_attribute_told_two_values": corr}


def shortcut_read(rows, cues, out):
    frozen = AD.mine_score().POLICIES
    by = pairs_of(rows)
    pairs = [(v["a"], v["b"]) for v in by.values() if set(v) == {"a", "b"}]

    def item(r):
        return {"value": None, "told": list(r["decision"]["told_in_order"]), "last_user": RC.parse_turns(r)[-1][0], "row": None}

    def passes(reply, it):
        return TS.passes(reply, it["_row"], cues)
    items = [(dict(item(a), _row=a), dict(item(b), _row=b)) for a, b in pairs]
    overall = SH.read(items, passes, frozen)
    fams = sorted({a["provenance"]["family"] for a, _ in pairs})
    per = {f: SH.read([x for x in items if x[0]["_row"]["provenance"]["family"] == f], passes, frozen) for f in fams}
    verdict = "held"
    try:
        check_shortcut_read(overall, SH.SHORTCUT_BAR_POINTS)
    except Exception as e:
        verdict = str(e)
    out["shortcut_read"] = {"bar_points": SH.SHORTCUT_BAR_POINTS, "definitions_sha": SH.definitions_sha(), "overall": overall, "by_family": per, "result": verdict, "pairs": len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--receipt")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--other-tables", nargs="*", default=[])
    ap.add_argument("--attr", required=True)
    ap.add_argument("--cues", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-rows", type=int, default=700)
    ap.add_argument("--shortcuts-only", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    out = {"producer": "Claude Opus 5.5, 7 Nov 2026", "rows_file": a.rows, "rows_sha256": sha_file(d / a.rows)}
    rows = read_rows(d / a.rows)
    cues = TS.load_cues(a.cues)
    if not a.shortcuts_only:
        out["1_inputs"] = {"tables": sha_file(a.tables) == INPUTS["tables"], "attr_meta": sha_file(a.attr) == INPUTS["attr_meta"]}
        problems = collections.Counter()
        check_format(rows, problems)
        out["1_format"] = {"rows": len(rows), "expected": a.expect_rows, "problems": dict(problems)}
        with open(a.tables, encoding="utf-8") as f:
            tables = json.load(f)
        with open(a.attr, encoding="utf-8") as f:
            attr = json.load(f)
        others = {}
        for p in a.other_tables:
            with open(p, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    others.setdefault(k, []).extend(v)
        check_entries(rows, tables, attr, others, out)
        check_values(rows, out)
        check_twins(rows, cues, out)
        check_counts(rows, STAGE_A, out)
        check_sequences(rows, out)
        if a.receipt:
            rec = json.load(open(d / a.receipt, encoding="utf-8"))
            out["8_hashes"] = {"rows_match_receipt": rec["output_sha256"].get(a.rows) == hashlib.sha256((d / a.rows).read_bytes()).hexdigest(), "inputs_match_receipt": rec["input_sha256"].get("T_E.json") == INPUTS["tables"] and rec["input_sha256"].get("attr_meta.json") == INPUTS["attr_meta"]}
    shortcut_read(rows, cues, out)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
