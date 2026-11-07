"""Tests for tools/lab. Synthetic fixtures only. Claude Opus 5.5, 7 Nov 2026.

    python3 -m pytest tools/lab -q     or     python3 -m unittest discover -s tools/lab
"""
import ast
import json
import os
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controller as C  # noqa: E402
import exposure as X  # noqa: E402
import firewall as FW  # noqa: E402
import gate as G  # noqa: E402
import history as H  # noqa: E402
import miner as M  # noqa: E402
import proposer as P  # noqa: E402
import rowcheck as SC  # noqa: E402
import shortcuts as SH  # noqa: E402
import te_merge as TM  # noqa: E402
import te_score as TS  # noqa: E402
import te_verify as TV  # noqa: E402
import sim as S  # noqa: E402
import synthetic as SY  # noqa: E402
import taxonomy as T  # noqa: E402
import verdict as V  # noqa: E402
from ledger import Ledger, LedgerCorrupt  # noqa: E402
from schema import GENESIS, Malformed, canon, sha  # noqa: E402

PROD, DATE = SY.PRODUCER, SY.DATE
JON = "Jonathan, 7 Nov 2026"


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "ledger.jsonl"
        self.led = Ledger(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def demo(self, legs=2):
        return SY.write_demo(self.path, legs)

    def rewrite(self, lines):
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def lines(self):
        return self.path.read_text(encoding="utf-8").split("\n")[:-1]


class TestLedger(Tmp):
    def test_chain_and_reload(self):
        self.demo()
        recs = self.led.load()
        self.assertEqual(len(recs), 9)
        self.assertEqual(recs[0]["prev"], GENESIS)
        self.assertEqual(recs[3]["prev"], recs[2]["sha256"])

    def test_tamper_any_field_is_refused(self):
        self.demo()
        lines = self.lines()
        rec = json.loads(lines[2])
        rec["body"]["seed"] = 99
        lines[2] = canon(rec)
        self.rewrite(lines)
        with self.assertRaises(LedgerCorrupt):
            self.led.load()

    def test_torn_tail_refused(self):
        self.demo()
        t = self.path.read_text(encoding="utf-8")
        self.path.write_text(t[:-20], encoding="utf-8")
        with self.assertRaises(LedgerCorrupt):
            self.led.load()

    def test_deleted_line_refused(self):
        self.demo()
        lines = self.lines()
        self.rewrite(lines[:3] + lines[4:])
        with self.assertRaises(LedgerCorrupt):
            self.led.load()

    def test_tail_truncation_caught_by_the_anchor(self):
        """Erasing a failed leg from the end of the file would clear a review: the anchor refuses it with no head passed in."""
        self.demo()
        self.rewrite(self.lines()[:-1])
        with self.assertRaises(LedgerCorrupt):
            self.led.load()

    def test_missing_anchor_refused(self):
        self.demo()
        os.unlink(self.led.anchor)
        with self.assertRaises(LedgerCorrupt):
            self.led.load()

    def test_expected_head_from_elsewhere(self):
        self.demo()
        head = self.led.head()
        self.led.load(expected_head=head)
        with self.assertRaises(LedgerCorrupt):
            self.led.load(expected_head="1" * 64)

    def test_no_update_or_delete_api(self):
        names = {n for n in dir(Ledger) if not n.startswith("_")}
        self.assertEqual(names, {"append", "head", "load"})

    def test_a_second_outcome_is_refused_however_it_is_written(self):
        ok = {"experiment_id": "x", "verdict": "GUARD_FAILED", "rule_sha": sha("r"), "detail": {}}
        self.led.append("outcome", ok, PROD, DATE, "registered")
        again = dict(ok, verdict="ADAPTIVE_BETTER")
        with self.assertRaises(Malformed):
            self.led.append("outcome", again, PROD, DATE, "registered", supersedes=0)
        with self.assertRaises(Malformed):
            self.led.append("outcome", again, PROD, DATE, "registered")
        with self.assertRaises(Malformed):
            self.led.append("leg", SY.leg("fixed", 0), PROD, DATE, "registered", supersedes=0)
        self.assertEqual(len(self.led.load()), 1)

    def test_outcome_verdict_is_a_known_word(self):
        with self.assertRaises(Malformed):
            self.led.append("outcome", {"experiment_id": "x", "verdict": "WIN", "rule_sha": sha("r"), "detail": {}}, PROD, DATE, "registered")

    def test_a_second_registration_is_refused(self):
        self.led.append("registration", SY.registration(), PROD, DATE, "synthetic")
        with self.assertRaises(Malformed):
            self.led.append("registration", SY.registration(), PROD, DATE, "synthetic")

    def test_failed_leg_kept_and_a_retry_supersedes_it(self):
        self.led.append("leg", SY.leg("fixed", 0, status="failed"), PROD, DATE, "registered")
        self.led.append("leg", SY.leg("fixed", 0), PROD, DATE, "registered", supersedes=0)
        recs = self.led.load()
        self.assertEqual(recs[0]["body"]["status"], "failed")
        self.assertEqual(len(recs), 2)

    def test_stale_lock_refused_not_broken(self):
        Path(str(self.path) + ".lock").write_text("")
        with self.assertRaises(LedgerCorrupt):
            self.led.append("leg", SY.leg("fixed", 0), PROD, DATE, "registered")

    def test_line_separators_in_a_quote_do_not_break_the_file(self):
        b = SY.leg("fixed", 0)
        b["generator_version"] = "a b\u0085c"
        self.led.append("leg", b, PROD, DATE, "synthetic")
        self.led.append("leg", SY.leg("fixed", 1), PROD, DATE, "synthetic")
        self.assertEqual(len(self.led.load()), 2)


class TestMalformed(Tmp):
    def bad(self, kind, body, msg=None, evidence="registered", producer=PROD, date=DATE):
        with self.assertRaises(Malformed) as cm:
            self.led.append(kind, body, producer, date, evidence)
        if msg:
            self.assertIn(msg, str(cm.exception))
        self.assertFalse(self.path.exists() and self.path.read_text())

    def test_unknown_kind(self):
        self.bad("gossip", {})

    def test_bare_claude_is_no_producer(self):
        self.bad("leg", SY.leg("fixed", 0), producer="Claude")

    def test_bad_date(self):
        self.bad("leg", SY.leg("fixed", 0), date="7 Nov")

    def test_nan_refused(self):
        b = SY.leg("fixed", 0)
        b["train"]["last_loss"] = float("nan")
        self.bad("leg", b)

    def test_infinite_loss_refused_in_read(self):
        b = SY.dev_read("fixed", 0)
        b["families"]["role_toy"]["mean"] = float("inf")
        self.bad("dev_read", b)

    def test_negative_exposure(self):
        b = SY.leg("fixed", 0)
        b["exposure"]["role_toy"]["sampled_supervised"] = -1
        self.bad("leg", b)

    def test_decision_tokens_past_supervised(self):
        b = SY.leg("fixed", 0)
        b["exposure"]["role_toy"]["sampled_decision"] = b["exposure"]["role_toy"]["sampled_supervised"] + 1
        self.bad("leg", b, "more decision tokens")

    def test_failed_leg_needs_a_mode(self):
        b = SY.leg("fixed", 0, status="failed")
        del b["failure_mode"]
        self.bad("leg", b)

    def test_short_hash(self):
        b = SY.leg("fixed", 0)
        b["dataset_manifest"] = "abc"
        self.bad("leg", b, "sha256")

    def test_leg_needs_a_plan_and_a_recipe(self):
        b = SY.leg("fixed", 0)
        del b["plan_sha"]
        self.bad("leg", b)
        b = SY.leg("fixed", 0)
        del b["recipe"]["optimiser"]
        self.bad("leg", b)

    def test_registration_one_facet(self):
        b = SY.registration()
        b["changed_facet"] = "allocation and depth"
        self.bad("registration", b, "one changed")

    def test_registration_needs_control(self):
        b = SY.registration()
        b["control_arm"] = "nobody"
        self.bad("registration", b)

    def test_registration_config_hash_must_match(self):
        b = SY.registration()
        b["config"]["eta"] = 0.9
        self.bad("registration", b, "config_sha")

    def test_decision_cannot_claim_execution(self):
        self.demo()
        recs = self.led.load()
        d = C.decide(recs, SY.config(), "fixed", 2, recs[-1]["sha256"])
        d["executed"] = True
        d["decision_sha"] = sha({k: v for k, v in d.items() if k != "decision_sha"})
        with self.assertRaises(Malformed):
            self.led.append("decision", d, PROD, DATE, "synthetic")

    def test_altered_decision_refused(self):
        self.demo()
        recs = self.led.load()
        d = C.decide(recs, SY.config(), "fixed", 2, recs[-1]["sha256"])
        d["shares"]["role_toy"] = 0.9
        with self.assertRaises(Malformed):
            self.led.append("decision", d, PROD, DATE, "synthetic")

    def test_eval_cannot_be_development_role(self):
        b = {"experiment_id": "x", "base_id": "b", "arm": "a", "instrument_id": "i", "instrument_role": "development", "set_sha": sha(1), "scorer_sha": sha(2), "results_sha": sha(3), "summary": {}, "compute": {"l4_minutes": 0, "cpu_minutes": 1}}
        self.bad("evaluation", b)


def state(reads, doses=None, multipliers=None, spent=0.0, legs=None):
    out = []
    for i in range(len(reads)):
        d = (doses or {}).get(i, {})
        out.append({"leg": i, "status": "ok", "exposure": {f: {"sampled_decision": d.get(f, 9000.0), "sampled_supervised": 27000.0} for f in reads[0]}})
    return {"reads": reads, "read_legs": list(range(len(reads))), "legs": out if legs is None else legs, "failed_legs": [], "spent_l4": spent, "multipliers": multipliers}


def read(values, se=0.04, n=24):
    v = dict(values)
    v.setdefault(SY.REPLAY, 1.0)
    return {f: {"n": n, "mean": x, "se": se} for f, x in v.items()}


class TestController(Tmp):
    def cfg(self, **kw):
        c = SY.config()
        c.update(kw)
        return c

    def test_leg_zero_is_balanced(self):
        d = C.adaptive_decide(state([]), self.cfg(), "adaptive", 0, GENESIS)
        self.assertEqual(d["action"], "ALLOCATE")
        self.assertTrue(all(v == 1.0 for v in d["multipliers"].values()))

    def test_hard_family_at_its_floor_earns_nothing(self):
        vals = {f: SY.TRACE[f][0] + 0.01 for f in SY.FAMILIES}
        vals["smalltalk_toy"] = 0.91
        vals["correction_toy"] = 1.5
        d = C.adaptive_decide(state([read(vals)]), self.cfg(), "adaptive", 1, GENESIS)
        self.assertEqual(d["families"]["smalltalk_toy"]["state"], "held")
        self.assertGreater(d["multipliers"]["correction_toy"], d["multipliers"]["smalltalk_toy"])

    def test_a_tiny_excess_earns_nothing(self):
        """Finding 9: 0.001 nats past the noise used to reach the cap, because the push was scaled by the mean."""
        vals = {f: SY.TRACE[f][0] for f in SY.FAMILIES}
        z, se = 2.0, (0.04 ** 2 + 0.03 ** 2) ** 0.5
        vals["correction_toy"] = SY.TRACE["correction_toy"][0] + z * se + 0.001
        d = C.adaptive_decide(state([read(vals)]), self.cfg(), "adaptive", 1, GENESIS)
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in d["multipliers"].values()))
        self.assertIn("saturated", d["flags"])

    def test_families_at_their_floor_rarely_move(self):
        """Noise alone, every family at its floor, many decisions: a weight moves in under 5% of them."""
        rng = random.Random(5)
        moved = 0
        for _ in range(400):
            vals = {f: SY.TRACE[f][0] + rng.gauss(0, 0.05) for f in SY.FAMILIES}
            d = C.adaptive_decide(state([read(vals, se=0.05)]), self.cfg(), "adaptive", 1, GENESIS)
            moved += any(abs(v - 1.0) > 1e-6 for v in d["multipliers"].values())
        self.assertLess(moved, 20)

    def test_saturation_stops_when_unmatched_and_keeps_the_envelope_when_matched(self):
        vals = {f: SY.TRACE[f][0] + 0.03 for f in SY.FAMILIES}
        d = C.adaptive_decide(state([read(vals)]), self.cfg(matched_pilot=False, arm_policies={"adaptive": "adaptive"}), "adaptive", 1, GENESIS)
        self.assertEqual(d["action"], "STOP_SATURATED")
        d = C.adaptive_decide(state([read(vals)]), self.cfg(matched_pilot=True), "adaptive", 1, GENESIS)
        self.assertEqual(d["action"], "ALLOCATE")
        self.assertTrue(all(v == 1.0 for v in d["multipliers"].values()))
        self.assertEqual(d["flags"], ["saturated"])

    def test_caps_hold_however_extreme(self):
        vals = {f: SY.TRACE[f][0] for f in SY.FAMILIES}
        vals["correction_toy"] = 50.0
        d = C.adaptive_decide(state([read(vals)]), self.cfg(), "adaptive", 1, GENESIS)
        for m in d["multipliers"].values():
            self.assertGreaterEqual(m, 0.5 - 1e-6)
            self.assertLessEqual(m, 2.0 + 1e-6)
        self.assertAlmostEqual(sum(d["shares"].values()), 1.0, places=4)
        self.assertEqual(d["replay_share"], 0.3)

    def test_stalled_family_goes_to_minimum_and_stays(self):
        a = {f: SY.TRACE[f][0] + 1.0 for f in SY.FAMILIES}
        b = dict(a, correction_toy=a["correction_toy"] - 0.5, recall_toy=a["recall_toy"] - 0.5)
        d = C.adaptive_decide(state([read(a), read(b)], doses={1: {f: 9000.0 for f in SY.FAMILIES}}), self.cfg(), "adaptive", 2, GENESIS)
        self.assertEqual(d["families"]["role_toy"]["state"], "stalled")
        self.assertAlmostEqual(d["multipliers"]["role_toy"], 0.5, places=4)
        self.assertEqual(d["families"]["correction_toy"]["state"], "pushed")
        s = state([read(a), read(b), read(dict(b, correction_toy=b["correction_toy"] - 0.3, recall_toy=b["recall_toy"] - 0.3))], doses={1: {f: 9000.0 for f in SY.FAMILIES}, 2: {f: 100.0 for f in SY.FAMILIES}})
        d = C.adaptive_decide(s, self.cfg(), "adaptive", 3, GENESIS)
        self.assertEqual(d["families"]["role_toy"]["state"], "stalled")

    def test_a_family_at_its_floor_is_held_not_stalled(self):
        a = {f: SY.TRACE[f][0] + 0.02 for f in SY.FAMILIES}
        a["correction_toy"] = 1.5
        b = dict(a, correction_toy=1.0)
        d = C.adaptive_decide(state([read(a), read(b)], doses={1: {f: 9000.0 for f in SY.FAMILIES}}), self.cfg(), "adaptive", 2, GENESIS)
        self.assertEqual(d["families"]["recall_toy"]["state"], "held")

    def test_no_stall_on_too_little_dose(self):
        a = {f: SY.TRACE[f][0] + 1.0 for f in SY.FAMILIES}
        d = C.adaptive_decide(state([read(a), read(a)], doses={1: {f: 100.0 for f in SY.FAMILIES}}), self.cfg(), "adaptive", 2, GENESIS)
        self.assertNotEqual(d["families"]["role_toy"]["state"], "stalled")

    def test_stall_at_too_much_noise_is_flagged(self):
        a = {f: SY.TRACE[f][0] + 1.0 for f in SY.FAMILIES}
        d = C.adaptive_decide(state([read(a, se=0.2), read(a, se=0.2)]), self.cfg(), "adaptive", 2, GENESIS)
        self.assertFalse(d["families"]["role_toy"]["stall_readable"])
        self.assertIn("cannot be read", "\n".join(C.explain(d)))

    def test_measured_gain_never_raises_a_weight(self):
        a = {f: SY.TRACE[f][0] + 1.0 for f in SY.FAMILIES}
        a2 = dict(a, recall_toy=a["recall_toy"] + 0.8)
        d = C.adaptive_decide(state([read(a2), read(a)], doses={1: {f: 100.0 for f in SY.FAMILIES}}), self.cfg(), "adaptive", 2, GENESIS)
        self.assertAlmostEqual(d["multipliers"]["recall_toy"], d["multipliers"]["role_toy"], places=4)

    def test_a_guard_keeps_the_arm_running_and_flags_it(self):
        """Finding 3: a guard in a matched pilot does not halt the arm, or the comparison voids instead of counting against it."""
        a = {f: 1.0 for f in SY.FAMILIES}
        b = dict(a, role_toy=1.6)
        for fn, arm in ((C.adaptive_decide, "adaptive"), (C.fixed_decide, "fixed")):
            d = fn(state([read(a), read(b)]), self.cfg(), arm, 2, GENESIS)
            self.assertEqual(d["action"], "ALLOCATE")
            self.assertEqual(d["flags"], ["guard_tripped"])
            self.assertTrue(all(v == 1.0 for v in d["multipliers"].values()))

    def test_a_guard_outside_a_matched_pilot_asks_for_review(self):
        a = {f: 1.0 for f in SY.FAMILIES}
        d = C.adaptive_decide(state([read(a), read(dict(a, role_toy=1.6))]), self.cfg(matched_pilot=False, arm_policies={"adaptive": "adaptive"}), "adaptive", 2, GENESIS)
        self.assertEqual(d["action"], "REVIEW_GUARD")

    def test_guard_covers_replay(self):
        a = {f: 1.0 for f in SY.FAMILIES}
        r1 = read(a)
        r2 = read(a)
        r2[SY.REPLAY]["mean"] = 1.8
        d = C.adaptive_decide(state([r1, r2]), self.cfg(), "adaptive", 2, GENESIS)
        self.assertEqual(d["flags"], ["guard_tripped"])

    def test_rise_inside_noise_does_not_fire(self):
        a = {f: 1.0 for f in SY.FAMILIES}
        b = dict(a, role_toy=1.3)
        d = C.adaptive_decide(state([read(a, se=0.1), read(b, se=0.1)]), self.cfg(), "adaptive", 2, GENESIS)
        self.assertNotIn("guard_tripped", d["flags"])

    def test_budget_cap_stops_both_policies(self):
        vals = {f: 1.0 for f in SY.FAMILIES}
        for fn in (C.adaptive_decide, C.fixed_decide):
            d = fn(state([read(vals)], spent=380.0), self.cfg(), "x", 1, GENESIS)
            self.assertEqual(d["action"], "STOP_BUDGET")

    def test_registered_legs_stop_both(self):
        vals = {f: 1.0 for f in SY.FAMILIES}
        for fn in (C.adaptive_decide, C.fixed_decide):
            self.assertEqual(fn(state([read(vals)] * 4), self.cfg(), "x", 4, GENESIS)["action"], "STOP_LEGS")

    def test_missing_read_fails_closed(self):
        vals = {f: 1.0 for f in SY.FAMILIES}
        d = C.adaptive_decide(state([read(vals)]), self.cfg(), "adaptive", 2, GENESIS)
        self.assertEqual(d["action"], "REVIEW_INPUT")

    def test_family_missing_from_any_read_fails_closed(self):
        vals = {f: 1.0 for f in SY.FAMILIES}
        short = read(vals)
        del short["role_toy"]
        d = C.adaptive_decide(state([short, read(vals)]), self.cfg(), "adaptive", 2, GENESIS)
        self.assertEqual(d["action"], "REVIEW_INPUT")

    def test_failed_leg_stops_for_review(self):
        s = state([])
        s["failed_legs"] = [{"failure_mode": "pod lost"}]
        self.assertEqual(C.fixed_decide(s, self.cfg(), "fixed", 1, GENESIS)["action"], "REVIEW_INPUT")

    def test_a_superseding_retry_clears_the_failed_leg(self):
        self.led.append("registration", SY.registration(), PROD, DATE, "synthetic")
        for arm in ("fixed",):
            self.led.append("leg", SY.leg(arm, 0, status="failed"), PROD, DATE, "synthetic")
            d = C.decide(self.led.load(), SY.config(), arm, 1, GENESIS)
            self.assertEqual(d["action"], "REVIEW_INPUT")
            self.led.append("leg", SY.leg(arm, 0), PROD, DATE, "synthetic", supersedes=1)
            self.led.append("dev_read", SY.dev_read(arm, 0), PROD, DATE, "synthetic")
            self.assertEqual(C.decide(self.led.load(), SY.config(), arm, 1, GENESIS)["action"], "ALLOCATE")

    def test_floor_from_another_base_refused(self):
        c = self.cfg()
        c["families"]["role_toy"]["floor_base_id"] = "another_base"
        with self.assertRaises(Malformed):
            C.load_config(c)

    def test_records_from_another_base_refused(self):
        self.demo()
        b = SY.dev_read("adaptive", 2)
        b["base_id"] = "another_base"
        self.led.append("dev_read", b, PROD, DATE, "synthetic")
        with self.assertRaises(Malformed):
            C.decide(self.led.load(), SY.config(), "adaptive", 3, GENESIS)

    def test_no_registration_no_decision(self):
        with self.assertRaises(Malformed):
            C.decide([], SY.config(), "adaptive", 0, GENESIS)

    def test_a_draft_registration_is_no_registration(self):
        self.led.append("registration", SY.registration(status="draft"), PROD, DATE, "synthetic")
        with self.assertRaises(Malformed):
            C.decide(self.led.load(), SY.config(), "adaptive", 0, GENESIS)

    def test_config_must_be_the_registered_one(self):
        self.demo()
        recs = self.led.load()
        c = SY.config()
        c["eta"] = 0.9
        with self.assertRaises(Malformed):
            C.decide(recs, c, "adaptive", 2, recs[-1]["sha256"])

    def test_policies_must_be_the_registered_ones(self):
        cfg = SY.config({"fixed": "adaptive", "adaptive": "adaptive"})
        self.led.append("registration", SY.registration(cfg), PROD, DATE, "synthetic")
        with self.assertRaises(Malformed):
            C.decide(self.led.load(), cfg, "adaptive", 0, GENESIS)

    def test_allow_synthetic_is_refused_on_a_registered_experiment(self):
        cfg = SY.config()
        self.led.append("registration", SY.registration(cfg), PROD, DATE, "registered")
        with self.assertRaises(Malformed):
            C.decide(self.led.load(), cfg, "adaptive", 0, GENESIS)

    def test_observational_cannot_drive_a_decision(self):
        cfg = SY.config()
        self.led.append("registration", SY.registration(cfg), PROD, DATE, "observational")
        with self.assertRaises(Malformed):
            C.decide(self.led.load(), cfg, "adaptive", 0, GENESIS)

    def test_a_read_of_another_checkpoint_is_refused(self):
        self.demo()
        self.led.append("leg", SY.leg("adaptive", 2), PROD, DATE, "synthetic")
        self.led.append("dev_read", SY.dev_read("adaptive", 2, checkpoint_sha=sha("elsewhere")), PROD, DATE, "synthetic")
        with self.assertRaises(Malformed):
            C.decide(self.led.load(), SY.config(), "adaptive", 3, GENESIS)

    def test_a_broken_checkpoint_chain_is_refused(self):
        self.led.append("registration", SY.registration(), PROD, DATE, "synthetic")
        b = SY.leg("adaptive", 0)
        b["checkpoint_in"] = sha("not the start")
        self.led.append("leg", b, PROD, DATE, "synthetic")
        with self.assertRaises(Malformed):
            C.decide(self.led.load(), SY.config(), "adaptive", 1, GENESIS)

    def test_a_second_read_of_a_leg_is_refused(self):
        self.demo()
        self.led.append("dev_read", SY.dev_read("adaptive", 1), PROD, DATE, "synthetic")
        with self.assertRaises(Malformed):
            C.decide(self.led.load(), SY.config(), "adaptive", 2, GENESIS)

    def test_a_start_read_is_kept_and_reaches_the_guard(self):
        self.led.append("registration", SY.registration(), PROD, DATE, "synthetic")
        vals = {f: 0.5 for f in SY.FAMILIES}
        self.led.append("dev_read", SY.dev_read("adaptive", -1, values=dict(vals, **{SY.REPLAY: 0.5})), PROD, DATE, "synthetic")
        self.led.append("leg", SY.leg("adaptive", 0), PROD, DATE, "synthetic")
        self.led.append("dev_read", SY.dev_read("adaptive", 0, values=dict(vals, **{SY.REPLAY: 1.5})), PROD, DATE, "synthetic")
        d = C.decide(self.led.load(), SY.config(), "adaptive", 1, GENESIS)
        self.assertEqual(d["flags"], ["guard_tripped"])

    def test_multipliers_come_from_the_decision_for_the_previous_leg(self):
        self.demo()
        cfg = SY.config()
        recs = self.led.load()
        first = C.decide(recs, cfg, "adaptive", 2, recs[-1]["sha256"])
        self.led.append("decision", first, PROD, DATE, "synthetic")
        with self.assertRaises(Malformed):
            self.led.append("decision", first, PROD, DATE, "synthetic")
        s = C.state_from_ledger(self.led.load(), cfg, "adaptive", 3)
        self.assertEqual(s["multipliers"], first["multipliers"])
        self.assertIsNone(C.state_from_ledger(self.led.load(), cfg, "adaptive", 2)["multipliers"])

    def test_unmapped_packet_family_fails_closed(self):
        """Finding 12: a key the map misses used to read as zero dose, which turned the stall off."""
        cfg = self.cfg()
        leg = {"exposure": {"stance:mystery": {"sampled_decision": 9000.0}}}
        with self.assertRaises(Malformed):
            C.family_dose(leg, "role_toy", cfg)
        cfg["family_map"] = {"stance:mystery": "role_toy"}
        self.assertEqual(C.family_dose(leg, "role_toy", cfg), 9000.0)

    def test_a_family_with_no_decision_tokens_can_use_supervised_dose(self):
        cfg = self.cfg()
        cfg["families"]["smalltalk_toy"]["dose_field"] = "sampled_supervised"
        self.assertEqual(C.family_dose({"exposure": {"smalltalk_toy": {"sampled_decision": 0.0, "sampled_supervised": 50.0}}}, "smalltalk_toy", cfg), 50.0)

    def test_two_arms_must_be_a_matched_pilot(self):
        with self.assertRaises(Malformed):
            C.load_config(self.cfg(matched_pilot=False))

    def test_parameter_bounds(self):
        for kw in ({"patience": 0}, {"z": 0.0}, {"multiplier_max": 1000.0}, {"multiplier_min": 0.001}, {"replay_share": 0.0}, {"min_gain_nats": 0.0}, {"min_excess": 0.0}):
            with self.assertRaises(Malformed, msg=str(kw)):
                C.load_config(self.cfg(**kw))

    def test_z_under_two_is_refused_on_a_registered_run(self):
        c = self.real()
        c["z"] = 1.0
        with self.assertRaises(Malformed):
            C.load_config(c)

    def real(self):
        c = self.cfg()
        del c["allow_synthetic"]
        for f in c["families"].values():
            f["floor_source"] = {"kind": "specialist", "checkpoint": sha("spec"), "set_sha": SY.DEV_SET, "metric": SY.METRIC}
        return c

    def test_floor_must_be_a_specialists_with_provenance(self):
        C.load_config(self.real())
        c = self.real()
        c["families"]["role_toy"]["floor_source"] = {"kind": "best_model_so_far", "checkpoint": sha("x"), "set_sha": SY.DEV_SET, "metric": SY.METRIC}
        with self.assertRaises(Malformed):
            C.load_config(c)
        c = self.real()
        c["families"]["role_toy"]["floor_source"] = "specialist"
        with self.assertRaises(Malformed):
            C.load_config(c)

    def test_priority_policy_refused_outside_the_harness(self):
        c = self.real()
        c["arm_policies"] = {"adaptive": "priority", "fixed": "fixed"}
        with self.assertRaises(Malformed):
            C.load_config(c)

    def test_balanced_shares_must_sum(self):
        c = self.cfg()
        c["families"]["role_toy"]["balanced_share"] = 0.5
        with self.assertRaises(Malformed):
            C.load_config(c)

    def test_explain_covers_every_family(self):
        self.demo()
        recs = self.led.load()
        d = C.decide(recs, SY.config(), "adaptive", 2, recs[-1]["sha256"])
        text = "\n".join(C.explain(d))
        for f in SY.FAMILIES:
            self.assertIn(f, text)
        self.assertIn("executed is False", text)


class TestReproducibility(Tmp):
    def test_same_inputs_same_decision(self):
        self.demo()
        recs = self.led.load()
        cfg = SY.config()
        a = C.decide(recs, cfg, "adaptive", 2, recs[-1]["sha256"])
        b = C.decide(json.loads(json.dumps(recs)), json.loads(json.dumps(cfg)), "adaptive", 2, recs[-1]["sha256"])
        self.assertEqual(a, b)
        self.assertEqual(a["decision_sha"], b["decision_sha"])

    def test_config_change_is_refused_not_merely_rehashed(self):
        self.demo()
        recs = self.led.load()
        cfg = SY.config()
        cfg["eta"] = 0.6
        with self.assertRaises(Malformed):
            C.decide(recs, cfg, "adaptive", 2, recs[-1]["sha256"])

    def test_recorded_decisions_recompute_from_their_prefix(self):
        self.demo()
        cfg = SY.config()
        for arm in ("fixed", "adaptive"):
            recs = self.led.load()
            self.led.append("decision", C.decide(recs, cfg, arm, 2, recs[-1]["sha256"]), PROD, DATE, "synthetic")
        recs = self.led.load()
        seen = 0
        for r in recs:
            if r["kind"] == "decision":
                self.assertEqual(C.decide(recs[: r["seq"]], cfg, r["body"]["arm"], r["body"]["next_leg"], r["prev"]), r["body"])
                seen += 1
        self.assertEqual(seen, 2)

    def test_plan_is_reproducible_and_lands_on_the_ledger(self):
        import lab
        self.demo()
        cfg, pool = SY.config(), SY.pool()
        dec, plan, _ = lab.do_dry_run(self.led, cfg, "adaptive", 2, pool)
        lab.record(self.led, dec, plan, pool, PROD, DATE, "synthetic")
        last = self.led.load()[-1]
        self.assertEqual(last["kind"], "leg_plan")
        self.assertEqual(last["body"]["plan_sha"], sha(X.plan_leg(dec, pool, cfg)))

    def test_sim_is_deterministic(self):
        w = S.worlds()["responsive_imbalance"]
        self.assertEqual(S.compare(w, 2), S.compare(w, 2))


class TestFirewall(Tmp):
    def manifest(self):
        return {"schema": "lab-splits-v1", "splits": {s: {"groups": [f"{s}{i}" for i in range(3)], "scenarios": [f"{s}s{i}" for i in range(3)]} for s in FW.SPLITS}}

    def test_disjoint_passes(self):
        self.assertEqual(FW.check_splits(self.manifest())["groups"]["sealed"], 3)

    def test_group_overlap_breaches(self):
        m = self.manifest()
        m["splits"]["development"]["groups"].append("sealed0")
        with self.assertRaises(FW.FirewallBreach):
            FW.check_splits(m)

    def test_scenario_overlap_breaches(self):
        m = self.manifest()
        m["splits"]["train"]["scenarios"].append("validations1")
        with self.assertRaises(FW.FirewallBreach):
            FW.check_splits(m)

    def sealed_file(self):
        p = self.dir / "sealed.json"
        p.write_text(json.dumps({"groups": ["g1", "g2"], "clusters": ["c9"], "secret": "unread"}))
        return p

    def test_sealed_pin_and_count(self):
        from schema import sha_file
        p = self.sealed_file()
        self.assertEqual(FW.sealed_groups(p, sha_file(p), 2), {"g1", "g2"})
        with self.assertRaises(FW.FirewallBreach):
            FW.sealed_groups(p)
        with self.assertRaises(FW.FirewallBreach):
            FW.sealed_groups(p, sha_file(p), 31)
        with self.assertRaises(FW.FirewallBreach):
            FW.sealed_groups(p, "", 2)

    def test_sealed_overlap_gives_no_id_and_no_count(self):
        with self.assertRaises(FW.FirewallBreach) as cm:
            FW.check_against_sealed(["g1", "x"], {"g1"})
        self.assertNotIn("g1", str(cm.exception))
        self.assertFalse(re.search(r"[0-9]", str(cm.exception)))
        self.assertEqual(FW.check_against_sealed(["x"], {"g1"})["overlap"], 0)

    def test_no_sealed_id_or_pinned_file_is_in_this_folder(self):
        """The committed folder holds the sealed hash, and no file named as the list or holding its structure."""
        names = [p.name for p in HERE.rglob("*") if p.is_file()]
        self.assertNotIn("sealed_templates.json", names)
        for p in HERE.rglob("*.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            self.assertFalse(isinstance(d, dict) and "clusters" in d, p.name)

    def test_sealed_read_anywhere_stops_every_decision(self):
        """Finding 6: a sealed read logged under another experiment used to be skipped."""
        self.demo()
        b = SY.dev_read("adaptive", 2, role="sealed")
        b["experiment_id"] = "some-other-experiment"
        self.led.append("dev_read", b, PROD, DATE, "synthetic")
        with self.assertRaises(FW.FirewallBreach):
            C.decide(self.led.load(), SY.config(), "adaptive", 2, GENESIS)

    def test_validation_read_stops_the_other_arm_too(self):
        self.demo()
        self.led.append("dev_read", SY.dev_read("fixed", 2, role="validation"), PROD, DATE, "synthetic")
        with self.assertRaises(FW.FirewallBreach):
            C.decide(self.led.load(), SY.config(), "adaptive", 2, GENESIS)

    def test_only_the_registered_read_reaches_the_controller(self):
        """Self-declared names are not the control: any read but the registered instrument, set and metric stops the run."""
        for kw in ({"instrument": "harmless_name"}, {"set_sha": sha("T_E under a new name")}, {"metric": "points"}):
            with self.subTest(kw):
                d = tempfile.TemporaryDirectory()
                led = Ledger(Path(d.name) / "l.jsonl")
                SY.write_demo(led.path)
                led.append("dev_read", SY.dev_read("adaptive", 2, **kw), PROD, DATE, "synthetic")
                with self.assertRaises(FW.FirewallBreach):
                    C.decide(led.load(), SY.config(), "adaptive", 2, GENESIS)
                d.cleanup()

    def test_frozen_names_caught_however_spelt(self):
        for name in ("chatflow_v001", "dev_chatflow_v001", " chatflow_v001", "analytic-rubric", "rubric_v2", "sealed_templates", "held_T_E", "T_C_candidate_pool", "T_E_v1"):
            self.assertTrue(FW.is_frozen_instrument(name), name)
        self.assertFalse(FW.is_frozen_instrument("T_M_family_loss_v1"))

    def test_other_arm_reads_do_not_cross(self):
        self.demo()
        recs = self.led.load()
        self.assertTrue(all(r["body"]["arm"] == "fixed" for r in FW.controller_view(recs, SY.EXP, "fixed", recs[0]["body"]["development"])))

    def test_look_cap_is_enforced_by_decide(self):
        cfg = SY.config()
        reg = SY.registration(cfg)
        reg["budget"]["max_dev_reads"] = 3
        self.led.append("registration", reg, PROD, DATE, "synthetic")
        for arm in ("fixed", "adaptive"):
            for i in range(2):
                self.led.append("leg", SY.leg(arm, i), PROD, DATE, "synthetic")
                self.led.append("dev_read", SY.dev_read(arm, i), PROD, DATE, "synthetic")
        with self.assertRaises(FW.FirewallBreach):
            C.decide(self.led.load(), cfg, "adaptive", 2, GENESIS)

    def test_row_parts(self):
        rows = [{"p": "T_C"}, {"p": "T_E"}]
        with self.assertRaises(FW.FirewallBreach):
            FW.check_row_parts(rows, {"T_C"}, lambda r: r["p"])

    def test_shortcuts(self):
        full = {p: 10.0 for p in FW.REQUIRED_SHORTCUTS}
        self.assertEqual(len(FW.check_shortcut_read(full, 40.0)), 7)
        with self.assertRaises(FW.FirewallBreach):
            FW.check_shortcut_read(dict(full, answer_dump=45.0), 40.0)
        del full["last_entity_copy"]
        with self.assertRaises(FW.FirewallBreach):
            FW.check_shortcut_read(full, 40.0)

    def test_prefix_stable(self):
        turns = ["my job is baking", "what do I do"]
        fair = lambda ts: list(range(len(ts)))  # noqa: E731
        peeking = lambda ts: [len(ts) for _ in ts]  # noqa: E731
        self.assertIsNone(FW.prefix_stable(fair, turns))
        self.assertEqual(FW.prefix_stable(peeking, turns), 0)


class TestExposure(Tmp):
    def decision(self):
        self.demo()
        recs = self.led.load()
        return C.decide(recs, SY.config(), "adaptive", 2, recs[-1]["sha256"])

    def test_plan_meets_both_token_targets(self):
        d = self.decision()
        cfg, pool = SY.config(), SY.pool()
        plan = X.plan_leg(d, pool, cfg)
        self.assertLess(abs(plan["expected_supervised_total"] / cfg["supervised_tokens_per_leg"] - 1), 0.005)
        self.assertLess(abs(plan["expected_decision_total"] / cfg["decision_tokens_per_leg"] - 1), 0.005)
        self.assertEqual(sum(v["rows"] for v in plan["families"].values()), plan["packet_training_rows"])
        self.assertFalse(plan["executed"])

    def test_plan_leaves_the_controllers_shares_alone(self):
        """Finding 4: planning used to reverse the controller's choices. Family shares of decision tokens now come out as they went in."""
        d = self.decision()
        plan = X.plan_leg(d, SY.pool(), SY.config())
        total = plan["expected_decision_total"]
        for f, s in d["shares"].items():
            self.assertAlmostEqual(plan["families"][f]["expected_decision"] / total, s, delta=0.002)

    def test_both_arms_meet_the_same_targets(self):
        self.demo()
        recs = self.led.load()
        cfg, pool = SY.config(), SY.pool()
        a = X.plan_leg(C.decide(recs, cfg, "adaptive", 2, recs[-1]["sha256"]), pool, cfg)
        f = X.plan_leg(C.decide(recs, cfg, "fixed", 2, recs[-1]["sha256"]), pool, cfg)
        self.assertLess(abs(a["expected_supervised_total"] / f["expected_supervised_total"] - 1), 0.005)
        self.assertLess(abs(a["expected_decision_total"] / f["expected_decision_total"] - 1), 0.005)

    def test_unreachable_target_fails_closed(self):
        d = self.decision()
        cfg = SY.config()
        for key, val in (("supervised_tokens_per_leg", 9000000.0), ("decision_tokens_per_leg", 400000.0), ("supervised_tokens_per_leg", 300000.0)):
            with self.subTest(key=key, val=val):
                c = dict(cfg, **{key: val})
                with self.assertRaises(X.ExposureUnmatched):
                    X.plan_leg(d, SY.pool(), c)

    def test_capacity_fails_closed(self):
        d = self.decision()
        pool = SY.pool()
        pool["role_toy"]["rows"] = 50
        with self.assertRaises(X.ExposureUnmatched):
            X.plan_leg(d, pool, SY.config())

    def test_no_plan_for_a_stop(self):
        d = self.decision()
        d["action"] = "STOP_BUDGET"
        with self.assertRaises(Malformed):
            X.plan_leg(d, SY.pool(), SY.config())

    def test_distinct_rows_formula(self):
        d = self.decision()
        plan = X.plan_leg(d, SY.pool(), SY.config())
        n, r, dr = plan["families"]["role_toy"]["rows"], plan["packet_training_rows"], plan["steps"] * plan["batch"]
        self.assertAlmostEqual(plan["families"]["role_toy"]["expected_distinct_rows"], n * (1 - (1 - 1 / r) ** dr), places=0)

    def test_prelaunch_recount(self):
        d = self.decision()
        plan = X.plan_leg(d, SY.pool(), SY.config())
        good = {f: {"sampled_supervised": v["expected_supervised"], "sampled_decision": v["expected_decision"]} for f, v in plan["families"].items()}
        self.assertEqual(X.prelaunch_recount(plan, good), [])
        off = json.loads(json.dumps(good))
        off["role_toy"]["sampled_decision"] *= 0.8
        self.assertTrue(X.prelaunch_recount(plan, off))
        low = {f: dict(v, sampled_supervised=v["sampled_supervised"] * 0.9) for f, v in good.items()}
        self.assertTrue(X.prelaunch_recount(plan, low))

    def legs(self, n=2):
        return [SY.leg("fixed", i) for i in range(n)], [SY.leg("adaptive", i) for i in range(n)]

    def test_audit_matched(self):
        f, a = self.legs()
        self.assertTrue(X.audit_pair(f, a, SY.START)["ok"])

    def test_audit_catches_hidden_token_gap(self):
        f, a = self.legs()
        a[0]["exposure"]["role_toy"]["sampled_supervised"] *= 3
        r = X.audit_pair(f, a, SY.START)
        self.assertFalse(r["ok"])
        self.assertIn("sampled_supervised", r["problems"][0])

    def test_audit_catches_offsetting_legs(self):
        """Finding 25: a leg at 1.5 times offset by one at 0.5 times passed a total."""
        f, a = self.legs()
        for v in a[0]["exposure"].values():
            v["sampled_supervised"] *= 1.5
        for v in a[1]["exposure"].values():
            v["sampled_supervised"] *= 0.5
        self.assertFalse(X.audit_pair(f, a, SY.START)["ok"])

    def test_audit_catches_misnumbered_legs(self):
        f, a = self.legs(3)
        a[1]["leg"], a[2]["leg"] = 5, 9
        self.assertFalse(X.audit_pair(f, a, SY.START)["ok"])

    def test_audit_catches_a_different_start(self):
        f, a = self.legs()
        self.assertFalse(X.audit_pair(f, a, sha("elsewhere"))["ok"])

    def test_audit_catches_a_changed_recipe(self):
        for field, val in (("recipe", {"lr": 3e-5, "warmup": 64, "optimiser": "adamw", "batch": 16}), ("source_commit", "def5678"), ("generator_version", "v2")):
            f, a = self.legs()
            a[0][field] = val
            self.assertFalse(X.audit_pair(f, a, SY.START)["ok"], field)

    def test_audit_warns_then_voids_on_repetition(self):
        f, a = self.legs()
        for x in a:
            for v in x["exposure"].values():
                v["distinct_scenarios"] *= 0.87
        r = X.audit_pair(f, a, SY.START)
        self.assertTrue(r["ok"])
        self.assertTrue(any("distinct_scenarios" in w for w in r["warnings"]))
        for x in a:
            for v in x["exposure"].values():
                v["distinct_scenarios"] *= 0.7
        self.assertFalse(X.audit_pair(f, a, SY.START)["ok"])

    def test_replay_floor(self):
        b = SY.leg("fixed", 0)
        self.assertEqual(X.replay_floor_held([b], {"role_toy"}, 0.30)[0]["leg"], 0)
        self.assertEqual(X.replay_floor_held([b], {SY.REPLAY}, 0.30), [])

    def test_minutes_come_from_receipts(self):
        self.demo()
        m = X.minutes_from_ledger(self.led.load(), SY.EXP, "adaptive")
        self.assertEqual(m["training_minutes"], 82.0)
        self.assertEqual(m["read_minutes"], 2.0)
        self.assertEqual(m["dev_reads"], 2)


class TestTaxonomy(unittest.TestCase):
    def test_all_named_categories_exist(self):
        for c in ("wrong_operation", "missing_answer", "attribution", "wrong_value", "stale_value", "unwarranted_abstention", "failure_to_abstain", "role_identity", "echo_stock", "format", "passage"):
            self.assertIn(c, T.CATEGORIES)

    def test_mine_labels(self):
        self.assertIsNone(T.from_mine_label("recall", "VALUE"))
        self.assertIsNone(T.from_mine_label("abstain", "ABSTAIN"))
        self.assertEqual(T.from_mine_label("recall", "ABSTAIN"), "unwarranted_abstention")
        self.assertEqual(T.from_mine_label("abstain", "INVENT_OR_OTHER"), "failure_to_abstain")
        self.assertEqual(T.from_mine_label("recall", "LISTS_OTHERS"), "answer_dump")
        self.assertEqual(T.from_mine_label("self_name", "OTHER"), "role_identity")

    def test_new_label_stops(self):
        with self.assertRaises(Malformed):
            T.from_mine_label("recall", "A_NEW_LABEL")

    def test_frozen_labels_are_categories(self):
        self.assertEqual(T.from_frozen_label("ROLE_LEAK"), "role_identity")
        self.assertIsNone(T.from_frozen_label("ABSTAIN"))
        self.assertTrue(all(v in T.CATEGORIES for v in T.FROZEN_LABELS.values()))

    def test_tier_and_history_checked(self):
        ok = T.FailureEvent(1, 2, 0, "recall", "recall|role", "wrong_value", "heuristic", True, "clean", "T_M")
        self.assertEqual(ok.record()["tier"], "heuristic")
        with self.assertRaises(Malformed):
            T.FailureEvent(1, 2, 0, "recall", "c", "wrong_value", "vibes", True, "clean", "T_M")
        with self.assertRaises(Malformed):
            T.FailureEvent(1, 2, 0, "recall", "c", "not_a_category", "heuristic", True, "clean", "T_M")
        with self.assertRaises(Malformed):
            T.FailureEvent(1, 2, 0, "recall", "c", "wrong_value", "heuristic", True, "pooled", "T_M")

    def test_counts_keep_tiers_and_histories_apart(self):
        ev = [T.FailureEvent(1, 2, 0, "recall", "c", "wrong_value", t, True, h, "T_M") for t in ("heuristic", "llm_judged") for h in ("clean", "model_induced")]
        rows = T.counts(ev)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["n"] == 1 for r in rows))


def t_m_rows(n=8, part="T_M"):
    return [{"provenance": {"turns": [], "correction_part": part}} for _ in range(n)]


class TestMiner(unittest.TestCase):
    def journal(self):
        rows = []
        for r in range(8):
            rows.append({"row": r, "turn": 1, "draw": 0, "k": "recall", "c": "role", "meta": {}, "reply": "x" if r % 4 else "ok"})
            rows.append({"row": r, "turn": 2, "draw": 0, "k": "abstain", "c": "role", "meta": {}, "reply": "ok"})
        return rows

    def score(self, reply, meta, metas, i):
        return (True, "VALUE") if reply == "ok" else (False, "WRONG_VALUE")

    def test_cells_and_events(self):
        ev, cells = M.label_clean(self.journal(), t_m_rows(), self.score)
        self.assertEqual(cells["recall|role"]["fail"], 6)
        self.assertEqual(len(ev), 6)
        self.assertTrue(all(e.tier == "heuristic" and e.history == "clean" for e in ev))

    def test_rows_from_another_part_are_refused_whatever_the_caller_says(self):
        """Finding 17: the part comes from each row's own provenance."""
        for part in ("T_E", "T_C", "sealed", None):
            with self.assertRaises(FW.FirewallBreach):
                M.label_clean(self.journal(), t_m_rows(part=part), self.score)

    def test_one_round_only(self):
        with self.assertRaises(FW.FirewallBreach):
            M.label_clean(self.journal(), t_m_rows(), self.score, round_no=2)

    def test_a_failure_with_no_category_stops(self):
        with self.assertRaises(Malformed):
            M.label_clean(self.journal(), t_m_rows(), lambda *a: (False, "VALUE"))

    def test_target_verified_is_the_callers_statement(self):
        ev, _ = M.label_clean(self.journal(), t_m_rows(), self.score, target_verified=False)
        self.assertFalse(ev[0].target_verified)

    CELLS = {"a": {"n": 40, "fail": 38, "halves": [[19, 20], [19, 20]]}, "b": {"n": 40, "fail": 2, "halves": [[1, 20], [1, 20]]}, "c": {"n": 40, "fail": 20, "halves": [[20, 20], [0, 20]]}}

    def test_selection_order(self):
        out = M.selection_table(self.CELLS, {"a": 0.4, "b": 0.3, "c": 0.3})
        self.assertAlmostEqual(sum(out.values()), 1.0, places=5)
        self.assertGreater(out["a"], out["c"])
        self.assertLess(out["b"], out["c"])

    def test_selection_stays_inside_the_clip(self):
        """Finding 16: renormalising after the clip broke the bound. Every cell stays within 0.25 and 3 times the verified share."""
        share = {"a": 0.02, "b": 0.49, "c": 0.49}
        cells = {"a": {"n": 40, "fail": 40, "halves": [[20, 20], [20, 20]]}, "b": {"n": 40, "fail": 0, "halves": [[0, 20], [0, 20]]}, "c": {"n": 40, "fail": 0, "halves": [[0, 20], [0, 20]]}}
        out = M.selection_table(cells, share)
        for k, s in share.items():
            self.assertLessEqual(out[k] / s, 3.0 + 1e-3)
            self.assertGreaterEqual(out[k] / s, 0.25 - 1e-3)

    def test_model_induced_is_refused(self):
        with self.assertRaises(FW.FirewallBreach):
            M.selection_table({}, {}, history="model_induced")

    def test_unmined_cell_refused(self):
        with self.assertRaises(FW.FirewallBreach):
            M.selection_table({"a": {"n": 2, "fail": 1, "halves": [[1, 1], [0, 1]]}}, {"z": 1.0})
        with self.assertRaises(FW.FirewallBreach):
            M.selection_table({"a": {"n": 0, "fail": 0, "halves": [[0, 0], [0, 0]]}}, {"a": 1.0})


def toy_row(value="Paris", asked=True, later=False):
    """An invented row in the segment format, with a state machine this test owns. Nothing here comes from any table or instrument."""
    us = ["I moved to " + value + ".", "Where did I move?"]
    meta = [{"k": "fact", "value": value}, {"k": "recall", "value": value}]
    reply2 = "You moved to " + value + "."
    if later:
        us = ["Lovely day.", "I moved to " + value + "."]
    seg = [{"loss": False, "text": "<|user|>" + us[0] + "<|assistant|>"}, {"loss": True, "text": "Noted."}, {"loss": False, "text": "<|endoftext|><|user|>" + us[1] + "<|assistant|>"}, {"loss": True, "text": reply2 if asked else "Hmm."}]
    return {"segments": seg, "provenance": {"turns": meta}}


class TestRowCheck(unittest.TestCase):
    def test_parse(self):
        t = SC.parse_turns(toy_row())
        self.assertEqual(len(t), 2)
        self.assertEqual(t[1][1], "You moved to Paris.")

    def test_good_row_holds_and_coverage_is_reported(self):
        self.assertFalse(SC.check_row(toy_row()))
        rep = SC.check_rows([toy_row()])
        self.assertEqual(rep["checks_run"]["told_before_asked"], 1)
        self.assertEqual(rep["checks_run"]["gold_carries"], 1)

    def test_nothing_checked_shows_as_nothing_checked(self):
        r = toy_row()
        r["provenance"]["turns"] = [{"k": "chat"}, {"k": "chat"}]
        self.assertEqual(SC.check_rows([r])["checks_run"], {})

    def test_gold_missing_value(self):
        self.assertEqual(SC.check_row(toy_row(asked=False))["gold_carries"], 1)

    def test_lookahead_is_caught(self):
        self.assertEqual(SC.check_row(toy_row(later=True))["no_lookahead"], 1)

    def test_a_value_only_in_the_question_is_caught(self):
        """Finding 14: `seen` used to take the current turn first."""
        r = toy_row()
        r["provenance"]["turns"] = [{"k": "chat"}, {"k": "recall", "value": "Rome"}]
        r["segments"][2]["text"] = "<|endoftext|><|user|>Did I move to Rome?<|assistant|>"
        r["segments"][3]["text"] = "You moved to Rome."
        self.assertEqual(SC.check_row(r)["told_before_asked"], 1)

    def test_abstention_naming_a_told_value(self):
        r = toy_row()
        r["provenance"]["turns"][1] = {"k": "abstain"}
        self.assertEqual(SC.check_row(r)["abstain_clean"], 1)

    def test_decision_without_a_value(self):
        r = toy_row()
        r["provenance"]["turns"][1] = {"k": "recall"}
        self.assertEqual(SC.check_row(r)["decision_without_value"], 1)

    def test_like_recall_and_both_names_are_checked(self):
        r = toy_row()
        r["provenance"]["turns"][1] = {"k": "like_recall", "likes": ["Paris", "Lima"]}
        self.assertEqual(SC.check_row(r)["gold_carries"], 1)
        r["provenance"]["turns"][1] = {"k": "both_names", "value": "Paris", "other": "Lima"}
        self.assertEqual(SC.check_row(r)["gold_carries"], 1)

    def test_misaligned(self):
        r = toy_row()
        r["provenance"]["turns"].append({"k": "chat"})
        self.assertEqual(SC.check_row(r)["turns_do_not_align"], 1)

    def test_checker_imports_no_generator(self):
        tree = ast.parse((HERE / "rowcheck.py").read_text(encoding="utf-8"))
        names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(names & {"mixed_gen", "contrast_gen", "contrast_stance_gen", "correction_split", "adapters"})


RULE = {"practical_points": 3.0, "pair_sd_points": 1.0, "min_seed_pairs": 3, "min_groups": 8, "boot_draws": 2000, "boot_seed": 1, "compute_tolerance": 0.05}


def inp(diffs, pairs=3, guards=None, **kw):
    task = [{f"g{i}": d for i, d in enumerate(diffs)} for _ in range(pairs)]
    base = {"exposure": {"ok": True, "problems": [], "warnings": []}, "compute": {"fixed_minutes": 100.0, "adaptive_minutes": 102.0, "fixed_dev_reads": 3, "adaptive_dev_reads": 3}, "task_pairs": task, "guards": guards or {"identity": {"holds": True}}, "exercised": True}
    base.update(kw)
    return base


WIN = [5, 6, 7, 5, 6, 7, 5, 6]


class TestVerdict(unittest.TestCase):
    def test_adaptive_wins_when_it_should(self):
        r = V.adaptive_verdict(RULE, inp(WIN))
        self.assertEqual(r["verdict"], "ADAPTIVE_BETTER")
        self.assertEqual(r["operating_choice"], "adaptive")

    def test_two_seed_pairs_are_never_a_claim(self):
        r = V.adaptive_verdict(RULE, inp(WIN, pairs=2))
        self.assertEqual(r["verdict"], "SCREEN_PROMISING")
        self.assertEqual(r["operating_choice"], "fixed")

    def test_a_rule_cannot_lower_the_floors(self):
        for kw in ({"min_seed_pairs": 1}, {"min_groups": 1}, {"pair_sd_points": 0.0}, {"practical_points": 0.0}, {"practical_points": -50.0}):
            with self.assertRaises(Malformed, msg=str(kw)):
                V.adaptive_verdict(dict(RULE, **kw), inp(WIN))

    def test_too_few_groups_void(self):
        self.assertEqual(V.adaptive_verdict(RULE, inp([9, 9, 9]))["verdict"], "VOID")

    def test_seed_noise_widens_the_interval(self):
        """Finding 2: a model-level shift is not signal. The same groups, with measured seed spread, stop being a claim."""
        self.assertEqual(V.adaptive_verdict(RULE, inp(WIN))["verdict"], "ADAPTIVE_BETTER")
        r = V.adaptive_verdict(dict(RULE, pair_sd_points=8.0), inp(WIN))
        self.assertNotEqual(r["verdict"], "ADAPTIVE_BETTER")
        self.assertEqual(r["operating_choice"], "fixed")

    def test_null_with_seed_shifts_rarely_claims_either_way(self):
        rng = random.Random(3)
        rule = dict(RULE, practical_points=7.0, pair_sd_points=6.9, boot_draws=300)
        claims = {"ADAPTIVE_BETTER": 0, "FIXED_BETTER": 0}
        for _ in range(300):
            pairs = []
            for _ in range(3):
                shift = rng.gauss(0, 6.9)
                pairs.append({f"g{i}": shift + rng.gauss(0, 15) for i in range(30)})
            res = V.adaptive_verdict(rule, dict(inp(WIN), task_pairs=pairs))
            if res["verdict"] in claims:
                claims[res["verdict"]] += 1
        self.assertLessEqual(claims["ADAPTIVE_BETTER"], 9)
        self.assertLessEqual(claims["FIXED_BETTER"], 9)

    def test_fixed_can_win(self):
        r = V.adaptive_verdict(RULE, inp([-x for x in WIN]))
        self.assertEqual(r["verdict"], "FIXED_BETTER")
        self.assertEqual(r["operating_choice"], "fixed")

    def test_a_tie_keeps_fixed(self):
        r = V.adaptive_verdict(RULE, inp([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]))
        self.assertEqual(r["verdict"], "NO_ADAPTIVE_BENEFIT")
        self.assertEqual(r["operating_choice"], "fixed")

    def test_wide_interval_is_unresolved_and_keeps_fixed(self):
        r = V.adaptive_verdict(RULE, inp([20, -18, 15, -12, 22, -20, 10, -9]))
        self.assertEqual(r["verdict"], "UNRESOLVED")
        self.assertEqual(r["operating_choice"], "fixed")

    def test_arm_that_never_moved_is_not_a_finding(self):
        r = V.adaptive_verdict(RULE, inp([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0], exercised=False))
        self.assertEqual(r["verdict"], "ADAPTIVE_NOT_EXERCISED")

    def test_guard_is_non_compensatory(self):
        r = V.adaptive_verdict(RULE, inp(WIN, guards={"identity": {"holds": False}, "role": {"holds": True}}))
        self.assertEqual(r["verdict"], "GUARD_FAILED")
        self.assertEqual(r["failed_guards"], ["identity"])
        self.assertEqual(r["operating_choice"], "fixed")

    def test_small_talk_cannot_offset(self):
        talk = [{f"g{i}": 40.0 for i in range(8)}] * 3
        r = V.adaptive_verdict(RULE, inp([-x for x in WIN], small_talk_pairs=talk))
        self.assertEqual(r["verdict"], "FIXED_BETTER")
        self.assertTrue(r["small_talk"]["reported_only"])

    def test_overhead_counts_at_the_boundary(self):
        comp = {"fixed_minutes": 100.0, "fixed_dev_reads": 3, "adaptive_dev_reads": 3}
        self.assertEqual(V.adaptive_verdict(RULE, inp(WIN, compute=dict(comp, adaptive_minutes=105.0)))["verdict"], "ADAPTIVE_BETTER")
        self.assertEqual(V.adaptive_verdict(RULE, inp(WIN, compute=dict(comp, adaptive_minutes=105.1)))["verdict"], "VOID")

    def test_large_read_share_is_flagged(self):
        r = V.adaptive_verdict(RULE, inp(WIN, compute={"fixed_minutes": 100.0, "adaptive_minutes": 100.0, "fixed_dev_reads": 3, "adaptive_dev_reads": 3, "read_share": 0.2}))
        self.assertTrue(any("plain fixed run" in w for w in r["warnings"]))

    def test_unequal_evaluation_opportunities_void(self):
        r = V.adaptive_verdict(RULE, inp(WIN, compute={"fixed_minutes": 100.0, "adaptive_minutes": 100.0, "fixed_dev_reads": 3, "adaptive_dev_reads": 9}))
        self.assertEqual(r["verdict"], "VOID")

    def test_exposure_failure_voids(self):
        self.assertEqual(V.adaptive_verdict(RULE, inp(WIN, exposure={"ok": False, "problems": ["tokens"], "warnings": []}))["verdict"], "VOID")

    def test_missing_inputs_void(self):
        self.assertEqual(V.adaptive_verdict(RULE, {})["verdict"], "VOID")

    def test_mismatched_groups_refused(self):
        with self.assertRaises(Malformed):
            V.pool_seed_pairs([{"a": 1.0}, {"b": 1.0}])

    def test_rule_hash_is_reported(self):
        self.assertEqual(V.adaptive_verdict(RULE, inp(WIN))["rule_sha"], sha(RULE))

    E_RULE = dict(RULE, min_failure_precision=0.85)

    def e_inp(self, e21, e10, pairs=3, **kw):
        base = {"exposure": {"ok": True, "problems": []}, "precision": 0.9, "chatflow_bar": True, "guards": {"identity": {"holds": True}}, "e2_minus_e1": [{f"g{i}": d for i, d in enumerate(e21)}] * pairs, "e1_minus_e0": [{f"g{i}": d for i, d in enumerate(e10)}] * pairs}
        base.update(kw)
        return base

    FLAT = [0, 1, 0, -1, 0, 0, 1, -1]

    def test_error_selection_can_win(self):
        self.assertEqual(V.error_verdict(self.E_RULE, self.e_inp(WIN, self.FLAT))["verdict"], "SELECTION_ADDS")

    def test_selection_needs_the_chatflow_bar(self):
        self.assertEqual(V.error_verdict(self.E_RULE, self.e_inp(WIN, self.FLAT, chatflow_bar=False))["verdict"], "SCREEN_PROMISING")

    def test_every_error_claim_needs_three_pairs(self):
        """Finding 15: one pair used to reach VERIFIED_ROWS_ONLY."""
        for a, b in ((WIN, self.FLAT), (self.FLAT, WIN)):
            self.assertEqual(V.error_verdict(self.E_RULE, self.e_inp(a, b, pairs=2))["verdict"], "SCREEN_PROMISING")

    def test_error_rule_refuses_a_negative_margin(self):
        with self.assertRaises(Malformed):
            V.error_verdict(dict(self.E_RULE, practical_points=-50.0), self.e_inp(WIN, self.FLAT))

    def test_verified_rows_alone(self):
        self.assertEqual(V.error_verdict(self.E_RULE, self.e_inp(self.FLAT, WIN))["verdict"], "VERIFIED_ROWS_ONLY")

    def test_neither(self):
        self.assertEqual(V.error_verdict(self.E_RULE, self.e_inp(self.FLAT, self.FLAT))["verdict"], "NEITHER_BEATS_ORDINARY")

    def test_low_failure_precision_voids(self):
        self.assertEqual(V.error_verdict(self.E_RULE, self.e_inp(WIN, self.FLAT, precision=0.5))["verdict"], "VOID")

    def test_no_shipped_text_claims_recursion(self):
        texts = [json.dumps(V.MEANING)] + [p.read_text(encoding="utf-8") for p in list((HERE / "examples").glob("*.json")) + list(HERE.glob("*.md"))]
        for t in texts:
            V.lint_claim(t)
        with self.assertRaises(Malformed):
            V.lint_claim("a demonstration of Recursive Self-Improvement")

    def test_trajectory(self):
        self.assertEqual(V.trajectory([(0, 0.0), (10, 10.0)]), 5.0)
        self.assertIsNone(V.trajectory([(0, 0.0)]))

    def cells(self, sign=-1.0, spread=0.5):
        dev = {(m, f): 1.0 + 3.0 * f + spread * m + (0.13 * m * f) % 0.07 for m in range(3) for f in range(6)}
        return dev, {k: 50.0 + sign * 10.0 * v for k, v in dev.items()}

    def test_signal_gate_passes_when_the_signal_tracks(self):
        dev, pri = self.cells()
        self.assertTrue(V.signal_gate(dev, pri)["passes"])

    def test_signal_gate_fails_when_the_signal_runs_backwards(self):
        dev, pri = self.cells(sign=1.0)
        self.assertFalse(V.signal_gate(dev, pri)["passes"])

    def test_family_difficulty_cannot_carry_the_gate(self):
        """Hard families read high loss and low score on every model. That rank is not the signal: within families these reads run backwards."""
        dev, pri = self.cells(sign=1.0)
        pri = {k: v - 40.0 * k[1] for k, v in pri.items()}
        self.assertFalse(V.signal_gate(dev, pri)["passes"])

    def test_no_range_is_not_a_pass(self):
        dev, pri = self.cells(spread=0.01)
        r = V.signal_gate(dev, pri)
        self.assertFalse(r["passes"])
        self.assertIn("not informative", r["note"])

    def test_too_few_cells_is_not_a_pass(self):
        self.assertFalse(V.signal_gate({("m", 1): 1.0}, {("m", 1): 1.0})["passes"])

    def test_the_bar_is_what_the_plan_states(self):
        self.assertEqual((V.SIGNAL_BAR, V.SIGNAL_MIN_CELLS, V.SIGNAL_MIN_RANGE_NATS, V.SIGNAL_MIN_RANGE_FAMILIES), (0.5, 10, 0.30, 5))
        text = (HERE / "PLAN.md").read_text(encoding="utf-8")
        for phrase in ("Spearman at least 0.5", "at least 10 cells", "at least 0.30 nats", "at least 5 families"):
            self.assertIn(phrase, text)


class TestVerdictFromLedger(Tmp):
    """The end of the path: nothing hand-composed. A ledger and a rule go in."""

    def rule(self, **kw):
        r = dict(SY.RULE, pair_sd_points=0.5, shortcut_read={p: 5.0 for p in FW.REQUIRED_SHORTCUTS}, shortcut_bar=40.0)
        r["pairs"] = [["fixed", "adaptive"], ["fixed2", "adaptive2"], ["fixed3", "adaptive3"]]
        r.update(kw)
        return r

    def build(self, gain=6.0, rule=None, fixed_guard=True, adaptive_guard=True, moved=True, skip=None):
        rule = rule or self.rule()
        arms = {"fixed": "fixed", "adaptive": "adaptive", "fixed2": "fixed", "adaptive2": "adaptive", "fixed3": "fixed", "adaptive3": "adaptive"}
        cfg = SY.config(arms)
        reg = SY.registration(cfg, rule=rule)
        reg["budget"]["max_dev_reads"] = 60
        self.led.append("registration", reg, PROD, DATE, "synthetic")
        groups = {f"g{i}": 50.0 + i for i in range(10)}
        for arm, pol in arms.items():
            seed = {"": 1, "2": 2, "3": 3}[arm.replace("fixed", "").replace("adaptive", "")]
            for i in range(2):
                self.led.append("leg", SY.leg(arm, i, seed=seed), PROD, DATE, "synthetic")
            mult = {f: (2.0 if moved and pol == "adaptive" and f == "correction_toy" else 1.0) for f in SY.FAMILIES}
            dec = {"schema": "lab-decision-v1", "experiment_id": SY.EXP, "arm": arm, "next_leg": 1, "policy": pol, "action": "ALLOCATE", "shares": {f: 0.2 for f in SY.FAMILIES}, "flags": [], "multipliers": mult, "families": {}, "reasons": ["toy"], "config_sha": sha(cfg), "ledger_head": GENESIS, "seed": 1, "executed": False}
            dec["decision_sha"] = sha(dec)
            self.led.append("decision", dec, PROD, DATE, "synthetic")
            shift = gain if pol == "adaptive" else 0.0
            summary = {"task_groups": {g: v + shift for g, v in groups.items()}, "small_talk_groups": {g: v for g, v in groups.items()}, "guards": {g: {"holds": (adaptive_guard if pol == "adaptive" else fixed_guard)} for g in reg["guards"]}}
            if skip != arm:
                self.led.append("evaluation", {"experiment_id": SY.EXP, "base_id": SY.BASE, "arm": arm, "instrument_id": "toy_eval_v1", "instrument_role": "validation", "set_sha": sha("toy eval"), "scorer_sha": sha("scorer"), "results_sha": sha(arm), "summary": summary, "compute": {"l4_minutes": 0, "cpu_minutes": 3}}, PROD, DATE, "synthetic")
        return rule

    def test_adaptive_better_end_to_end(self):
        r = V.verdict_from_ledger(self.led.load() if False else (self.build() and self.led.load()), SY.EXP, self.rule())
        self.assertEqual(r["verdict"], "ADAPTIVE_BETTER")

    def test_fixed_better_end_to_end(self):
        """The brief's requirement, on the real path: the code reaches the conclusion that fixed is better."""
        rule = self.build(gain=-6.0)
        r = V.verdict_from_ledger(self.led.load(), SY.EXP, rule)
        self.assertEqual(r["verdict"], "FIXED_BETTER")
        self.assertEqual(r["operating_choice"], "fixed")
        self.led.append("outcome", {"experiment_id": SY.EXP, "verdict": r["verdict"], "rule_sha": r["rule_sha"], "detail": r}, PROD, DATE, "synthetic")
        self.assertEqual(self.led.load()[-1]["body"]["verdict"], "FIXED_BETTER")

    def test_an_unregistered_rule_voids(self):
        rule = self.build()
        r = V.verdict_from_ledger(self.led.load(), SY.EXP, dict(rule, practical_points=1.0))
        self.assertEqual(r["verdict"], "VOID")

    def test_a_missing_guard_voids(self):
        rule = self.build()
        recs = self.led.load()
        for x in recs:
            if x["kind"] == "evaluation":
                del x["body"]["summary"]["guards"]["role"]
        self.assertEqual(V.verdict_from_ledger(recs, SY.EXP, rule)["verdict"], "VOID")

    def test_a_failed_guard_counts_against_adaptive(self):
        rule = self.build(adaptive_guard=False)
        self.assertEqual(V.verdict_from_ledger(self.led.load(), SY.EXP, rule)["verdict"], "GUARD_FAILED")

    def test_a_guard_tripped_by_adaptive_alone_counts_against_it(self):
        rule = self.build()
        recs = self.led.load()
        for x in recs:
            if x["kind"] == "decision" and x["body"]["arm"] == "adaptive2":
                x["body"]["flags"] = ["guard_tripped"]
        r = V.verdict_from_ledger(recs, SY.EXP, rule)
        self.assertEqual(r["verdict"], "GUARD_FAILED")
        self.assertEqual(r["failed_guards"], ["family_regression"])

    def test_a_missing_evaluation_voids(self):
        rule = self.build(skip="adaptive3")
        self.assertEqual(V.verdict_from_ledger(self.led.load(), SY.EXP, rule)["verdict"], "VOID")

    def test_missing_shortcut_read_voids(self):
        rule = self.build(rule=self.rule(shortcut_read={}))
        self.assertEqual(V.verdict_from_ledger(self.led.load(), SY.EXP, rule)["verdict"], "VOID")

    def test_replay_under_its_floor_voids(self):
        rule = self.build(rule=self.rule(replay_floor=0.95))
        self.assertEqual(V.verdict_from_ledger(self.led.load(), SY.EXP, rule)["verdict"], "VOID")

    def test_an_arm_that_never_moved(self):
        rule = self.build(moved=False)
        self.assertEqual(V.verdict_from_ledger(self.led.load(), SY.EXP, rule)["verdict"], "ADAPTIVE_NOT_EXERCISED")

    def test_attempts_are_counted(self):
        rule = self.build()
        self.assertEqual(V.verdict_from_ledger(self.led.load(), SY.EXP, rule)["attempts_under_this_hypothesis"], 1)


class TestShortcuts(unittest.TestCase):
    """Definitions fixed before any T_E row exists. Invented items; the frozen policies are stubbed unless the workspace is here."""
    STUB = {"constant_identity": lambda v, o, r: "identity line", "constant_abstain": lambda v, o, r: "abstain line", "yes_always": lambda v, o, r: "yes line", "list_all_told": lambda v, o, r: " ".join(o), "last_told": lambda v, o, r: o[-1] if o else ""}

    def item(self, key, told, last="toy question?"):
        return {"value": key, "told": told, "last_user": last, "key": key}

    def pairs(self):
        return [(self.item("aa", ["aa", "bb"]), self.item("bb", ["aa", "bb"])) for _ in range(4)]

    def passes(self, text, item):
        others = [t for t in item["told"] if t != item["key"]]
        return item["key"] in text.split() and not any(o in text.split() for o in others)

    def test_every_required_shortcut_is_defined(self):
        self.assertEqual(SH.check_complete(), sorted(FW.REQUIRED_SHORTCUTS))

    def test_definitions_are_pinned(self):
        """A change to any definition or the bar has to change this line too, so it shows in review."""
        self.assertEqual(SH.definitions_sha(), "b06b42f5e2320ebe592ed95c2263d4d3f02ec26e3c2f201b295992b5d78bd40e")
        self.assertEqual(SH.SHORTCUT_BAR_POINTS, 25.0)

    def test_no_policy_can_pass_both_twins_of_a_keyed_pair(self):
        read = SH.read(self.pairs(), self.passes, self.STUB)
        self.assertEqual(sorted(read), sorted(FW.REQUIRED_SHORTCUTS))
        self.assertTrue(all(v == 0.0 for v in read.values()), read)
        self.assertEqual(len(FW.check_shortcut_read(read, SH.SHORTCUT_BAR_POINTS)), 7)

    def test_a_policy_that_passes_both_twins_is_caught(self):
        def lenient(text, item):
            return True
        read = SH.read(self.pairs(), lenient, self.STUB)
        with self.assertRaises(FW.FirewallBreach):
            FW.check_shortcut_read(read, SH.SHORTCUT_BAR_POINTS)

    def test_echo_and_ack(self):
        it = self.item("aa", ["aa"], "toy question?")
        self.assertEqual(SH.reply("echo_question", it, self.STUB), "toy question?")
        self.assertEqual(SH.reply("generic_acknowledgement", it, self.STUB), SH.GENERIC_ACK)

    def test_no_policy_is_handed_the_key(self):
        seen = []
        stub = dict(self.STUB, constant_identity=lambda v, o, r: seen.append((v, o, r)) or "x")
        SH.reply("constant_identity", {"value": None, "told": [], "last_user": "q", "row": None, "key": "secret"}, stub)
        self.assertNotIn("secret", str(seen))

    def test_unknown_policy_refused(self):
        with self.assertRaises(Malformed):
            SH.reply("cleverness", self.item("a", []), self.STUB)


CUES = TS.compile_cues({"abstain": ["no clue"], "no": ["no", "not"]})


def te_row(k, accept, refuse, last="Toy question?", pol=None, told=("Paris", "Lima"), fam="toyfam", pid="p1", var="a", gold="Toy reply."):
    """An invented decision row. Nothing here comes from any table or instrument."""
    segs = [{"text": "<|user|>Toy opener.<|assistant|>", "loss": False}, {"text": "Noted.", "loss": True}, {"text": "<|endoftext|><|user|>" + last + "<|assistant|>", "loss": False}, {"text": gold, "loss": True}]
    metas = [{"k": "chat"}, {"k": k, "c": "none"}]
    if pol:
        metas[1]["pol"] = pol
    return {"conversation_id": f"te-{fam}-01-{var}", "segments": segs, "provenance": {"family": fam, "pair_id": pid, "variant": var, "turns": metas, "values": {"told": list(told)}, "changed_turns": [1]},
            "decision": {"turn": 1, "k": k, "c": "none", "form": "D", "template_group": "0" * 12, "key": {"accept": list(accept), "refuse": list(refuse)}, "told_in_order": list(told)}}


class TestTEScore(unittest.TestCase):
    def test_accept_and_refuse(self):
        r = te_row("recall", ["Paris"], ["Lima"])
        self.assertTrue(TS.passes("You said Paris.", r, CUES))
        self.assertFalse(TS.passes("You said Lima.", r, CUES))
        self.assertFalse(TS.passes("You said Paris or Lima.", r, CUES))
        self.assertFalse(TS.passes("", r, CUES))

    def test_a_refused_word_inside_the_accepted_value_is_no_mention(self):
        r = te_row("recall", ["New York"], ["York"])
        self.assertTrue(TS.passes("You said New York.", r, CUES))
        self.assertFalse(TS.passes("You said York.", r, CUES))

    def test_article_is_ignored(self):
        self.assertTrue(TS.passes("You are a baker.", te_row("recall", ["a baker"], []), CUES))

    def test_abstention_needs_a_cue_and_no_value(self):
        r = te_row("abstain", [], ["Paris", "Lima"])
        self.assertTrue(TS.passes("I have no clue.", r, CUES))
        self.assertFalse(TS.passes("Okay.", r, CUES))
        self.assertFalse(TS.passes("No clue, maybe Paris.", r, CUES))

    def test_a_confirmation_holds_its_polarity(self):
        yes, no = te_row("like_affirm", ["tea"], [], pol="yes"), te_row("like_affirm", ["tea"], [], pol="no")
        self.assertTrue(TS.passes("You like tea.", yes, CUES))
        self.assertFalse(TS.passes("You do not like tea.", yes, CUES))
        self.assertTrue(TS.passes("You do not like tea.", no, CUES))
        self.assertFalse(TS.passes("You like tea.", no, CUES))

    def test_a_copied_question_fails_whatever_it_names(self):
        r = te_row("like_affirm", ["tea"], [], last="Do I like tea?", pol="yes")
        self.assertFalse(TS.passes("Do I like tea?", r, CUES))
        self.assertFalse(TS.passes("do i like TEA", r, CUES))
        self.assertTrue(TS.passes("You like tea.", r, CUES))

    def test_a_constant_earns_nothing_on_a_keyed_pair(self):
        a, b = te_row("recall", ["Paris"], ["Lima"], var="a"), te_row("recall", ["Lima"], ["Paris"], var="b")
        self.assertEqual(TS.pair_points([(a, b)], lambda r: "You said Paris.", CUES), 0.0)
        self.assertEqual(TS.pair_points([(a, b)], lambda r: "You said " + r["decision"]["key"]["accept"][0] + ".", CUES), 100.0)

    def test_the_cue_list_is_pinned(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text('{"abstain": ["x"], "no": ["y"]}')
            with self.assertRaises(Malformed):
                TS.load_cues(p)

    def test_the_scorer_shares_no_code_with_mining_or_the_author(self):
        tree = ast.parse((HERE / "te_score.py").read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(names & {"build_te", "mine_score", "adapters", "mixed_gen"})


class TestTEVerify(unittest.TestCase):
    def named(self, value, k="both_names", user_says=False, in_meta=True):
        r = te_row(k, [value], [])
        first = "My name is " + value + "." if user_says else "Toy opener."
        r["segments"][0]["text"] = "<|user|>" + first + "<|assistant|>"
        r["segments"][1]["text"] = "I'm " + value + "."
        r["provenance"]["turns"][0] = {"k": "rp_nameless", "name": value if in_meta else None}
        return r

    def values(self, rows):
        out = {}
        TV.check_values(rows, out)
        return out["3_values"]

    def test_3a_a_users_value_must_come_from_a_user_turn(self):
        r = te_row("recall", ["Paris"], [])
        r["segments"][1]["text"] = "Paris it is."
        self.assertEqual(self.values([r])["3a_defects"], {"toyfam/recall": 1})

    def test_3b_admits_the_characters_name(self):
        v = self.values([self.named("Zorp")])
        self.assertEqual((v["3b_character_names_held"], v["3b_defects"]), (1, {}))

    def test_3b_refuses_a_value_that_is_not_the_characters_name(self):
        v = self.values([self.named("Zorp", in_meta=False)])
        self.assertEqual(v["3b_character_names_held"], 0)
        self.assertEqual(sum(v["3b_defects"].values()), 1)

    def test_the_assistants_own_name_is_out_of_scope_and_nothing_else_is(self):
        r = te_row("both_names", ["Paris", "Ember"], [], told=("Paris",))
        r["segments"][0]["text"] = "<|user|>I am Paris.<|assistant|>"
        v = self.values([r])
        self.assertEqual((v["self_name_values_out_of_scope"], v["3b_defects"]), (1, {}))
        other = te_row("both_names", ["Paris", "Zorp"], [], told=("Paris",))
        other["segments"][0]["text"] = "<|user|>I am Paris.<|assistant|>"
        self.assertEqual(sum(self.values([other])["3b_defects"].values()) + sum(self.values([other])["no_source_in_any_turn"].values()), 1)

    def test_a_value_the_user_said_is_3a(self):
        v = self.values([self.named("Zorp", user_says=True)])
        self.assertEqual(v["3b_character_names_held"], 0)
        self.assertEqual(v["3a_user_values_held"], 1)

    def test_the_assistants_own_decisions_are_counted_apart(self):
        self.assertEqual(self.values([te_row("self_name", ["Ember"], [])])["assistant_own_decisions_not_checked_here"], {"self_name": 1})

    def twins(self, a, b):
        out = {}
        TV.check_twins([a, b], CUES, out)
        return out["4_twins"]

    def test_good_twins(self):
        a = te_row("recall", ["Paris"], ["Lima"], last="Where?", var="a", gold="Paris.")
        b = te_row("recall", ["Lima"], ["Paris"], last="And there?", var="b", gold="Lima.")
        t = self.twins(a, b)
        self.assertEqual(t["structural_defects"], {})
        self.assertEqual((t["gold_passes_own_key"], t["gold_fails_twin_key"]), ("2 of 2", "2 of 2"))

    def test_twins_with_one_key_are_caught(self):
        a = te_row("recall", ["Paris"], [], last="Where?", var="a", gold="Paris.")
        b = te_row("recall", ["Paris"], [], last="And there?", var="b", gold="Paris.")
        t = self.twins(a, b)
        self.assertEqual(t["structural_defects"]["same_key"], 1)
        self.assertEqual(t["gold_fails_twin_key"], "0 of 2")

    def test_recorded_change_must_be_the_real_one(self):
        a = te_row("recall", ["Paris"], ["Lima"], last="Where?", var="a", gold="Paris.")
        b = te_row("recall", ["Lima"], ["Paris"], last="And there?", var="b", gold="Lima.")
        a["provenance"]["changed_turns"] = b["provenance"]["changed_turns"] = [0]
        self.assertEqual(self.twins(a, b)["structural_defects"]["changed_turns_not_the_difference"], 1)

    def test_fits_a_slot_with_a_digit_and_sentence_joins(self):
        exact, rx = TV.frames({"toy": ["Hello {e1} friend.", "Fixed line."], "pool:x": ["Paris"]}, {})
        self.assertTrue(TV.fits("Hello big friend.", exact, rx))
        self.assertTrue(TV.fits("Hello big friend. Fixed line.", exact, rx))
        self.assertFalse(TV.fits("Quite new wording.", exact, rx))

    def test_the_verb_s_slot_may_be_empty(self):
        exact, rx = TV.frames({"toy": ["{X} always cheer{s_} me."]}, {})
        self.assertTrue(TV.fits("Tea always cheers me.", exact, rx))
        self.assertTrue(TV.fits("Teas always cheer me.", exact, rx))

    def test_the_verifier_shares_no_code_with_the_author(self):
        tree = ast.parse((HERE / "te_verify.py").read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("build_te", names)


class TestTEMerge(unittest.TestCase):
    def pair(self, pid, op="recall", c="none", fam="toyfam"):
        out = []
        for var in ("a", "b"):
            r = te_row(op, ["Paris"], [], fam=fam, pid=pid, var=var)
            r["decision"]["c"] = c
            r["provenance"]["values"]["pair_op"] = op
            out.append(r)
        return out

    SPEC = {"toyfam": {("recall", "none"): 2, ("abstain", "none"): 1}}

    def test_first_file_first_then_the_top_up_within_each_cell(self):
        first = self.pair("p1") + self.pair("p2") + self.pair("p3")
        top = self.pair("p1") + self.pair("p9", op="abstain")
        kept, short, src = TM.merge([("s0", first), ("s1", top)], self.SPEC)
        self.assertEqual([r["provenance"]["pair_id"] for r in kept][::2], ["s0_p1", "s0_p2", "s1_p9"])
        self.assertEqual(short, {})
        self.assertEqual(src, {"s0": 2, "s1": 1})

    def test_a_short_cell_stays_short_and_borrows_nothing(self):
        kept, short, _ = TM.merge([("s0", self.pair("p1") + self.pair("p2") + self.pair("p3"))], self.SPEC)
        self.assertEqual(len(kept), 4)
        self.assertEqual(short, {"toyfam/abstain/none": 1})

    def test_the_same_id_in_two_files_stays_two_pairs(self):
        kept, _, _ = TM.merge([("s0", self.pair("p1")), ("s1", self.pair("p1"))], self.SPEC)
        self.assertEqual(len({r["provenance"]["pair_id"] for r in kept}), 2)
        self.assertEqual(len({r["conversation_id"] for r in kept}), 4)

    def test_half_a_pair_is_refused(self):
        with self.assertRaises(Malformed):
            TM.merge([("s0", self.pair("p1")[:1])], self.SPEC)

    def test_nothing_but_the_filter_selects(self):
        """The merge imports no scorer, and reads no key."""
        src = (HERE / "te_merge.py").read_text(encoding="utf-8")
        self.assertNotIn("te_score", src)
        self.assertNotIn('["key"]', src)


class TestReplaySim(unittest.TestCase):
    """The verdict code on synthetic worlds, through the whole path. These are behaviours of the code, and no evidence about any model."""

    def test_fixed_wins_when_the_proxy_is_invalid(self):
        r = S.compare(S.worlds()["proxy_invalid"], 10)
        self.assertGreaterEqual(r["verdicts"].get("FIXED_BETTER", 0), 8)
        self.assertLess(r["mean_gain_points"], 0)

    def test_adaptive_can_win_where_it_should(self):
        r = S.compare(S.worlds()["responsive_imbalance"], 10)
        self.assertGreaterEqual(r["verdicts"].get("ADAPTIVE_BETTER", 0), 2)
        self.assertGreater(r["mean_gain_points"], 3.0)

    def test_a_null_world_rarely_claims_adaptive(self):
        r = S.compare(S.worlds()["uniform"], 40)
        self.assertLessEqual(r["verdicts"].get("ADAPTIVE_BETTER", 0), 2)

    def test_at_the_floor_the_arm_barely_moves(self):
        """Weights must not chase noise: with every family at its floor the arm should read as not exercised, not swing to the caps."""
        r = S.compare(S.worlds()["at_floor"], 10, pairs=2)
        self.assertGreaterEqual(r["verdicts"].get("ADAPTIVE_NOT_EXERCISED", 0), 8)

    def test_no_decision_in_the_harness_is_ignored(self):
        w = S.worlds()["noisy_read"]
        arm = S.run_arm(w, "adaptive", 9)
        self.assertEqual(len(arm["legs"]), 4)
        self.assertEqual(len(arm["decisions"]), 4)
        self.assertTrue(all(d["action"] == "ALLOCATE" for d in arm["decisions"]))

    def test_three_reads_as_the_draft_says(self):
        arm = S.run_arm(S.worlds()["uniform"], "adaptive", 9)
        self.assertEqual(len(arm["legs"]), 4)

    def test_the_priority_policy_runs_only_with_allow_synthetic(self):
        self.assertTrue(S.base_config(S.worlds()["uniform"]["fams"], "priority")["allow_synthetic"])

    def test_hard_family_gets_no_extra_dose_on_any_leg_or_seed(self):
        w = S.worlds()["hard_floor"]
        for seed in range(6):
            for m in S.run_arm(w, "adaptive", seed)["mults"]:
                self.assertLessEqual(m["fam_a"], 1.01)


class TestProposer(unittest.TestCase):
    def entries(self):
        return json.loads((HERE / "examples" / "interventions.json").read_text(encoding="utf-8"))["interventions"]

    def facts(self):
        return json.loads((HERE / "examples" / "facts_2026-11-07.json").read_text(encoding="utf-8"))["facts"]

    def all_true(self):
        return {p: True for e in self.entries() for p in e["prerequisites"]}

    def test_every_entry_declares_everything(self):
        for e in self.entries():
            P.validate(e)

    def test_missing_declaration_refused(self):
        e = dict(self.entries()[0])
        del e["control_condition"]
        with self.assertRaises(Malformed):
            P.validate(e)

    def test_nothing_may_launch(self):
        res = P.rank(self.entries(), self.all_true())
        self.assertTrue(res["ranked"])
        self.assertTrue(all(r["may_launch"] is False and r["needs_authorisation"] for r in res["ranked"]))

    def test_unknown_prerequisite_blocks(self):
        res = P.rank(self.entries(), {})
        self.assertFalse(res["ranked"])
        self.assertEqual(len(res["blocked"]), len(self.entries()))

    def test_saturation_puts_pretraining_above_sft(self):
        facts = dict(self.all_true(), sft_saturated=False)
        order = [r["id"] for r in P.rank(self.entries(), dict(facts, sft_saturated=True))["ranked"]]
        self.assertLess(order.index("more_pretraining"), order.index("fixed_balanced_stronger"))
        self.assertLess(order.index("more_pretraining"), order.index("adaptive_three_pairs_stronger"))

    def test_saturation_is_read_from_the_ledger(self):
        recs = [{"kind": "decision", "body": {"experiment_id": "x", "arm": "a", "flags": ["saturated"]}}]
        self.assertTrue(P.saturated_from_ledger(recs))
        recs.append({"kind": "decision", "body": {"experiment_id": "x", "arm": "a", "flags": []}})
        self.assertFalse(P.saturated_from_ledger(recs))

    def test_a_measurement_has_no_performance(self):
        rows = {r["id"]: r for r in P.rank(self.entries(), self.all_true())["ranked"]}
        self.assertEqual(rows["noise_pair_pk_sft_v1"]["expected_gain_points"], 0.0)

    def test_scores_stay_within_their_group(self):
        rows = P.rank(self.entries(), self.all_true())["ranked"]
        self.assertGreater(len({r["group"] for r in rows}), 1)
        for g in {r["group"] for r in rows}:
            self.assertLessEqual(max(r["score"] for r in rows if r["group"] == g), 1.0)

    def test_the_noise_pair_comes_before_either_pilot(self):
        for pid in ("error_directed_pk_sft_v1", "adaptive_three_pairs_pk_sft_v1"):
            e = [x for x in self.entries() if x["id"] == pid][0]
            self.assertIn("noise_floor_measured", e["prerequisites"])

    def test_s54_blocks_the_error_pilot_until_its_read_exists(self):
        res = P.rank(self.entries(), self.facts())
        row = [r for r in res["blocked"] if r["id"] == "error_directed_pk_sft_v1"][0]
        self.assertIn("s54_j1_read", row["unmet"])

    def test_prior_uncertainty_and_producer_print(self):
        r = P.rank(self.entries(), self.all_true())["ranked"][0]
        self.assertIn("+/-", r["prior"])
        self.assertIn("Opus 5.5", r["prior_producer"])


class TestGate(Tmp):
    def action(self, **kw):
        a = {"kind": "gpu_training", "experiment_id": SY.EXP, "spec_sha": sha("spec"), "provider": "runpod", "l4_minutes": 170.0, "usd": 2.0}
        a.update(kw)
        return a

    def reg(self, evidence="registered", **kw):
        b = SY.registration()
        b.update(kw)
        return self.led.append("registration", b, PROD, DATE, evidence)

    def review(self, reg, reviewer="an independent reviewer, 7 Nov 2026", result="passed"):
        return self.led.append("review", {"experiment_id": SY.EXP, "registration_sha": reg["sha256"], "reviewer": reviewer, "result": result, "findings_sha": sha("findings")}, reviewer if "2026" in reviewer else PROD, DATE, "registered")

    def grant(self, action, reg, producer=JON, evidence="registered", **kw):
        b = {"action_sha": G.action_sha(action), "registration_sha": reg["sha256"], "granted_by": "Jonathan", "quote": "go", "granted": "2026-11-07", "expires": "2026-11-14", "cap_l4_minutes": 200.0, "cap_usd": 3.0}
        b.update(kw)
        return self.led.append("authorisation", b, producer, DATE, evidence)

    def ready(self, **kw):
        r = self.reg()
        self.review(r)
        a = self.action()
        self.grant(a, r, **kw)
        return a

    def denied(self, action, today="2026-11-08"):
        with self.assertRaises(G.Denied) as cm:
            G.check(action, self.led.load(), today)
        return str(cm.exception)

    def test_empty_ledger_denies(self):
        self.assertIn("no registration", self.denied(self.action()))

    def test_draft_denies(self):
        self.reg(status="draft")
        self.assertIn("draft", self.denied(self.action()))

    def test_synthetic_registration_denies(self):
        self.reg(evidence="synthetic")
        self.assertIn("draft", self.denied(self.action()))

    def test_no_review_denies(self):
        self.reg()
        self.assertIn("review", self.denied(self.action()))

    def test_failed_review_denies(self):
        self.review(self.reg(), result="failed")
        self.assertIn("review", self.denied(self.action()))

    def test_author_cannot_review_itself(self):
        self.review(self.reg(), reviewer=PROD)
        self.assertIn("producer", self.denied(self.action()))

    def test_no_grant_denies(self):
        self.review(self.reg())
        self.assertIn("no authorisation", self.denied(self.action()))

    def test_full_chain_permits_and_executes_nothing(self):
        a = self.ready()
        p = G.check(a, self.led.load(), "2026-11-08")
        self.assertTrue(p["permit"])
        self.assertEqual(sorted(p), ["action_sha", "authorisation_seq", "note", "permit"])

    def test_changed_action_denies(self):
        self.ready()
        self.assertIn("no authorisation", self.denied(self.action(l4_minutes=171.0)))

    def test_only_jonathans_own_line_grants(self):
        r = self.reg()
        self.review(r)
        a = self.action()
        self.grant(a, r, producer=PROD)
        self.assertIn("own registered line", self.denied(a))

    def test_agent_grant_in_jonathans_name_is_denied(self):
        r = self.reg()
        self.review(r)
        a = self.action()
        self.grant(a, r, granted_by="Claude Opus 5.5")
        self.assertIn("own registered line", self.denied(a))

    def test_synthetic_grant_denies(self):
        r = self.reg()
        self.review(r)
        a = self.action()
        self.grant(a, r, evidence="synthetic")
        self.assertIn("own registered line", self.denied(a))

    def test_a_zero_cost_action_is_refused(self):
        with self.assertRaises(G.Denied):
            G.action_sha(self.action(usd=0.0))

    def test_grant_for_another_registration_denies(self):
        r = self.reg()
        self.review(r)
        a = self.action()
        self.grant(a, r, registration_sha=sha("another"))
        self.assertIn("another registration", self.denied(a))

    def test_expired_denies(self):
        a = self.ready(expires="2026-11-07")
        self.assertIn("out of date", self.denied(a))

    def test_cap_denies(self):
        a = self.ready(cap_usd=1.0)
        self.assertIn("caps", self.denied(a))

    def test_registered_budget_denies(self):
        r = self.reg()
        self.review(r)
        a = self.action(l4_minutes=401.0, usd=2.0)
        self.grant(a, r, cap_l4_minutes=500.0)
        self.assertIn("budget", self.denied(a))

    def test_consume_is_single_use(self):
        a = self.ready()
        p = G.consume(a, self.led, "2026-11-08", PROD, DATE)
        self.assertTrue(p["permit"])
        self.assertIn("consumed", self.denied(a))
        with self.assertRaises(G.Denied):
            G.consume(a, self.led, "2026-11-08", PROD, DATE)

    def test_no_override_in_the_source(self):
        code = (HERE / "gate.py").read_text(encoding="utf-8").split('"""', 2)[2]
        for word in ("environ", "getenv", "force", "override"):
            self.assertNotIn(word, code)


FORBIDDEN_IMPORTS = {"subprocess", "socket", "urllib", "http", "requests", "ftplib", "paramiko", "asyncio", "webbrowser", "smtplib", "multiprocessing", "ctypes", "importlib", "pty", "telnetlib"}
FORBIDDEN_CALLS = {"system", "popen", "spawn", "spawnl", "spawnv", "execv", "execvp", "startfile", "fork", "__import__"}
EXEC_ALLOWED = {"adapters.py"}   # load_frozen: one exec of a hash-pinned frozen file


class TestNoLauncher(unittest.TestCase):
    def sources(self):
        return [p for p in HERE.glob("*.py") if p.name != "test_lab.py"] + list((HERE / "examples").glob("*.py"))

    def test_no_process_or_network_module_is_imported(self):
        for p in self.sources():
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]] + ([a.name for a in node.names] if node.module == "os" else [])
                self.assertFalse(FORBIDDEN_IMPORTS & set(names), f"{p.name} imports {names}")
                self.assertFalse(FORBIDDEN_CALLS & set(names), f"{p.name} imports {names}")

    def test_no_shell_or_dynamic_call(self):
        for p in self.sources():
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    self.assertNotIn(name, FORBIDDEN_CALLS, p.name)
                    if name in ("exec", "eval"):
                        self.assertIn(p.name, EXEC_ALLOWED, f"{p.name} calls {name}")

    def test_the_one_exec_is_behind_the_pin(self):
        src = (HERE / "adapters.py").read_text(encoding="utf-8")
        self.assertLess(src.index("is not at its pin"), src.index("exec(compile"))

    def test_no_provider_is_named(self):
        for p in self.sources():
            t = p.read_text(encoding="utf-8").lower()
            for word in ("colab_driver.json", "runner_multi", "runpod_api", "colab new", "generativeai"):
                self.assertNotIn(word, t, p.name)


class TestHistory(unittest.TestCase):
    def snap(self):
        fams = ["a", "b", "tail:human"]
        loss = {"a": {"P": 3.0, "S23": 3.0, "S30": 2.9, "S34": 0.3, "S51": 0.4, "S52": 0.3}, "b": {"P": 3.0, "S23": 3.0, "S30": 2.0, "S34": 2.9, "S51": 0.4, "S52": 3.0}, "tail:human": {"P": 2.8, "S23": 2.3, "S30": 2.3, "S34": 2.3, "S51": 2.4, "S52": 2.3}}
        se = {f: {m: 0.05 for m in ("S23", "S30", "S34", "S51", "S52")} for f in fams}
        return {"banner": H.BANNER, "families": fams, "per_family_rows": 24, "loss": loss, "gain_se": se, "s51_minus_s52": {"a": [-0.1, -0.2, 0.0], "b": [2.6, 2.4, 2.8], "tail:human": [-0.1, -0.2, 0.0]},
                "leg_dose": {"M1_on_S34": {"a": 10, "b": 90, "tail:human": 300}, "C0_on_S34": {"a": 90, "b": 0, "tail:human": 900}}, "leg_supervised_total": {"M1_on_S34": 100, "C0_on_S34": 178},
                "outcomes": {"j1_relevance": {"S23": 2.9, "S30": 2.5, "S34": 2.1, "S51": 2.0, "S52": 2.1}, "told_of_108": {"S30": 31, "S34": 28, "S51": 37, "S52": 30}, "analytic_v2_task": {"S23": 0.3, "S30": 0.2, "S34": 0.5, "S51": 0.3, "S52": 0.5},
                             "analytic_v2_small_talk": {"S23": 0.8, "S30": 0.7, "S34": 0.6, "S51": 0.4, "S52": 0.5}, "a19_verdict": "guard", "a19_reach_gain_items": 7, "a19_reach_bar": 13}}

    def test_says_observational(self):
        s = self.snap()
        self.assertIn("OBSERVATIONAL", H.replay(s)["banner"])
        self.assertIn("OBSERVATIONAL", H.signal_check(s)["banner"])
        self.assertEqual(H.replay(s)["evidence"], "observational")

    def test_floors_use_no_later_model(self):
        r = H.replay(self.snap())
        self.assertEqual(r["floors_then"]["b"], 2.0)
        self.assertIn("b", r["families_pushed"])

    def test_spearman(self):
        self.assertEqual(H.spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(H.spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_receipt_import_keeps_numbers_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "result_X1.json").write_text(json.dumps({"T": 1024, "wall_seconds": 10.0, "name": "not kept", "note": "not kept"}))
            (p / "home.json").write_text(json.dumps({"X1": sha("gguf")}))
            b = H.observation_from_receipt(p, "X1", "b")
            self.assertNotIn("note", b["data"])
            self.assertEqual(b["data"]["T"], 1024)


class TestNoWording(unittest.TestCase):
    def test_committed_json_has_no_long_text_outside_known_fields(self):
        p = HERE / "examples" / "history_observational_v1.json"
        if not p.exists():
            self.skipTest("no snapshot")
        d = json.loads(p.read_text(encoding="utf-8"))
        strings = []

        def walk(x):
            if isinstance(x, str):
                strings.append(x)
            elif isinstance(x, dict):
                for k, v in x.items():
                    if k not in ("banner", "producer"):
                        walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(d)
        self.assertTrue(all(len(s) < 90 for s in strings), [s for s in strings if len(s) >= 90][:2])


@unittest.skipUnless(os.environ.get("LAB_SCREENS_DIR") or Path("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude").is_dir(), "no workspace root on this machine")
class TestFrozenAdapters(unittest.TestCase):
    def test_pins_hold(self):
        import adapters as AD
        self.assertEqual(len(AD.verify_pins()), 5)

    def test_changed_file_refused(self):
        import adapters as AD
        with tempfile.TemporaryDirectory() as d:
            q = Path(d) / "analysis" / "correction_base_receipts"
            q.mkdir(parents=True)
            (q / "mine_score.py").write_text("x = 1")
            with self.assertRaises(AD.FrozenChanged):
                AD.load_frozen("analysis/correction_base_receipts/mine_score.py", d)

    def test_every_mine_label_is_mapped(self):
        import adapters as AD
        m = AD.mine_score()
        src = Path(m.__file__).read_text(encoding="utf-8")
        labels = set(re.findall(r'"([A-Z][A-Z_]{3,})"', src)) & {"VALUE", "WRONG_VALUE", "LISTS_OTHERS", "ABSTAIN", "OTHER", "PARTIAL", "DENY", "ABST_WITH_VALUE", "INVENT_OR_OTHER", "NO_VALUES", "UNSCORED"}
        self.assertGreaterEqual(len(labels), 10)
        ops = {"recall": ("VALUE", "WRONG_VALUE", "LISTS_OTHERS", "ABSTAIN", "OTHER"), "like_recall": ("VALUE", "PARTIAL", "NO_VALUES"), "like_affirm": ("DENY", "OTHER"), "abstain": ("ABSTAIN", "INVENT_OR_OTHER", "ABST_WITH_VALUE"), "self_name": ("OTHER",)}
        seen = set()
        for op, ls in ops.items():
            for la in ls:
                T.from_mine_label(op, la)
                seen.add(la)
        self.assertEqual(labels - seen - {"UNSCORED"}, set())

    def test_importing_the_scorer_starts_no_analysis(self):
        import adapters as AD
        self.assertTrue(callable(AD.mine_score().score))

    def test_the_five_frozen_shortcuts_exist_in_the_pinned_scorer(self):
        import adapters as AD
        names = set(AD.mine_score().POLICIES)
        self.assertTrue(set(SH.FROM_MINE_SCORE.values()) <= names)


if __name__ == "__main__":
    unittest.main()
