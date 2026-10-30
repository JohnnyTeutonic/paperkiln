"""Prospective, local-only S-full/HS0 three-seed finalizer; never rents compute or launches follow-ups."""

from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import finalize_core as common
import prepare_sfull as prep

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROOF = HERE / "receipts/sfull_prospective_20261031"
MANIFEST = PROOF / "manifest.json"
LAUNCH = PROOF / "launch_pin.json"
TRIGGER = PROOF / "trigger_hs1_five_seed.json"
REFERENCE = PROOF / "data_reference.json"
REPORT = "analysis_hs0.json"
SEEDS = [1, 2, 3]
CELLS = ["B", "Sgist", "Sfull"]
SCOPE = "REGISTERED CONDITIONAL HS0 ONLY; exact initial seeds1-3; five-seed HS1 trigger preserved"


def sources():
    m = common.load_json(MANIFEST)
    return [
        Path(__file__),
        HERE / "prepare_sfull.py",
        HERE / "finalize_core.py",
        HERE / "analyze.py",
        HERE / "PREREGISTRATION.md",
        HERE.parent / "tiny_chat_2026-10/analyze.py",
        HERE.parent / "tiny_chat_2026-10/analyze_x5.py",
        common.CPU_BUDGET,
        MANIFEST,
        LAUNCH,
        REPO / "corpus/eval/carry12_v1.jsonl",
        *[REPO / p for p in m["artifacts_sha256"]],
    ]


def parity(row):
    g = row["carry12"]["gate"]
    n, p = g.get("replies"), g.get("argmax_identical")
    if type(n) is not int or type(p) is not int or not 0 <= p <= n or n <= 0:
        raise ValueError("invalid full-history gate inventory")
    return {"identical": p, "replies": n, "nonidentical": n - p, "all_pass": n == p}


def calculate(art, out):
    a = common.load_module("sfull_registered", HERE / "analyze.py")
    runs = {}
    for c in CELLS:
        for s in SEEDS:
            runs[c, s] = common.load_json(
                Path(art) / f"c12_{c}_r{s}" / f"post_c12_{c}_r{s}/summary.json"
            )
    floors = {c: a.floor_ok(runs, c, SEEDS) for c in CELLS}
    hs0 = (
        a.decide(a.paired(runs, "Sfull", "B", SEEDS, "Y12"), a.SESOI, +1)
        if floors["B"] and floors["Sfull"]
        else {"verdict": "FLOOR", "cells": floors}
    )
    mask = (
        a.decide(a.paired(runs, "Sgist", "Sfull", SEEDS, "Y12"), a.SESOI, +1)
        if floors["Sgist"] and floors["Sfull"]
        else {"verdict": "FLOOR", "cells": floors}
    )
    gates = {f"{c}{s}": parity(runs[c, s]) for c in CELLS for s in SEEDS}
    historical = common.load_json(PROOF / "historical_history_parity.json")
    common.atomic_json(
        out,
        {
            "scope": SCOPE,
            "partial": False,
            "cells": CELLS,
            "seeds": SEEDS,
            "missing": [],
            "validity_gate": floors,
            "cell_means": a.cell_table(runs, CELLS, SEEDS),
            "HS0_Sfull_minus_B": {**hs0, "prediction": "EQUIVALENT within3points"},
            "mask_cost_Sgist_minus_Sfull": {
                **mask,
                "label": "REPORTED BESIDE HS0; unchanged registered helper direction",
            },
            "five_seed_HS1_trigger_sha256": common.sha(TRIGGER),
            "global_core_unchanged": True,
            "five_seed_HS1_unchanged": True,
            "full_history_parity": gates,
            "all_history_checks_pass": all(g["all_pass"] for g in gates.values()),
            "historical_HS1_nonidentical": len(historical["nonidentical"]),
            "interpretation_requires_parity_review": bool(historical["nonidentical"])
            or not all(g["all_pass"] for g in gates.values()),
            "no_automatic_followup_launch": True,
            "primary_extension_flag_is_report_only": True,
        },
    )


class Finalizer(common.Finalizer):
    def __init__(
        self,
        art,
        state_dir,
        output=None,
        max_hours=96,
        campaign_state_root="/home/jonat",
        source_freeze=None,
    ):
        super().__init__(art, state_dir, output, max_hours)
        self.campaign_state_root = Path(campaign_state_root).resolve()
        self.source_freeze = Path(source_freeze).resolve() if source_freeze else None

    def state(self):
        actual = {str(p.resolve()): common.sha(p) for p in sources()}
        if self.source_freeze is not None:
            frozen = common.load_json(self.source_freeze)
            if frozen.get("source_hashes") != actual:
                raise ValueError("prospective finalizer source freeze differs")
            actual[str(self.source_freeze)] = common.sha(self.source_freeze)
        if self.state_path.exists():
            s = super().state()
            if s.get("cpu_budget") != str(common.CPU_BUDGET.resolve()):
                raise ValueError("CPU budget wrapper path changed")
            if s["campaign_state_root"] != str(self.campaign_state_root):
                raise ValueError("campaign-state root changed")
            return s
        s = {
            "schema": 1,
            "created": time.time(),
            "status": "WAITING",
            "art": str(self.art),
            "output": str(self.output),
            "campaign_state_root": str(self.campaign_state_root),
            "source_hashes": actual,
            "funding_or_parity_approval_not_inferred": True,
            "cpu_budget": str(common.CPU_BUDGET.resolve()),
        }
        common.atomic_json(self.state_path, s)
        return s

    def inputs(self):
        prep.trigger(TRIGGER.read_bytes())
        m = common.load_json(MANIFEST)
        pin = common.load_json(LAUNCH)
        if pin["parent"] != prep.PIN or pin["manifest_sha256"] != common.sha(MANIFEST):
            raise ValueError("launch manifest binding differs")
        for path, h in m["artifacts_sha256"].items():
            if common.sha(REPO / path) != h:
                raise ValueError("packet artifact changed:" + path)
        pending = []
        files = {}
        pairs = []

        def add(p):
            p = Path(p)
            if not p.is_file():
                pending.append(str(p))
                return None
            files[str(p)] = common.sha(p)
            return common.load_json(p)

        for path, h in m["baseline_artifact_sha256"].items():
            p = self.art / path
            if add(p) is not None and files[str(p)] != h:
                raise ValueError("audited baseline changed:" + path)
        wanted = {(c, s) for c in CELLS for s in SEEDS}
        found = {}
        for p in self.art.glob("*/post_c12_*/summary.json"):
            r = common.load_json(p)
            key = (r.get("cell"), r.get("seed"))
            if r.get("cell") == "Sfull" and (
                type(r.get("seed")) is not int or r["seed"] not in SEEDS
            ):
                raise ValueError("unexpected Sfull seed")
            if key in wanted:
                if key in found:
                    raise ValueError("duplicate selected summary:" + str(key))
                found[key] = p
        ref = common.load_json(REFERENCE)
        for seed in SEEDS:
            base = self.art / f"c12_Sfull_r{seed}"
            p = base / f"post_c12_Sfull_r{seed}/summary.json"
            r = add(p)
            if r is not None:
                if (
                    r.get("cell") != "Sfull"
                    or type(r.get("seed")) is not int
                    or r["seed"] != seed
                    or r.get("mode") != "post"
                    or type(r.get("Y12")) not in (int, float)
                    or not math.isfinite(r["Y12"])
                    or r.get("provenance", {}).get("paperkiln_commit") != pin["commit"]
                    or r.get("arch")
                    != {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240}
                ):
                    raise ValueError("Sfull identity/shape/source differs")
                for key, steps in [("pt_result", 10000), ("sft_result", 1500)]:
                    v = r.get(key, {})
                    if any(
                        v.get(k) != x
                        for k, x in [
                            ("T", 1024),
                            ("batch", 16),
                            ("accum", 1),
                            ("seed", seed),
                            ("final_step", steps),
                            ("steps_requested", steps),
                            ("early_stopped", False),
                        ]
                    ):
                        raise ValueError("training budget differs:" + key)
                c = r.get("carry12", {})
                meta = c.get("meta", {})
                engine = c.get("gen_engine", {})
                if (
                    c.get("context") != 1024
                    or meta.get("items") != 1452
                    or meta.get("max_new") != 48
                    or meta.get("self_scope") != "target"
                    or meta.get("near_self") is not False
                    or meta.get("scorer") != "mtstudio"
                    or meta.get("spot") != 20
                    or meta.get("sha256")
                    != "de24f2c726a1006e8e61542dd3b2dfbf8677b9e6fe689277cca035305bd32797"
                    or "error" in c
                ):
                    raise ValueError("Carry12 configuration differs/incomplete")
                if (
                    engine.get("engine") not in ("ember", "mtstudio")
                    or (
                        engine["engine"] == "ember"
                        and (engine.get("spot") or {}).get("pass") is not True
                    )
                    or (engine["engine"] == "mtstudio" and not engine.get("flag"))
                ):
                    raise ValueError("unchecked generation engine")
                if not parity(r)["all_pass"]:
                    raise ValueError(
                        "new Sfull full-history parity failed; no automatic retry"
                    )
                pairs.append(["Sfull", seed])
            raw_path = base / f"post_c12_Sfull_r{seed}/carry12/m/items.jsonl"
            gen_path = base / f"post_c12_Sfull_r{seed}/carry12/m/gen.jsonl"
            corpus = [
                json.loads(x)
                for x in (REPO / "corpus/eval/carry12_v1.jsonl")
                .read_text()
                .splitlines()
            ]
            expected_ids = {x["id"] for x in corpus}
            expected_targets = {x["id"]: x["target_turn"] for x in corpus}
            for path in (raw_path, gen_path):
                if not path.is_file():
                    pending.append(str(path))
                    continue
                files[str(path)] = common.sha(path)
                rows = [json.loads(x) for x in path.read_text().splitlines()]
                if (
                    len(rows) != 1452
                    or len({x["id"] for x in rows}) != 1452
                    or {x["id"] for x in rows} != expected_ids
                ):
                    raise ValueError("raw generation/item inventory differs")
                if path == raw_path:
                    flags = [v for x in rows for v in x["gate"]]
                    if (
                        len(flags) != 17424
                        or any(
                            len(x.get("gate", [])) != expected_targets[x["id"]]
                            for x in rows
                        )
                        or not all(v is True for v in flags)
                        or any(x.get("server_errors") != 0 for x in rows)
                    ):
                        raise ValueError(
                            "new Sfull raw full-history parity failed; no automatic retry"
                        )
                    if r is not None and r["carry12"]["gate"] != {
                        "replies": 17424,
                        "argmax_identical": 17424,
                    }:
                        raise ValueError("summary/raw history gate count differs")
                elif any(
                    x.get("self_scope") != "target"
                    or x.get("near_self")
                    or len(x.get("self", [])) != expected_targets[x["id"]]
                    for x in rows
                ):
                    raise ValueError("raw generation scope differs")
            ledger = add(base / "data/data.ledger.json")
            if ledger is not None and common.data_ledger(ledger) != common.data_ledger(
                ref
            ):
                raise ValueError("Sfull data ledger differs")
            for path in [base / "gate/gate.json", base / "cuda_gate/cuda_gate.json"]:
                g = add(path)
                if g is not None and (
                    g.get("ok") is not True
                    or (
                        g.get("failed")
                        if path.parent.name == "cuda_gate"
                        else g.get("problems")
                    )
                    != []
                ):
                    raise ValueError("execution/CUDA gate failed")
            g = add(base / "sfull_inputs/verified_inputs.json")
            if g is not None and (
                g.get("verified") is not True
                or g.get("reference_sha256") != prep.REFERENCE_SHA
                or g.get("outputs_sha256")
                != {k: v["sha256"] for k, v in ref["outputs"].items()}
                or g.get("no_data_mutation") is not True
                or g.get("vocab_sha256") != ref["vocab_sha256"]
                or g.get("raw_human_sources_verified") != len(ref["human"]["files"])
                or g.get("evaluation_sources_verified") != 2
            ):
                raise ValueError("missing exact pre-training data proof")
            for path in [
                base / f"pt_c12_Sfull_r{seed}/pt_c12_Sfull_r{seed}_spec.json",
                base / f"post_c12_Sfull_r{seed}/sft_spec.json",
            ]:
                spec = add(path)
                if spec is not None and (
                    spec.get("arch", {}).get("state_tokens")
                    != {"k": 4, "mask": "causal"}
                    or spec.get("arch", {}).get("custom")
                    != {"d": 768, "layers": 8, "heads": 12, "d_ff": 2240}
                ):
                    raise ValueError("runtime Sfull mask/shape differs")
            home = add(
                self.campaign_state_root / f"pk_c12_sfull_r{seed}_runner/state.json"
            )
            if home is not None:
                if home.get("campaign_home") is not True:
                    pending.append("campaign not home:" + str(seed))
                elif (
                    not home.get("jobs")
                    or not all(
                        v.get("home") is True and v.get("status") == "FINISHED"
                        for v in home["jobs"].values()
                    )
                    or home.get("runpod", {}).get("pod")
                    or home.get("runpod", {}).get("volume")
                ):
                    raise ValueError("contradictory campaign-home proof")
        return {
            "ready": not pending,
            "pending": pending,
            "selected_new_pairs": pairs,
            "sha256": files,
        }

    @staticmethod
    def verify_outputs(folder):
        r = common.load_json(Path(folder) / REPORT)
        if (
            r.get("scope") != SCOPE
            or r.get("partial") is not False
            or r.get("seeds") != SEEDS
            or r.get("cells") != CELLS
            or r.get("missing") != []
            or r.get("five_seed_HS1_trigger_sha256") != prep.TRIGGER_SHA
            or r.get("no_automatic_followup_launch") is not True
            or any(type(s) is not int for s in r.get("seeds", []))
        ):
            raise ValueError("not complete scoped HS0 output")
        for key in ["HS0_Sfull_minus_B", "mask_cost_Sgist_minus_Sfull"]:
            if r[key]["verdict"] != "FLOOR" and (
                r[key].get("n") != 3 or len(r[key].get("per_seed", [])) != 3
            ):
                raise ValueError("HS0 requires exactly3pairs")
        if set(r.get("full_history_parity", {})) != {
            f"{c}{s}" for c in CELLS for s in SEEDS
        }:
            raise ValueError("missing parity disclosure")

    def tick(self):
        s = self.state()
        if s["status"] == "HALTED":
            return "HALTED"
        try:
            self.verify_sources(s)
            published = self.output / "reports"
            if published.exists():
                c = common.load_json(published / "completion.json")
                self.verify_outputs(published)
                if (
                    c["source_hashes"] != s["source_hashes"]
                    or c["input_manifest"] != self.inputs()
                ):
                    raise ValueError("published source/input drift")
                for n, h in c["report_hashes"].items():
                    if common.sha(published / n) != h:
                        raise ValueError("published report changed")
                s.update(status="COMPLETE", completed=c["completed"])
                common.atomic_json(self.state_path, s)
                return "COMPLETE"
            if time.time() - s["created"] > self.max_hours * 3600:
                raise ValueError("HS0 deadline exceeded")
            before = self.inputs()
            for p, h in s.get("observed_inputs", {}).items():
                if before["sha256"].get(p) != h:
                    raise ValueError("observed input changed:" + p)
            # Active runner state is mutable until home; do not latch a partial state as immutable evidence.
            s["observed_inputs"] = {
                p: h
                for p, h in before["sha256"].items()
                if not p.endswith("/state.json")
            }
            if not before["ready"]:
                s.update(
                    status="WAITING",
                    pending_count=len(before["pending"]),
                    checked=common.now(),
                )
                common.atomic_json(self.state_path, s)
                return "WAITING"
            if len(before["selected_new_pairs"]) != 3:
                raise ValueError("expected exactly3new summaries")
            s.update(status="ANALYSING")
            common.atomic_json(self.state_path, s)
            stage = self.output / ".incoming"
            stage.mkdir(exist_ok=True)
            common.atomic_json(stage / "input_receipt.json", before)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--calculate",
                "--art",
                str(self.art),
                "--out",
                str(stage / REPORT),
            ]
            common.atomic_json(stage / "commands.json", [cmd])
            wrapper = [
                sys.executable,
                str(common.CPU_BUDGET),
                "run",
                "--session",
                "paperkiln",
                "--job",
                "registered conditional HS0 analysis",
                "--cores",
                "1",
                "--resume",
                "existing Sfull finalizer watchdog entry",
                "--",
                *cmd,
            ]
            with (stage / "analysis.log").open("w") as log:
                subprocess.run(
                    wrapper,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
            self.verify_sources(s)
            self.verify_outputs(stage)
            if self.inputs() != before:
                raise ValueError("input changed during analysis")
            c = {
                "completed": common.now(),
                "source_hashes": s["source_hashes"],
                "input_manifest": before,
                "report_hashes": {
                    n: common.sha(stage / n)
                    for n in [REPORT, "input_receipt.json", "commands.json"]
                },
            }
            common.atomic_json(stage / "completion.json", c)
            stage.rename(published)
            s.update(status="COMPLETE", completed=c["completed"])
            common.atomic_json(self.state_path, s)
            return "COMPLETE"
        except (Exception, SystemExit) as e:
            s.update(status="HALTED", error=str(e), checked=common.now())
            common.atomic_json(self.state_path, s)
            (self.home / "HALTED").write_text(str(e) + "\n")
            self.log("HALTED:" + str(e))
            return "HALTED"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-budget", type=Path, default=common.CPU_BUDGET)
    p.add_argument("--art", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--state-dir")
    p.add_argument("--campaign-state-root", default="/home/jonat")
    p.add_argument("--source-freeze", type=Path, default=PROOF / "source_freeze.json")
    p.add_argument("--max-hours", type=float, default=96)
    p.add_argument("--calculate", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=240)
    a = p.parse_args()
    if a.calculate:
        calculate(a.art, a.out)
        return 0
    common.CPU_BUDGET = a.cpu_budget.resolve()
    if not common.CPU_BUDGET.is_file():
        p.error("--cpu-budget must name the workspace CPU budget wrapper")
    if not a.state_dir:
        p.error("--state-dir required")
    f = Finalizer(
        a.art, a.state_dir, a.out, a.max_hours, a.campaign_state_root, a.source_freeze
    )
    with common.DriverLock(f.home):
        while True:
            status = f.tick()
            if status == "COMPLETE":
                return 0
            if a.once:
                return 1 if status == "HALTED" else 2
            time.sleep(max(10, a.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
