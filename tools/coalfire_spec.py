#!/usr/bin/env python3
"""paperkiln spec.json -> coalfire.cpp transformer_config.json (workstream C2).

The point of this tool is NOT convenience. It is that a cross-engine
comparison is only evidence if the two configurations mean the same
thing, and the two engines DO NOT use the same vocabulary for the same
concepts. A hand-mapped config produces runs that differ for reasons
nobody wrote down, and a difference nobody wrote down gets attributed
to whatever the experiment was nominally about. That is the same
failure shape as the SRD confound: an artifact read as a result.

So this translator REFUSES rather than approximates. Every knob lands
in exactly one bucket, and the buckets are printed as a parity report
that is meant to be committed alongside any cross-engine claim:

  MATCHED    - same concept, same units, copied across
  CONVERTED  - same concept, DIFFERENT units; formula shown
  REFUSED    - no faithful equivalent; translation aborts
  UNPAIRED   - exists on one side only, with a default that changes
               the comparison; listed so it can never be silent

    python tools/coalfire_spec.py spec.json --out cf_config.json
    python tools/coalfire_spec.py spec.json --report-only
    python tools/coalfire_spec.py --selftest

KNOWN SEMANTIC TRAPS THIS TOOL EXISTS TO CATCH (verified 2026-08-06
against coalfire src/attention.cpp:593 and config/transformer_config.json):

1. WINDOW UNITS. coalfire masks with `(i - j) > window_size / 2`, so
   its visible span is [i - window_size/2, i] -- window_size/2 + 1
   tokens. microtorch's `window` IS the visible token count, span
   [i - window + 1, i]. Equal visibility therefore requires
   `window_size = 2 * (window - 1)`, NOT `window_size = window`.
   Mapping them 1:1 silently halves coalfire's context.

2. ATTENTION SINKS. microtorch's swa keeps the first `sinks` tokens
   globally visible (StreamingLLM-style). coalfire has NO sink
   support (grep 'sink' over src/ include/ returns nothing). A
   window+sink lane and a window-only lane are DIFFERENT MECHANISMS,
   so any spec with sinks > 0 is refused outright.

3. TRAINING REGIME. coalfire trains in EPOCHS with a warmup/decay LR
   schedule, dropout 0.1 and fp16. microtorch runs a fixed STEP count
   at flat LR, no dropout, fp32. These are refused or reported as
   UNPAIRED rather than quietly bridged, because each one alone can
   move val loss by more than the effects we are measuring.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

# --------------------------------------------------------------------------
# The mapping table. Each entry: paperkiln path -> (coalfire path, kind,
# converter, note). kind is "match" or "convert".
DIRECT = [
    ("arch.custom.d", "model.hidden_size", "match", None,
     "embedding width"),
    ("arch.custom.layers", "model.num_layers", "match", None,
     "block count"),
    ("arch.custom.heads", "model.num_heads", "match", None,
     "attention heads"),
    ("arch.custom.d_ff", "model.intermediate_size", "match", None,
     "FFN width"),
    ("data.T", "model.max_seq_length", "match", None,
     "sequence length"),
]

# Knobs with no faithful coalfire equivalent. Presence of the paperkiln
# value listed (or any non-default value) aborts translation.
REFUSALS = {
    "arch.custom.residual": (
        lambda v: v not in (None, "", "residual"),
        "coalfire's residual stream is fixed; highway/plain variants are "
        "microtorch flex-family features (registry #0001) with no coalfire "
        "equivalent"),
    "arch.custom.sinks": (
        lambda v: v not in (None, 0),
        "coalfire has NO attention-sink support (verified: no 'sink' symbol "
        "in src/ or include/). window+sink and window-only are different "
        "mechanisms, not different parameters. Set sinks=0 to compare "
        "window-only lanes across engines, and register the sink lane as "
        "microtorch-only."),
    "arch.custom.attention": (
        lambda v: v not in (None, "exact", "swa"),
        "only 'exact' and 'swa' exist on both engines. kimi / srd / attnres "
        "are microtorch-only mechanisms with no coalfire counterpart."),
    "train.optimizer": (
        lambda v: v not in (None, "adamw", "adam"),
        "coalfire's trainer is Adam-family; muon has no counterpart there."),
}

# Concepts that exist on ONE side with a default that materially changes
# the comparison. Never silent: always reported.
UNPAIRED = [
    ("training.dropout_rate", 0.1,
     "coalfire applies dropout 0.1 by default; microtorch's parity models "
     "use none. Set to 0.0 for a comparable run."),
    ("optimization.use_fp16", True,
     "coalfire defaults to fp16; microtorch is fp32 throughout. Mixed "
     "precision alone can move val loss by more than the effects measured "
     "in the S1 cells."),
    ("training.learning_rate", "schedule",
     "coalfire uses warmup + decay (warmup_steps/peak_lr/decay_factor); "
     "microtorch's spec carries a FLAT lr. A flat-lr spec cannot be "
     "expressed as a schedule without choosing warmup behaviour, which "
     "changes the run. Set warmup_steps=0 and peak_lr=initial_lr=min_lr "
     "for the closest match, and record that you did."),
    ("training.num_epochs", 10,
     "coalfire counts EPOCHS; microtorch counts STEPS. The two coincide "
     "only if tokens-per-epoch is known and fixed; compute it explicitly "
     "and record the arithmetic rather than assuming."),
    ("optimization.gradient_accumulation_steps", 8,
     "coalfire accumulates 8 micro-batches by default; microtorch's "
     "train.accum defaults to 1. Effective batch differs by 8x unless set."),
]


def get_path(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def set_path(d, path, value):
    cur = d
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def window_to_coalfire(window):
    """microtorch `window` (visible token count) -> coalfire `window_size`.

    coalfire visible span is [i - window_size/2, i], i.e.
    window_size//2 + 1 tokens. Solving window_size//2 + 1 == window
    gives window_size = 2*(window - 1).
    """
    return 2 * (int(window) - 1)


def coalfire_visible(window_size):
    """Inverse: how many tokens a coalfire window_size actually sees."""
    return window_size // 2 + 1


def translate(spec, base_config=None):
    """Returns (config_or_None, report). report is a list of
    (bucket, detail) tuples. config is None iff any REFUSED entry fired."""
    report = []
    cfg = copy.deepcopy(base_config) if base_config else {}

    # --- refusals first: never emit a partially-valid config ---
    refused = False
    for path, (trips, why) in REFUSALS.items():
        val = get_path(spec, path)
        if trips(val):
            report.append(("REFUSED", f"{path} = {val!r}: {why}"))
            refused = True

    # --- preset-only specs cannot be translated ---
    # mtstudio resolves arch.preset to dims SERVER-SIDE, so a spec carrying
    # only a preset name has no dims for this tool to read. Guessing them
    # from a local copy of the preset table is exactly the kind of silent
    # drift this translator exists to prevent (the table changes; the guess
    # would not).
    if get_path(spec, "arch.preset") and get_path(spec, "arch.custom.d") is None:
        report.append((
            "REFUSED",
            f"arch.preset = {get_path(spec, 'arch.preset')!r} with no explicit "
            "arch.custom.d: preset dims are resolved server-side by mtstudio "
            "and are not readable here. Resolve the preset first (run once "
            "and read the 'model' event, which records d/layers/heads/d_ff) "
            "and pass those explicitly."))
        refused = True

    # --- direct + converted ---
    for pk_path, cf_path, kind, _conv, note in DIRECT:
        val = get_path(spec, pk_path)
        if val is None:
            continue
        set_path(cfg, cf_path, val)
        report.append(("MATCHED", f"{pk_path} -> {cf_path} = {val}  ({note})"))

    # head_dim is derived on the coalfire side and must stay consistent
    d, heads = get_path(spec, "arch.custom.d"), get_path(spec, "arch.custom.heads")
    if d and heads:
        if d % heads:
            report.append(("REFUSED",
                           f"d={d} is not divisible by heads={heads}"))
            refused = True
        else:
            set_path(cfg, "model.head_dim", d // heads)
            report.append(("CONVERTED",
                           f"model.head_dim = d/heads = {d}//{heads} = {d//heads}"))

    # --- the window trap ---
    attention = get_path(spec, "arch.custom.attention")
    window = get_path(spec, "arch.custom.window")
    if attention == "swa":
        if window is None:
            report.append(("REFUSED", "attention='swa' with no window set"))
            refused = True
        else:
            cf_ws = window_to_coalfire(window)
            set_path(cfg, "attention.use_sliding_window", True)
            set_path(cfg, "attention.window_size", cf_ws)
            report.append((
                "CONVERTED",
                f"arch.custom.window = {window} (visible tokens) -> "
                f"attention.window_size = 2*({window}-1) = {cf_ws}, which "
                f"coalfire reads as visible span {cf_ws}//2+1 = "
                f"{coalfire_visible(cf_ws)} tokens. A 1:1 mapping here would "
                f"have given {coalfire_visible(window)} -- silently halving "
                f"the context."))
    elif attention == "exact":
        set_path(cfg, "attention.use_sliding_window", False)
        report.append(("MATCHED",
                       "attention='exact' -> use_sliding_window = False"))

    # --- unpaired concepts, always reported ---
    for cf_path, default, why in UNPAIRED:
        report.append(("UNPAIRED", f"{cf_path} (coalfire default "
                                   f"{default!r}): {why}"))

    return (None if refused else cfg), report


def format_report(report):
    order = {"REFUSED": 0, "CONVERTED": 1, "MATCHED": 2, "UNPAIRED": 3}
    lines = ["=" * 72, "CROSS-ENGINE PARITY REPORT  (paperkiln -> coalfire.cpp)",
             "=" * 72]
    for bucket in sorted({b for b, _ in report}, key=lambda b: order.get(b, 9)):
        lines.append(f"\n[{bucket}]")
        for b, detail in report:
            if b == bucket:
                lines.append(f"  - {detail}")
    n_ref = sum(1 for b, _ in report if b == "REFUSED")
    lines.append("")
    lines.append("=" * 72)
    lines.append("TRANSLATION ABORTED — see [REFUSED]" if n_ref else
                 "Translation OK. [UNPAIRED] items are NOT resolved by this "
                 "tool and must be set deliberately before any cross-engine "
                 "claim is registered.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def selftest():
    # 1. the window conversion round-trips to equal visibility
    for w in (16, 32, 64, 128, 256):
        ws = window_to_coalfire(w)
        assert coalfire_visible(ws) == w, (w, ws, coalfire_visible(ws))
    # and the naive mapping is genuinely wrong (the trap is real)
    assert coalfire_visible(64) == 33 != 64

    spec = {"arch": {"custom": {"attention": "swa", "window": 64, "sinks": 0,
                                "d": 128, "layers": 2, "heads": 4}},
            "data": {"T": 256}, "train": {"optimizer": "adamw"}}
    cfg, rep = translate(spec)
    assert cfg is not None, format_report(rep)
    assert cfg["attention"]["window_size"] == 126
    assert coalfire_visible(cfg["attention"]["window_size"]) == 64
    assert cfg["model"]["head_dim"] == 32
    assert cfg["model"]["max_seq_length"] == 256

    # 2. sinks are refused (different mechanism, not a parameter)
    spec_sink = copy.deepcopy(spec)
    spec_sink["arch"]["custom"]["sinks"] = 1
    cfg2, rep2 = translate(spec_sink)
    assert cfg2 is None
    assert any(b == "REFUSED" and "sink" in d.lower() for b, d in rep2)

    # 3. microtorch-only mechanisms are refused
    for mech in ("kimi", "srd", "attnres"):
        s = copy.deepcopy(spec)
        s["arch"]["custom"]["attention"] = mech
        c, r = translate(s)
        assert c is None, mech

    # 4. muon is refused
    s = copy.deepcopy(spec)
    s["train"]["optimizer"] = "muon"
    assert translate(s)[0] is None

    # 5. indivisible d/heads is refused
    s = copy.deepcopy(spec)
    s["arch"]["custom"]["heads"] = 5
    assert translate(s)[0] is None

    # 6. exact maps to window off
    s = copy.deepcopy(spec)
    s["arch"]["custom"]["attention"] = "exact"
    c, _ = translate(s)
    assert c["attention"]["use_sliding_window"] is False

    # 7. unpaired items are ALWAYS reported, even on a clean translation
    _, rep_clean = translate(spec)
    unpaired = [d for b, d in rep_clean if b == "UNPAIRED"]
    assert len(unpaired) == len(UNPAIRED)
    assert any("fp16" in d for d in unpaired)
    assert any("EPOCHS" in d for d in unpaired)

    print("SELFTEST OK: window unit conversion (5 scales, naive mapping shown "
          "wrong), sink refusal, mechanism refusals, optimizer refusal, "
          "head-divisibility refusal, exact mapping, unpaired always reported")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?", help="paperkiln spec.json")
    ap.add_argument("--out", help="write translated coalfire config here")
    ap.add_argument("--base", help="coalfire config to use as the base "
                                   "(defaults left untouched where unmapped)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return 0
    if not a.spec:
        ap.error("spec is required (or --selftest)")

    spec = json.load(open(a.spec, encoding="utf-8"))
    base = json.load(open(a.base, encoding="utf-8")) if a.base else None
    cfg, report = translate(spec, base)
    print(format_report(report))
    if cfg is None:
        return 2
    if a.out and not a.report_only:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
